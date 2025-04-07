import os
import json
import itertools
from typing import Dict, List, Tuple, Literal
from .memory import Memory
from .general_model import GeneralLawBase, GeneralLaw
from .specific_model import SpecificModel, ConservedInfo, ZeroInfo, CQCalculator
from .object_model import ObjectModel
from .regression import search_type, search_relations_ver1
from .interface import Knowledge
from .interface import (
    DataStruct, ExpStructure, MeasureType, Proposition,
    Exp, Concept, AtomExp, ExpData, Intrinsic
)
from .manager import Manager
from .parsing import extract_expression_pattern
from tqdm import tqdm
import random
import copy
import time
from multiprocessing import Queue, Process
import multiprocessing
import numpy as np
import sympy as sp
import torch


def run_method_in_process(instance: "Theorist",
                          method_name: str,
                          q: Queue,
                          shared_list: List,
                          *args):
    method = getattr(instance, method_name)
    try:
        reward_tot: int = method(*args)
    except Exception as e:
        print(f"Error occurred: {e}", flush=True)
        shared_list.append((None, 0))
        q.put(shared_list)
    else:
        shared_list.append((instance, reward_tot))
        q.put(shared_list)


class Theorist(Manager):
    """
    Theorist 类，由一个主要 Knowledge 类 `knowledge` 
    和一系列特殊 Model 类 `specific`,`objmodel` 组成，
    除此以外它还包括一个 Memory 类 `memory` 用于存储 AI 对各个动作的记忆
    （为了在接下来的探索中更高效率有更高回报地选择动作）。

    knowledge 是一个全局的知识库。
    general 代表了关于物理世界的“一般模型”。
    specific 代表了关于每一个实验的“具体模型”。
    objmodel 代表了关于每一个物体的模型（刻画了对物体的认识）。

    理论家的主要工作就是不断地做实验，然后从实验中发现一些规律，
    将这些规律注册到 knowledge 和 具体的 specific 中，
    如果这个过程中发现一些关于物理对象的具体知识，这个知识会被注册到 objmodel 中。

    memory 负责记忆 AI 对各个动作的“期待”（或者说“倾向性”）和“熟悉度”，
    它采用的是非平稳老虎机的一个变种公式来更新动作的期待值和熟悉值，来辅助 AI 对动作进行选择。
    """
    knowledge: Knowledge
    memory: Memory
    general: GeneralLawBase
    specific: Dict[str, SpecificModel]
    objmodel: Dict[str, ObjectModel]
    num_threads: int

    def __init__(self, init_time_limit: int | float | None = None, num_threads: int = 32):
        self.knowledge = Knowledge.default()
        self.memory = Memory(self.knowledge, init_time_limit)
        experiment_list = self.knowledge.fetch_exps
        self.specific = {}
        for name in experiment_list:
            self.specific[name] = SpecificModel(name, self.knowledge)
            self.memory.register_experiment(name)
            for concept in self.specific[name].experiment.original_concept:
                self.register_basic_concept(concept)
            self.specific[name].num_threads = num_threads
        self.objmodel = {}
        self.general = GeneralLawBase(self.knowledge, self.specific, self.memory, self.objmodel)
        self.num_threads = num_threads

    def __getstate__(self):
        s1 = self.knowledge.__getstate__()
        s2 = self.memory.__getstate__()
        s3 = self.general.to_json()
        s4 = {key: value.to_json() for key, value in self.specific.items()}
        s5 = {key: value.to_json() for key, value in self.objmodel.items()}
        return {
            "knowledge": s1,
            "memory": s2,
            "general": s3,
            "specific": s4,
            "objmodel": s5,
            "num_threads": self.num_threads
        }

    def __setstate__(self, state: dict[str, dict]):
        self.knowledge = object.__new__(Knowledge)
        self.knowledge.__setstate__(state["knowledge"])
        self.memory = object.__new__(Memory)
        self.memory.knowledge = self.knowledge
        self.memory.__setstate__(state["memory"])
        self.specific = {}
        for name in self.knowledge.fetch_exps:
            self.specific[name] = SpecificModel(name, self.knowledge, reset=False)
            self.specific[name].load_json(state["specific"][name])
        self.objmodel = {key: ObjectModel.from_json(value, self.knowledge)
                         for key, value in state["objmodel"].items()}
        self.general = GeneralLawBase.from_json(state["general"], self.knowledge,
                                                self.specific, self.memory, self.objmodel)
        self.num_threads = state.get("num_threads", 32)
        for spm in self.specific.values():
            spm.num_threads = self.num_threads

    @staticmethod
    def read_from_file(filename: str, reset_experiment: bool = True,
                       num_threads: int = 32) -> "Theorist":
        filename_for_knowledge = filename + "_knowledge.txt"
        filename_for_memory = filename + "_memory.json"
        filename_for_general_model = filename + "_general_model.json"
        filename_for_specific_model = filename + "_specific_model.json"
        filename_for_object_model = filename + "_object_model.json"
        file_name_for_policy_nn = filename + "_policy_nn.pth"

        obj = object.__new__(Theorist)

        if os.path.exists(filename_for_knowledge):
            obj.knowledge = Knowledge.read_from_file(filename_for_knowledge)
        else:
            print("Knowledge file not found. Use default knowledge.", flush=True)
            obj.knowledge = Knowledge.default()

        if os.path.exists(filename_for_memory):
            with open(filename_for_memory, "r") as f:
                memory_dict = json.load(f)
            obj.memory = Memory.from_json(memory_dict, obj.knowledge)
        else:
            print("Memory file not found. Use default memory.", flush=True)
            obj.memory = Memory(obj.knowledge)

        if os.path.exists(filename_for_specific_model):
            with open(filename_for_specific_model, "r") as f:
                specific_model_dict = json.load(f)
            obj.specific = {}
            for name in obj.knowledge.fetch_exps:
                obj.specific[name] = SpecificModel(name, obj.knowledge, reset=reset_experiment)
                obj.specific[name].load_json(specific_model_dict[name])
        else:
            print("Specific model file not found. Use default specific model.", flush=True)
            obj.specific = {}
            for name in obj.knowledge.fetch_exps:
                obj.specific[name] = SpecificModel(name, obj.knowledge, reset=reset_experiment)
                obj.memory.register_experiment(name)
                for concept in obj.specific[name].experiment.original_concept:
                    obj.register_basic_concept(concept)

        obj.objmodel = {}
        if os.path.exists(filename_for_object_model):
            with open(filename_for_object_model, "r") as f:
                objmodel_dict = json.load(f)
            for obj_type in objmodel_dict:
                obj.objmodel[obj_type] = ObjectModel.from_json(objmodel_dict[obj_type], obj.knowledge)

        if os.path.exists(filename_for_general_model):
            with open(filename_for_general_model, "r") as f:
                general_dict = json.load(f)
            obj.general = GeneralLawBase.from_json(
                general_dict, obj.knowledge, obj.specific, obj.memory, obj.objmodel)
        else:
            print("General law file not found. Use default general laws.", flush=True)
            obj.general = GeneralLawBase(obj.knowledge, obj.specific, obj.memory, obj.objmodel)

        if os.path.exists(file_name_for_policy_nn):
            obj.memory.action_bandit.policy_nn.load_state_dict(torch.load(file_name_for_policy_nn,
                                                                          weights_only=True))

        obj.num_threads = num_threads
        for spm in obj.specific.values():
            spm.num_threads = num_threads

        return obj

    def save_to_file(self, filename: str | None):
        if filename is None:
            return
        filename_for_knowledge = filename + "_knowledge.txt"
        filename_for_memory = filename + "_memory.json"
        filename_for_general_model = filename + "_general_model.json"
        filename_for_specific_model = filename + "_specific_model.json"
        filename_for_object_model = filename + "_object_model.json"
        self.knowledge.save_to_file(filename_for_knowledge)
        with open(filename_for_memory, "w") as f:
            json.dump(self.memory.to_json(), f, indent=4)
        with open(filename_for_general_model, "w") as f:
            json.dump(self.general.to_json(), f, indent=4)
        specific_dict = {
            key: value.to_json() for key, value in self.specific.items()
        }
        with open(filename_for_specific_model, "w") as f:
            json.dump(specific_dict, f, indent=4)
        object_dict = {
            key: value.to_json() for key, value in self.objmodel.items()
        }
        with open(filename_for_object_model, "w") as f:
            json.dump(object_dict, f, indent=4)

    def save_policy_nn(self, file_path: str | None):
        if file_path is None:
            return
        filename_for_policy_nn = file_path + "_policy_nn.pth"
        torch.save(self.memory.action_bandit.policy_nn.state_dict(), filename_for_policy_nn)

    def perform_discovery(self,
                          exp_select_mode: Literal["random", "ucb"] = "ucb",
                          cpt_lim_mul: float = 2.,
                          n0: int = None,
                          max_actions: int = 5,
                          max_cpt_lim: float = 240.,
                          cpt_lim_bias: float = 0.0,
                          max_epochs: int = 1000,
                          save_path: str = None):
        # Print the process id
        print(f"Process ID: {os.getpid()}", flush=True)

        ver = 'ver3'

        if n0 is None:  # NOTE: This may be modified by itself
            n0 = int(np.sqrt(len(self.knowledge.fetch_exps))) + 1
        # ucb_bound: float = 1 / np.sqrt(1.0 + n0)
        ucb_bound: float = 0.36
        print(f"initial ucb_bound: {ucb_bound}", flush=True)

        while not self.memory.experiments_pool:
            self.memory.experiments_pool = self.filter_feasible_experiments(ver,
                                                                            max_actions,
                                                                            max_cpt_lim,
                                                                            cpt_lim_bias)
            if not self.memory.experiments_pool:
                self.reset_time_limit(max_cpt_lim, cpt_lim_mul)
                print(' \033[1m' +
                      f"Computational limit extended to {self.memory.computational_limit}s" +
                      '\033[0m ',
                      flush=True)
        self.save_to_file(save_path)

        while self.memory.epoch < max_epochs:
            self.memory.epoch += 1

            # Select experiment
            # In each period, all experiments should be selected at least once
            # This is integrated in the pick_experiment method
            # When the current period comes to an end, the ucb value of all experiments will be reset
            exp_name = self.memory.pick_experiment(verbose=True, from_subset=True) if exp_select_mode == "ucb" \
                else random.choice(self.knowledge.fetch_exps)
            print(f"\n Epoch {self.memory.epoch}/{max_epochs}: {exp_name}", flush=True)

            cpt_lim_old = copy.deepcopy(self.memory.computational_limit)
            spm_old = {expname: {'experiment': copy.deepcopy(specific.experiment),
                                 'experiment_control': copy.deepcopy(specific.experiment_control)}
                       for expname, specific in self.specific.items()}
            concepts_num_old = len(self.knowledge.fetch_concepts)

            q: Queue = Queue()
            manager = multiprocessing.Manager()
            shared_list = manager.list([])

            process = Process(target=run_method_in_process,
                              args=(self, "theoretical_analysis",
                                    q, shared_list,
                                    exp_name, ver, None, max_actions))
            process.start()
            t1 = time.time()
            cpt_lim: float = min(self.memory.computational_limit, max_cpt_lim) + cpt_lim_bias
            process.join(timeout=cpt_lim)
            t2 = time.time()

            if process.is_alive():
                process.terminate()
                process.join()
                print(f"Discovery for {exp_name} is terminated due to computational limit({cpt_lim}s)",
                      flush=True)
                self.memory.experiment_bandit.actions[exp_name].update(0.0)
            else:
                print(f"Discovery for {exp_name} is finished in {t2 - t1}s( < {cpt_lim}s )",
                      flush=True)
                try:
                    updated_self, reward_tot = q.get()[0]
                except Exception as e:
                    print(f"Error occurred in analyzing {exp_name}: {e}. Reset experiments.", flush=True)
                    for expname in self.specific:
                        self.specific[expname].experiments_reset()
                    continue
                else:
                    if updated_self is None:
                        print(f"Error occurred in analyzing {exp_name}. Reset experiments.", flush=True)
                        for expname in self.specific:
                            self.specific[expname].experiments_reset()
                        self.memory.experiment_bandit.actions[exp_name].count += 1
                        continue
                updated_self: Theorist
                reward_tot: int
                self.knowledge = updated_self.knowledge
                self.memory = updated_self.memory
                self.general = updated_self.general
                self.specific = updated_self.specific
                for expname in self.specific:
                    if expname not in spm_old:
                        self.specific[expname].experiments_reset()
                    else:
                        self.specific[expname].experiment = spm_old[expname]["experiment"]
                        self.specific[expname].experiment_control = spm_old[expname]["experiment_control"]
                self.objmodel = updated_self.objmodel

                if save_path is not None:
                    self.save_policy_nn(save_path)

                self.memory.update_policy_nn(save_path, self.memory.epoch, concepts_num_old)

                # Obliviate randomly
                if random.random() < 1 / len(self.knowledge.fetch_exps):
                    self.obliviate()

                # Save to file
                if save_path is not None:
                    self.save_to_file(save_path)

            # Update computational limit.
            # Enter next period
            if all([self.memory.experiment_bandit.actions[exp_name].ucb() < ucb_bound
                    for exp_name in self.memory.experiments_pool]):
                self.memory.computational_limit *= cpt_lim_mul
                if self.memory.computational_limit > max_cpt_lim:
                    self.memory.computational_limit = max_cpt_lim
                print(' \033[1m' +
                      f"Computational limit {cpt_lim_old}s -> {self.memory.computational_limit}s" +
                      '\033[0m ',
                      flush=True)
                for name, action in self.memory.experiment_bandit.actions.items():
                    action.reset(for_count=1.0)
                self.obliviate()  # NOTE: obliviate strategy should be improved
                # Determine n0 and ucb_bound for the next period
                # n0 = int(np.sqrt(len(self.memory.action_bandit.actions))) + 1
                # ucb_bound: float = 1 / np.sqrt(1.0 + n0)

                self.memory.experiments_pool = []
                self.memory.experiments_pool = self.filter_feasible_experiments(ver,
                                                                                max_actions,
                                                                                self.memory.computational_limit,
                                                                                cpt_lim_bias)
                while not self.memory.experiments_pool:
                    self.reset_time_limit(max_cpt_lim, cpt_lim_mul)
                    print(' \033[1m' +
                          f"Computational limit {cpt_lim_old}s -> {self.memory.computational_limit}s" +
                          '\033[0m ',
                          flush=True)
                    for name, action in self.memory.experiment_bandit.actions.items():
                        action.reset(for_count=1.0)
                    self.memory.experiments_pool = self.filter_feasible_experiments(ver,
                                                                                    max_actions,
                                                                                    self.memory.computational_limit,
                                                                                    cpt_lim_bias)
                self.save_to_file(save_path)

        print("Discovery finished", flush=True)

    def reset_time_limit(self, max_cpt_lim: float, cpt_lim_mul: float):
        self.memory.computational_limit *= cpt_lim_mul
        if self.memory.computational_limit > max_cpt_lim:
            self.memory.computational_limit = max_cpt_lim

    def filter_feasible_experiments(self,
                                    ver,
                                    max_actions,
                                    max_cpt_lim,
                                    cpt_lim_bias
                                    ) -> list[str]:
        print('\n \033[1m' + "Filter feasible experiments" + '\033[0m \n', flush=True)
        feasible_experiments: list[str] = []
        error_flag: bool = False
        for exp_name in self.knowledge.fetch_exps:
            q: Queue = Queue()
            manager = multiprocessing.Manager()
            shared_list = manager.list([])

            process = Process(target=run_method_in_process,
                              args=(self, "theoretical_analysis",
                                    q, shared_list,
                                    exp_name, ver, None, max_actions))
            process.start()
            cpt_lim: float = min(self.memory.computational_limit, max_cpt_lim) + cpt_lim_bias
            t1 = time.time()
            process.join(timeout=cpt_lim)
            t2 = time.time()

            if process.is_alive():
                process.terminate()
                process.join()
                print(f"Discovery for {exp_name} is terminated due to computational limit({cpt_lim}s)",
                      flush=True)
            else:
                print(f"Discovery for {exp_name} is finished in {t2 - t1}s ( < {cpt_lim}s )",
                      flush=True)
                try:
                    updated_self, reward_tot = q.get()[0]
                except Exception as e:
                    print(f"Error occurred in analyzing {exp_name}: {e}.", flush=True)
                    error_flag = True
                    continue
                else:
                    if updated_self is None:
                        print(f"Error occurred in analyzing {exp_name}.", flush=True)
                        error_flag = True
                        continue
                print("Experiment", " \033[1m" + exp_name + '\033[0m ', "is feasible in the current period",
                      flush=True)
                feasible_experiments.append(exp_name)

        if error_flag:
            print("Reset experiments", flush=True)
            for expname in self.specific:
                self.specific[expname].experiments_reset()

        print('\n \033[1m' + "Finish filtering feasible experiments" + '\033[0m',
              flush=True)
        print('\n \033[1m' + "Feasible experiments" + '\033[0m' + ":",
              flush=True)
        print(feasible_experiments, flush=True)

        return feasible_experiments

    def theoretical_analysis(self, exp_name: str, ver: str | None = None,
                             name_list: List[str] | None = None,
                             max_actions: int = 5) -> int:
        assert (exp_name in self.specific)
        print('\n', flush=True)
        print('#'*10 + ' \033[1m' + exp_name + '\033[0m ' + '#'*10, flush=True)
        print('\n', flush=True)
        print(f"Process ID: {os.getpid()} of experiment {exp_name}", flush=True)
        """
        动作一：选择抽取的概念组的数量 num_concept_groups
        """
        num_groups: int = self.memory.pick_num_concept_groups(exp_name)
        """
        动作二：选择抽取的概念和它们生成的表达式 exprs
        """
        # Initialize the reward groups and pickout concepts
        if name_list is None:
            action_groups, ppuct = self.memory.pick_concept_groups(exp_name, num_groups)
        else:
            action_groups = [tuple(name_list)]
            ppuct = None
        if ppuct is not None:
            self.memory.concept_group_history.append({tuple(self.memory.action_bandit.group_complexity(grp)): 0.0
                                                      for grp in action_groups})
            self.memory.action_bandit.current_group = [grp for grp in action_groups]
        else:
            self.memory.concept_group_history.append({})
            self.memory.action_bandit.current_group = []

        reward_groups, props = self.doit(
            exp_name,
            ver,
            action_groups,
            max_actions
        )
        self.memory.update_rewards(exp_name, reward_groups)

        rewards_total = self.doit_again(
            exp_name,
            props
        )

        self.cluster_concepts(exp_name)
        self.general.compensate_general_laws(exp_name)

        reward_gl: int = self.general.service.reduce_general_laws()
        rewards_total += reward_gl
        """
        最终奖励结算
        """
        print(f"Total rewards: {rewards_total}", flush=True)
        self.memory.update_reward_total(exp_name, rewards_total)

        return rewards_total

    def obliviate(self):
        """
        清空除了 specific model、object model 和 general laws 以外的所有记忆
        """
        useful_concept = set()
        # Specific model 中涉及的概念不能被遗忘
        for exp_name in self.specific:
            for conclusion in self.specific[exp_name].conclusions.values():
                for atom in conclusion.unwrap_exp.all_atoms:
                    useful_concept.add(atom.name)
                expr_conclusion: Exp = conclusion.unwrap_exp
                possible_name: str | None = \
                    self.knowledge.find_similar_concept(self.knowledge.generalize(exp_name, expr_conclusion))
                if possible_name is not None:
                    useful_concept.add(possible_name)
                possible_name: str | None = \
                    self.knowledge.find_similar_concept(self.knowledge.generalize_to_normal_concept(exp_name, expr_conclusion))
                if possible_name is not None:
                    useful_concept.add(possible_name)
        # Basic concepts 不能被遗忘
        for exp_name in self.specific:
            for concept in self.specific[exp_name].experiment.original_concept:
                useful_concept.add(concept.atomexp_name)
        # Object model 中涉及的内禀概念不能被遗忘
        for obj_type in self.objmodel:
            for name in self.objmodel[obj_type].attr:
                useful_concept.add(name)
        # 依赖于这些概念的概念也不能被遗忘
        useful_names = set()
        for name in useful_concept:
            concept = self.knowledge.K.fetch_concept_by_name(name)
            if concept.expr_type == 'Concept':
                useful_names |= self.knowledge.K.dependence_of_concept(self.knowledge.fetch_concept_by_name(name))
            else:
                useful_names |= self.knowledge.K.dependence_of_intrinsic(self.knowledge.fetch_intrinsic_by_name(name))
        # General laws 中涉及的概念不能被遗忘
        for gnm_name, gnm in self.general.general_laws.items():
            useful_names |= self.knowledge.K.dependence_of_proposition(gnm.prop)
        useful_names |= useful_concept
        # Concepts in memory.concept_clusters should be considered as a whole
        useless_names = set(self.knowledge.fetch_concepts.keys()) - useful_names
        for name in useless_names:
            for patt, symb_direcs in self.memory.concept_clusters.items():
                if name in symb_direcs:
                    if not (set(symb_direcs.keys()) <= useless_names):
                        useful_names |= set(symb_direcs.keys())
                    break
        # 开始遗忘
        self.knowledge.K.obliviate(useful_names)
        self.memory.obliviate(useful_names)

    def doit(self, exp_name: str, ver: str,
             action_groups: List[Tuple[str]],
             max_actions: int,
             ) -> Tuple[Dict[Tuple[str], float], List[Proposition]]:
        """
        根据动作组，进行符号回归和发现新的（在当前实验中成立的）关系，
        返回动作组的奖励值，以及发现的新关系
        """
        knowledge: Knowledge = self.knowledge
        spm: SpecificModel = self.specific[exp_name]
        spm.experiments_reset()

        gnm_check_result, incomplete_gnm = self.general.check_general_laws(exp_name)
        gnm_check_result: dict[str, bool]
        incomplete_gnm: dict[str, ExpData | None]

        reward_groups: Dict[Tuple[str], float] = {}
        name_list: list[str] = list(set.union(*[set(i) for i in action_groups])) if len(action_groups) > 0 else []
        res_zero, res_consts, name_list = self.filter_selected_concepts(name_list, exp_name)
        res_zero: list[Exp]
        res_consts: list[AtomExp]
        action_groups: list[tuple[str, ...]] = []
        if len(name_list) > max_actions:
            # Ramdonly select max_actions concepts in the list
            name_list = random.sample(name_list, max_actions)
        for i in range(len(name_list)):
            for j in range(i):
                for k in range(j):
                    action_group = tuple(sorted([name_list[i], name_list[j], name_list[k]]))
                    action_groups.append(action_group)
                    reward_groups[action_group] = 0.0
        print(f"Selected concepts: {name_list}", flush=True)

        # Try to complete invalid general laws
        print("Try to complete invalid general laws", flush=True)
        lst_exprs: list[AtomExp] = list(itertools.chain(*[knowledge.specialize_concept(name, exp_name)
                                                          for name in name_list]))
        lst_intrinsic_append: list[AtomExp] = list(itertools.chain(
            *[knowledge.specialize_concept(con_name, exp_name)
                for con_name in [k for k, v in knowledge.fetch_intrinsic_concepts.items()
                                 if (k not in self.universal_constants)
                                 and len(v.input_objtypes) == 1
                                 and str(v.input_objtypes[0]) in str(spm.experiment.obj_info)]]
        ))
        concepts_before: set[str] = set(knowledge.fetch_concepts.keys())
        incomplete_gnm = self.complete_general_laws(exp_name, incomplete_gnm, list(set(lst_exprs + lst_intrinsic_append)))
        concepts_diff: set[str] = set(knowledge.fetch_concepts.keys()) - concepts_before
        if len(concepts_diff) > 0:
            print("Try to complete invalid general laws again", flush=True)
            lst_exprs_diff: list[AtomExp] = list(itertools.chain(*[knowledge.specialize_concept(name, exp_name)
                                                                   for name in concepts_diff]))
            incomplete_gnm = self.complete_general_laws(exp_name, incomplete_gnm,
                                                        lst_intrinsic_append + lst_exprs_diff)
        # Try to convert conserved general laws to zero ones
        print("Try to convert conserved general laws to zero ones", flush=True)
        self.general.convert_conserved_to_zero(exp_name, list(set(lst_intrinsic_append + res_consts)))
        print("Finish trying to convert conserved general laws to zero ones", flush=True)

        # Investigate trivial conservatism of concepts
        cons_result: list[Tuple[Exp, ExpData]] = self.check_trivial_conservatism(exp_name, name_list)

        # Search for new relations
        name_list = self.remove_meaningless_sums(spm, name_list)
        exprs: itertools.chain = itertools.chain(*[knowledge.specialize_concept(name, exp_name) for name in name_list])
        ds = spm.generate_data_struct(exprs)
        result: list[Tuple[Exp, ExpData]] = search_type[ver](ds)
        assert all([data.is_conserved for _, data in result])

        # Convert the results to propositions
        props_direct_zero: List[Proposition] = list(set([
            Proposition.IsZero(i) for i in res_zero
        ]))
        props: List[Proposition] = list(set([
            Proposition.IsZero(i) if j.is_zero else Proposition.IsConserved(i)
            for i, j in result + cons_result]
        ))
        props = list(set(props + props_direct_zero))
        print(f"Found {len(props)} relations", flush=True)
        props = [prop for prop in props if not self.memory.exist_exp(exp_name, prop.unwrap_exp)]
        for prop in props:
            self.memory.record_exp(exp_name, prop.unwrap_exp, 'zero' if prop.is_zero else 'conserved')
        props = spm.filter_relations(props)
        props = [prop for prop in props if spm.test_on_test_experiment(prop.unwrap_exp, 'conserved' if prop.is_conserved else 'zero')]
        print(f"After filtering, {len(props)} relations left", flush=True)
        for prop in props:
            print(f"Propose new proposition: {prop}", flush=True)
        for prop in props:
            group = tuple(sorted(i.name for i in prop.unwrap_exp.all_atoms))
            reward_groups[group] = 1.0
        return reward_groups, props

    def doit_again(self, exp_name, props: List[Proposition]) -> int:
        """
        用当前实验中发现的关系更新 Specific model 和知识库，并返回总的奖励值。
        并根据新发现的关系，Propose 新的 general laws.
        """
        spm = self.specific[exp_name]
        conclusion_before = set(spm.conclusions.keys())
        # Try to add the propositions to the specific model one by one
        for prop in tqdm(props, desc="Add to Specific model"):
            # Comparison is made based on the Exp object
            expr: Exp = prop.unwrap_exp
            if prop.is_zero:
                name = spm.append_zero_exp(expr)
            elif prop.is_conserved:
                name = spm.append_conserved_exp(expr)
        """
        求结论的 minimal 表示
        """
        spm.reduce_conclusions(debug=False)
        """
        第二步，寻找守恒量之间进一步的关系
        """
        self.cqCalc = CQCalculator(exp_name, self.knowledge)
        for name, info in spm.conclusions.conserved_list.items():
            self.cqCalc.insert_cq_info(info)
        for obj_type, id in spm.experiment.obj_info.values():
            if not self.objmodel.__contains__(str(obj_type)):
                continue
            for name in self.objmodel[str(obj_type)].attr:
                expr = Exp.Atom(AtomExp.VariableIds(name, [id]))
                info = ConservedInfo(exp_name, str(expr), expr, True, {id})
                self.cqCalc.insert_cq_info(info)

        res_const, res_zero = self.cqCalc.calc_relations()
        print(f"Found {len(res_const) + len(res_zero)} relations between conserved quantities",
              flush=True)
        for expr in res_const:
            expr_new = exp_replace_by_dict(expr, spm.conclusions.conserved_list)
            if self.memory.exist_exp(exp_name, expr_new, 'intrinsic'):
                continue
            info = spm.make_conserved_info(str(expr_new), expr_new)
            if info.is_intrinsic:
                spm.intrinsic_buffer[str(expr_new)] = info
                self.memory.record_exp(exp_name, expr_new, 'intrinsic')
        for expr in res_zero:
            expr_new = exp_replace_by_dict(expr, spm.conclusions.conserved_list)
            if self.memory.exist_exp(exp_name, expr_new):
                continue
            name = spm.append_zero_exp(expr_new)
            self.memory.record_exp(exp_name, expr_new, 'zero')
        spm.reduce_conclusions(debug=False)

        # Try to generalize new laws
        conclusion_after = set(spm.conclusions.keys())
        if len(conclusion_after) > len(conclusion_before):
            for name in conclusion_after - conclusion_before:
                self.general.propose_new_general_law(law_name=name, specific_model=spm, exp_name=exp_name)
        self.general.fix_general_laws()
        """
        比较前后的 conclusions 的变化，计算 reward 并注册概念
        """
        rewards_total = 0
        conclusion_after = set(spm.conclusions.keys())
        conclusion_diff = conclusion_after - conclusion_before
        for name in conclusion_diff:
            expr: Exp = spm.conclusions.get(name).unwrap_exp
            if str(expr) in spm.intrinsic_buffer:
                continue
            if self.is_initial_condition(expr, spm) or self.possess_large_coes(expr, spm):
                continue
            concepts = self.knowledge.extract_concepts(exp_name, expr)
            for concept in concepts:
                if self.is_concept_pure_geometric(concept, spm):
                    continue
                new_concept_name = self.register_concept(concept, self.specific[exp_name])
                if new_concept_name is not None:
                    # Update self.memory.concept_group_history
                    group: tuple[str, ...] = tuple(sorted(i.name for i in expr.all_atoms))
                    for grp in self.memory.action_bandit.current_group:
                        if set(group) <= set(grp):
                            complex_grp = tuple(self.memory.action_bandit.group_complexity(grp))
                            self.memory.concept_group_history[-1][complex_grp] = 1.0
            rewards_total += 1
        # 将 intrinsic_buffer 中的内禀概念注册到知识库中
        registered_name_lst = self.register_intrinsics(spm.intrinsic_buffer, 1,
                                                       is_intrinsic_concept=spm.possible_intrinsic())
        rewards_total += len(registered_name_lst)
        for name in spm.intrinsic_buffer:
            cqinfo = spm.intrinsic_buffer[name]
            if self.memory.exist_exp(exp_name, cqinfo.exp, 'intrinsic'):
                continue
            self.memory.record_exp(exp_name, cqinfo.exp, 'intrinsic')
        spm.intrinsic_buffer.clear()

        # Check concepts
        abundant_concepts: set[str] = self.memory.action_bandit.actions - set(self.knowledge.fetch_concepts.keys())
        for name in abundant_concepts:
            self.memory.action_bandit.actions.remove(name)

        return rewards_total

    def cluster_concepts(self, exp_name: str):
        knowledge: Knowledge = self.knowledge
        spm: SpecificModel = self.specific[exp_name]

        concept_to_gen: dict[str, dict[str, set[str]]] = {}  # {concept_name: {direction: {patterns}}}
        for name in knowledge.fetch_concepts.keys():
            if any([name in set(patt_symbols.keys()) for patt_symbols in self.memory.concept_clusters.values()]):
                continue
            concept: Concept | None = knowledge.fetch_concept_by_name(name)
            if concept is None or not self.knowledge.specialize(concept, exp_name):
                continue
            direction_patterns: dict[str, set[str]] = self.check_single_relevant_direction(concept.exp, spm)
            # If the concept involves only one direction, cluster it with that of other directions into a group
            if len(direction_patterns) == 1:
                concept_to_gen[name] = direction_patterns

        concept_clusters_inversed = {patt: {direction: atom
                                            for atom, direction in equivs.items()}
                                     for patt, equivs in self.memory.concept_clusters.items()}  # {pattern: {direction: atom}}
        for name, patt_dict in concept_to_gen.items():
            new_pattern_directions: dict[str, str] = {name: list(patt_dict.keys())[0]}  # {concept_name: direction}
            exist_concept: Concept | None = knowledge.fetch_concept_by_name(name)
            if exist_concept is None:
                continue
            patts: set[str] = list(patt_dict.values())[0]
            dir2patt: dict[str, set[str]] = {direction: patts for direction in {"posx", "posy", "posz"} - set(patt_dict.keys())}
            template = extract_expression_pattern([str(exist_concept.exp)],
                                                  {patt: set(direc.keys())
                                                   for patt, direc in self.memory.concept_clusters.items()
                                                   if direc})[0]
            if template is None:
                continue
            flag = True
            for direction, set_patts in dir2patt.items():
                concept_exp_str: str = template
                for patt in set_patts:
                    concept_exp_str = concept_exp_str.replace(patt, concept_clusters_inversed[patt][direction])
                concept_exp: Exp = Exp(concept_exp_str)
                concept: Concept | None = knowledge.generalize_to_normal_concept(exp_name, concept_exp)
                if concept is None:
                    flag = False
                    break
                if exist_concept.is_sum:
                    concept = Concept.Mksum(list(exist_concept.objtype_id_map.keys())[0], concept)
                if not knowledge.specialize(concept, exp_name):
                    flag = False
                    break
                concept_name: str | None = self.knowledge.find_similar_concept(concept)
                if concept_name is None:
                    flag = False
                    break
                new_pattern_directions[concept_name] = direction
            if flag:
                if any([new_pattern_directions == patt for patt in self.memory.concept_clusters.values()]):
                    continue
                patt_name: str = f"${len(self.memory.concept_clusters) + 1}$"
                self.memory.concept_clusters[patt_name] = new_pattern_directions

    def filter_selected_concepts(self, name_list: list[str], exp_name: str) -> tuple[list[Exp], list[AtomExp], list[str]]:
        spm: SpecificModel = self.specific[exp_name]

        irr_basic_concepts: list[str] = self.extract_irrelevant_basic_concepts(exp_name)
        res_zero: list[Exp] = []
        res_not_zero: list[str] = []
        res_const: list[AtomExp] = []

        intrinsics: list[Intrinsic] = [self.knowledge.fetch_intrinsic_by_name(nm)
                                       for nm in self.universal_constants]
        intrinsics_def: list[Intrinsic] = [intr for intr in intrinsics if exp_name == intr.measure_experiment]
        intrinsics_def_name_lst: list[str] = []
        for intr in intrinsics_def:
            try:
                intrinsic_exp: Exp = Exp(str(intr).split('|- ')[1][:-1])
                def_concept: Concept = self.knowledge.generalize_to_normal_concept(exp_name, intrinsic_exp)
            except Exception:
                continue
            if def_concept is not None:
                def_concept_name: str = self.knowledge.find_similar_concept(def_concept)
                if def_concept_name is not None and def_concept_name in name_list:
                    name_list.remove(def_concept_name)
                    intrinsics_def_name_lst.append(def_concept_name)

        for name in name_list:
            if self.knowledge.fetch_concept_by_name(name) is not None:
                spec_exprs: list[Exp] = self.knowledge.specialize(self.knowledge.fetch_concept_by_name(name),
                                                                  exp_name)
            elif self.knowledge.fetch_intrinsic_by_name(name) is not None:
                spec_exprs: list[Exp] = [Exp.Atom(it) for it in self.knowledge.specialize_concept(name, exp_name)]
            else:
                continue
            sp_raw_exprs: list[sp.Expr] = [spm._sympy_of_raw_defi(spm.expand_exp(expr))
                                           for expr in spec_exprs]
            if not sp_raw_exprs or all([item.is_Number for item in sp_raw_exprs]) or \
                    any(["/Derivative(0, t_0)" in str(raw_expr) for raw_expr in sp_raw_exprs]):  # NOTE:
                continue
            has_intr_def_flag: bool = False
            for intr_def_name in set(intrinsics_def_name_lst):
                if any([intr_def_name in str(raw_expr) for raw_expr in sp_raw_exprs]):
                    has_intr_def_flag = True
                    break
            if has_intr_def_flag:
                continue
            has_irr_flag: bool = False
            for irr_basic in irr_basic_concepts:
                if any([irr_basic in str(raw_expr) for raw_expr in sp_raw_exprs]):
                    has_irr_flag = True
                    break
            if has_irr_flag:
                continue
            vals: list[ExpData] = [self.knowledge.eval(expr, spm.experiment)
                                   for expr in spec_exprs]
            if all([val.is_err for val in vals]):
                continue
            if all([(val.is_zero or val.__powi__(2).is_zero) for val in vals]):
                res_zero.extend([spec_expr for spec_expr in spec_exprs
                                 if str(sp.simplify(spm._sympy_of_raw_defi(spm.expand_exp(spec_expr)))) != '0'])
            else:
                res_not_zero.append(name)

        for name in res_not_zero:
            if name in self.universal_constants:
                continue
            atom_exprs: list[AtomExp] = self.knowledge.specialize_concept(name, exp_name)
            res_const.extend([atm for atm in atom_exprs
                              if self.knowledge.eval(Exp.Atom(atm), spm.experiment).is_const])

        return res_zero, res_const, res_not_zero

    def complete_general_laws(self, exp_name: str,
                              incomplete_gl: dict[str, ExpData | None],
                              lst_exprs: list[AtomExp]) -> dict[str, ExpData | None]:
        completed_gl = []
        for gl_name, gl_data in incomplete_gl.items():
            print(f"- Try to complete {gl_name}", flush=True)
            _, is_completed = self.general.try_complete_general_law(gl_name, gl_data, lst_exprs, exp_name)
            if is_completed:
                completed_gl.append(gl_name)
        # Remove the completed general laws
        for name in completed_gl:
            del incomplete_gl[name]
        print("Finish trying to complete incomplete general laws", flush=True)
        self.general.fix_general_laws()

        return incomplete_gl

    def extract_irrelevant_basic_concepts(self, exp_name: str) -> list[str]:
        """
        从 Specific model 中提取出不相关的基础概念
        """
        spm: SpecificModel = self.specific[exp_name]
        knowledge: Knowledge = self.knowledge
        res: list[str] = []
        basic_names: list[str] = [nm
                                  for nm in knowledge.fetch_concepts.keys()
                                  if not nm.startswith("C_")]
        for name, lst_atoms in {nm: knowledge.specialize_concept(nm, exp_name)
                                for nm in basic_names}.items():
            if not lst_atoms:
                res.append(name)
                continue
            vals: list[ExpData] = [knowledge.eval(Exp.Atom(atm), spm.experiment)
                                   for atm in lst_atoms]
            if all([(val.is_err or val.is_zero) for val in vals]):
                res.append(name)

        return res

    def check_trivial_conservatism(self, exp_name: str, name_list: str) -> list[tuple[Exp, ExpData]]:
        res: list[tuple[Exp, ExpData]] = []
        spm: SpecificModel = self.specific[exp_name]
        for name in name_list:
            # Consider basic concepts only
            if self.knowledge.fetch_concept_by_name(name) is None or name == 't' or name == 'dist' or name.startswith('C_'):
                continue
            spec_exprs: list[Exp] = self.knowledge.specialize(self.knowledge.fetch_concept_by_name(name),
                                                              exp_name)
            flag: bool = False
            for expr in spec_exprs:
                # If the expression is already conserved, skip it
                if self.knowledge.eval(expr, spm.experiment).is_conserved:
                    continue
                expr_dt1: Exp = expr.__difft__(1)
                val_dt: ExpData = self.knowledge.eval(expr_dt1, spm.experiment)
                # If the first derivative is conserved, add it to the result
                if val_dt.is_zero:
                    continue
                elif val_dt.is_conserved:
                    res.append((expr_dt1, val_dt))
                # If the second derivative is conserved, register it as a new concept and add it to the result
                if flag:
                    continue
                val_dtt = self.knowledge.eval(expr.__difft__(2), spm.experiment)
                if val_dtt.is_conserved and not val_dtt.is_zero:
                    concept_dt: Concept | None = self.knowledge.generalize_to_normal_concept(exp_name,
                                                                                             expr.__difft__(1))
                    if concept_dt is None:
                        continue
                    concept_dt_name: str | None = self.register_concept(concept_dt, spm)
                    if concept_dt_name is None:
                        continue
                    exprs_dt: list[Exp] = [Exp.Atom(atm)
                                           for atm in self.knowledge.specialize_concept(concept_dt_name, exp_name)]
                    for expr_dt in exprs_dt:
                        expr_dtt: Exp = expr_dt.__difft__(1)
                        val_expr_dtt: ExpData = self.knowledge.eval(expr_dtt, spm.experiment)
                        if val_expr_dtt.is_conserved and not val_expr_dtt.is_zero:
                            res.append((expr_dtt, val_expr_dtt))
                    # If the second derivative is not only conserved, but also zero, propose a new general law
                    if val_dtt.is_zero:
                        expr_dtt: Exp = Exp.Atom(self.knowledge.specialize_concept(concept_dt_name, exp_name)[0]).__difft__(1)
                        concept_dtt: Concept | None = self.knowledge.generalize_to_normal_concept(exp_name, expr_dtt)
                        if concept_dtt is None:
                            continue
                        self.general.propose_new_general_law(exp_name=exp_name,
                                                             concept=concept_dtt,
                                                             law_type="IsZero")
                    flag = True

        return res

    # ------------------------------ Test functions ------------------------------ #

    def print_concepts(self):
        sorted_keys = sorted(list(self.knowledge.fetch_concepts.keys()))
        concepts_dict = self.knowledge.fetch_concepts
        for name in sorted_keys:
            print(name, str(concepts_dict[name]))

    def print_general_laws(self):
        for name, gl in self.general.general_laws.items():
            print(name, str(gl))


def stupid_analysis(knowledge: Knowledge, exp_name: str) -> ExpStructure:
    """
    一个非常简单粗暴的函数 （用于测试，详见 test8.py ）
    将一个理论家记忆中的所有概念实例化 （specialize） 到一个实验中的具体表达式
    再对具体表达式进行各种加减乘除求导的拼凑组合求值，
    如果结果守恒，就将这个表达式注册为新的概念 （generalize）
    """
    exp = knowledge.fetch_expstruct(exp_name)
    exp.random_settings()
    exp.collect_expdata(MeasureType.default())
    for key in knowledge.fetch_concepts:
        specific_exprs: list[AtomExp] = knowledge.specialize_concept(key, exp_name)
        for i in specific_exprs:
            knowledge.eval(str(i), exp)
    data_info: DataStruct = exp.data_info
    print(data_info)
    res: List[Tuple[Exp, ExpData]] = search_relations_ver1(data_info)
    for (expr, expdata) in res:
        if expdata.is_zero:
            prop = Proposition.IsZero(expr)
            knowledge.register_conclusion(str(prop))
        elif expdata.is_conserved:
            prop = Proposition.IsConserved(expr)
            knowledge.register_conclusion(str(prop))
        concept: Concept | None = knowledge.generalize(exp_name, str(expr))
        if concept is not None:
            knowledge.register_expr(str(concept))
    return exp


def exp_replace_by_dict(exp: Exp, atom_to_exp: Dict[str, ConservedInfo | ZeroInfo]) -> "Exp":
    atom_set = exp.all_atoms
    res = exp.copy()
    for atom in atom_set:
        if atom.name in atom_to_exp:
            res = res.replace_atom_by_exp(atom, atom_to_exp[atom.name].exp)
    return res
