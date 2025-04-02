from typing import Dict, Any, List, Tuple, Set
import multiprocessing
import numpy as np
from .ucb import ExperimentUCB
from .interface import AtomExp, Exp, Knowledge
from .recommendation_engine.concept_group import ConceptGroupPolicy
from .recommendation_engine.update_concept_group_policy import update_network_parameters
import os
import random
import torch


def dict_to_json(d: Dict[str, Any]) -> Dict[str, str]:
    return {k: str(v) for k, v in d.items()}


def fetch_nonempty_subsets(x: Tuple[str]) -> List[Tuple[str]]:
    match len(x):
        case 0:
            return []
        case 1:
            return [x]
        case 2:
            return [(x[0],), (x[1],), x]
        case 3:
            return [(x[0],), (x[1],), (x[2],), (x[0], x[1]), (x[0], x[2]), (x[1], x[2]), x]
        case _:
            result = fetch_nonempty_subsets(x[1:])
            result2 = [(x[0],) + i for i in result]
            return [(x[0],)] + result + result2


class ExperimentRecommander:
    """
    非平稳多臂老虎机，单动作的推荐系统
    """
    actions: Dict[str, ExperimentUCB]

    def __init__(self):
        self.actions: Dict[str, ExperimentUCB] = {}

    def to_json(self):
        return {key: value.to_json() for key, value in self.actions.items()}

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "ExperimentRecommander":
        obj = object.__new__(ExperimentRecommander)
        obj.actions = {key: ExperimentUCB.from_json(value) for key, value in data.items()}
        return obj

    def register_action(self, name: str, info: str = None, init: float = 1.0):
        """
        新注册一个动作
        """
        if name not in self.actions:
            self.actions[name] = ExperimentUCB(info)
            self.actions[name].update(init)

    def choose_action(self, verbose: bool = False,
                      subset: list[str] | None = None) -> str:
        if subset is not None:
            pool: dict[str, ExperimentUCB] = {k: v for k, v in self.actions.items() if k in subset}
        else:
            pool = self.actions
        actions_have_not_been_tried = [i for i in pool
                                       if pool[i].count < 1.1]
        # If an action hasn't been tried, choose it
        if actions_have_not_been_tried:
            res: str = random.choice(actions_have_not_been_tried)
            if verbose:
                print(f"Experiment {res} is selected for the first time")
            return res
        v = [(i, pool[i].ucb()) for i in pool]
        if verbose:
            print("Experiments' UCB values: ", v)
        # Randomly choose an action according to the probability proportional to its ucb value
        sorted_v = sorted(v, key=lambda x: x[1], reverse=True)
        # picked: str = random.choices(v, [i[1] ** 2 for i in v])[0][0]
        picked: str = sorted_v[0][0]
        return picked

    def update_reward(self, name: str, reward: float):
        """
        选定的动作获得了回报，更新动作的 nsUCB。
        """
        self.update_rewards({name: reward})

    def choose_actions(self, num: int, random_percent: float = None) -> List[str]:
        """
        以 ub 值为排序，选择最优的 num 个动作
        """
        v = [(i, self.actions[i].ucb()) for i in self.actions]
        v.sort(key=lambda x: x[1], reverse=True)
        if random_percent is not None:
            num_ucb, num_random = min(int(num * (1 - random_percent)), num - 1), max(int(num * random_percent), 1)
            ucbs = random.sample([i[0] for i in v[:num]], min(num_ucb, len(v[:num])))
            randoms = random.sample([i[0] for i in v], min(num_random, len(v)))
            return list(set(ucbs + randoms))
        return [i[0] for i in v[:num]]

    def update_rewards(self, rewards: Dict[str, float]):
        """
        选定的动作获得了回报，更新每个动作的 nsUCB。
        """
        for key, item in self.actions.items():
            item.update(rewards.get(key, None))


class GroupBandit:
    """
    组合动作的推荐系统
    """
    cpuct: float
    actions: Set[str]
    groups_V: Dict[Tuple[str], float]
    groups_N: Dict[str, Dict[Tuple[str], float]]
    policy_nn: "ConceptGroupPolicy"
    knowledge: Knowledge
    current_group: List[Tuple[str]]

    def __init__(self, knowledge: Knowledge) -> None:
        self.actions = set()
        self.groups_V = {}
        self.groups_N = {}
        self.cpuct = 1.0
        self.policy_nn = ConceptGroupPolicy()
        self.knowledge = knowledge
        self.current_group = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "actions": list(self.actions),
            "groups_V": {
                str(key): item
                for key, item in self.groups_V.items()
            },
            "groups_N": {
                exp_name: {
                    str(key): item
                    for key, item in value.items()
                }
                for exp_name, value in self.groups_N.items()
            }
        }

    @staticmethod
    def from_json(data: Dict[str, Any], knowledge: Knowledge) -> "GroupBandit":
        obj = object.__new__(GroupBandit)
        obj.actions = set(data["actions"])
        obj.groups_V = {
            eval(key): item
            for key, item in data["groups_V"].items()
        }
        obj.groups_N = {
            exp_name: {
                eval(key): item
                for key, item in value.items()
            }
            for exp_name, value in data["groups_N"].items()
        }
        obj.cpuct = 10.0
        obj.policy_nn = ConceptGroupPolicy()
        obj.knowledge = knowledge
        obj.current_group = None
        return obj

    def register_action(self, name: str, init: float = 1.0):
        """
        新注册一个动作
        """
        if name not in self.actions:
            self.actions.add(name)
            self.groups_V[(name,)] = init

    def choose_action_groups(self, exp_name: str, num: int, knowledge: Knowledge,
                             exclude_list: List[str] = None) -> tuple[List[Tuple[str]], torch.Tensor | None]:
        """
        Consider 3 actions (concepts) as a group.

        Args:
            exp_name: Name of the experiment.
            num: Number of groups to choose.
            knowledge: Knowledge base.
            exclude_list: List of actions to exclude.

        Returns:
            List of groups of actions.
        """
        if exclude_list is None:
            exclude_list = []
        name_list = [name for name in self.actions
                     if len(knowledge.specialize_concept(name, exp_name)) > 0 and name not in exclude_list]
        if len(name_list) < 3:
            return [name_list], None
        ucb_list: list[Tuple[Tuple[str], float]] = []  # [((concept1, ...), ucb)]
        if not self.groups_N.__contains__(exp_name):
            self.groups_N[exp_name] = {}
        ppuct_lst = []
        # Move self.policy_nn to GPU if available
        self.policy_nn.to('cuda')
        for i in range(len(name_list)):
            for j in range(i + 1, len(name_list)):
                for k in range(j + 1, len(name_list)):
                    group = tuple(sorted([name_list[i], name_list[j], name_list[k]]))
                    group_complexity = self.group_complexity(group) + [len(self.knowledge.fetch_concepts)]
                    Q = 0
                    subsets = fetch_nonempty_subsets(group)
                    for subset in subsets:
                        Q += self.groups_V.get(subset, 0)
                    Q /= len(subsets)
                    N = self.groups_N[exp_name].get(group, 0)
                    input_vec = torch.tensor(group_complexity, dtype=torch.float32, device='cuda')
                    ppuct: torch.Tensor = self.policy_nn(input_vec)
                    ppuct_lst.append(ppuct)
                    ucb = (0.1 + ppuct.clone().cpu().detach().numpy()) * Q + self.cpuct * np.sqrt(1.0 / (1 + N))
                    ucb_list.append((group, ucb))
        ucb_list.sort(key=lambda x: x[1], reverse=True)
        # Randomly choose an action according to the probability proportional to its ucb value
        picked: list[Tuple[Tuple[str], float]] = random.choices(ucb_list, [i[1] ** 2 for i in ucb_list], k=num)
        # ppuct_avg = torch.stack(ppuct_lst).mean(dtype=torch.float32) if len(ppuct_lst) > 1 else None
        return [i[0] for i in picked], torch.stack(ppuct_lst) if len(ppuct_lst) > 1 else None

    def update_rewards(self, exp_name: str, rewards: Dict[Tuple[str], float]):
        rewards_aux = {}
        for group, reward in rewards.items():
            for subset in fetch_nonempty_subsets(group):
                if subset not in rewards_aux:
                    rewards_aux[subset] = 0
                rewards_aux[subset] += reward
        if not self.groups_N.__contains__(exp_name):
            self.groups_N[exp_name] = {}
        for group, reward in rewards_aux.items():
            self.groups_V[group] = self.groups_V.get(group, 0) * (1 - 0.02) + reward
            self.groups_N[exp_name][group] = self.groups_N[exp_name].get(group, 0) * (1 - 0.001) + 1
        for group in self.groups_N[exp_name]:
            if group not in rewards_aux:
                self.groups_N[exp_name][group] = self.groups_N[exp_name][group] * (1 - 0.001)

    def group_complexity(self, group: Tuple[str]) -> list[int]:
        return sorted([self.knowledge.fetch_concept_by_name(name).complexity
                       if self.knowledge.fetch_concept_by_name(name) is not None else 10
                       for name in group])


class Memory:
    """
    AI 的记忆仓库，目的是在传入具体实验名称 exp_name 时，
    回想起一些相关的原子表达式，并根据权重抽取原子表达式 （pick_relevant_exprs）。
    抽取算法是基于非平稳多臂老虎机的，
    一个老虎机有一个动作空间，每一个动作都对应一个原子表达式。
    不同实验的经验是可以相互迁移的，因此有一个通用的老虎机 action_bandit，
    它的动作空间是所有实验的并集。如果在实验 A 中概念 v 有很好的表现，那么在其他实验中会有更高的倾向去选择 v。
    """
    epoch: int
    knowledge: Knowledge
    history: Dict[str, Dict[str, str]]  # Dict[exp_name, Dict[exp, type(conserved/zero/...)]]
    experiment_bandit: ExperimentRecommander
    action_bandit: GroupBandit
    computational_limit: int | float | None
    experiments_pool: list[str]
    general_conclusion_attempts: dict[str, dict[str, set[tuple[str, ...]]]]
    concept_clusters: dict[str, dict[str, str]]
    concept_group_history: list[dict]

    def __init__(self, knowledge: Knowledge, computational_limit: int | float | None = None):
        self.epoch = 0
        self.knowledge = knowledge
        self.history = {}
        self.experiment_bandit = ExperimentRecommander()
        self.action_bandit = GroupBandit(knowledge=knowledge)
        self.num_concept_groups = 20
        self.computational_limit = 4 if computational_limit is None else computational_limit
        self.experiments_pool = []
        self.general_conclusion_attempts = {}
        self.concept_clusters = {"$1$": {"posx": "posx", "posy": "posy", "posz": "posz"}}
        self.concept_group_history = []

    def __getstate__(self):
        # self.action_bandit.policy_nn shall be pickled alone
        # Others are pickled together via to_json
        return {
            **self.to_json(),
            "policy_nn": self.action_bandit.policy_nn.state_dict()
        }

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(Memory.from_json(state, self.knowledge).__dict__)
        self.action_bandit.policy_nn.load_state_dict(state["policy_nn"])

    def to_json(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "experiment_bandit": self.experiment_bandit.to_json(),
            "action_bandit": self.action_bandit.to_json(),
            "history": self.history,
            "computational_limit": self.computational_limit,
            "experiments_pool": self.experiments_pool,
            "general_conclusion_attempts": {en: {gcn: list(attempts) for gcn, attempts in edict.items()}
                                            for en, edict in self.general_conclusion_attempts.items()},
            "concept_clusters": self.concept_clusters,
            "concept_group_history": [[(k, v) for k, v in d.items()] for d in self.concept_group_history]
        }

    @staticmethod
    def from_json(data: Dict[str, Any], knowledge: Knowledge) -> "Memory":
        obj = object.__new__(Memory)
        obj.epoch = data.get("epoch", 0)
        obj.knowledge = knowledge
        obj.experiment_bandit = ExperimentRecommander.from_json(data["experiment_bandit"])
        obj.action_bandit = GroupBandit.from_json(data["action_bandit"], knowledge)
        obj.history = data["history"]
        obj.computational_limit = data.get("computational_limit", 4)
        obj.experiments_pool = data.get("experiments_pool", [])
        obj.general_conclusion_attempts = {en: {gcn: set([tuple(comb)
                                                          for comb in attempts])
                                                for gcn, attempts in edict.items()}
                                           for en, edict in data["general_conclusion_attempts"].items()} \
            if "general_conclusion_attempts" in data else {}
        obj.concept_clusters = data.get("concept_clusters", {"$1$": {"posx": "posx", "posy": "posy", "posz": "posz"}})
        obj.num_concept_groups = 20
        obj.concept_group_history = data.get("concept_group_history", [])
        if obj.concept_group_history:
            obj.concept_group_history = [{tuple(k): v for k, v in d} for d in obj.concept_group_history]
        return obj

    def pick_num_concept_groups(self, exp_name: str) -> int:
        """
        选择当前实验中需要抽取的概念组合的数量
        """
        # TODO: 抽取概念的数量需要视情况而定
        return self.num_concept_groups

    def pick_experiment(self, verbose: bool = False,
                        from_subset: bool = False) -> str:
        """
        有方向地选择一个实验，期望有较高的 reward。
        """
        return self.experiment_bandit.choose_action(verbose, self.experiments_pool if from_subset else None)

    def pick_concept_groups(self, exp_name: str, num: int) -> tuple[List[Tuple[str]], torch.Tensor | None]:
        """
        选择当前实验中需要抽取的概念组合
        """
        return self.action_bandit.choose_action_groups(exp_name, num, self.knowledge)

    def register_action(self, action_name: str, init: float = 1.0):
        self.action_bandit.register_action(action_name, init)

    def register_experiment(self, exp_name: str):
        self.experiment_bandit.register_action(exp_name)

    def update_reward_total(self, exp_name: str, reward_total: float):
        self.experiment_bandit.update_reward(exp_name, reward_total)

    def update_rewards(self, exp_name: str, rewards: Dict[Tuple[str], float]):
        self.action_bandit.update_rewards(exp_name, rewards)

    def update_policy_nn(self, file_path: str, epoch: int, concepts_num: int,
                         max_batch_size: int = 256):
        nn_path: str = file_path + "_policy_nn.pth"
        epoch_loss_path: str = file_path + "_concept_grp_epoch_loss.txt"
        # Check if nn_path exists
        if not (os.path.exists(nn_path) and self.concept_group_history and self.concept_group_history[-1]):
            return
        # If the number of concept group history exceeds the maximum batch size,
        # pop the first few elements to keep the size within the limit
        if len(self.concept_group_history) > max_batch_size:
            for i in range(len(self.concept_group_history) - max_batch_size):
                self.concept_group_history.pop(0)
        grp_comps_sc = torch.tensor([[*grp, concepts_num, sc]
                                     for i in range(len(self.concept_group_history))
                                     for grp, sc in self.concept_group_history[i].items()
                                     if self.concept_group_history[i]],
                                    dtype=torch.float32).clone().detach()
        epoch_decay = torch.tensor([[0.97 ** (len(self.concept_group_history) - i - 1)]
                                    for i in range(len(self.concept_group_history))
                                    for _ in range(len(self.concept_group_history[i]))
                                    if self.concept_group_history[i]],
                                   dtype=torch.float32).clone().detach()
        input_vec = grp_comps_sc[:, :-1]
        find_new = grp_comps_sc[:, -1]
        # Generate a torch tensor `delta_vec`, whose shape is the same as `input_weight`
        # Elements corresponding to 0 are set to 0, and others are set to 1
        delta_vec = torch.where(find_new == 0, torch.tensor(0.0), torch.tensor(1.0))
        # Weights should be softmaxed
        print("Start updating concept group policy_nn")
        process = multiprocessing.Process(
            target=update_network_parameters,
            args=(nn_path, epoch_loss_path, epoch, input_vec, delta_vec, epoch_decay)
        )
        process.start()
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join()
            print(f"Concept group policy_nn update failed withing {60} seconds")
        else:
            print("Concept group policy_nn updated successfully")
        self.action_bandit.policy_nn.load_state_dict(torch.load(nn_path, weights_only=True))
        self.action_bandit.policy_nn.eval()
        # self.concept_group_history = []

    # Operate on history
    def record_exp(self, exp_name: str, exp: Exp, info: str):
        if exp_name not in self.history:
            self.history[exp_name] = {}  # dict[str, str]
        self.history[exp_name][str(exp)] = info  # conserved/zero/...

    def exist_exp(self, exp_name: str, exp: Exp, info: str = None) -> bool:
        if info is None:
            return str(exp) in self.history.get(exp_name, {})
        return self.history.get(exp_name, {}).get(str(exp), None) == info

    def record_gc_attempts(self,
                           exp_name: str,
                           gc_name: str,
                           item: set[tuple[str, ...]] | list[tuple[str, ...]] | tuple[str, ...]):
        if isinstance(item, list):
            item = set(item)
        elif isinstance(item, tuple):
            item = {item}
        elif not isinstance(item, set):
            raise TypeError("item should be a set, list or tuple")

        if exp_name not in self.general_conclusion_attempts:
            self.general_conclusion_attempts[exp_name] = {}
        if gc_name not in self.general_conclusion_attempts[exp_name]:
            self.general_conclusion_attempts[exp_name][gc_name] = set()

        self.general_conclusion_attempts[exp_name][gc_name] |= item

    def obliviate(self, useful_name: Set[str]):
        self.action_bandit.groups_V = {}
        # self.action_bandit.groups_N = {}
        self.action_bandit.actions = self.action_bandit.actions & useful_name
        for action in self.action_bandit.actions:
            self.action_bandit.groups_V[(action,)] = 1.0
        # for action, ucb in self.experiment_bandit.actions.items():
        #     ucb.reset()
        #     ucb.update(1.0)
