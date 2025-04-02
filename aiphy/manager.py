"""
Functions for updating the knowledge base and memory system.
"""
from tqdm import tqdm
import fractions
import sympy as sp
from .memory import Memory
from .interface import (
    Knowledge, Concept, Expression, Intrinsic,
    IExpConfig, SExp, AtomExp, Exp, ExpData, ExpStructure
)
from .core import sentence, ConstData
from .specific_model import SpecificModel, ConservedInfo
from .object_model import ObjectModel
from typing import Dict, List
import re


class Manager:
    """
    The Manager class is a subclass of Knowledge and Memory, which is responsible for
    updating the knowledge base and memory system and do query operations.
    """
    knowledge: Knowledge
    memory: Memory

    def register_basic_concept(self, concept: Concept):
        name = self.knowledge.register_basic_concept(concept)
        if name is not None:
            tqdm.write("\033[1m" + f"Registered New Basic Concept: {name} = {concept}" + "\033[0m")
            self.memory.register_action(name, init=2.0)

    def register_concept(self, concept: Concept,
                         specific: SpecificModel = None,
                         register_action: bool = True) -> str | None:
        """
        Theorist 类中新注册一个概念。
        如果概念已经存在（可以通过 find_similar 查询已存在的名字），则返回 None，否则返回概念的名字。
        """
        if self.is_concept_pure_geometric(concept, specific) or self.is_initial_condition(concept, specific):
            return None
        if 'Sum' not in str(concept) and self.is_concept_propto_const(concept, specific):
            return None
        expression: Expression = Expression.Concept(concept=concept)
        name = self.knowledge.register_expr(expression)
        if name is not None:
            tqdm.write("\033[1m" + f"Registered New Concept: {name} = {concept}" + "\033[0m")
            init = 4.0 if concept.is_sum else 1.0
            if register_action:
                self.memory.register_action(name, init=init)
        return name

    def register_concept_or_return_name(self, concept: Concept,
                                        specific: SpecificModel = None,
                                        register_action: bool = True) -> str | None:
        """
        Theorist 类中新注册一个概念，并返回概念的名字。
        如果概念已经存在（可以通过 find_similar_concept 查询），则返回相似的概念的名字。
        如果注册失败且概念不存在，则返回 None。
        """
        name = self.register_concept(concept, specific=specific, register_action=register_action)
        if name is not None:
            return name
        return self.knowledge.find_similar_concept(concept)

    def register_universal_constant(self, intrinsic: Intrinsic,
                                    def_exp: Exp,
                                    specific: SpecificModel) -> str | None:
        # An universal constant can't be registered via another universal constant
        expr_raw: Exp = sentence.parse_sympy_to_exp(str(sp.simplify(specific._sympy_of_raw_defi(specific.expand_exp(def_exp)))))
        raw_atoms: set[str] = {str(item) for item in expr_raw.all_atoms}
        # If the potential univ. const. is essentially a number or a combination of universal constants,
        # it should be ignored.
        if expr_raw.type == "Number" or \
            raw_atoms.issubset(set(self.universal_constants)) or \
                self.is_too_complicated(def_exp, specific):
            return None
        # If the potential univ. const.'s definition contains universal constants,
        # it should be checked if the dependent universal constants are really necessary.
        if any([its in str(expr_raw) for its in self.universal_constants]):
            if self.check_univ_dependence(expr_raw, specific):
                return None
        res = self.knowledge.eval_manybody_intrinsic(intrinsic, [])
        if self.is_numerically_reduntant(res, specific):
            return None
        rationalize_res = fractions.Fraction(res.mean).limit_denominator(100)
        if res.std < 1e-3 and abs(res.mean - rationalize_res) < res.std * 5:
            return None
        name = self.knowledge.register_expr(Expression.Intrinsic(intrinsic))
        if name is not None:
            print("\033[1m" + f"Registered New Universal Constant: {name} = {res} defined by {intrinsic}" + "\033[0m")
            self.memory.register_action(name, init=5.0)
        return name

    def check_univ_dependence(self, expr_raw: Exp, specific: SpecificModel):
        univs: set[str] = {str(item) for item in expr_raw.all_atoms} & set(self.universal_constants)
        if not univs:
            raise ValueError("No universal constants found in the expression.")
        primes: list[int] = list(sp.primerange(1, len(univs) * 10))
        subs_dict1: dict[str, float] = {name: primes[i] for i, name in enumerate(univs)}
        subs_dict2: dict[str, float] = {name: primes[-(i+1)] for i, name in enumerate(univs)}
        str_raw1: str = str(expr_raw)
        str_raw2: str = str(expr_raw)
        for name in univs:
            pattern: str = rf"{re.escape(name)}(?!\d)"
            str_raw1 = re.sub(pattern, str(subs_dict1[name]), str_raw1)
            str_raw2 = re.sub(pattern, str(subs_dict2[name]), str_raw2)
        expr_raw1: Exp = Exp(str_raw1)
        expr_raw2: Exp = Exp(str_raw2)
        res1: ExpData = self.knowledge.eval(expr_raw1, specific.experiment)
        res2: ExpData = self.knowledge.eval(expr_raw2, specific.experiment)
        if (res1 - res2).is_zero:
            return False
        return True

    specific: Dict[str, SpecificModel]
    objmodel: Dict[str, ObjectModel]

    def register_intrinsics(self, CQinfos: Dict[str, ConservedInfo], intrinsic_mode: int,
                            is_intrinsic_concept: bool) -> List[str]:
        registered_name_list = []
        for info_name, info in CQinfos.items():
            assert info.is_intrinsic and info.relevant_id is not None
            exp_name = info.exp_name
            specific: SpecificModel = self.specific[exp_name]
            experiment = self.specific[exp_name].experiment
            relevant_id = list(info.relevant_id)
            expr = info.exp
            if self.is_too_complicated(expr, specific):
                continue
            if len(relevant_id) == 0:
                # print(f"Found universal constant defined by: {expr}")
                iexp_config = IExpConfig.From(exp_name)
                intrinsic = Intrinsic.From(SExp.Mk(iexp_config, expr))
                name = self.register_universal_constant(intrinsic,
                                                        expr,
                                                        self.specific[exp_name])
            elif len(relevant_id) == 1 and is_intrinsic_concept:
                print(f"Found intrinsic relation: {expr} with relevant_id = {relevant_id}")
                expr_raw: Exp = sentence.parse_sympy_to_exp(str(sp.simplify(specific._sympy_of_raw_defi(specific.expand_exp(expr)))))
                if ({str(item) for item in expr_raw.all_atoms} & set(self.universal_constants)):
                    continue
                id, obj_type = relevant_id[0], str(experiment.get_obj_type(relevant_id[0]))
                iexp_config = IExpConfig.Mk(
                    obj_type,
                    IExpConfig.From(exp_name),
                    id
                )
                intrinsic = Intrinsic.From(SExp.Mk(iexp_config, expr))
                name = self.register_new_intrinsic(obj_type, intrinsic, experiment)
                if name is not None:
                    registered_name_list.append(info_name)
            if intrinsic_mode == 2 and len(relevant_id) == 2 and is_intrinsic_concept:
                # intrinsic_mode = 2 意味着通过选定“标准物体”的方式来定义内禀概念
                # 非必要情况下，不建议使用这种方式来定义内禀概念，因为它会引入更多的物理学常数增加回归的复杂度。
                print(f"Found intrinsic relation: {expr} with relevant_id = {relevant_id}")

                id, obj_type = relevant_id[1], str(experiment.get_obj_type(relevant_id[1]))
                iexp_config = IExpConfig.Mk(
                    obj_type, IExpConfig.From(exp_name), id
                )
                id1, obj_type1 = relevant_id[0], str(experiment.get_obj_type(relevant_id[0]))
                standard_obj = experiment.get_obj(relevant_id[0])
                standard_obj.random_settings()
                standard_object_name = self.knowledge.register_object(standard_obj)
                iexp_config = IExpConfig.Mkfix(
                    standard_object_name, iexp_config, id1
                )
                intrinsic = Intrinsic.From(SExp.Mk(iexp_config, expr))
                name = self.register_new_intrinsic(obj_type, intrinsic, experiment)
                if name is None:
                    continue
                registered_name_list.append(info_name)
                new_exp = Exp.Atom(AtomExp.VariableIds(name, [id])) / expr
                new_info = self.specific[exp_name].make_conserved_info(None, new_exp)
                if new_info.is_intrinsic and new_info.relevant_id == {id1}:
                    new_iexp_config = IExpConfig.Mk(
                        obj_type1, IExpConfig.From(exp_name), id1
                    )
                    intrinsic = Intrinsic.From(SExp.Mk(new_iexp_config, new_exp))
                    name = self.register_new_intrinsic(obj_type1, intrinsic, experiment)
                    if name is not None:
                        registered_name_list.append(info_name)
        return registered_name_list

    def register_new_intrinsic(self, obj_type: str, intrinsic: Intrinsic,
                               experiment: ExpStructure) -> str:
        if not self.objmodel.__contains__(obj_type):
            self.objmodel[obj_type] = self.newObjectModel(obj_type)
        name = self.objmodel[obj_type].register_intrinsic(intrinsic, experiment)
        if name is not None:
            print("\033[1m" + f"Registered New Concept: {name} = {intrinsic}" + "\033[0m")
            self.memory.register_action(name, init=8.0)
        return name

    def register_experiments(self, exp_structures: list[ExpStructure]):
        for exp_structure in exp_structures:
            self.register_experiment(exp_structure)

    def update_experiment(self, exp_structure: ExpStructure):
        # 不推荐使用 (many TODOs)
        # 知识库中已经有老版本的实验（exp_name 相同），需要更新实验配置文件。
        name = exp_structure.exp_name
        if name not in self.specific:
            print(f"Warning: Experiment {name} doesn't exist, update failed.")
            return
        self.knowledge.update_expstruct(name, exp_structure)
        self.specific[name] = SpecificModel(name, self.knowledge)
        # TODO, obliviate the old knowledge about the experiment, including specific model, memory.history and so on. 
        # TODO, update the general conclusions' valid experiments list.
        props = exp_structure.geometry_info
        for concept in self.specific[name].experiment.original_concept:
            self.register_basic_concept(concept)
        # Try to add the propositions to the specific model one by one
        for prop in tqdm(props, desc=f"geommetry info in \033[1m<{name}>\033[0m Add to Specific model"):
            # Comparison is made based on the Exp object
            expr: Exp = prop.unwrap_exp
            if prop.is_zero:
                _ = self.specific[name].append_zero_exp(expr, trust_me_and_no_test=True)
            elif prop.is_conserved:
                _ = self.specific[name].append_conserved_exp(expr, trust_me_and_no_test=True)

    def register_experiment(self, exp_structure: ExpStructure):
        name = exp_structure.exp_name
        if name in self.specific:
            print(f"Warning: Experiment {name} already exists, register failed.")
            return
        self.knowledge.register_expstruct(name, exp_structure)
        # 初始化 specific model 和 memory
        # 这一部分代码保持与 __init__ 中的一致
        self.specific[name] = SpecificModel(name, self.knowledge)
        self.memory.register_experiment(name)
        for concept in self.specific[name].experiment.original_concept:
            self.register_basic_concept(concept)
        # Try to add the propositions to the specific model one by one
        props = exp_structure.geometry_info
        for prop in tqdm(props, desc=f"geommetry info in \033[1m<{name}>\033[0m Add to Specific model"):
            # Comparison is made based on the Exp object
            expr: Exp = prop.unwrap_exp
            if prop.is_zero:
                _ = self.specific[name].append_zero_exp(expr)
            elif prop.is_conserved:
                _ = self.specific[name].append_conserved_exp(expr)

    def newObjectModel(self, obj_type: str) -> ObjectModel:
        return ObjectModel(obj_type, self.knowledge)

    def calc_expr(self, expr: Exp | str, exp_name: str) -> ExpData:
        if self.specific[exp_name].experiment is None:
            raise ValueError("The experiment is not initialized.")
        expr = Exp(expr) if isinstance(expr, str) else expr
        return self.knowledge.eval(self.specific[exp_name].expand_exp(expr),
                                   self.specific[exp_name].experiment)

    @property
    def universal_constants(self) -> list[str]:
        obj_intrinsic: list[str] = []
        for obj_type, obj_model in self.objmodel.items():
            obj_intrinsic.extend(list(obj_model.attr.keys()))
        return [name for name in self.knowledge.fetch_intrinsic_concepts.keys()
                if name not in obj_intrinsic]

    def is_numerically_reduntant(self, res: ConstData,
                                 specific: SpecificModel) -> bool:
        exist_univs_vals: list[ConstData] = [self.knowledge.eval(Exp(name), specific.experiment).const_data
                                             for name in self.universal_constants]
        ansatz_power: list[tuple[int, int]] = [(-2, 0), (0, -2), (-1, -1),
                                               (-1, 0), (0, -1),
                                               (-1, 1), (1, -1),
                                               (0, 1), (1, 0), (-1, 2), (2, -1),
                                               (0, 2), (2, 0), (1, 1)]
        ansatz_coe: list[tuple[int, int]] = [(1, -1), (1, 1), (2, -1), (1, -2),
                                             (3, -1), (1, -3),]
        ansatz_multiple: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for exist_univ_val in exist_univs_vals:
            if ExpData.from_const_data(res - exist_univ_val).is_zero:
                return True
        for i in range(len(exist_univs_vals)):
            if any([ExpData.from_const_data(i2*res + j2*exist_univs_vals[i].__powi__(i1)).is_zero
                    for (i1, j1) in ansatz_power for (i2, j2) in ansatz_coe]):
                return True
        for i in range(len(exist_univs_vals)):
            for j in range(i):
                if any([ExpData.from_const_data(i2*res + j2*exist_univs_vals[i].__powi__(i1) * exist_univs_vals[j].__powi__(j1)).is_zero
                        for (i1, j1) in ansatz_power for (i2, j2) in ansatz_coe]):
                    return True
        for i in range(len(exist_univs_vals)):
            if any([(ExpData.from_const_data(res - mul * exist_univs_vals[i].__powi__(-1)).is_zero
                     or ExpData.from_const_data(res + mul * exist_univs_vals[i].__powi__(-1)).is_zero)
                    for mul in ansatz_multiple]):
                return True
        return False

    def is_initial_condition(self, concept: Concept | Exp, specific: SpecificModel) -> bool:
        def closure(expr: sp.Expr) -> bool:
            t0 = sp.Symbol("t_0")
            b = sp.Wild("b")
            try:
                match_plus = True if expr.find(t0*sp.Derivative(b, t0) - b) else False
            except Exception:
                match_plus = False
            try:
                match_minus = True if expr.find(-t0*sp.Derivative(b, t0) + b) else False
            except Exception:
                match_minus = False
            return match_plus or match_minus
        if isinstance(concept, Exp):
            exps_lst: list[Exp] = [concept]
        else:
            exps_lst: list[Exp] = self.knowledge.specialize(concept, specific.exp_name)
        if exps_lst is None:
            return False
        exprs_lst: list[sp.Expr] = [sp.simplify(specific._sympy_of_raw_defi(specific.expand_exp(exp)))
                                    for exp in exps_lst]
        if all([closure(expr) for expr in exprs_lst]):
            return True
        return False

    def is_concept_pure_geometric(self, concept: Concept | Exp, specific: SpecificModel) -> bool:
        if specific is None:
            return False
        if "Concept_Mk0" in str(type(concept)):
            return False  # NOTE:
        if isinstance(concept, Exp):
            lst_atom_names: list[str] = [atomexpr.name
                                         for atomexpr in concept.all_atoms]
        else:
            lst_atom_names: list[str] = [atomexpr.name
                                         for atomexpr in concept.exp.all_atoms]
        geo_vars: list[str] = [i.name for i in specific.fetch_geometry_variables]
        if not lst_atom_names:
            return True
        if all([name in geo_vars for name in lst_atom_names]):
            return True
        return False

    def is_concept_propto_const(self, concept: Concept, specific: SpecificModel):
        exprs: list[Exp] = self.knowledge.specialize(concept, specific.exp_name)
        if not exprs:
            return True
        # An universal constant can't be registered via another universal constant
        exprs_raw: list[Exp] = [sentence.parse_sympy_to_exp(str(sp.simplify(specific._sympy_of_raw_defi(specific.expand_exp(exp)))))
                                for exp in exprs]
        raw_atoms: set[str] = set.union(*[{str(item) for item in expr_raw.all_atoms}
                                          for expr_raw in exprs_raw])
        univs: set[str] = set(self.universal_constants) & raw_atoms
        if not univs:
            return False
        flag_propto: bool = True
        for expr_raw in exprs_raw:
            str_raw: str = str(expr_raw)
            for name in univs:
                pattern: str = rf"{re.escape(name)}(?!\d)"
                str_raw = re.sub(pattern, str(0), str_raw)
            expr_raw: Exp = Exp(str_raw)
            res: ExpData = self.knowledge.eval(expr_raw, specific.experiment)
            if not (res.is_err or res.is_zero):
                flag_propto = False
                break

        return flag_propto

    def is_too_complicated(self, expr: Exp, specific: SpecificModel) -> bool:
        if self.possess_large_coes(expr, specific):
            return True

        expr_raw_sp: sp.Expr = sp.simplify(specific._sympy_of_raw_defi(specific.expand_exp(expr)))
        pattern = r'\(t_0, ([1-9]\d*)\)'
        matches = re.finditer(pattern, str(expr_raw_sp))
        results = []
        for match in matches:
            full_str = match.group(0)      # 完整匹配的字符串，如 "(t_0, 123)"
            number = int(match.group(1))   # 提取数字并转为整数
            results.append((full_str, number))
        if any([number > 2 for _, number in results]):
            return True

        # For pattern Derivative(t_0**#, t_0)
        pattern_2 = r'Derivative\(t_0\*\*([1-9]\d*), t_0\)'
        matches_2 = re.finditer(pattern_2, str(specific._sympy_of_raw_defi(specific.expand_exp(expr))))
        results_2 = []
        for match in matches_2:
            full_str = match.group(0)      # 完整匹配的字符串，如 "Derivative(t_0**123, t_0)"
            number = int(match.group(1))   # 提取数字并转为整数
            results_2.append((full_str, number))
        if any([number > 1 for _, number in results_2]):
            return True

        return False

    def possess_large_coes(self, expr: Exp, spm: SpecificModel,
                           upper_lim: float = 24.0) -> bool:
        expr_raw_sp: sp.Expr = sp.simplify(spm._sympy_of_raw_defi(spm.expand_exp(expr)))
        numbers = []
        for node in sp.preorder_traversal(expr_raw_sp):
            if isinstance(node, sp.Number):
                numbers.append(float(node))
        if any([abs(number) > upper_lim for number in numbers]):
            return True
        return False

    def remove_meaningless_sums(self, spm: SpecificModel, name_list: list[str]):
        obj_type_num: dict[str, int] = {}  # {obj_type: num}
        for obj in spm.experiment.obj_info.values():
            tp: str = str(obj[0])
            if tp not in obj_type_num:
                obj_type_num[tp] = 0
            obj_type_num[tp] += 1

        res_name_list: list[str] = []
        for name in name_list:
            concept: Concept | None = self.knowledge.fetch_concept_by_name(name)
            if concept is None or 'Sum' not in str(concept):
                res_name_list.append(name)
                continue
            flag_meaningless: bool = False
            for tp in concept.objtype_id_map:
                if tp not in obj_type_num:
                    flag_meaningless = True
                    break
                if obj_type_num[tp] == 1 or len(concept.objtype_id_map[tp]) >= obj_type_num[tp]:
                    flag_meaningless = True
                    break
            if not flag_meaningless:
                res_name_list.append(name)

        return res_name_list

    def check_single_relevant_direction(self, expr: Exp, spm: SpecificModel) -> dict[str, set[str]]:
        all_atom_names: set[str] = {atm.name for atm in expr.all_atoms}
        relevant_directions = {}  # {direction: set[pattern]}
        for atom in all_atom_names - {"t"} - {intr for intr in self.knowledge.fetch_intrinsic_concepts.keys()
                                              if "pos" not in str(
                                                  spm._sympy_of_raw_defi(
                                                      spm.expand_exp(
                                                          Exp(str(self.knowledge.fetch_intrinsic_by_name(intr)).split("|- ")[1][:-1])
                                                        )
                                                    )
                                                )
                                              }:
            flag = False
            for patt, equivs in self.memory.concept_clusters.items():
                if atom in equivs:
                    if equivs[atom] not in relevant_directions:
                        relevant_directions[equivs[atom]] = set()
                    relevant_directions[equivs[atom]].add(patt)
                    flag = True
                    break
            if not flag:
                relevant_directions = {}
                break
            if len(relevant_directions) > 1:
                break
        return relevant_directions
