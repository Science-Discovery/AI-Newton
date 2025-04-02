import sympy as sp
from fractions import Fraction
from .symbolic import partial_diff, GeoInfoCache

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from copy import deepcopy
from tqdm import tqdm
from typing import List, Tuple, Dict, Set, Any, Literal, Callable
from _collections_abc import Iterator
from .memory import dict_to_json, Memory
from .interface import (
    Knowledge, ExpData, DataStruct, ExpStructure, ConstData, ObjType_Clock,
    Exp, AtomExp, Proposition, Concept, MeasureType, sentence,
    KeyValueHashed
)
from .regression import search_relations_by_pca
from .diffalg import DifferentialRing, diffalg
from .parsing import expand_expression


class ZeroInfo:
    """
    ZeroInfo 类是一个用来存储零量的信息的类
    """
    exp_name: str
    name: str
    exp: Exp

    def __str__(self):
        return f"{self.name}: {self.exp}"

    def from_json(data: Tuple[str, str, str]) -> "ZeroInfo":
        obj = object.__new__(ZeroInfo)
        obj.exp_name = data[0]
        obj.name = data[1]
        obj.exp = Exp(data[2])
        return obj

    def to_json(self) -> Tuple[str, str, str]:
        return self.exp_name, self.name, str(self.exp)


class ConservedInfo:
    """
    ConservedInfo 类是一个用来存储守恒量的信息的类，
    这些信息包括它的取值是否是内禀的，以及它依赖的实验对象编号
    """
    exp_name: str
    name: str
    exp: Exp
    is_intrinsic: bool
    relevant_id: Set[int]

    def __str__(self):
        return f"{self.name}: {self.exp}"

    def __new__(cls, exp_name: str, name: str, exp: Exp,
                is_intrinsic: bool, relevant_id: Set[int]):
        obj = object.__new__(ConservedInfo)
        obj.exp_name = exp_name
        obj.name = name
        obj.exp = exp
        obj.is_intrinsic = is_intrinsic
        obj.relevant_id = relevant_id
        return obj

    def from_json(data: Tuple[str, str, str, bool, List[int]]) -> "ConservedInfo":
        obj = object.__new__(ConservedInfo)
        obj.exp_name = data[0]
        obj.name = data[1]
        obj.exp = Exp(data[2])
        obj.is_intrinsic = data[3]
        obj.relevant_id = set(data[4]) if obj.is_intrinsic else None
        return obj

    def to_json(self) -> Tuple[str, str, str, bool, List[int]]:
        res = (
            self.exp_name, self.name, str(self.exp),
            self.is_intrinsic, list(self.relevant_id) if self.relevant_id is not None else []
        )
        return res


class ConclusionSet:
    """
    ConclusionSet 类是一个用来存储一组结论的类
    """
    exp_name: str
    knowledge: Knowledge
    conclusion: Dict[str, Proposition]
    conclusion_id: int
    zero_list: Dict[str, ZeroInfo]
    conserved_list: Dict[str, ConservedInfo]

    def __init__(self, knowledge: Knowledge, exp_name: str):
        self.exp_name = exp_name
        self.knowledge = knowledge
        self.conclusion = {}
        self.conclusion_id = 0
        self.zero_list = {}
        self.conserved_list = {}

    def __iter__(self) -> Iterator[str]:
        return iter(self.conclusion)

    def to_json(self) -> Dict[str, Any]:
        return {
            "conclusion": dict_to_json(self.conclusion),
            "conclusion_id": self.conclusion_id,
            "conserved_list": [item.to_json() for item in self.conserved_list.values()],
            "zero_list": [item.to_json() for item in self.zero_list.values()],
        }

    def load_json(self, data: Dict[str, Any]):
        self.conclusion = {k: Proposition(v) for k, v in data["conclusion"].items()}
        self.conclusion_id = data["conclusion_id"]
        self.conserved_list = {}
        for item in data["conserved_list"]:
            info = ConservedInfo.from_json(item)
            self.conserved_list[info.name] = info
        self.zero_list = {}
        for item in data["zero_list"]:
            info = ZeroInfo.from_json(item)
            self.zero_list[info.name] = info

    def keys(self):
        return self.conclusion.keys()

    def values(self):
        return self.conclusion.values()

    def get(self, key: str) -> Proposition:
        return self.conclusion.get(key)

    def remove_conclusion(self, name: str):
        if name in self.conclusion:
            del self.conclusion[name]
        if name in self.zero_list:
            del self.zero_list[name]
        if name in self.conserved_list:
            del self.conserved_list[name]

    def __register_conclusion(self, prop: Proposition):
        self.conclusion_id += 1
        name = f"P{self.conclusion_id}"
        self.conclusion[name] = prop
        return name

    def print_conclusions(self):
        for name, prop in self.conclusion.items():
            print(name, prop)

    def exp_hashed(self, exp: Exp) -> KeyValueHashed:
        try:
            return self.knowledge.K.eval_exp_keyvaluehashed(exp, self.exp_name)
        except Exception:
            raise Exception(f'Error occurs in exp_hashed\n  Args: {exp}')

    def already_exist(self, exp: Exp, exp_type: Literal["zero", "const"]) -> bool:
        hashed_value = self.exp_hashed(exp)
        match exp_type:
            case "zero":
                if hashed_value.is_none or hashed_value.is_zero:
                    return True
                for _, info in self.zero_list.items():
                    if self.exp_hashed(info.exp) == hashed_value:
                        return True
            case "const":
                if hashed_value.is_none or hashed_value.is_const:
                    return True
                for _, info in self.conserved_list.items():
                    if self.exp_hashed(info.exp) == hashed_value:
                        return True
        return False

    def append_conserved_exp(self, conserved_exp: Exp, info: ConservedInfo,
                             name: str | None = None) -> str:
        if name is None:
            info.name = self.__register_conclusion(Proposition.IsConserved(conserved_exp))
        self.conserved_list[info.name] = info
        return info.name

    def append_zero_exp(self, zero_exp: Exp, info: ZeroInfo,
                        name: str | None = None) -> str:
        if name is None:
            info.name = self.__register_conclusion(Proposition.IsZero(zero_exp))
        self.zero_list[info.name] = info
        return info.name

    def _sympy_of_raw_defi(self, exp: Exp) -> sp.Expr:
        return self.knowledge.sympy_of_raw_defi(exp, self.exp_name)

    def fetch_differential_ring(self) -> Tuple[sp.Symbol, Set[sp.Symbol], Set[sp.Symbol], Set[sp.Function], Set[sp.Symbol], DifferentialRing]:
        """
        This function is used to fetch the differential ring for the current conclusions set.
        Return a tuple of:
        1. argument: sp.Symbol, the argument of the differential ring
        2. all_normal_symbols: Set[sp.Symbol], all normal symbols in the differential ring
        3. all_intrinsic_symbols: Set[sp.Symbol], all intrinsic symbols in the differential ring
        4. all_functions: Set[sp.Function], all functions in the differential ring
        5. all_symbols_ne_zero: Set[sp.Symbol], all symbols except the zero symbols
        6. ring: DifferentialRing, the differential ring
        """
        all_normal_symbols = set()
        all_intrinsic_symbols = set()
        all_functions = set()
        for name, info in self.conserved_list.items():
            if info.is_intrinsic:
                all_intrinsic_symbols.add(sp.Symbol(name))
            else:
                all_normal_symbols.add(sp.Symbol(name))
        all_symbols_ne_zero = all_normal_symbols | all_intrinsic_symbols
        argument = sp.Symbol("t_0")
        for value in self.values():
            symb_atoms = self._sympy_of_raw_defi(value.unwrap_exp).atoms(sp.Symbol)
            if symb_atoms.__contains__(argument):
                symb_atoms.remove(argument)
            all_symbols_ne_zero |= symb_atoms
            all_intrinsic_symbols |= symb_atoms
            all_functions |= self._sympy_of_raw_defi(value.unwrap_exp).atoms(sp.Function)
        for value in self.zero_list.values():
            symb = self._sympy_of_raw_defi(value.exp)
            if symb.is_Symbol:
                all_symbols_ne_zero.remove(symb)
        ring = DifferentialRing([('lex', list(all_functions)),
                                 ('lex', list(all_normal_symbols)),
                                 ('lex', list(all_intrinsic_symbols))])
        return argument, all_normal_symbols, all_intrinsic_symbols, all_functions, all_symbols_ne_zero, ring

    def diffalg_representation(self,
                               prop_complexity_list: List[Tuple[int, str]] = None,
                               debug=False) -> Tuple[Dict[sp.Expr, sp.Expr], Dict[sp.Expr, sp.Expr],
                                                     List[Tuple[int, diffalg]]]:
        """
        这个函数的目的是将当前实验中的 conserved 和 zero 的表达式整理并取 minimal 表示
        Return a tuple of:
        1. subs_dict: Dict[sp.Expr, sp.Expr], the substitution dictionary
        2. inverse_dict: Dict[sp.Expr, sp.Expr], the inverse substitution dictionary
        3. result: List[Tuple[int, diffalg]], the minimal representation of the conserved and zero expressions
        """
        name_list: List[str] = list(self.keys())
        if prop_complexity_list is not None:
            name_list = prop_complexity_list
        else:
            name_list: List[Tuple[int, str]] = [(self.knowledge.raw_complexity(self.get(x), self.exp_name), x) for x in name_list]
        name_list = sorted(name_list, key=lambda x: x[0])
        # 第一步：提取 DifferentialRing
        argument, _, _, _, all_symbols_ne_zero, ring = self.fetch_differential_ring()
        # 第二步：把无意义的 conclusion 去掉
        ideal: diffalg = diffalg(ring)
        ideal.insert_new_ineqs(argument)
        added_ineqs = {argument}
        subs_dict = dict()
        inverse_dict = dict()
        result = []
        for complexity, name in name_list:
            prop = self.get(name)
            sp_expr = self._sympy_of_raw_defi(prop.unwrap_exp).subs(subs_dict, simultaneous=True)\
                .doit().subs(inverse_dict, simultaneous=True)
            if prop.prop_type == "IsConserved":
                if sp_expr.is_Function:
                    subs_dict[sp_expr] = sp.Symbol(sp_expr.name)
                    inverse_dict[sp.Symbol(sp_expr.name)] = sp_expr
                start = time.time()
                ideal = insert_to_ideal(ideal, sp_expr - sp.Symbol(name), all_symbols_ne_zero, added_ineqs, debug)
                if time.time() - start > 5:
                    print(f'Warning: time consuming in diffalg_representation: {time.time() - start}')
                    break
            elif prop.prop_type == "IsZero":
                new_eq = sp_expr.as_numer_denom()[0]
                if new_eq.func == sp.Add and len(new_eq.args) == 2:
                    atom1 = new_eq.args[0]
                    atom2 = new_eq.args[1]

                    def is_neg_symbol(x: sp.Expr):
                        return x.func == sp.Mul and x.args[0] == -1 and x.args[1].is_Symbol
                    if is_neg_symbol(atom2):
                        atom2 = -atom2
                    else:
                        atom1 = -atom1
                    subs_dict[atom2] = atom1
                start = time.time()
                ideal = insert_to_ideal(ideal, sp_expr, all_symbols_ne_zero, added_ineqs, debug)
                if time.time() - start > 5:
                    print(f'Warning: time consuming in diffalg_representation: {time.time() - start}')
                    break
            result.append((complexity, deepcopy(ideal)))
        return subs_dict, inverse_dict, result


class CQCalculator:
    knowledge: Knowledge
    experiment: ExpStructure
    experiment_control: Dict[int, List[ExpStructure]]
    info: Dict[int, Dict[str, ConservedInfo]]
    data: Dict[int, DataStruct]
    repeat_time: int = 20

    def __init__(self, exp_name: str, knowledge: Knowledge):
        self.knowledge = knowledge
        self.exp_name = exp_name
        self.experiments_reset()
        self.info = {}
        self.data = {}

    def experiments_reset(self):
        if not hasattr(self, 'experiment') or self.experiment is None:
            self.experiment = self.knowledge.fetch_expstruct(self.exp_name)
        self.experiment.random_settings()
        self.experiment.collect_expdata(MeasureType.default())
        self.experiment_control = {}

    def insert_cq_info(self, info: ConservedInfo):
        """
        check if `sp_expr` can be represented as combinations of simpler conserved info
        """
        ids = info.relevant_id if info.is_intrinsic else self.experiment.all_ids | {-1}
        for id in ids:
            if id != -1 and self.experiment.get_obj_type(id) == ObjType_Clock:
                continue
            if not self.info.__contains__(id):
                self.info[id] = {}
            self.info[id][info.name] = info

    def complement_experiments(self, id: int, num: int):
        if not self.experiment_control.__contains__(id):
            self.experiment_control[id] = []
        while len(self.experiment_control[id]) < num:
            new_exp = self.experiment.copy()
            if id == -1:
                new_exp.random_set_exp_para()
            else:
                new_exp.random_set_obj(id)
            new_exp.collect_expdata(MeasureType.default())
            self.experiment_control[id].append(new_exp)

    def complement_experiments_test(self, num: int = 20) -> list[ExpStructure]:
        if not hasattr(self, 'experiment_test'):
            self.experiment_test: list[ExpStructure] = []
        while len(self.experiment_test) < num:
            experiment_test = self.experiment.copy()
            experiment_test.random_settings()
            experiment_test.collect_expdata(MeasureType.default())
            self.experiment_test.append(experiment_test)

    def calc_relations(self, debug=False) -> Tuple[Set[Exp], Set[Exp]]:
        list_info = set()
        for id in self.info.keys():
            list_info = list_info | set(self.info[id].keys())
        print('calc_relations between ', list_info)
        res_const, res_zero = set(), set()
        for id in self.info.keys():
            if len(self.info[id]) < 2:
                continue
            self.complement_experiments(-1, 20)
            self.complement_experiments(id, 10)
            self.data[id] = DataStruct.empty()
            for name, info in self.info[id].items():
                lst = []
                test_experiment_list = self.experiment_control[-1]
                if id != -1:
                    test_experiment_list = test_experiment_list + self.experiment_control[id]
                for experiment in test_experiment_list:
                    expdata = self.knowledge.eval(info.exp, experiment)
                    lst.append(expdata.const_data if expdata.is_const else None)
                wrapped_data: ExpData = ExpData.wrapped_list_of_const_data(lst, self.repeat_time)
                self.data[id].add_data(AtomExp(name), wrapped_data)
            if debug:
                print(f"Data for {id}: {self.data[id]}")
            res = search_relations_by_pca(self.data[id])
            for exp, value in res:
                if value.is_const:
                    res_const.add(exp)
                elif value.is_zero:
                    res_zero.add(exp)
        self.result = res_const, res_zero
        return self.result


class SpecificModel:
    """
    SpecificModel 类是掌管特定实验和实验对照组的物理学家模型，
    一个 experiment 对象，存储了这个实验的具体信息和某个随机参数下的实验结果；
    以及一个 experiment_control 字典，是相对于 experiment 的实验对照组，表达了在控制变量的条件下做实验获得的新结果。
    """
    exp_name: str
    knowledge: Knowledge
    memory: Memory
    experiment: ExpStructure
    experiment_control: Dict[int, List[ExpStructure]]
    # 保持其他实验对象不变，改变实验对象 id 并进行实验获得的结果存储在 experiment_control[id] 中
    # id = -1 代表保持所有实验对象不变，只改变实验控制参数
    conclusions: ConclusionSet
    intrinsic_buffer: Dict[str, ConservedInfo]
    # 保证 conserved_list 与 zero_list 对应了 memory.conclusion 中的结论
    num_threads: int

    def __init__(self, exp_name: str, knowledge: Knowledge,
                 reset: bool = True):
        """
        初始化一个 SpecificModel 对象，需要提供实验的名称和实验的结构
        """
        self.exp_name = exp_name
        self.knowledge = knowledge
        if reset:
            self.experiments_reset()
        self.conclusions = ConclusionSet(self.knowledge, self.exp_name)
        self.intrinsic_buffer = {}
        self.num_threads = None

    def experiments_reset(self):
        if not hasattr(self, 'experiment') or self.experiment is None:
            self.experiment = self.knowledge.fetch_expstruct(self.exp_name)
        self.experiment.random_settings()
        self.experiment.collect_expdata(MeasureType.default())
        self.experiment_control = {}

    def complement_experiments(self, id: int, num: int):
        if not self.experiment_control.__contains__(id):
            self.experiment_control[id] = []
        while len(self.experiment_control[id]) < num:
            new_exp = self.experiment.copy()
            if id == -1:
                new_exp.random_set_exp_para()
            else:
                new_exp.random_set_obj(id)
            new_exp.collect_expdata(MeasureType.default())
            self.experiment_control[id].append(new_exp)

    def complement_experiments_test(self, num: int = 20):
        if not hasattr(self, 'experiment_test'):
            self.experiment_test: list[ExpStructure] = []
        while len(self.experiment_test) < num:
            experiment_test = self.experiment.copy()
            experiment_test.random_settings()
            experiment_test.collect_expdata(MeasureType.default())
            self.experiment_test.append(experiment_test)

    def generate_data_struct(self, exprs: List[AtomExp]) -> DataStruct:
        """
        这个函数的目的是根据一组原子表达式在 self.experiment 下求值生成一个 DataStruct 对象
        """
        DS = DataStruct.empty()
        for atom_exp in exprs:
            try:
                expdata: ExpData = self.knowledge.eval(Exp.Atom(atom_exp), self.experiment)
                # (All evaluation in self.knowledge.eval should be performed on Exp objects, not AtomExp objects)
            except Exception:
                # 表达式中有些变量缺少实验数据，所以无法求值
                continue
            DS.add_data(atom_exp, expdata)
        return DS

    def to_json(self) -> Dict[str, str]:
        return {
            "exp_name": self.exp_name,
            "conclusions": self.conclusions.to_json(),
        }

    def load_json(self, data: Dict[str, Any]):
        assert data["exp_name"] == self.exp_name
        self.conclusions.load_json(data["conclusions"])

    def reduce_conclusions(self, debug=False, parallel=True):
        """
        这个函数的目的是将当前实验中的所有的 conserved 和 zero 的表达式整理并取 minimal 表示
        """
        print(f"Reducing {len(self.conclusions.keys())} conclusions")
        name_list: List[str] = list(self.conclusions.keys())
        name_list: List[Tuple[int, str]] = [(self.knowledge.complexity(self.conclusions.get(x), self.exp_name), x) for x in name_list]
        name_list = sorted(name_list, key=lambda x: x[0])
        if debug:
            print("Conclusions:")
            for complexity, name in name_list:
                print(name, self.conclusions.get(name), complexity)
        # 第一步：提取 DifferentialRing
        argument, _, all_intrinsic_symbols, _, all_symbols_ne_zero, ring = self.conclusions.fetch_differential_ring()

        # 第二步：把无意义的 conclusion 去掉
        ideal: diffalg = diffalg(ring)
        ideal.insert_new_ineqs(argument)
        added_ineqs = {argument}
        if debug:
            print('prepare ring', ring)
        subs_dict = dict()
        inverse_dict = dict()
        maple_out_of_time = False

        def remove_prop(name: str, ideal: diffalg):
            prop: Proposition = self.conclusions.get(name)
            sp_expr = self._sympy_of_raw_defi(prop.unwrap_exp)\
                .subs(subs_dict, simultaneous=True)\
                .doit().subs(inverse_dict, simultaneous=True)
            if prop.prop_type == "IsConserved":
                if sp_expr.is_Function:
                    subs_dict[sp_expr] = sp.Symbol(sp_expr.name)
                    inverse_dict[sp.Symbol(sp_expr.name)] = sp_expr
                info: ConservedInfo = self.conclusions.conserved_list.get(name)
                try:
                    is_new_intrinsic = reduce_conserved_info_by_ideal(ideal, info, sp_expr, all_intrinsic_symbols, argument, debug)
                    if is_new_intrinsic is None:
                        return name
                except Exception:
                    raise Exception(f'Error occurs in reduce_conserved_info_by_ideal\n  Args 1: {ideal}\
                                    \n  Args 2: {info}  \n  Args 3: {sp_expr}  \n  Args 4: {all_intrinsic_symbols}')
            elif prop.prop_type == "IsZero":
                new_eq = sp_expr.as_numer_denom()[0]
                reduce_new_eq = ideal.reduce(new_eq)
                if reduce_new_eq is None or reduce_new_eq.is_zero:
                    return name
            return None

        def parallel_remove_prop(remove_prop_names, name_list, ideal):
            # Get num of threads can be used within the device
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(remove_prop, name, ideal) for complexity, name in name_list]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f'Error occurs in process_prop: {e}. Skip this prop')
                        continue
                    if result is not None:
                        remove_prop_names.add(result)
            return remove_prop_names

        remove_prop_names = set()
        # ideal可更新的循环
        for i, (complexity, name) in enumerate(tqdm(name_list, desc='Reduce Conclusions')):
            if name in remove_prop_names or maple_out_of_time:
                self.conclusions.remove_conclusion(name)
                continue
            prop: Proposition = self.conclusions.get(name)
            sp_expr = self._sympy_of_raw_defi(prop.unwrap_exp)\
                .subs(subs_dict, simultaneous=True)\
                .doit().subs(inverse_dict, simultaneous=True)
            if prop.prop_type == "IsConserved":
                if sp_expr.is_Function:
                    subs_dict[sp_expr] = sp.Symbol(sp_expr.name)
                    inverse_dict[sp.Symbol(sp_expr.name)] = sp_expr
                info: ConservedInfo = self.conclusions.conserved_list.get(name)
                if info.is_intrinsic:
                    self.intrinsic_buffer[name] = info
                start = time.time()
                ideal = insert_to_ideal(ideal, sp_expr - sp.Symbol(name), all_symbols_ne_zero, added_ineqs, debug)
                if time.time() - start > 5:
                    print(f'Warning: time consuming in reduce_conclusions: {time.time() - start}')
                    maple_out_of_time = True
                    continue
                tqdm.write(f'Insert to ideal: {prop.unwrap_exp} is conserved')
            elif prop.prop_type == "IsZero":
                new_eq = sp_expr.as_numer_denom()[0]
                if debug:
                    tqdm.write(f"Check Zero equation: {new_eq} = 0")
                if new_eq.func == sp.Add and len(new_eq.args) == 2:
                    atom1 = new_eq.args[0]
                    atom2 = new_eq.args[1]

                    if atom2.func == sp.Mul and atom2.args[0] == -1 and atom2.args[1].is_Symbol:
                        atom2 = -atom2
                    else:
                        atom1 = -atom1
                    subs_dict[atom2] = atom1
                start = time.time()
                ideal = insert_to_ideal(ideal, sp_expr, all_symbols_ne_zero, added_ineqs, debug)
                if time.time() - start > 5:
                    print(f'Warning: time consuming in reduce_conclusions: {time.time() - start}')
                    maple_out_of_time = True
                    continue
                tqdm.write(f'Insert to ideal: {prop.unwrap_exp} is zero')
            # update the remove name set with new ideal
            remove_prop_names = parallel_remove_prop(remove_prop_names, name_list[i+1:], ideal)
        if debug:
            self.debug_diffalg = ideal

    def possible_intrinsic(self):
        exp_struct: ExpStructure = self.knowledge.fetch_expstruct(self.exp_name)
        if exp_struct.spdim != 1:
            return False
        obj_type = {}
        flag = True
        for obj, info in exp_struct.obj_info.items():
            if str(info[0]) not in obj_type:
                obj_type[str(info[0])] = [obj]
            else:
                flag = False
                break
        if flag:
            return flag

    def check_intrinsic(self, exp: Exp) -> Tuple[bool, Set[int] | None]:
        """
        这个函数的目的是检查一个表达式是否是内禀概念（取值仅依赖于实验对象）
        如果是，返回 True 和它依赖的实验对象编号
        否则，返回 False 和 None
        """
        try:
            expdata: ExpData = self.knowledge.eval(exp, self.experiment)
            if not expdata.is_const:
                return False, None
            self.complement_experiments(-1, 20)
            expdata_list = [expdata.const_data]
            for new_exp in self.experiment_control[-1]:
                new_expdata = self.knowledge.eval(exp, new_exp)
                expdata_list.append(new_expdata.const_data if new_expdata.is_const else None)
            # count the `None` in the list, if more than 10% of the list is `None`, then return False
            if expdata_list.count(None) > len(expdata_list) / 10:
                return False, None
            if not ExpData.wrapped_list_of_const_data(expdata_list, 20).is_const:
                return False, None

            relevant_ids = set()

            ids = self.experiment.all_ids
            for id in ids:
                if self.experiment.get_obj_type(id) == ObjType_Clock:
                    continue
                self.complement_experiments(id, 20)
                expdata_list = [expdata.const_data]
                for new_exp in self.experiment_control[id]:
                    new_expdata = self.knowledge.eval(exp, new_exp)
                    expdata_list.append(new_expdata.const_data if new_expdata.is_const else None)
                if not ExpData.wrapped_list_of_const_data(expdata_list, 20).is_const:
                    relevant_ids.add(id)

            return True, relevant_ids
        except Exception:
            raise Exception(f'Error occurs in check_intrinsic\n  Args: {exp}')

    def make_zero_info(self, name: str, exp: Exp) -> ZeroInfo:
        obj = object.__new__(ZeroInfo)
        obj.exp_name = self.exp_name
        obj.name = name
        obj.exp = exp
        return obj

    def make_conserved_info(self, name: str, exp: Exp,
                            from_gc: bool = False) -> ConservedInfo:
        obj = object.__new__(ConservedInfo)
        obj.exp_name = self.exp_name
        obj.name = name
        obj.exp = exp
        if from_gc:
            obj.is_intrinsic, obj.relevant_id = False, None
        else:
            obj.is_intrinsic, obj.relevant_id = self.check_intrinsic(exp)
        # print('check intrinsic', exp, obj.is_intrinsic, obj.relevant_id)
        return obj

    def test_on_test_experiment(self, exp: Exp, type: Literal['conserved', 'zero'],
                                debug: bool = False) -> bool:
        """
        这个函数的目的是在测试实验上对一个表达式求值
        """
        if not hasattr(self, 'experiment_test'):
            self.experiment_test = []
            for _ in range(10):
                experiment_test = self.experiment.copy()
                experiment_test.random_settings()
                experiment_test.collect_expdata(MeasureType.default())
                self.experiment_test.append(experiment_test)
        score = 10
        for i in range(10):
            data = self.knowledge.eval(exp, self.experiment_test[i])
            if type == 'conserved' and not data.is_conserved:
                score -= 1
            if type == 'zero' and not data.is_zero:
                score -= 1
            if not debug and score < 8:
                return False
        if debug:
            print(f"Test if {exp} is {type}: {score}/10")
        return score >= 8

    def append_conserved_exp(self, conserved_exp: Exp, trust_me_and_no_test: bool = False) -> str | None:
        # Test if the conclusion already exists
        if self.conclusions.already_exist(conserved_exp, "const"):
            return None
        # Test if the conclusion is correct
        if not trust_me_and_no_test:
            if not self.test_on_test_experiment(conserved_exp, 'conserved'):
                return None
        return self.conclusions.append_conserved_exp(conserved_exp, self.make_conserved_info(None, conserved_exp))

    def append_zero_exp(self, zero_exp: Exp, trust_me_and_no_test: bool = False) -> str | None:
        # Test if the conclusion already exists
        if self.conclusions.already_exist(zero_exp, "zero"):
            return None
        # Test if the conclusion is correct
        if not trust_me_and_no_test:
            if not self.test_on_test_experiment(zero_exp, 'zero'):
                return None
        return self.conclusions.append_zero_exp(zero_exp, self.make_zero_info(None, zero_exp))

    def print_conclusion(self):
        print(f"Exp's name = {self.exp_name}, conclusions:")
        self.conclusions.print_conclusions()

    def print_full_conclusion(self):
        for name, exp in self.conclusions.zero_list:
            print(name, "zero:", exp, "=", self.knowledge.K.raw_definition_exp(exp))
        for name, exp in self.conclusions.conserved_list:
            print(name, "conserved:", exp, "=", self.knowledge.K.raw_definition_exp(exp))

    def _sympy_of_raw_defi(self, exp: Exp) -> sp.Expr:
        return self.knowledge.sympy_of_raw_defi(exp, self.exp_name)

    def print_sympy_conclusion(self):
        for name, info in self.conclusions.zero_list.items():
            print(name, "zero:", info.exp, "=", self._sympy_of_raw_defi(info.exp))
        for name, info in self.conclusions.conserved_list.items():
            print(name, "conserved:", info.exp, "=", self._sympy_of_raw_defi(info.exp))

    def list_sympy_conclusion(self) -> List[Tuple[str, sp.Expr]]:
        res = []
        for name, info in self.conclusions.zero_list.items():
            res.append((name, self._sympy_of_raw_defi(info.exp)))
        for name, info in self.conclusions.conserved_list.items():
            res.append((name, self._sympy_of_raw_defi(info.exp)))
        return res

    def filter_relations(self, props: List[Proposition], parallel=True) -> List[Proposition]:
        """
        对 propositions 做一次集体 complexity sort and filter，去掉那些被 specific model 认为是无用的关系
        """
        print("Begin filtering relations")
        # Remove some useless relations from numerical problems
        props = [prop for prop in props
                 if not (prop.prop_type == 'IsZero'
                         and "D" not in str(prop)
                         and ('*' in str(prop) or '/' in str(prop))
                         and ('+' not in str(prop) and '-' not in str(prop))
                         and len(expand_expression(str(prop.unwrap_exp))) == 1)]
        subs_dict, inverse_dict, diffalg_list = self.conclusions.diffalg_representation()
        prop_complexity_list: List[Tuple[Proposition, int]] = [
            (prop, 0) for prop in self.experiment.geometry_info
        ] + [
            (prop, self.knowledge.raw_complexity(prop, self.exp_name)) for prop in props
        ]
        prop_complexity_list.sort(key=lambda x: x[1])
        result_props = []

        def process_prop(prop, complexity):
            idd = -1
            for id in range(len(diffalg_list)):
                if diffalg_list[id][0] <= complexity:
                    idd = id
            sp_expr = self._sympy_of_raw_defi(prop.unwrap_exp).subs(subs_dict, simultaneous=True)\
                .doit().subs(inverse_dict, simultaneous=True)
            ideal = diffalg_list[idd][1] if idd >= 0 else None
            # calculate running time
            now = time.time()
            if ideal is None:
                if not sp_expr.is_zero and not sp_expr.diff(sp.Symbol("t_0")).is_zero:
                    return prop
            elif prop.is_conserved:
                if not reduce_conserved_by_ideal(ideal, sp_expr, sp.Symbol("t_0")):
                    return prop
            elif prop.is_zero:
                reduce_new_eq = ideal.reduce(sp_expr.as_numer_denom()[0])
                if reduce_new_eq is not None and not reduce_new_eq.is_zero:
                    return prop
            # if time.time() - now > 5:
            #     print(f'Warning: time consuming in filter_relations: {time.time() - now}')
            return None

        if parallel:
            # 开启并行
            # Get num of threads can be used within the device
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(process_prop, prop, complexity) for prop, complexity in prop_complexity_list]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f'Error occurs in process_prop: {e}. Skip this prop')
                        continue
                    if result is not None:
                        result_props.append(result)
        else:
            for prop, complexity in prop_complexity_list:
                prop_out = process_prop(prop, complexity)
                if prop_out is not None:
                    result_props.append(prop_out)

        result_props = sorted(result_props, key=lambda x: self.knowledge.complexity(x, self.exp_name))
        return result_props

    @property
    def fetch_geo_info(self) -> GeoInfoCache:
        spatial_coords = self.fetch_spatial_coords
        if hasattr(self, 'geoinfo'):
            return self.geoinfo
        symbs = [
            sp.sympify(self.knowledge.K.parse_exp_to_sympy_str(Exp.Atom(atom), 't_0'))
            for atom in spatial_coords
        ]
        geometry_info = []
        index = 0
        for prop in self.experiment.geometry_info:
            index += 1
            exp = sp.sympify(self.knowledge.K.parse_exp_to_sympy_str(prop.unwrap_exp, 't_0'))
            if prop.is_zero:
                geometry_info.append(exp)
            elif prop.is_conserved:
                geometry_info.append(exp - sp.Symbol(f'Γ_{index}'))
        self.geoinfo = GeoInfoCache(geometry_info, symbs)
        return self.geoinfo

    def is_spatial_partial_diff_trivial(self, exp: Exp) -> bool:
        expr = self._sympy_of_raw_defi(exp)
        func: Set[sp.Function] = expr.atoms(sp.Function)
        return not (len(func & set(self.fetch_geo_info.funcs)) > 0)

    @property
    def fetch_spatial_coords(self):
        return list(filter(
            lambda x: x.name in ["posx", "posy", "posz"],
            self.experiment.original_data
        ))

    @property
    def fetch_geometry_variables(self):
        return list(filter(
            lambda x: x.name != 't',
            self.experiment.original_data
        ))

    def spatial_partial_diff(self, exp: Exp) -> Dict[AtomExp, Exp]:
        try:
            expr = self.knowledge.sympy_of_raw_defi(exp, self.exp_name)
            spatial_coords = self.fetch_spatial_coords
            geoinfo = self.fetch_geo_info
            partial_diffs = partial_diff(expr, geoinfo)
            partial_diffs = [sentence.parse_sympy_to_exp(str(expr)) for expr in partial_diffs]
        except Exception:
            raise Exception(f'Error occurs in spatial_partial_diff\n  Args: {str(exp)}')
        return {spatial_coords[i]: partial_diffs[i] for i in range(len(spatial_coords))}

    def expand_exp(self, exp: Exp) -> Exp:
        """
        这个函数的目的是将一个表达式充分展开为 raw_definition，并实际进行偏导数 Partial 的计算、替换和化简
        """
        res = self.knowledge.K.raw_definition_exp(exp, self.exp_name).doit()
        partials = res.all_partials
        for partial in partials:
            exp0, atom0 = partial.unwrap_partial
            exp_after = self.spatial_partial_diff(exp0)[atom0]
            res = res.replace_partial_by_exp(partial, exp_after)
        return res


def reduce_conserved_by_ideal(ideal: diffalg, sp_expr: sp.Expr, argument: sp.Symbol) -> bool:
    diff_eq = sp.diff(sp_expr, argument).as_numer_denom()[0]
    reduce_diff_eq_result: sp.Expr | None = ideal.gb[0].reduce(diff_eq)
    if reduce_diff_eq_result is None:
        return True
    if not reduce_diff_eq_result.is_zero:
        return False
    eq_reduced = ideal.reduce(sp_expr)
    if eq_reduced is None:
        return True
    if eq_reduced.diff(argument).is_zero:
        return True
    else:
        return False


def reduce_conserved_info_by_ideal(ideal: diffalg, info: ConservedInfo,
                                   sp_expr: sp.Expr, all_intrinsic_symbols: Set[sp.Symbol], argument: sp.Symbol, debug=False
                                   ) -> Tuple[bool]:
    """
    The return value is a tuple of three bools:
    1. if the CQ is a new relation to insert to ideal: bool,
    2. if the CQ is a new intrinsic concept: bool
    3. if the conclusion need to be add to minimal representation of specific model: bool
    If maple ran out of time, the return value is None
    """
    diff_eq = sp.diff(sp_expr, argument).as_numer_denom()[0]
    reduce_diff_eq_result: sp.Expr | None = ideal.gb[0].reduce(diff_eq)
    if reduce_diff_eq_result is None:
        return None
    if not reduce_diff_eq_result.is_zero:
        return info.is_intrinsic
    eq_reduced = ideal.reduce(sp_expr)
    if debug:
        tqdm.write(f'{sp_expr} eq_reduced = {eq_reduced}')
    if eq_reduced is None:
        return None
    if eq_reduced.diff(argument).is_zero:
        # if eq_reduced is composed by all const value, then remove it
        if info.is_intrinsic:
            symbs = eq_reduced.atoms(sp.Symbol)
            if symbs.issubset(all_intrinsic_symbols):
                return None
            else:
                # A new intrinsic relation should be added to the ideal, so the first argument is True
                return True
        return None
    else:
        return info.is_intrinsic


def insert_to_ideal(ideal: diffalg, new_eq: sp.Expr, all_symbols_ne_zero: Set[sp.Symbol], added_ineqs: Set[sp.Symbol], debug=False) -> diffalg:
    new_ineqs = []
    for atom in new_eq.atoms(sp.Symbol):
        if atom in all_symbols_ne_zero and atom not in added_ineqs:
            new_ineqs.append(atom)
            added_ineqs.add(atom)
    if debug:
        tqdm.write(f'add new eq to ideal {new_eq} = 0 and {new_ineqs} <> 0')
    new_ideal = ideal._insert_new_eqs_and_ineqs([new_eq], new_ineqs)
    if new_ideal is None or len(new_ideal.gb) == 0:
        return ideal
    """
    TODO
    if len(new_ideal.gb) > 1:
        # need new inequations to determine which regular differential chain is the correct.
    """
    return new_ideal
