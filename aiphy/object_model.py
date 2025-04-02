from typing import Dict, List
from .interface import Expression, Intrinsic, Proposition, Knowledge, ConstData
from .interface import Objstructure, ExpData, ExpStructure, Exp


class ObjectModel:
    knowledge: Knowledge
    obj_type: str
    attr: Dict[str, Intrinsic]

    # 比如对于 obj_type = 'Particle'， 可能注册两个内禀概念：引力质量和惯性质量，
    # 经过实验发现，这两种方式定义的质量是相等的。那么就可以得到一个结论：引力质量 = 惯性质量。
    # 将这样的结论存入 conclusion_about_attr 中，并在最终注册进 Knowledge 时只注册其中一个（内禀概念的判重）。
    conclusion_about_attr: Dict[str, Proposition]

    # 每个 attr 测量了 obj_type 物体的某个属性。但是测量是有量程的，一旦超过这个量程，测量就会失败。
    # 所以倾向于选择量程大的测量方式。在此基础上进行判重。

    def __init__(self, obj_type: str, knowledge: Knowledge):
        self.knowledge = knowledge
        self.obj_type = obj_type
        self.attr = {}
        self.conclusion_about_attr = {}

    def to_json(self) -> Dict:
        return {
            'obj_type': self.obj_type,
            'attr': {name: str(attr) for name, attr in self.attr.items()},
            'conclusion_about_attr': {name: str(prop) for name, prop in self.conclusion_about_attr.items()}
        }

    @classmethod
    def from_json(cls, data: Dict, knowledge: Knowledge):
        obj = cls(data['obj_type'], knowledge)
        obj.attr = {name: Intrinsic(attr) for name, attr in data['attr'].items()}
        obj.conclusion_about_attr = {name: Proposition(prop) for name, prop in data['conclusion_about_attr'].items()}
        return obj

    @property
    def universal_constants(self) -> list[str]:
        return [name for name, con in self.knowledge.fetch_intrinsic_concepts.items()
                if "->" not in str(con)]

    def _is_new_intrinsic(self, intrinsic: Intrinsic, experiment: ExpStructure,
                          debug: bool = False) -> bool:
        input_objtypes = intrinsic.input_objtypes
        if not (len(input_objtypes) == 1 and str(input_objtypes[0]) == self.obj_type):
            raise ValueError(f"Input objtypes of intrinsic {intrinsic} should be [{self.obj_type}]")
        id = intrinsic.input_objids[0]

        arg_N = 50
        arg_repeat = 20

        # 获取这个内禀概念的测量是在哪个实验中进行的
        measure_expname: str = intrinsic.measure_experiment
        # 从那个实验里获取这个物体的结构体
        objstructure = self.knowledge.fetch_objstruct_from_expstruct(measure_expname, id)
        result_list = []
        attr_result_list: Dict[str, List[ConstData | None]] = {attr_name: [] for attr_name in self.attr}
        # Universal constants should also be considered
        univ_result_dict: Dict[str, ExpData] = {univ: self.knowledge.eval(Exp(univ), experiment)
                                                for univ in self.universal_constants}
        for _ in range(arg_N):
            # 在结构体的参数范围内随机选取一个物体
            objstructure.random_settings()
            # 测量计算内禀概念的值
            result: ConstData | None = self.knowledge.eval_onebody_intrinsic(intrinsic, objstructure)
            # 如果不出意外没有数值错误的话，那么 result 应该是 ConstData 类型的
            result_list.append(result)
            # 测量计算 self.attr
            for attr_name in attr_result_list:
                result: ConstData | None = self.knowledge.eval_onebody_intrinsic(self.attr[attr_name], objstructure)
                attr_result_list[attr_name].append(result)
        # 将长度为 50 的 result_list 压缩成一个大小为 20*50 的 ExpData
        expdata: ExpData = ExpData.wrapped_list_of_const_data(result_list, repeat_time=arg_repeat)
        if not expdata.is_normal:
            return False
        attr_result_list: Dict[str, ExpData] = {
            attr_name: ExpData.wrapped_list_of_const_data(
                attr_result_list[attr_name], repeat_time=arg_repeat) for attr_name in attr_result_list}
        if debug:
            attr_result_list['debug'] = expdata
            self.debug_attr_result_list = attr_result_list

        def sub_mean(data: ExpData) -> ExpData:
            constdata = data.calc_mean
            return data - ExpData.from_elem(constdata.mean, constdata.std, arg_N, arg_repeat)

        # a very trivial way to judge whether the intrinsic is duplicate
        data0 = sub_mean(expdata)
        for attr_name in attr_result_list:
            if attr_name == 'debug':
                continue
            expdata1: ExpData = attr_result_list[attr_name]
            if (data0 / sub_mean(expdata1)).is_const:
                return False
            if (expdata / (expdata1 * expdata1)).is_const:
                return False
            if (expdata * expdata1).is_const:
                return False
            for univ_data in univ_result_dict.values():
                dt1 = ExpData.from_const(1, 0)
                if (expdata - expdata1 - univ_data).is_zero:
                    return False
                if (expdata - expdata1 + univ_data).is_zero:
                    return False
                if (expdata + expdata1 - univ_data).is_zero:
                    return False
                if (expdata + expdata1 + univ_data).is_zero:
                    return False
                if (expdata - dt1 / expdata1 - univ_data).is_zero:
                    return False
                if (expdata - dt1 / expdata1 + univ_data).is_zero:
                    return False
                if (expdata + dt1 / expdata1 - univ_data).is_zero:
                    return False
                if (expdata + dt1 / expdata1 + univ_data).is_zero:
                    return False
        if len(attr_result_list) > 1:
            attr_name_list = list(attr_result_list.keys())
            for i in range(len(attr_name_list)):
                for j in range(i + 1, len(attr_name_list)):
                    expdata1 = attr_result_list[attr_name_list[i]]
                    expdata2 = attr_result_list[attr_name_list[j]]
                    if (expdata / (expdata1 * expdata2)).is_const:
                        return False
                    if (expdata / (expdata1 / expdata2)).is_const:
                        return False
                    if (expdata / (expdata2 / expdata1)).is_const:
                        return False

        return True

    def register_intrinsic(self, intrinsic: Intrinsic, experiment: ExpStructure,
                           debug: bool = False) -> str | None:
        if not self._is_new_intrinsic(intrinsic, experiment, debug):
            return None
        name = self.knowledge.register_expr(Expression.Intrinsic(intrinsic))
        if name is not None:
            print(f"\033[1m" + f"Registered New Onebody Intrinsic Concept: {name} = {intrinsic}" + f"\033[0m")
            self.attr[name] = intrinsic
        return name
