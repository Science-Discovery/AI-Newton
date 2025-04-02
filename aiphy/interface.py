from typing import List, Dict, Set, Tuple
import json
import sympy as sp
import copy
import aiphy.core as aiphy
from aiphy.core import (
    DoExpType,
    Proposition,
    Exp,
    SExp,
    Concept,
    AtomExp,
    IExpConfig,
    Intrinsic,
    Expression,
    DataStruct,
    Parastructure,
    ExpConfig,
    ExpStructure,
)
from aiphy.core import (
    ExpData,
    ConstData,
    NormalData,
    KeyValueHashed,
    is_conserved_const_list,
    is_conserved_mean_and_std
)
from aiphy.core import (
    MeasureType,    # .default()
    Objstructure,    # .make_particle() .make_spring()
    ObjType,
    DATA
)
# parser
from aiphy.core import (
    sentence
)


class Knowledge:
    """
    Knowledge 类，是 ai_physicist.Knowledge （ built-in 的方法是在 rust 中实现的 ） 的一个包装类，提供了一些便捷的方法，
    存储各种知识，包括实验、概念、结论等。
    可以在这个知识库中注册和查询概念与结论。

    查看当前知识库信息的方法：    
    fetch_expstruct()  fetch_exps()  fetch_concepts()  print_concepts()  print_conclusions()
    注册新的概念、结论、实验的方法：
    register_expr()  register_conclusion()  register_expstruct()
    """
    K: aiphy.Knowledge
    concept_id: int = 0
    conclusion_id: int = 0
    object_id: int = 0

    @staticmethod
    def default() -> "Knowledge":
        """
        创建一个新的 Knowledge 对象，
        内部包含一些默认的实验 （ 程序内置的实验 ） 。
        """
        obj = object.__new__(Knowledge)
        obj.K = aiphy.Knowledge.default()
        return obj

    @staticmethod
    def read_from_file(filename: str) -> "Knowledge":
        """
        从文件中读取知识库。
        """
        obj = object.__new__(Knowledge)
        with open(filename, "r") as f:
            s = f.read().strip()
            # 首先去掉最后一行
            s1 = s[:s.rfind("\n")]
            s2 = s[s.rfind("\n") + 1:]
            # try:
            obj.K = aiphy.Knowledge.parse_from(s1)
            # except:
            #     print("Failed to load knowledge from string")
            #     print(s1)
            #     raise Exception("Failed to load knowledge from string")
            id_dict = json.loads(s2)
            obj.concept_id = id_dict["concept_id"]
            obj.conclusion_id = id_dict["conclusion_id"]
            obj.object_id = id_dict["object_id"]

        return obj

    def save_to_file(self, filename: str):
        """
        将当前知识库保存到文件中。
        """
        self.remove_useless_objects()
        print("Saving knowledge to", filename, '\n')
        with open(filename, "w") as f:
            f.write(str(self.K)+'\n')
            json.dump({
                "concept_id": self.concept_id,
                "conclusion_id": self.conclusion_id,
                "object_id": self.object_id
            }, f)

    def __getstate__(self):
        return {
            "K": str(self.K),
            "concept_id": self.concept_id,
            "conclusion_id": self.conclusion_id,
            "object_id": self.object_id
        }

    def __setstate__(self, state):
        self.K = aiphy.Knowledge.parse_from(state["K"])
        self.concept_id = state["concept_id"]
        self.conclusion_id = state["conclusion_id"]
        self.object_id = state["object_id"]

    @property
    def fetch_exps(self) -> List[str]:
        return self.K.fetch_experiments

    @property
    def fetch_concepts(self) -> Dict[str, Expression]:
        return self.K.fetch_concepts

    @property
    def fetch_intrinsic_concepts(self) -> Dict[str, Expression]:
        return self.K.fetch_intrinsic_concepts

    @property
    def fetch_conclusions(self) -> Dict[str, Proposition]:
        return self.K.fetch_conclusions

    def update_expstruct(self, name: str, expstruct: ExpStructure):
        self.K.update_experiment(name, expstruct)

    def register_expstruct(self, name: str, expstruct: ExpStructure):
        self.K.register_experiment(name, expstruct)

    def fetch_expstruct(self, name: str) -> ExpStructure:
        return self.K.fetch_expstruct(name)

    def fetch_objstruct_from_expstruct(self, expname: str, objid: int) -> Objstructure:
        return self.K.fetch_objstructure_in_expstruct(expname, objid)

    def find_similar_concept(self, concept: Concept | None) -> str | None:
        if concept is None:
            return None
        return self.K.find_similar_concept(concept)

    def eval_onebody_intrinsic(self, intrinsic: Intrinsic, obj: Objstructure) -> ConstData | None:
        """
        在特定的实验数据结构 （ 包含测量数据 ） 下，计算一个内禀概念的值。
        """
        return self.K.eval_intrinsic(intrinsic, [obj])

    def eval_manybody_intrinsic(self, intrinsic: Intrinsic, obj_list: List[Objstructure]) -> ConstData | None:
        """
        在特定的实验数据结构 （ 包含测量数据 ） 下，计算一个多体内禀概念的值。
        """
        return self.K.eval_intrinsic(intrinsic, obj_list)

    def eval(self, expr: Exp | str, expstruct: ExpStructure) -> ExpData:
        """
        在特定的实验数据结构 （ 包含测量数据 ） 下，计算一个表达式的值。
        """
        if isinstance(expr, str):
            expr = Exp(expr)
        try:
            return self.K.eval(expr, expstruct)
        except Exception:
            raise Exception(f"Failed to eval {expr}")

    def register_object(self, objstruct: Objstructure, name: str = None) -> str:
        """
        以 name 为名字，注册一个具体的物理对象 objstruct。
        """
        name = self.auto_object_name() if name is None else name
        self.K.register_object(name, objstruct)
        return name

    def register_basic_concept(self, concept: Concept | str) -> str | None:
        if isinstance(concept, str):
            concept = Concept(concept)
        if self.K.register_basic_concept(concept):
            return concept.atomexp_name
        else:
            return None

    def register_expr(self, definition: Expression | str, name: str = None) -> str | None:
        """
        以 name 为名字，注册一个概念 definition ，
        1. 它可以是 Concept （普通概念）， 例如
        "(1->Particle) (2->Clock) |- D[posx[1]]/D[t[2]]"
        2. 或者是 Intrinsic （内禀概念）， 例如
        "[#oscillation (1->Particle) [2->Obj_02] |- D[posx[1]]/D[posx[1]'']]"
        """
        name = self.auto_concept_name() if name is None else name
        expr: Expression = Expression(definition) if isinstance(definition, str) else definition
        if self.K.register_expression(name, expr):
            # 概念注册成功
            if expr.expr_type == "Intrinsic":
                # 如果是内禀概念，那么将其相关的物体名字加入 obj_tmp
                intr: Intrinsic = expr.unwrap_intrinsic
                for obj_name in intr.relevant_objs:
                    self.obj_tmp.add(obj_name)
            return name
        else:
            return None

    def register_conclusion(self, definition: str | Proposition, name: str = None) -> str | None:
        """
        以 name 为名字，注册一个结论 definition ，
        1. 它可以是 Exp(...) is zero， 例如
            "(posx[1] - posr[2]) is zero"
        2. 它可以是 Exp(...) is conserved， 例如
            "(m[1] * v[1] + m[2] * v[2]) is conserved"
        3. 或者 Concept(...) is zero，例如
            "(1 -> Particle) |- m[1] * a[1] + Partial[V]/Partial[x[1]] is zero"
        4. 或者 Concept(...) is conserved，例如
            "|- T + V is conserved"
        """
        name = self.auto_conclusion_name() if name is None else name
        prop: Proposition = Proposition(definition) if isinstance(definition, str) else definition
        if self.K.register_conclusion(name, prop):
            # 结论注册成功
            return name
        else:
            return None

    def remove_conclusion(self, name: str):
        """
        移除一个结论。
        """
        self.K.remove_conclusion(name)

    def remove_concept(self, name: str):
        """
        移除一个概念。
        """
        self.K.remove_concept(name)

    def remove_useless_objects(self):
        """
        移除那些没有被任何内禀概念引用的物体。
        """
        for obj_name in self.K.fetch_object_keys:
            if obj_name not in self.obj_tmp:
                self.K.remove_object(obj_name)

    def auto_object_name(self) -> str:
        self.object_id += 1
        return "Obj_{:02d}".format(self.object_id)

    def auto_concept_name(self) -> str:
        self.concept_id += 1
        return "C_{:02d}".format(self.concept_id)

    def auto_conclusion_name(self) -> str:
        self.conclusion_id += 1
        return "R_{:02d}".format(self.conclusion_id)

    def extract_concepts(self, exp_name: str, exp: Exp | str) -> Set[Concept]:
        exp = Exp(exp) if isinstance(exp, str) else exp
        return self.K.extract_concepts(exp, exp_name)

    def generalize(self, exp_name: str, exp: Exp | str) -> Concept | None:
        try:
            exp: Exp = Exp(exp) if isinstance(exp, str) else exp
            return self.K.generalize(exp, exp_name)
        except Exception:
            print("Failed to generalize", exp_name, exp)

    def generalize_to_normal_concept(self, exp_name: str, exp: Exp | str) -> Concept | None:
        try:
            exp: Exp = Exp(exp) if isinstance(exp, str) else exp
            return self.K.generalize_to_normal_concept(exp, exp_name)
        except Exception:
            print("Failed to generalize to normal concept", exp_name, exp)

    def specialize(self, concept: Concept | str, exp_name: str) -> List[Exp]:
        try:
            concept: Concept = Concept(concept) if isinstance(concept, str) else concept
            return self.K.specialize(concept, exp_name)
        except Exception:
            print("Failed to specialize", concept, exp_name)

    def fetch_concept_by_name(self, concept_name: str) -> Concept | None:
        res = self.K.fetch_concept_by_name(concept_name)
        return res.unwrap_concept if res.expr_type == 'Concept' else None

    def fetch_intrinsic_by_name(self, expression_name: str) -> Intrinsic | None:
        res = self.K.fetch_concept_by_name(expression_name)
        return res.unwrap_intrinsic if res.expr_type == 'Intrinsic' else None

    def specialize_concept(self, concept_name: str, exp_name: str) -> List[AtomExp]:
        return self.K.specialize_concept(concept_name, exp_name)

    def print_concepts(self):
        """
        打印当前知识库中的所有概念。
        """
        self.K.list_concepts()

    def print_conclusions(self):
        """
        打印当前知识库中的所有结论。
        """
        self.K.list_conclusions()

    def sympy_of_raw_defi(self, exp: Exp | str, exp_name: str) -> sp.Expr:
        exp = Exp(exp) if isinstance(exp, str) else exp
        return sp.sympify(self.K.parse_exp_to_sympy_str(
            self.K.raw_definition_exp(exp, exp_name), "t_0"
        ))

    def raw_complexity(self, exp: Exp | Proposition, exp_name: str | None = None) -> int:
        if isinstance(exp, Exp):
            return self.K.raw_definition_exp(exp, exp_name).complexity
        elif isinstance(exp, Proposition):
            return self.K.raw_definition_prop(exp, exp_name).complexity
        else:
            raise Exception("Invalid type")

    def complexity(self, exp: Exp | Proposition, exp_name: str) -> int:
        """
        计算一个表达式的复杂度，
        第一关键字：raw definition 的复杂度
        第二关键字：exp 的复杂度
        """
        if isinstance(exp, Exp):
            return self.K.raw_definition_exp(exp, exp_name).complexity * 100 + exp.complexity
        elif isinstance(exp, Proposition):
            return self.K.raw_definition_prop(exp, exp_name).complexity * 100 + exp.complexity
        else:
            raise Exception("Invalid type")

    def gen_atom_concept(self, concept_name: str) -> Concept:
        return self.K.gen_atom_concept_by_name(concept_name)

    def find_similar(self, expr: str | Concept | Intrinsic | Proposition) -> str | None:
        if isinstance(expr, str):
            expr = Expression(expr)
            match expr.expr_type:
                case "Concept":
                    expr = expr.unwrap_concept
                case "Intrinsic":
                    expr = expr.unwrap_intrinsic
                case "Proposition":
                    expr = expr.unwrap_proposition
        if isinstance(expr, Concept):
            name = self.K.find_similar_concept(expr)
            if name is not None:
                return name
            return self.K.find_similar_concept(-expr)
        elif isinstance(expr, Intrinsic):
            name = self.K.find_similar_intrinsic(expr)
            if name is not None:
                return name
            return self.K.find_similar_intrinsic(-expr)
        elif isinstance(expr, Proposition):
            name = self.K.find_similar_conclusion(expr)
            if name is not None:
                return name
            return self.K.find_similar_conclusion(-expr)
        else:
            raise Exception("Invalid type")


ObjType_Clock = ObjType("Clock")
