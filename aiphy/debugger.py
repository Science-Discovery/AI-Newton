"""
Debugger module for aiphy
"""

from typing import Tuple, Any, Literal, List
from aiphy import (
    Theorist, Knowledge,
    Exp, Concept, Intrinsic, Proposition, Expression
)


class DebuggerItem:
    human_name: str
    machine_name: str
    obj: Concept
    repr: str

    def __init__(self, human_name: str, machine_name: str, obj: Concept, repr: str | None = None):
        if not isinstance(obj, Concept):
            raise Exception("Invalid type, only Concept is supported")
        self.human_name = human_name
        self.machine_name = machine_name
        self.obj = obj
        self.repr = repr or self.obj.__repr__()

    def __str__(self):
        return self.repr

    def __repr__(self):
        return self.repr

    def __call__(self, *args: int) -> Exp:
        return self.obj.subst_by_vec(list(args))

    def __add__(self, other: "DebuggerItem") -> "DebuggerItem":
        return DebuggerItem(None, None, self.obj + other.obj)

    def __sub__(self, other: "DebuggerItem") -> "DebuggerItem":
        return DebuggerItem(None, None, self.obj - other.obj)

    def __mul__(self, other: "DebuggerItem") -> "DebuggerItem":
        return DebuggerItem(None, None, self.obj * other.obj)

    def __rmul__(self, other: int) -> "DebuggerItem":
        return DebuggerItem(None, None, other * self.obj)

    def __truediv__(self, other: "DebuggerItem") -> "DebuggerItem":
        return DebuggerItem(None, None, self.obj / other.obj)

    def __pow__(self, other: int) -> "DebuggerItem":
        return DebuggerItem(None, None, self.obj.__powi__(other))

    def __neg__(self) -> "DebuggerItem":
        return DebuggerItem(None, None, -self.obj)

    def diff(self, ord: int = 1) -> "DebuggerItem":
        return DebuggerItem(None, None, self.obj.__difft__(ord))

    def partial(self, other: "DebuggerItem") -> "DebuggerItem":
        return DebuggerItem(None, None, self.obj.__partial__(other.obj))

    def mksum(self, objtype: str) -> "DebuggerItem":
        return DebuggerItem(None, None, Concept.Mksum(objtype, self.obj))

    def gen_prop(self, prop_type: Literal["Conserved", "Zero"]) -> Proposition:
        func = Proposition.Conserved if prop_type == "Conserved" else Proposition.Zero
        return func(self.obj)


class Debugger:
    theorist: Theorist
    knowledge: Knowledge

    def __init__(self, theorist: Theorist):
        self.theorist = theorist
        self.knowledge = theorist.knowledge

    def work(self, exp_name: str,
             name_list: List[DebuggerItem | str],
             ver: Literal['ver1', 'ver2', 'ver3'] = 'ver3',
             intrinsic_mode: int = 0,):
        self.theorist.theoretical_analysis(exp_name, ver,
                                           [item.machine_name if isinstance(item, DebuggerItem) else item
                                            for item in name_list])

    def check_concept_discovered(self, debug_item: DebuggerItem | Any, human_name: str) -> DebuggerItem:
        obj = debug_item.obj if isinstance(debug_item, DebuggerItem) else debug_item
        name, obj, repr = self.__find_similar(obj)
        return DebuggerItem(human_name, name, obj, repr)

    def check_law(self, prop: str | Proposition) -> str:
        if isinstance(prop, str):
            prop = Proposition(prop)
        if isinstance(prop, Proposition):
            return self.knowledge.K.find_similar_conclusion(prop) or \
                self.knowledge.K.find_similar_conclusion(-prop)
        else:
            raise Exception(f"No proposition <{prop}> found")

    def __find_similar(self, expr: str | Concept | Intrinsic | Proposition
                       ) -> Tuple[str | None, Concept | Intrinsic | Proposition, str]:
        if isinstance(expr, str):
            expr: Expression = Expression(expr)
            match expr.expr_type:
                case "Concept":
                    expr = expr.unwrap_concept
                case "Intrinsic":
                    expr = expr.unwrap_intrinsic
                case _:
                    raise Exception(f"Invalid type {expr.__class__}")
        if isinstance(expr, Concept):
            name = self.knowledge.K.find_similar_concept(expr)
            if name is not None:
                return name, self.knowledge.gen_atom_concept(name), name
            name = self.knowledge.K.find_similar_concept(-expr)
            if name is not None:
                return name, -self.knowledge.gen_atom_concept(name), "-" + name
            name = self.knowledge.K.find_similar_concept(expr.__powi__(-1))
            if name is not None:
                return name, self.knowledge.gen_atom_concept(name).__powi__(-1), "1/" + name
            name = self.knowledge.K.find_similar_concept(-expr.__powi__(-1))
            if name is not None:
                return name, -self.knowledge.gen_atom_concept(name).__powi__(-1), "-1/" + name
            raise Exception(f"No concept <{expr}> found")
        elif isinstance(expr, Intrinsic):
            name = self.knowledge.K.find_similar_intrinsic(expr)
            if name is not None:
                return name, self.knowledge.gen_atom_concept(name), name
            name = self.knowledge.K.find_similar_intrinsic(-expr)
            if name is not None:
                return name, -self.knowledge.gen_atom_concept(name), "-" + name
            name = self.knowledge.K.find_similar_intrinsic(expr.__inv__())
            if name is not None:
                return name, self.knowledge.gen_atom_concept(name).__powi__(-1), "1/" + name
            name = self.knowledge.K.find_similar_intrinsic(-expr.__inv__())
            if name is not None:
                return name, -self.knowledge.gen_atom_concept(name).__powi__(-1), "-1/" + name
            raise Exception(f"No intrinsic <{expr}> found")
        else:
            raise Exception("Invalid type")

    def raw_definition(self,
                       expr: str | Exp | Concept | Intrinsic | Proposition,
                       exp_name: str | None = None) -> str:
        return str(self.knowledge.K.raw_definition(Expression(str(expr)), exp_name=exp_name))

    def sympy_raw_definition(self,
                             expr: Exp | str,
                             exp_name: str):
        return self.knowledge.sympy_of_raw_defi(
            self.theorist.specific[exp_name].expand_exp(Exp(str(expr))),
            exp_name=exp_name
        )

    def law_info(self, prop_name: str) -> str:
        general_law = self.theorist.general.general_laws[prop_name]
        not_valid = list(set(self.theorist.knowledge.fetch_exps) - set(general_law.valid_experiments))
        return general_law.__str__() + '\n' + str(general_law.prop) + '\n' + self.raw_definition(str(general_law.prop)) + '\n' + \
            "Not valid experiments: " + "\033[1m" + \
            "[" + ', '.join(not_valid) + ']' + "\033[0m"
