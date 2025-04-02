import sympy as sp
from typing import Optional

from typing import Tuple, Dict, Set, Any
from _collections_abc import Iterator
from .memory import dict_to_json
from .interface import (
    Knowledge, Exp, Proposition,
    KeyValueHashed
)
from .diffalg import DifferentialRing, diffalg


class DiffalgService:
    exp_name: str
    knowledge: Knowledge
    conclusion: Dict[str, Proposition]
    conclusion_id: int
    diffalg_ideal: Optional[diffalg]

    def __init__(self, knowledge: Knowledge, exp_name: str):
        self.exp_name = exp_name
        self.knowledge = knowledge
        self.conclusion = {}
        self.conclusion_id = 0
        self.diffalg_ideal = None

    def __iter__(self) -> Iterator[str]:
        return iter(self.conclusion)

    def to_json(self) -> Dict[str, Any]:
        return {
            "exp_name": self.exp_name,
            "conclusion": dict_to_json(self.conclusion),
            "conclusion_id": self.conclusion_id,
            "diffalg_ideal": self.diffalg_ideal.to_json() if self.diffalg_ideal is not None else None
        }

    @classmethod
    def from_json(cls, knowledge: Knowledge, data: Dict[str, Any]) -> "DiffalgService":
        obj = cls(knowledge, "")
        obj.exp_name = data["exp_name"]
        obj.conclusion = {k: Proposition(v) for k, v in data["conclusion"].items()}
        obj.conclusion_id = data["conclusion_id"]
        obj.diffalg_ideal = diffalg.from_json(data["diffalg_ideal"]) if data["diffalg_ideal"] is not None else None
        return obj

    def keys(self):
        return self.conclusion.keys()

    def values(self):
        return self.conclusion.values()

    def get(self, key: str) -> Proposition:
        return self.conclusion.get(key)

    def remove_law(self, name: str):
        if name in self.conclusion:
            del self.conclusion[name]

    def register_law(self, prop: Proposition):
        self.conclusion_id += 1
        name = f"P{self.conclusion_id}"
        self.conclusion[name] = prop
        self.diffalg_ideal = None
        return name

    def print_conclusions(self):
        for name, prop in self.conclusion.items():
            print(name, prop)

    def exp_hashed(self, exp: Exp) -> KeyValueHashed:
        try:
            return self.knowledge.K.eval_exp_keyvaluehashed(exp, self.exp_name)
        except Exception:
            raise Exception(f'Error occurs in exp_hashed\n  Args: {exp}')

    def _sympy_of_raw_defi(self, exp: Exp) -> sp.Expr:
        return self.knowledge.sympy_of_raw_defi(exp, self.exp_name)

    def fetch_differential_ring(self) -> Tuple[sp.Symbol, Set[sp.Symbol], Set[sp.Symbol], Set[sp.Symbol], DifferentialRing]:
        """
        This function is used to fetch the differential ring for the current conclusions set.
        Return a tuple of:
        1. argument: sp.Symbol, the argument of the differential ring
        2. all_normal_symbols: Set[sp.Symbol], all normal symbols in the differential ring
        3. all_functions: Set[sp.Function], all functions in the differential ring
        4. all_symbols_ne_zero: Set[sp.Symbol], all symbols except the zero symbols
        5. ring: DifferentialRing, the differential ring
        """
        all_normal_symbols = set()
        all_functions = set()
        for name, prop in self.conclusion.items():
            if prop.prop_type == "IsConserved":
                all_normal_symbols.add(sp.Symbol(name))
        all_symbols_ne_zero = all_normal_symbols
        argument = sp.Symbol("t_0")
        for value in self.values():
            symb_atoms = self._sympy_of_raw_defi(value.unwrap_exp).atoms(sp.Symbol)
            if argument in symb_atoms:
                symb_atoms.remove(argument)
            all_symbols_ne_zero |= symb_atoms
            all_functions |= self._sympy_of_raw_defi(value.unwrap_exp).atoms(sp.Function)
        for value in self.values():
            if value.prop_type == "IsZero":
                symb = self._sympy_of_raw_defi(value.unwrap_exp)
                if symb.is_Symbol:
                    all_symbols_ne_zero.remove(symb)
        ring = DifferentialRing([('lex', list(all_functions)),
                                 ('lex', list(all_normal_symbols))])
        return argument, all_normal_symbols, all_functions, all_symbols_ne_zero, ring

    def calculate_diffalg(self, prop_exp: Exp, timeout: float = 20.0) -> diffalg:
        if self.diffalg_ideal is not None:
            return self.diffalg_ideal
        argument, _, _, all_symbols_ne_zero, ring = self.fetch_differential_ring()
        eqs = []
        for name, prop in self.conclusion.items():
            if not ((prop.unwrap_exp.all_atoms & prop_exp.all_atoms) - Exp("t[0]").all_atoms):
                continue
            expr: sp.Expr = self._sympy_of_raw_defi(prop.unwrap_exp)
            match prop.prop_type:
                case "IsConserved":
                    eqs.append(expr - sp.Symbol(name))
                case "IsZero":
                    eqs.append(expr)
        ineqs = list(all_symbols_ne_zero) + [argument]
        self.diffalg_ideal = diffalg.from_eqs(ring, eqs, ineqs,
                                              timeout=timeout)
        return self.diffalg_ideal

    def can_reduce(self, prop: Proposition, timeout: float = 20.0) -> bool:
        expr: sp.Expr = self._sympy_of_raw_defi(prop.unwrap_exp).doit()
        if expr.is_number or expr.diff(sp.Symbol("t_0")).is_zero:
            return False
        match prop.prop_type:
            case "IsConserved":
                return reduce_conserved_by_ideal(self.calculate_diffalg(prop.unwrap_exp, timeout), expr, sp.Symbol("t_0"),
                                                 timeout)
            case "IsZero":
                reduce_new_eq = self.calculate_diffalg(prop.unwrap_exp, timeout).reduce(expr.as_numer_denom()[0],
                                                                                        timeout=timeout)
                return reduce_new_eq is None or reduce_new_eq.is_zero
            case _:
                raise ValueError(f"Unknown prop_type: {prop.prop_type}")


def reduce_conserved_by_ideal(ideal: diffalg, sp_expr: sp.Expr, argument: sp.Symbol,
                              timeout: float = 20.0) -> bool:
    if not ideal.eqs:
        return False
    diff_eq = sp.diff(sp_expr, argument).as_numer_denom()[0]
    reduce_diff_eq_result: sp.Expr | None = ideal.gb[0].reduce(diff_eq, timeout=timeout)
    if reduce_diff_eq_result is None:
        return True
    if not reduce_diff_eq_result.is_zero:
        return False
    eq_reduced = ideal.reduce(sp_expr, timeout=timeout)
    if eq_reduced is None:
        return True
    if eq_reduced.diff(argument).is_zero:
        return True
    else:
        return False
