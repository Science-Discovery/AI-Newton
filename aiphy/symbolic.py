import sympy as sp
from typing import List, Tuple
from copy import deepcopy
import deprecated


class GeoInfoCache:
    """
    This class is used to cache the geometric information, including the whole spatial coordinates,
    the geometric constraints, and the differential information of the sub-manifold defined by the constraints.
    """
    geo_info: List[sp.Expr]
    symbs: List[sp.Function]

    funcs: List[sp.Function]
    lagrange_multipliers: List[sp.Function]
    lagrange_item: List[sp.Expr]
    langange_nabla: List[List[sp.Expr]]
    nabla_matrix: sp.Matrix
    nabla_reduced_matrix: sp.Matrix
    pivot_tuple: Tuple[int]

    jacobian: sp.Matrix

    def __init__(self, geo_info: List[sp.Expr], symbs: List[sp.Function]):
        self.geo_info = deepcopy(geo_info)
        for symb in symbs:
            if not symb.is_Function:
                raise Exception(f'Error, {symb} is not a function')
        self.symbs = deepcopy(symbs)
        self.generate_cache()
        result = []
        for func in self.funcs:
            result.append(self._partial_diff(func))
        self.jacobian = sp.Matrix(result)

    def generate_cache(self):
        funcs = set()
        for constraint in self.geo_info:
            funcs |= constraint.atoms(sp.Function)
        funcs: list[sp.Function] = self.symbs + list(funcs - set(self.symbs))

        lagrange_multipliers = []
        lagrange_item = [sp.Number(0) for _ in funcs]
        langange_nabla = []

        for i in range(len(self.geo_info)):
            # 引入拉格朗日乘子
            lagrange_multipliers.append(sp.Function(f'λ_{i}')(sp.Symbol('t_0')))
            nabla = sp.derive_by_array(self.geo_info[i], funcs)
            langange_nabla.append(nabla)
            lagrange_item = [
                lagrange_item[j] + lagrange_multipliers[-1] * nabla[j]
                for j in range(len(funcs))
            ]

        nabla_matrix: sp.Matrix = sp.Matrix(langange_nabla)[:, ::-1]
        nabla_reduced_matrix, pivot_tuple = nabla_matrix.rref(simplify=True)
        nabla_reduced_matrix: sp.Matrix = nabla_reduced_matrix[:, ::-1]
        pivot_tuple: Tuple[int] = tuple(len(funcs)-1-i for i in pivot_tuple)

        self.funcs = funcs
        self.lagrange_multipliers = lagrange_multipliers
        self.lagrange_item = lagrange_item
        self.langange_nabla = langange_nabla
        self.nabla_matrix = nabla_matrix
        self.nabla_reduced_matrix = nabla_reduced_matrix
        self.pivot_tuple = pivot_tuple

    def _partial_diff(self, func: sp.Function) -> List[sp.Expr]:
        # It is required that (nabla dot expr + nabla dot lagrange_item) dot langange_nabla = 0
        # to assure that the solution is unique up to geometric constraints
        assert func in self.funcs
        character_func = [sp.Number(func == f) for f in self.funcs]
        # eqs = self.geo_info.copy()
        eqs = []
        for i in range(self.nabla_reduced_matrix.shape[0]):
            if self.pivot_tuple[i] >= len(self.symbs):
                continue
            # print(nabla_reduced_matrix[i,:])
            eq = sp.Number(0)
            for j in range(len(self.symbs)): 
                eq = eq + self.nabla_reduced_matrix[i, j] * (self.lagrange_item[j] + character_func[j])
            eqs.append(eq)
        for i in range(len(self.symbs), len(self.funcs)):
            eq = self.lagrange_item[i] + character_func[i]
            eqs.append(eq)

        args = self.lagrange_multipliers
        solution = sp.solve(list(eqs), list(args))
        if args and not solution:
            print('eqs:')
            for eq in eqs:
                print(eq)
            print()
            print('args:')
            for arg in args:
                print(arg)
            raise Exception(f'Error occurs in eval_partial\n  Args: expr={func}\n  symbs={self.symbs}\n  geo_info={self.geo_info}\n \
                            Cannot solve the lagrange multipliers above')

        result = []
        for i in range(len(self.symbs)):
            res = character_func[i] + self.lagrange_item[i]
            res = res.subs(solution, simultaneous=True)
            result.append(res.simplify())
        return result


def partial_diff(expr: sp.Expr, geoinfo: GeoInfoCache) -> List[sp.Expr]:
    """
    This Function is used to calculate the partial derivative of `expr` with respect to `symbs`
    under the geometric constraints `geo_info`

    From mathematical view, `geo_info` limits the space of `symbs` to a sub-manifold we concern,
    and we calculate the component of partial derivative on this sub-manifold.
    """
    if expr.is_number:
        return [sp.Number(0) for _ in geoinfo.symbs]

    expr_nabla = sp.Matrix([sp.derive_by_array(expr.doit(), geoinfo.funcs)])
    result = list(expr_nabla * geoinfo.jacobian)

    return result


@deprecated.deprecated(version='0.1.0', reason='This function is deprecated, and the new version is `partial_diff`')
def partial_diff_old(expr: sp.Expr, symbs: list[sp.Function], geo_info: list[sp.Expr]) -> list[sp.Expr]:
    """
    这个函数的目的是在一个特定的实验，考虑所有几何约束，一个表达式对 symbs 的偏导数
    This function is deprecated, and the new version is `partial_diff`
    """
    for symb in symbs:
        if not symb.is_Function:
            raise Exception(f'Error occurs in eval_partial\n  Args: {expr} {symbs}\nReason: {symb} is not a function')

    constraints: list[sp.Expr] = []
    funcs = expr.atoms(sp.Function)
    for exp_constraint in geo_info:
        constraints.append(exp_constraint)
        funcs |= exp_constraint.atoms(sp.Function)
    funcs = funcs - set(symbs)
    # func: symbs 的函数，期待求出它们对 symbs 的偏导数
    # 在 sympy 表达式的水平上将 func 表示为 symbs 的函数
    sub_dict = {}
    for func in funcs:
        sub_dict[func] = sp.Function(func.name)(*symbs)
    inverse_dict = {v: k for k, v in sub_dict.items()}
    expr = expr.subs(sub_dict, simultaneous=True)
    lagrange_multipliers = []
    lagrange_item = [sp.Number(0) for _ in symbs]
    langange_nabla = []
    new_constraints = []
    eqs = []
    for i in range(len(constraints)):
        if constraints[i].atoms(sp.Function) & funcs:
            new_constraints.append(constraints[i].subs(sub_dict, simultaneous=True))
        else:
            # symbs 间存在约束，所以需要引入拉格朗日乘子
            lagrange_multipliers.append(sp.Function(f'λ_{i}')(sp.Symbol('t_0')))
            langange_nabla.append(sp.derive_by_array(constraints[i], symbs))
            lagrange_item = [
                lagrange_item[j] + lagrange_multipliers[-1] * langange_nabla[-1][j]
                for j in range(len(symbs))
            ]
            eqs.append(constraints[i])
    expr_nabla = sp.derive_by_array(expr, symbs)

    args = set(lagrange_multipliers)
    for constraint in new_constraints:
        for symb in symbs:
            eqs.append(sp.diff(constraint, symb))
            args |= eqs[-1].atoms(sp.Derivative)

    for i in range(len(lagrange_multipliers)):
        eq = sp.Number(0)
        for j in range(len(symbs)):
            eq = eq + langange_nabla[i][j] * (expr_nabla[j] + lagrange_item[j])
        eqs.append(eq)

    solution = sp.solve(list(eqs), list(args))
    result = []
    for i in range(len(symbs)):
        res = expr_nabla[i] + lagrange_item[i]
        res = res.subs(solution, simultaneous=True)
        res = res.subs(inverse_dict, simultaneous=True)
        result.append(res.simplify())
    return result
