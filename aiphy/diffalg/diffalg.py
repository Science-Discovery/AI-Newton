from .mapleIO import mapleIO
import sympy as sp
from typing import List, Dict, Union, Tuple, Literal


def aux(s: sp.Symbol | sp.Function) -> str:
    temp = str(s).replace(' ', '')
    return (temp + '()') if s.is_Symbol else temp


class DifferentialRing:
    # derivations define the arguments of the functions
    derivations: List[sp.Symbol]
    # blocks define the block order of the variables
    blocks: List[Tuple[str, List[Union[sp.Symbol, sp.Function]]]]

    def to_json(self) -> dict:
        return {
            'derivations': [str(deriv) for deriv in self.derivations],
            'blocks': [
                (block[0], [str(var) for var in block[1]])
                for block in self.blocks
            ]
        }

    @classmethod
    def from_json(cls, json_data: dict) -> 'DifferentialRing':
        obj = object.__new__(cls)
        obj.derivations = [sp.Symbol(deriv) for deriv in json_data['derivations']]
        obj.blocks = [
            (block[0], [sp.sympify(var) for var in block[1]])
            for block in json_data['blocks']
        ]
        return obj

    def __str__(self):
        return 'DifferentialRing:\n' + '\n'.join([block[0] + ': ' + ', '.join([str(var) for var in block[1]]) for block in self.blocks])

    def __init__(self, blocks: List[Tuple[str, List[Union[sp.Symbol, sp.Function]]]]):
        self.blocks = [block for block in blocks if len(block[1]) > 0]
        derivs = set()
        for item in self.blocks:
            assert item[0] in ['grlexA', 'grlexB', 'degrevlexA', 'degrevlexB', 'lex']
            assert len(item[1]) > 0
            for var in item[1]:
                if var.is_Function:
                    derivs |= set(var.args)
        self.derivations = list(derivs)

    @classmethod
    def default(cls, vars: List[Union[sp.Symbol, sp.Function]]):
        return cls([('lex', vars)])

    @property
    def temp_symbol(self) -> sp.Function:
        return sp.Function('temp')(*self.derivations)

    def ring_to_maple(self, trans_table: Literal['ver1', 'ver2'] = 'ver1',
                      symbs: set[sp.Symbol | sp.Function] = None) -> str:
        derivs_arg = '[' + ', '.join([deriv.name for deriv in self.derivations]) + ']'
        blocks = [(block[0], [var for var in block[1] if symbs is None or var in symbs]) for block in self.blocks]
        blocks = [block for block in blocks if len(block[1]) > 0]
        if self.temp_symbol in symbs:
            # 在 diffalg.reduce 中，会用到 temp 变量来辅助化简
            blocks = [('lex', [self.temp_symbol])] + blocks
        remain_symbs = symbs - set.union(*[set(block[1]) for block in blocks])
        remain_symbs = remain_symbs - set(self.derivations)
        if remain_symbs:
            blocks = blocks + [('lex', list(remain_symbs))]
        if (trans_table == 'ver1'):
            blocks_arg = '[' + ', '.join([block[0] + '[' +
                                          ','.join([var.name for var in block[1]]) +
                                          ']' for block in blocks]) + ']'
        else:
            blocks_arg = '[' + ', '.join(['[' +
                                          ','.join([aux(var) for var in block[1]]) +
                                          ']' for block in blocks]) + ']'
        return f'DifferentialRing(blocks = {blocks_arg}, derivations = {derivs_arg})'


class RegularDifferentialChain:
    ring: DifferentialRing
    gb: List[sp.Expr]

    def __init__(self, ring: DifferentialRing, gb: List[sp.Expr] = None):
        self.ring = ring
        self.gb = gb if gb is not None else []

    def __str__(self):
        return '[' + ', '.join([str(i) for i in self.gb]) + ']'

    def to_json(self) -> dict:
        return {
            'ring': self.ring.to_json(),
            'gb': [str(i) for i in self.gb]
        }

    @classmethod
    def from_json(cls, json_data: dict) -> 'RegularDifferentialChain':
        obj = object.__new__(cls)
        obj.ring = DifferentialRing.from_json(json_data['ring'])
        obj.gb = [sp.sympify(i) for i in json_data['gb']]
        return obj

    def reduce(self, eq: sp.Expr,
               timeout: int = 10) -> sp.Expr | None:
        if not self.gb:
            return eq
        solver = mapleIO()
        solver.import_lib('DifferentialAlgebra')
        solver.import_lib('Tools')
        symbs = eq.atoms(sp.Symbol, sp.Function)
        for i in self.gb:
            symbs |= i.atoms(sp.Symbol, sp.Function)
        solver.append_command(f'R := {self.ring.ring_to_maple(trans_table="ver2", symbs=symbs)}')
        eqs_arg = ', '.join([eq_to_maple(self.ring, i, trans_table="ver2") for i in self.gb])
        solver.append_command(f'eqs := [{eqs_arg}]')
        solver.append_command(f'ideal := PretendRegularDifferentialChain(eqs, R)')
        eq_arg = eq_to_maple(self.ring, eq=eq, trans_table="ver2")
        solver.append_command(f'eq := NormalForm({eq_arg}, ideal)')
        solver.append_command(f'print(eq)')
        try:
            stdout = solver.exec_maple(timeout=timeout)
        except Exception:
            return None
        return eq_from_maple(self.ring, stdout[-1])


class diffalg:
    ring: DifferentialRing
    gb: List[RegularDifferentialChain]
    eqs: List[sp.Expr]
    ineqs: List[sp.Expr]

    def __init__(self, ring: DifferentialRing, gb: List[RegularDifferentialChain] = None,
                 eqs: List[sp.Expr] = None, ineqs: List[sp.Expr] = None):
        self.ring = ring
        self.gb = [RegularDifferentialChain(ring)] if gb is None else gb
        self.eqs = [] if eqs is None else eqs
        self.ineqs = [] if ineqs is None else ineqs

    def __str__(self):
        return 'DifferentialAlgebra:\n' + '\n'.join([i.__str__() for i in self.gb])

    def to_json(self) -> dict:
        return {
            'ring': self.ring.to_json(),
            'gb': [i.to_json() for i in self.gb],
            'eqs': [str(i) for i in self.eqs],
            'ineqs': [str(i) for i in self.ineqs]
        }

    @classmethod
    def from_json(cls, json_data: dict) -> 'diffalg':
        obj = object.__new__(cls)
        obj.ring = DifferentialRing.from_json(json_data['ring'])
        obj.gb = [RegularDifferentialChain.from_json(i) for i in json_data['gb']]
        obj.eqs = [sp.sympify(i) for i in json_data['eqs']]
        obj.ineqs = [sp.sympify(i) for i in json_data['ineqs']]
        return obj

    @classmethod
    def from_eqs(cls, ring: DifferentialRing, eqs: List[sp.Expr] = None, ineqs: List[sp.Expr] = None,
                 timeout: int = 10):
        """
        return None when TIMEOUT or ERROR in maple
        """
        if eqs is None or eqs == []:
            return cls(ring, gb=None, eqs=None, ineqs=ineqs)
        if ineqs is None:
            ineqs = []
        solver = mapleIO()
        solver.import_lib('DifferentialAlgebra')
        symbs = set()
        for i in eqs:
            symbs |= i.atoms(sp.Symbol, sp.Function)
        for i in ineqs:
            symbs |= i.atoms(sp.Symbol, sp.Function)
        solver.append_command(f'R := {ring.ring_to_maple(symbs=symbs)}')
        eqs_args = [eq_to_maple(ring, i) for i in eqs] + [eq_to_maple(ring, i) + '<> 0' for i in ineqs]
        args = ', '.join(eqs_args)
        solver.append_command(f'eqs := [{args}]')
        solver.append_command(f'ideal := RosenfeldGroebner(eqs, R)')
        solver.append_command(f'print(Equations(ideal))')
        try:
            stdout = solver.exec_maple(timeout=timeout)
        except Exception:
            return None
        return cls(ring, gb_from_maple(ring, stdout[-1]), eqs, ineqs)

    def insert_new_eq(self, eq: sp.Expr) -> 'diffalg':
        """
        return None when TIMEOUT or ERROR in maple
        """
        if self.belongs_to(eq):
            return self
        return diffalg.from_eqs(self.ring, self.eqs + [eq], self.ineqs)

    def _insert_new_eq(self, eq: sp.Expr) -> 'diffalg':
        return diffalg.from_eqs(self.ring, self.eqs + [eq], self.ineqs)

    def insert_new_ineqs(self, eq: sp.Expr) -> 'diffalg':
        if eq in self.ineqs:
            return self
        if self.belongs_to(eq):
            return diffalg(self.ring, gb=None, eqs=self.eqs, ineqs=self.ineqs + [eq])
        return diffalg.from_eqs(self.ring, self.eqs, self.ineqs + [eq])

    def _insert_new_ineqs(self, eq: sp.Expr) -> 'diffalg':
        return diffalg.from_eqs(self.ring, self.eqs, self.ineqs + [eq])

    def _insert_new_eqs_and_ineqs(self, eqs: List[sp.Expr], ineqs: List[sp.Expr]) -> 'diffalg':
        return diffalg.from_eqs(self.ring, self.eqs + eqs, self.ineqs + ineqs)

    def belongs_to(self, eq: sp.Expr) -> bool | None:
        """
        return None when TIMEOUT or ERROR in maple
        """
        if self.eqs is None or not self.eqs:
            return False
        symbols = set()
        for i in self.gb:
            for j in i.gb:
                symbols |= (j.free_symbols & set(self.ring.derivations))
        if not (eq.free_symbols & set(self.ring.derivations)).issubset(symbols):
            return False
        solver = mapleIO()
        solver.import_lib('DifferentialAlgebra')
        solver.import_lib('Tools')
        symbs = eq.atoms(sp.Symbol, sp.Function)
        for i in self.eqs:
            symbs |= i.atoms(sp.Symbol, sp.Function)
        for i in self.ineqs:
            symbs |= i.atoms(sp.Symbol, sp.Function)
        solver.append_command(f'R := {self.ring.ring_to_maple(trans_table="ver2", symbs=symbs)}')
        arg_lst = []
        for ideal in self.gb:
            eqs_arg = ', '.join([eq_to_maple(self.ring, i, trans_table="ver2") for i in ideal.gb])
            arg_lst.append(f'PretendRegularDifferentialChain([{eqs_arg}], R)')
        arg = ', '.join(arg_lst)
        solver.append_command(f'ideal := [{arg}]')
        eq_arg = eq_to_maple(self.ring, eq=eq, trans_table="ver2")
        solver.append_command(f'print(BelongsTo([{eq_arg}], ideal))')
        try:
            stdout = solver.exec_maple(timeout=10)
        except Exception:
            return None
        result = stdout[-1].strip('[] ')
        assert result in ['true', 'false']
        return result == 'true'

    def reduce(self, eq: sp.Expr, debug: bool = False,
               timeout: int = 10) -> sp.Expr | None:
        if not self.eqs:
            return eq
        solver = mapleIO()
        solver.import_lib('DifferentialAlgebra')
        temp = self.ring.temp_symbol
        symbs = {temp} | eq.atoms(sp.Symbol, sp.Function)
        for i in self.eqs:
            symbs |= i.atoms(sp.Symbol, sp.Function)
        for i in self.ineqs:
            symbs |= i.atoms(sp.Symbol, sp.Function)
        solver.append_command(f'R := {self.ring.ring_to_maple(symbs=symbs)}')
        eqs_args = [eq_to_maple(self.ring, i) for i in self.eqs]
        eqs_args.append(eq_to_maple(self.ring, eq - temp))
        eqs_args.extend([eq_to_maple(self.ring, i) + '<> 0' for i in self.ineqs])
        args = ', '.join(eqs_args)
        solver.append_command(f'eqs := [{args}]')
        solver.append_command(f'ideal := RosenfeldGroebner(eqs, R)')
        solver.append_command(f'print(Equations(ideal))')
        eq_arg = eq_to_maple(self.ring, temp)
        solver.append_command(f'eq := NormalForm({eq_arg}, ideal[1])')
        solver.append_command(f'print(eq)')
        try:
            stdout = solver.exec_maple(timeout=timeout)
        except Exception:
            return None
        if debug:
            for c in solver.commands:
                print(c + ';')
        result = eq_from_maple(self.ring, stdout[-1])
        if result == sp.Symbol('temp'):
            return eq
        return result


def eq_to_maple(ring: DifferentialRing, eq: sp.Expr, trans_table: Literal['ver1', 'ver2'] = 'ver1') -> str:
    if eq == sp.oo or eq == -sp.oo or eq == sp.zoo:
        return '0'
    if eq.is_Number:
        return str(eq)
    if eq.is_Symbol or eq.is_Function:
        if trans_table == 'ver1' or eq in ring.derivations:
            return str(eq)
        return aux(eq)
    if eq.is_Add:
        return ' + '.join(['(' + eq_to_maple(ring, arg, trans_table) + ')' for arg in eq.args])
    if eq.is_Mul:
        return ' * '.join(['(' + eq_to_maple(ring, arg, trans_table) + ')' for arg in eq.args])
    if eq.is_Pow:
        return f'({eq_to_maple(ring, eq.base, trans_table)})^({eq_to_maple(ring, eq.exp, trans_table)})'
    if eq.is_Derivative:
        derivs_arg = eq_to_maple(ring, eq.args[1][0], trans_table)
        if eq.args[1][1] != sp.S.One:
            derivs_arg += '$' + str(eq.args[1][1])
        return f'diff({eq_to_maple(ring, eq.args[0], trans_table)}, {derivs_arg})'
    raise ValueError(f'Error! Parse expression {eq} failed! Func = ', eq.func)


def eq_from_maple(ring: DifferentialRing, eq: str, trans_table: Literal['ver1', 'ver2'] = 'ver1',
                  debug: bool = False) -> sp.Expr:
    try:
        eq = eq.replace('diff', 'Derivative')
        eq = eq.replace('()', '')
        eq = eq.replace('$', ',')
        return sp.sympify(eq)
    except Exception as e:
        print('Error! Failed to parse the output of Maple!')
        if debug:
            print('Output:', eq)
        raise e


def gb_from_maple(ring: DifferentialRing, gb: str) -> List[RegularDifferentialChain]:
    if gb == '[]':
        return []
    stack = []
    temp = []
    for i in range(len(gb)):
        if gb[i] in ['[', '(', '{']:
            stack.append(i)
            if gb[i] == '[' and len(stack) == 2:
                temp.append([])
                temp[-1].append('')
        elif gb[i] == ']':
            assert (gb[stack.pop()] == '[')
        elif gb[i] == ')':
            assert (gb[stack.pop()] == '(')
        elif gb[i] == '}':
            assert (gb[stack.pop()] == '{')
        if len(stack) == 2 and gb[i] == ',':
            temp[-1].append('')
        elif len(stack) >= 2 and gb[i] != '[':
            temp[-1][-1] += gb[i]
    try:
        gb = [RegularDifferentialChain(ring, [eq_from_maple(ring, j) for j in i]) for i in temp]
    except Exception:
        print('Error! Failed to parse the output of Maple!')
        print('Output:', temp)
        raise RuntimeError
    return gb
