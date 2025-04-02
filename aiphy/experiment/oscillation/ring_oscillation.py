import sympy as sp


def create_eqs(n):
    t = sp.symbols('t')
    ms = [sp.symbols('m'+str(i+1)) for i in range(n)]
    ks = [sp.symbols('k'+str(i+1)) for i in range(n-1)]
    ls = [sp.symbols('l'+str(i+1)) for i in range(n-1)]
    xts = [sp.Function('x'+str(i+1))(t) for i in range(n)]
    xs = [sp.symbols('x'+str(i+1)) for i in range(n)]
    vels = [sp.symbols('v'+str(i+1)) for i in range(n)]
    acs = [sp.symbols('a'+str(i+1)) for i in range(n)]
    vxs = [x.diff(t) for x in xts]
    v2s = [vx**2 for vx in vxs]
    dxs = [x2-x1-l for x2, x1, l in zip(xts[1:], xts[:-1], ls)]
    Ts = [m*v2/2 for m, v2 in zip(ms, v2s)]
    Vs = [k*dx/2 for k, dx in zip(ks, dxs)]
    L = sum(Ts) - sum(Vs)

    eqs = [sp.diff(sp.diff(L, xt.diff(t)), t) - sp.diff(L, xt) for xt in xts]
    substitutions = [(sp.diff(xt, t, t), ac) for xt, ac in zip(xts, acs)]
    eqs = [eq.subs(substitutions) for eq in eqs]
    substitutions = [(sp.diff(xt, t), vel) for xt, vel in zip(xts, vels)]
    eqs = [eq.subs(substitutions) for eq in eqs]
    substitutions = [(xt, x) for xt, x in zip(xts, xs)]
    eqs = [eq.subs(substitutions) for eq in eqs]

    eqs = [sp.sympify(eq) for eq in eqs]
    return eqs


def numeric_solve(eqs, numeric):
    acs = [sp.symbols('a'+str(i+1)) for i in range(len(eqs))]
    eqs_n = [eq.subs(numeric) for eq in eqs]
    free_symbols = [eq.free_symbols for eq in eqs_n]
    variables = free_symbols[0]
    for symbol in free_symbols:
        variables = variables.union(symbol)
    if variables != set(acs):
        raise ValueError('wrong numeric')

    try:
        solutions = sp.solve(eqs_n, acs)
        solutions = [var.subs(solutions) for var in acs]
        return solutions
    except Exception as e:
        print(f'can\'t solve the dymanic system: : {e}')
        return None
