import sympy as sp


def create_eqs(n):
    t = sp.symbols('t')
    G = sp.symbols('G')
    ms = [sp.symbols('m'+str(i+1)) for i in range(n)]

    xts = [sp.Function('x'+str(i+1))(t) for i in range(n)]
    xs = [sp.symbols('x'+str(i+1)) for i in range(n)]
    vxs = [sp.symbols('vx'+str(i+1)) for i in range(n)]
    axs = [sp.symbols('ax'+str(i+1)) for i in range(n)]
    yts = [sp.Function('y'+str(i+1))(t) for i in range(n)]
    ys = [sp.symbols('y'+str(i+1)) for i in range(n)]
    vys = [sp.symbols('vy'+str(i+1)) for i in range(n)]
    ays = [sp.symbols('ay'+str(i+1)) for i in range(n)]
    v2s = [vx**2 + vy**2 for vx, vy in zip([x.diff(t) for x in xts], [y.diff(t) for y in yts])]
    Ts = [m*v2/2 for m, v2 in zip(ms, v2s)]

    V = 0
    for i in range(n):
        for j in range(i+1, n):
            r = sp.sqrt((xts[i]-xts[j])**2+(yts[i]-yts[j])**2)
            V = V - G*ms[i]*ms[j]/r
    L = sum(Ts) - V

    eqs = [sp.diff(sp.diff(L, xt.diff(t)), t) - sp.diff(L, xt) for xt in xts + yts]
    substitutions = [(sp.diff(xt, t, t), ax) for xt, ax in zip(xts + yts, axs + ays)]
    eqs = [eq.subs(substitutions) for eq in eqs]
    substitutions = [(sp.diff(xt, t), vx) for xt, vx in zip(xts + yts, vxs + vys)]
    eqs = [eq.subs(substitutions) for eq in eqs]
    substitutions = [(xt, x) for xt, x in zip(xts + yts, xs + ys)]
    eqs = [eq.subs(substitutions) for eq in eqs]

    eqs = [sp.sympify(eq) for eq in eqs]
    return eqs


def numeric_solve(eqs, numeric):
    n = int(len(eqs)/2)
    acs = [sp.symbols('ax'+str(i+1)) for i in range(n)] + [sp.symbols('ay'+str(i+1)) for i in range(n)]
    eqs_n = [eq.subs(numeric) for eq in eqs]
    free_symbols = [eq.free_symbols for eq in eqs_n]
    variables = free_symbols[0]
    for symbol in free_symbols:
        variables = variables.union(symbol)
    if variables != set(acs):
        raise ValueError('wrong numeric')

    try:
        solutions = sp.solve(eqs, acs)
        solutions = [var.subs(solutions) for var in acs]
        return solutions
    except Exception as e:
        print(f'can\'t solve the dymanic system: : {e}')
        return None
