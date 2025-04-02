import sympy as sp


def create_eqs(n):
    t = sp.symbols('t')
    g = sp.symbols('g')
    ms = [sp.symbols('m'+str(i+1)) for i in range(n)]
    ks = [sp.symbols('k'+str(i+1)) for i in range(n-1)]
    ls = [sp.symbols('l'+str(i+1)) for i in range(n)]
    rts = [sp.Function('r'+str(i+1))(t) for i in range(n)]
    rs = [sp.symbols('r'+str(i+1)) for i in range(n)]
    vs = [sp.symbols('v'+str(i+1)) for i in range(n)]
    acs = [sp.symbols('a'+str(i+1)) for i in range(n)]
    thetats = [sp.Function('theta'+str(i+1))(t) for i in range(n)]
    thetas = [sp.symbols('theta'+str(i+1)) for i in range(n)]
    betas = [sp.symbols('beta'+str(i+1)) for i in range(n)]
    omegas = [sp.symbols('omega'+str(i+1)) for i in range(n)]
    dxs = [rts[i] * sp.sin(thetats[i]) for i in range(n)]
    dys = [-rts[i] * sp.cos(thetats[i]) for i in range(n)]
    xs = [sum(dxs[:i+1]) for i in range(n)]
    ys = [sum(dys[:i+1]) for i in range(n)]
    vxs = [x.diff(t) for x in xs]
    vys = [y.diff(t) for y in ys]
    v2s = [vx**2 + vy**2 for vx, vy in zip(vxs, vys)]
    Ts = [m*v2 / 2 for m, v2 in zip(ms, v2s)]
    V1s = [m*g*y for m, y in zip(ms, ys)]
    V2s = [k*(r-l)**2 for k, r, l in zip(ks, rts, ls)]
    L = sum(Ts) - sum(V1s) - sum(V2s)

    qs = rts + thetats
    eqs = [sp.diff(sp.diff(L, q.diff(t)), t) - sp.diff(L, q) for q in qs]
    substitutions = [(sp.diff(theta, t, t), beta) for theta, beta in zip(qs, acs+betas)]
    eqs = [eq.subs(substitutions) for eq in eqs]
    substitutions = [(sp.diff(theta, t), omega) for theta, omega in zip(qs, vs+omegas)]
    eqs = [eq.subs(substitutions) for eq in eqs]
    substitutions = [(thetat, theta) for thetat, theta in zip(qs, rs+thetas)]
    eqs = [eq.subs(substitutions) for eq in eqs]

    eqs = [sp.sympify(eq) for eq in eqs]
    return eqs


def numeric_solve(eqs, numeric):
    n = int(len(eqs))/2
    acs = [sp.symbols('a'+str(i+1)) for i in range(n)] + [sp.symbols('beta'+str(i+1)) for i in range(n)]
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
