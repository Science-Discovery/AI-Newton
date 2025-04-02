import sympy as sp


def create_eqs(n):
    t = sp.symbols('t')
    g = sp.symbols('g')
    ms = [sp.symbols('m'+str(i+1)) for i in range(n)]
    ls = [sp.symbols('l'+str(i+1)) for i in range(n)]
    thetats = [sp.Function('theta'+str(i+1))(t) for i in range(n)]
    thetas = [sp.symbols('theta'+str(i+1)) for i in range(n)]
    betas = [sp.symbols('beta'+str(i+1)) for i in range(n)]
    omegas = [sp.symbols('omega'+str(i+1)) for i in range(n)]
    dxs = [ls[i] * sp.sin(thetats[i]) for i in range(n)]
    dys = [-ls[i] * sp.cos(thetats[i]) for i in range(n)]
    xs = [sum(dxs[:i+1]) for i in range(n)]
    ys = [sum(dys[:i+1]) for i in range(n)]
    vxs = [x.diff(t) for x in xs]
    vys = [y.diff(t) for y in ys]
    v2s = [vx**2 + vy**2 for vx, vy in zip(vxs, vys)]
    Ts = [m*v2 / 2 for m, v2 in zip(ms, v2s)]
    Vs = [m*g*y for m, y in zip(ms, ys)]
    L = sum(Ts) - sum(Vs)

    eqs = [sp.diff(sp.diff(L, theta.diff(t)), t) - sp.diff(L, theta) for theta in thetats]
    substitutions = [(sp.diff(theta, t, t), beta) for theta, beta in zip(thetats, betas)]
    eqs = [eq.subs(substitutions) for eq in eqs]
    substitutions = [(sp.diff(theta, t), omega) for theta, omega in zip(thetats, omegas)]
    eqs = [eq.subs(substitutions) for eq in eqs]
    substitutions = [(thetat, theta) for thetat, theta in zip(thetats, thetas)]
    eqs = [eq.subs(substitutions) for eq in eqs]

    eqs = [sp.simplify(eq) for eq in eqs]
    return eqs


def numeric_solve(eqs, numeric):
    betas = [sp.symbols('beta'+str(i+1)) for i in range(len(eqs))]
    eqs_n = [eq.subs(numeric) for eq in eqs]
    free_symbols = [eq.free_symbols for eq in eqs_n]
    variables = free_symbols[0]
    for symbol in free_symbols:
        variables = variables.union(symbol)
    if variables != set(betas):
        raise ValueError('wrong numeric')

    try:
        solutions = sp.solve(eqs_n, betas)
        solutions = [var.subs(solutions) for var in betas]
        return solutions
    except Exception as e:
        print(f'can\'t solve the dymanic system: : {e}')
        return None
