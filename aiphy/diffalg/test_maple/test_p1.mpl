with(DifferentialAlgebra);
with (Tools);

R := DifferentialRing(blocks=[
    [x(t), v(t), a(t), p(t), e(t)], [Expr1_s1(t), m()]],
    derivations=[t]);
eqs := [
    diff(x(t), t) - v(t),
    diff(v(t), t) - a(t),
    m * v(t) - p(t),
    m * v(t)^2 - e(t),
    Expr1_s1(t) - p(t)^2 / (2*m)
];
ideal := RosenfeldGroebner(eqs, R);
ReducedForm(diff(x(t), t), ideal);
ReducedForm(m*diff(x(t), t$2)^2, ideal);
ReducedForm(diff(x(t), t$2)*diff(x(t), t), ideal);

NormalForm(diff(x(t), t), ideal);
NormalForm(m*diff(x(t), t$2)^2, ideal);
NormalForm(diff(x(t), t$2)*diff(x(t), t), ideal);

# eqs := [
#     -p(t) + m*diff(x(t), t),
#     v(t)*m - p(t),
#     a(t)*m - diff(p(t), t),
#     Expr1_s1(t) + p(t)
# ];
# ideal := PretendRegularDifferentialChain(eqs, R);
# eq := diff(x(t), t$2);
# ReducedForm(eq, ideal);
# NormalForm(eq, ideal);

# eq := diff(p(t), t) - a(t)*m;
# NormalForm(eq, ideal);