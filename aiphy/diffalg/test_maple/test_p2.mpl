with(DifferentialAlgebra);
with (Tools);

R := DifferentialRing(blocks=[
    [x(t), xr(t), v(t), a(t), p(t), eV(t), eT(t)], [Expr(t), xl(), m(), k(), l0()]],
    derivations=[t]);
eqs := [
    x(t) - xr(t),
    -v(t) + diff(x(t), t),
    -a(t) + diff(v(t), t),
    - p(t) + m * v(t),
    -eT(t) + m * v(t)^2,
    -eV(t) + k * Expr(t)^2,
    diff(eT(t) + eV(t), t),
    -Expr(t) + (l0 - (xr(t) - xl))
];
ideal := RosenfeldGroebner(eqs, R);
NormalForm(diff(x(t), t), ideal);
NormalForm(m*diff(x(t), t$2)^2, ideal);
NormalForm(diff(x(t), t$2)*diff(x(t), t), ideal);
ReducedForm(x(t) - xl - l0, ideal);
ReducedForm(a(t), ideal);
NormalForm(a(t), ideal);

rr := DifferentialRing(blocks=[
    [Expr(t), xl(), p(t), eV(t), eT(t)], [m(), k(), l0(), xr(t), x(t), v(t), a(t)]],
    derivations=[t]);
JJ := RosenfeldGroebner(Equations(ideal[1]), rr);

NormalForm(diff(x(t), t), JJ);
NormalForm(m*diff(x(t), t$2)^2, JJ);
NormalForm(diff(x(t), t$2)*diff(x(t), t), JJ);
NormalForm(x(t) - xl - l0, JJ);
NormalForm(a(t), JJ);
