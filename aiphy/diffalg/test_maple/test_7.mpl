with(DifferentialAlgebra);

R := DifferentialRing(blocks=[lex[x1, x2], lex[p0, e0, d0], lex[m1, m2]], derivations=[t]);

v1 := diff(x1(t), t);
v2 := diff(x2(t), t);
p1 := m1 * v1;
p2 := m2 * v2;
e1 := m1 * v1 ^ 2 / 2;
e2 := m2 * v2 ^ 2 / 2;

eq1 := p1 + p2 - p0;
eq2 := e1 + e2 - e0;
eq3 := (v1-v2)^2 - d0;

# eqs := [eq1, eq2, eq3, m1<>0, m2<>0, d0<>0];
eqs := [eq1, eq2, eq3];
ideal := RosenfeldGroebner(eqs, R);
Equations(ideal);
BelongsTo([diff(x1(t), t$2)], ideal);
BelongsTo([diff((v1-v2)^2, t)], ideal);

ReducedForm([diff(x1(t), t$2)], ideal);
ReducedForm([diff((v1-v2)^2, t)], ideal);