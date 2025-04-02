with(DifferentialAlgebra);

R := DifferentialRing(blocks=[[m1, m2, x1, x2]], derivations=[t]);

p1 := m1(t) * diff(x1(t), t);
p2 := m2(t) * diff(x2(t), t);
e1 := m1(t) * diff(x1(t), t) ^ 2 / 2;
e2 := m2(t) * diff(x2(t), t) ^ 2 / 2;

eq1 := diff(m1(t), t);
eq2 := diff(m2(t), t);
eq3 := diff(p1+p2, t);
eq4 := diff(e1+e2, t);

eqs := [eq1, eq2, eq3, eq4, m1(t)<>0, m2(t)<>0];
ideal := RosenfeldGroebner(eqs, R);
Equations(ideal);
BelongsTo([diff(x1(t), t$2)], ideal);

eqs := [op(eqs), m1(t) + m2(t) <> 0];
ideal := RosenfeldGroebner(eqs, R);
Equations(ideal);
BelongsTo([diff(x1(t), t$2)], ideal);
BelongsTo([(diff(x1(t), t) - diff(x2(t), t))^2], ideal);