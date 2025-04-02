with(DifferentialAlgebra);
v := diff(x(t), t);
a := diff(v, t);
p1 := v - a * t;
p2 := diff(a, t);
p3 := diff(v^2 - 2 * a * x(t), t);
R := DifferentialRing(blocks=[x], derivations=[t]);

eqs := [p1, p2, p3];

ideal := RosenfeldGroebner(eqs, R);

Equations(ideal);

p4 := x(t) - 1/2 * a * t^2 - p1 * t;
ReducedForm(p4, ideal);