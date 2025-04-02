with(DifferentialAlgebra);
v := diff(x(t), t);
a := diff(v, t);
p1 := v0 + a * t - v;
p2 := a - a0;
p3 := diff(v^2 - 2 * a * x(t), t);
p4 := x0 + v0 * t + 1/2 * a0 * t^2 - x(t);
R := DifferentialRing(blocks=[[x], [x0, v0, a0]], derivations=[t]);

eqs := [p1, p2, p3, p4];

ideal := RosenfeldGroebner(eqs, R);

Equations(ideal);

p4 := x(t) + 1/2 * a * t^2 - v * t;
ReducedForm(p4, ideal);


p5 := x(t) - v * t / 2;
ReducedForm(p5, ideal);


p6 := v^2 - 2 * a * x(t);
ReducedForm(p6, ideal);


p7 := v^2 + (2 * a * x(t));
ReducedForm(p7, ideal);