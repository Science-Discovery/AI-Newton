with(DifferentialAlgebra);
p0 := diff(mass(t), t);
p1 := pos(t) - posr(t);
k := diff(pos(t), t) / diff(pos(t), [t$3]) / mass(t);
p2 := diff(k, t);
R := DifferentialRing(blocks=[[pos, posr, mass]], derivations=[t]);

ideal := RosenfeldGroebner([p0, p1, p2], R);

Equations(ideal);

p3 := numer(diff(diff(posr(t), t) / diff(pos(t), [t$3]), t));

BelongsTo(p3, ideal);
