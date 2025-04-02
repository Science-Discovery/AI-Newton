with(DifferentialAlgebra);
p1 := pos(t) - posr(t);
p2 := diff(pos(t), t) / diff(pos(t), [t$3]) / mass - k;
p3 := diff(pos(t), [t$3]) / diff(posr(t), t) - k2;
p4 := diff(posr(t), [t$3]) ** 2 / diff(pos(t), t) ** 2 - k3;
R := DifferentialRing(blocks=[lex[pos, posr], lex[k, k2, k3], mass], derivations=[t]);

eqs := [p1, p2, p3, p4, k3<>0];

ideal := RosenfeldGroebner(eqs, R);

Equations(ideal);

p3 := numer(diff(diff(pos(t), t$2) / diff(posr(t), [t$4]), t));

BelongsTo(p3, ideal);

p4 := numer(diff((mass + diff(pos(t), t$2)) * diff(pos(t), t) / diff(posr(t), [t$3]), t));

ReducedForm(p4, ideal);
