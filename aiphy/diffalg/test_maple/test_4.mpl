with(DifferentialAlgebra);
p1 := pos(t) - posr(t);
k := diff(pos(t), t) / diff(pos(t), [t$3]) / mass;
p2 := diff(k, t);
R := DifferentialRing(blocks=[posr, lex[pos , mass]], derivations=[t]);

eqs := [(p0 + p2)^3, p1^2, p2^5 - p0 - p1 - p2, (p0 + 1/(p1+1) - 1 - p2)^2];

ideal := RosenfeldGroebner(eqs, R);

Equations(ideal);

p3 := numer(diff(diff(pos(t), t$2) / diff(posr(t), [t$4]), t));

BelongsTo(p3, ideal);

p4 := numer(diff((mass + diff(pos(t), t$2)) * diff(pos(t), t) / diff(posr(t), [t$3]), t));

ReducedForm(p4, ideal);

# with(DifferentialAlgebra);
# p1 := pos(t) - posr(t);
# k := diff(pos(t), t) / diff(pos(t), [t$3]) / mass;
# p2 := diff(k, t);
# R := DifferentialRing(blocks=[[posr(t), pos(t)] , [mass]], derivations=[t]);

# eqs := [numer((p0 + p2)^3), numer(p1^2), numer(p2^5 - p0 - p1 - p2), numer((p0 + 1/(p1+1) - 1 - p2)^2)];

# ideal := RosenfeldGroebner(eqs, R);

# Equations(ideal);

# p3 := numer(diff(diff(pos(t), t$2) / diff(posr(t), [t$4]), t));

# BelongsTo(p3, ideal);

# p4 := numer(diff((mass + diff(pos(t), t$2)) * diff(pos(t), t) / diff(posr(t), [t$3]), t));

# ReducedForm(p4, ideal);