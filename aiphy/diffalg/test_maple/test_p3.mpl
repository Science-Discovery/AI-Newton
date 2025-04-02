with(DifferentialAlgebra);
with (Tools);

R := DifferentialRing(blocks=[
    [pos_s1(t), pos_s2(t)], [Const_1, Const_2, Const_3, Const_4, MPi1_s2(t)]],
    derivations=[t]);

eqs := [
    -Const_1 + MPi1_s2(t),
    -Const_2 + diff(pos_s1(t), t)**2 - 2*diff(pos_s1(t), t)*diff(pos_s2(t), t) + diff(pos_s2(t), t)**2,
    diff(pos_s2(t), t$2),
    diff(pos_s1(t), t$2),
    -2*Const_3*pos_s1(t)*diff(pos_s2(t), t)*diff(pos_s2(t), t$2) - Const_3*diff(pos_s1(t), t)*diff(pos_s2(t), t)**2 + pos_s2(t)*diff(pos_s1(t), t)*diff(pos_s2(t), t$2) + pos_s2(t)*diff(pos_s1(t), t$2)*diff(pos_s2(t), t) + diff(pos_s1(t), t)*diff(pos_s2(t), t)**2,
    -2*Const_4*pos_s2(t)*diff(pos_s1(t), t)*diff(pos_s1(t), t$2) - Const_4*diff(pos_s1(t), t)**2*diff(pos_s2(t), t) + pos_s1(t)*diff(pos_s1(t), t)*diff(pos_s2(t), t$2) + pos_s1(t)*diff(pos_s1(t), t$2)*diff(pos_s2(t), t) + diff(pos_s1(t), t)**2*diff(pos_s2(t), t),
    Const_1 <> 0,
    Const_2 <> 0,
    Const_3 <> 0,
    Const_4 <> 0,
    MPi1_s1(t) <> 0
    ];
ideal := RosenfeldGroebner(eqs, R);
