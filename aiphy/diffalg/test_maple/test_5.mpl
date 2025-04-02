with(DifferentialAlgebra);
p1 := x - y + z;
p2 := z^2 - x^2 - 2*y^2;
p3 := x^3 - yx + z;
R := DifferentialRing(blocks=[z, lex[x, y]], derivations=[]);

eqs := [p1, p2, p3];

ideal := RosenfeldGroebner(eqs, R);

Equations(ideal);