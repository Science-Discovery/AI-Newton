with(DifferentialAlgebra);

R := DifferentialRing(blocks=[[f]], derivations=[x, y]);

u := f(x, y);
p := (diff(u, [x$4]) - u)^7 * (diff(u, [x$2]) + diff(u, x))^5;
q := (diff(u, [y$3]) - diff(u, [y$2]))^4 * (diff(u, y) + u)^5;

ideal := RosenfeldGroebner([p, q], R);

Equations(ideal);
