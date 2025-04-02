with(DifferentialAlgebra);
p1 := diff(v(x,y), y) + u(x, y);
p2 := diff(u(x,y), x) - diff(u(x,y), y);
R := DifferentialRing(blocks=[[u, v]], derivations=[x, y]);
sys := expand([p1=0, p2=0, diff(p1*p2, x)=0, diff(p2, y)=0]);
simplified_sys := RosenfeldGroebner(sys, R);
Equations(simplified_sys);

p3 := diff(v(x,y), x) + u(x,y) - y;
RosenfeldGroebner({p1, p2, p3}, R);