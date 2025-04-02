with (DifferentialAlgebra);
R := DifferentialRing(blocks = [lex[temp], lex[posx_2,posx_1], lex[P8], lex[P1,P5,C_07_1,C_07_2]], derivations = [t_0]);
eqs := [((-1) * (P1)) + (((diff(posx_1(t_0), t_0))^(-1)) * (((t_0) * (diff(posx_1(t_0), t_0$2))) + (diff(posx_1(t_0), t_0)))), ((-1) * (P5)) + (((diff(posx_2(t_0), t_0))^(-1)) * (((t_0) * (diff(posx_2(t_0), t_0$2))) + (diff(posx_2(t_0), t_0)))), ((-1) * (P8)) + ((C_07_1) * (diff(posx_1(t_0), t_0))) + ((C_07_2) * (diff(posx_2(t_0), t_0))), ((-1) * (temp)) + (((diff(posx_1(t_0), t_0))^(2)) * (posx_2(t_0))) + ((-1) * ((((t_0) * (diff(posx_1(t_0), t_0$2))) + (diff(posx_1(t_0), t_0)))^(2)) * (posx_2(t_0)))];
ideal := RosenfeldGroebner(eqs, R);
# ERROR HERE!

print(Equations(ideal));
eq := NormalForm(temp, ideal[1]);
print(eq);