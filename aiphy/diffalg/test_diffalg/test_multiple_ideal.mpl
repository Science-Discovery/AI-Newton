with (DifferentialAlgebra);
R := DifferentialRing(blocks = [lex[temp], lex[length_2,posx_1], lex[C_03_2,P1,C_10_1,C_35_2]], derivations = [t_0]);
eqs := [((-1) * (P1)) + (((diff(posx_1(t_0), t_0$2))^(-1)) * ((C_03_2) + ((-1) * (length_2(t_0))))), ((-1) * (temp(t_0))) + ((C_10_1) * (diff(posx_1(t_0), t_0$2))) + ((-1) * (C_35_2) * ((C_03_2) + ((-1) * (length_2(t_0))))),
    C_03_2<> 0, P1<> 0, C_10_1<>0, C_35_2<>0];
ideal := RosenfeldGroebner(eqs, R);
print(Equations(ideal[1]));
print(Equations(ideal[2]));
