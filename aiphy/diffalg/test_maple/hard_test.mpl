with (DifferentialAlgebra);
R := DifferentialRing(blocks = [lex[temp], lex[posx_1,posx_2], lex[P44,P38,P42,P32,P27], lex[C_28_2,C_28_1]], derivations = [t_0]);
eqs := [
    diff(posx_1(t_0), t_0$2), diff(posx_2(t_0), t_0$2),
    ((-1) * (P27)) + ((C_28_1) * (diff(posx_1(t_0), t_0))) + ((C_28_2) * (diff(posx_2(t_0), t_0))),
    ((-1) * (P32)) + ((C_28_1) * ((diff(posx_1(t_0), t_0))^(2))) + ((C_28_2) * ((diff(posx_2(t_0), t_0))^(2))),
    ((-1) * (P38)) + ((C_28_1) * (((-1) * (posx_1(t_0))) + ((t_0) * (diff(posx_1(t_0), t_0))))) + ((C_28_2) * (((-1) * (posx_2(t_0))) + ((t_0) * (diff(posx_2(t_0), t_0))))),
    ((-1) * (P42)) + ((C_28_1) * ((((-1) * (posx_1(t_0))) + ((t_0) * (diff(posx_1(t_0), t_0))))^(2))) + ((C_28_2) * ((((-1) * (posx_2(t_0))) + ((t_0) * (diff(posx_2(t_0), t_0))))^(2))),
    ((-1) * (P44)) + ((C_28_1) * (((-1) * (posx_1(t_0))) + ((t_0) * (diff(posx_1(t_0), t_0)))) * (diff(posx_1(t_0), t_0))) + ((C_28_2) * (((-1) * (posx_2(t_0))) + ((t_0) * (diff(posx_2(t_0), t_0)))) * (diff(posx_2(t_0), t_0))),
    ((-1) * (temp)) + (((C_28_2)^(2)) * (((C_28_1) * ((((-1) * (posx_1(t_0))) + ((t_0) * (diff(posx_1(t_0), t_0))))^(2))) + ((C_28_2) * ((((-1) * (posx_2(t_0))) + ((t_0) * (diff(posx_2(t_0), t_0))))^(2)))))
];
ideal := RosenfeldGroebner(eqs, R);
print(Equations(ideal));
eq := NormalForm(temp, ideal[1]);
print(eq);