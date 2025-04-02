with (DifferentialAlgebra);
R := DifferentialRing(blocks = [lex[temp], lex[posl,posr,x], lex[P102,P259,P2], lex[m,kinv,freeL,minv]], derivations = [t]);
eqs := [
    ((-1) * (posr(t))) + (x(t)),
    ((-1) * (P2)) + (posl(t)),
    ((-1) * (P102)) + ((m) * (diff(x(t), t$2))) + (((kinv)^(-1)) * (x(t))),
    ((-1) * (P259)) + ((kinv) * (diff(x(t), t$2))) + ((minv) * (x(t))),
    ((m) * (diff(x(t), t$2))) + (((kinv)^(-1)) * (((-1) * (freeL)) + ((-1) * (posl(t))) + (posr(t)))),
    ((-1) * (temp)) + ((m) * ((diff(x(t), t))^(2))) + (((kinv)^(-1)) * ((((-1) * (freeL)) + ((-1) * (posl(t))) + (posr(t)))^(2))),
    P2 <> 0,
    P102 <> 0,
    P259 <> 0,
    minv <> 0,
    kinv <> 0,
    freeL <> 0
];
ideal := RosenfeldGroebner(eqs, R);
print(Equations(ideal));
print(Inequations(ideal));
eq := NormalForm(temp, ideal);
eq := NormalForm(m*minv, ideal);