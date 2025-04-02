# 关于 PretendRegularDifferentialChain 的 bug 
# PretendRegularDifferentialChain 的作用是将一个方程组直接设置为一个微分多项式链（假装它满足微分多项式链具有的性质）。

with (DifferentialAlgebra);
with (Tools);
R := DifferentialRing(blocks = [[x1(t),x2(t)], [P0(),E0(),D0()], [mass1(),mass2()]], derivations = [t]);

I1 := PretendRegularDifferentialChain([((-1) * (P0())) + ((mass1()) * (diff(x1(t), t))) + ((mass2()) * (diff(x2(t), t)))], R);
I2 := PretendRegularDifferentialChain([((-1) * (P0())) + ((mass2()) * (diff(x2(t), t))), mass1()], R);
I3 := PretendRegularDifferentialChain([P0(), mass1(), mass2()], R);
ideal := [I1, I2, I3];
eq := -E0() + ((1/2) * (mass1()) * ((diff(x1(t), t))^(2))) + ((1/2) * (mass2()) * ((diff(x2(t), t))^(2)));
print(BelongsTo([eq], ideal));
print(ReducedForm([eq], ideal));



# 下面这样写会报错。 PretendRegularDifferentialChain 似乎会导致常数变量被识别成函数变量。

# with (DifferentialAlgebra);
# with (Tools);
# R := DifferentialRing(blocks = [[x1,x2], [P0,E0,D0], [mass1,mass2]], derivations = [t]);

# I1 := PretendRegularDifferentialChain([((-1) * (P0)) + ((mass1) * (diff(x1(t), t))) + ((mass2) * (diff(x2(t), t)))], R);
# I2 := PretendRegularDifferentialChain([((-1) * (P0)) + ((mass2) * (diff(x2(t), t))), mass1], R);
# I3 := PretendRegularDifferentialChain([P0, mass1, mass2], R);
# ideal := [I1, I2, I3];
# eq := -E0 + ((1/2) * (mass1) * ((diff(x1(t), t))^(2))) + ((1/2) * (mass2) * ((diff(x2(t), t))^(2)));
# print(BelongsTo([eq], ideal));
# print(ReducedForm([eq], ideal));
