from aiphy.core import struct_oscillation
from aiphy.core import SExp, Exp, Intrinsic, MeasureType, Knowledge
exp_oscillation = struct_oscillation()
sexp = SExp("#oscillation(1->Particle)|-D[posx[1]'']/D[posx[1]]")
print(str(sexp))
intrinsic = Intrinsic.From(sexp)
print(SExp("#oscillation|-D[posx[1]'']/D[posx[1]]"))
KK = Knowledge()
exp_oscillation.random_settings()
expdata = exp_oscillation.collect_expdata(MeasureType.default())
exp = Exp("D[posx[1]'']/D[posx[1]]")
result = KK.eval(exp0=exp, context=exp_oscillation)
print(result.is_conserved, result.is_zero)
print(result)