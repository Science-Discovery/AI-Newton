from aiphy.core import struct_oscillation
from aiphy.core import SExp, Intrinsic, Objstructure, Knowledge

sexp = SExp("#oscillation (1 -> Particle) (2 -> Spring) |- D[posx[1]'']/D[posx[1]]")
intrinsic = Intrinsic.From(sexp)
print(str(sexp))
mp = Objstructure.make_particle(2, 5)
mp.random_settings()
sp = Objstructure.make_spring(1.8, 2.2, 4, 6)
sp.random_settings()
print(str(mp))
print(str(sp))
newexp = struct_oscillation()
newexp.random_settings()

KK = Knowledge.default()
expdata1 = KK.eval_intrinsic(intrinsic, [mp, sp])
print(expdata1)

KK2 = Knowledge.default()
expdata2 = KK2.eval_intrinsic(intrinsic, [mp, sp])
print(expdata2)

print(expdata1 - expdata2)