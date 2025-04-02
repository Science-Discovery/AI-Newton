from aiphy.core import ExpData
import numpy as np
# random a 2d array
data = np.random.rand(50, 100) / 10 + np.array([[i*i for i in range(100)] for j in range(50)])
data = ExpData(data)
# print(data.__difftau__().data)
print(data.__difftau__().is_conserved)
print(data.__difftau__().__difftau__())
print(data.__difftau__().__difftau__().__difftau__())