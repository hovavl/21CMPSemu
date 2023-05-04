import numpy as np
from scipy.stats import qmc
import pickle
import os

F_STAR10 = [-3.0, -0.0]
ALPHA_STAR = [-0.5, 1]
F_ESC10 = [-3.0, -0.0]
ALPHA_ESC = [-1, 0.5]
M_TURN = [8.0, 10.0]
t_STAR = [0.0, 1.0]
L_X = [30, 42]
E_0 = [100, 1500]
X_RAY_SPEC_INDEX = [-1, 3]

l_bounds = [F_STAR10[0], ALPHA_STAR[0], F_ESC10[0], ALPHA_ESC[0], M_TURN[0], t_STAR[0], L_X[0], E_0[0],
            X_RAY_SPEC_INDEX[0]]
u_bounds = [F_STAR10[1], ALPHA_STAR[1], F_ESC10[1], ALPHA_ESC[1], M_TURN[1], t_STAR[1], L_X[1], E_0[1],
            X_RAY_SPEC_INDEX[1]]
sampler = qmc.LatinHypercube(d=9)
sample = sampler.random(n=5000)
paramspace = qmc.scale(sample, l_bounds, u_bounds)

# n = len(paramspace) / 2500
n = 1
data = []
for i in range(int(n)):
    outputName = os.getcwd() + '/new_samples/21cmfastData_batch43.pk'
    samp = []
    for params in paramspace[2500 * i: 2500 * (i + 2)]:
        samp += [params]
    data += [samp]
    #pickle.dump(data, open(outputName, 'ab'))
data = np.array(data)
print(data.shape)

# data = []
#
# for i in range(8, 10):
#     samp = []
#     path = '/Users/hovavlazare/Downloads/21cmfastData_batch' + str(i) + '.pk'
#     f = open(path, 'rb')
#     try:
#         while True:
#             tmp = pickle.load(f)
#             for obj in tmp:
#                 samp += [obj]
#     except Exception:
#         f.close()
#     samp = np.reshape(samp, (2500, 9))
#     data += [samp]
#
# data = np.array(data)
# print(data.shape)
a = data[0]
b = data[1]
print(a.shape)
print(b.shape)
c = np.concatenate([data[0], data[1]], axis=0)
print(c.shape)
tmp1 = []
for val in c:
    tmp1 += [tuple(val)]
a = set(tmp1)
print(len(a))
