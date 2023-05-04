from scipy.stats import qmc
import pickle
import os

# Press the green button in the gutter to run the script.

# F_STAR10 = [-3.0, -0.5]
# F_STAR7_MINI = [-3.5, -1]
# ALPHA_STAR = [-0.2, 1]
# ALPHA_STAR_MINI = [-0.5, 0.5]
# F_ESC10 = [-3.0, -0.0]
# F_ESC7_MINI = [-3.0, -0.0]
# ALPHA_ESC = [-1, 1]
# M_TURN = [8.0, 9.5]
# L_X = [38, 42]
# E_0 = [100, 1500]

F_STAR10 = [-2.0, -1.0]
F_STAR7_MINI = [-3.2, -2.0]
ALPHA_STAR = [-0.0, 0.6]
ALPHA_STAR_MINI = [-3.0, 0.4]
F_ESC10 = [-1.85, -1.55]
F_ESC7_MINI = [-2.8, -1.5]
ALPHA_ESC = [0.3, 0.9]
M_TURN = [9.0,10]
L_X = [38.5, 41.5]
E_0 = [100, 1500]

l_bounds = [F_STAR10[0], F_STAR7_MINI[0], ALPHA_STAR[0], ALPHA_STAR_MINI[0], F_ESC10[0], F_ESC7_MINI[0], ALPHA_ESC[0],
            M_TURN[0], L_X[0], E_0[0]]
u_bounds = [F_STAR10[1], F_STAR7_MINI[1], ALPHA_STAR[1], ALPHA_STAR_MINI[1], F_ESC10[1], F_ESC7_MINI[1], ALPHA_ESC[1],
            M_TURN[1], L_X[1], E_0[1]]
sampler = qmc.LatinHypercube(d=10)
sample = sampler.random(n=5000)
paramspace = qmc.scale(sample, l_bounds, u_bounds)

n = 1
for i in range(int(n)):
    outputName = os.getcwd() + '/new_samples_mini/21cmfastData_batch44.pk'
    for params in paramspace:

        pickle.dump(params, open(outputName, 'ab'))