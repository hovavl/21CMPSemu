import matplotlib.pyplot as plt
import emcee
import numpy as np
from schwimmbad import MPIPool
import mpi4py
from scipy import interpolate
import pickle
import warnings
import copy
import hera_pspec as hp
import corner
import json
from scipy import special
import sys
import datetime

sys.path.insert(1, '/gpfs0/elyk/users/hovavl/21CMPSemu')

from read_LF_mini import likelihood
from NN_emulator import emulator
from Classifier import SignalClassifier


warnings.filterwarnings('ignore', module='hera_sim')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# HERA-Stack

warnings.filterwarnings('ignore')

h0 = 0.698


even_k = np.array([i % 2 == 0 for i in range(24)])
odd_k = np.array([i % 2 == 1 for i in range(24)])

kbins_104 = np.array([0.23, 0.29, 0.36, 0.42, 0.48, 0.54, 0.60, 0.66, 0.73, 0.79,
                      0.85, 0.91, 0.97, 1.04, 1.10, 1.16, 1.22, 1.28, 1.35, 1.41,
                      1.47, 1.53, 1.59, 1.65])*h0
kbins_79 = np.array([0.27, 0.34, 0.41, 0.48, 0.55, 0.62, 0.69, 0.76, 0.83, 0.90,
                     0.97, 1.05, 1.12, 1.19, 1.26, 1.33, 1.40, 1.47, 1.54, 1.61,
                     1.68, 1.75, 1.83, 1.90])*h0

delta_104 = np.array([13006, 2325, 0, 6866, 0, 25094, 17496, 14751, 0, 0, 0, 0,
                      26219, 18461, 112195, 0, 197836, 0, 62242, 0, 158372,
                      69220, 0, 0])[even_k]  # field D
err_104 = np.array([677, 1011, 1748, 2888, 4423, 6301, 8595, 11370, 14727, 18791,
                    23439, 29005, 35219, 42035, 50065, 58914, 68913, 79627, 91821,
                    105107, 119620, 134761, 151666, 169976])[even_k]  # field D

delta_79 = np.array([330, 44, 0, 736, 1174, 0, 0, 1636, 8578, 10570, 1360, 12516,
                     0, 0 ,0, 4083, 19670, 0, 0, 0, 0, 0, 25762, 0])[odd_k]  # field C
err_79 = np.array([129, 206, 360, 595, 921, 1305, 1778, 2341, 3048, 3868, 4824, 5958,
                   7158, 8473, 10058, 11855, 13891, 16094, 18511, 21178, 24006, 27134,
                   30636, 34297])[odd_k]  # field C



uvp = hp.UVPSpec()
field = 1
#uvp.read_hdf5('/Users/hovavlazare/PycharmProjects/21CM Project/data/ps_files/pspec_h1c_idr2_field1.h5')
uvp.read_hdf5('/gpfs0/elyk/users/hovavl/jobs/21cm_mcmc_job/data_for_mcmc/ps_files/pspec_h1c_idr2_field1.h5')

# print the two available keys
band1_key, band2_key = uvp.get_all_keys()

# get data
band2_wfn = uvp.get_window_function(band2_key)
w_mat_79 = band2_wfn[0, 3:27, 3:27]
band1_wfn = uvp.get_window_function(band1_key)
w_mat_104 = band1_wfn[0, 3:27, 3:27]

# model constants:
TAU_MEAN = 0.0569
TAU_STD_HIGH = 0.0081  # not True
TAU_STD_LOW = 0.0066
XH_MEAN = 0.06
XH_STD = 0.05

# data points
k_min = 0.03005976
k_max = 1.73339733
emulator_k_modes = 10**(np.linspace(np.log10(k_min) , np.log10(k_max) , num=100))[30:]
# luminosity function real data
# with open('/gpfs0/elyk/users/hovavl/jobs/21cm_mcmc_job/UV_LU_data_reduced_new.json', 'r') as openfile:
#     # Reading from json file
#     UV_LU_data = json.load(openfile)

# restore NN

nn_dir = '/gpfs0/elyk/users/hovavl/21CMPSemu/mini_halos/mini_halos_NN'


nn_ps = emulator(restore=True, use_log=False,
                 files_dir=f'{nn_dir}/retrained_model_files_7-9_v2',
                 name='emulator_7-9_mini')
nn_ps104 = emulator(restore=True, use_log=False,
                    files_dir=f'{nn_dir}/retrained_model_files_10-4_v2',
                    name='emulator_10-4_mini')
nn_tau = emulator(restore=True, use_log=False,
                  files_dir=f'{nn_dir}/tau_model_files',
                  name='tau_emulator')
nn_xH = emulator(restore=True, use_log=False,
                 files_dir=f'{nn_dir}/xH_model_files',
                 name='xH_emulator')
myClassifier79 = SignalClassifier(restore=True,
                                  files_dir=f'{nn_dir}/classifier_model_files_7-9',
                                  name='classify_NN_mini_7-9')
myClassifier104 = SignalClassifier(restore=True,
                                   files_dir=f'{nn_dir}/classifier_model_files_10-4',
                                   name='classify_NN_mini_10-4')

def culcPS(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, M_TURN, L_X, NU_X_THRESH, = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}
    label_pred = np.around(myClassifier79.predict(params)[0])[0]
    if label_pred == 0:
        return np.clip(np.random.randn(int(kbins_79.shape[0]/2)) * 1 + 2, 0, 3)
    predicted_testing_spectra = nn_ps.predict(params)
    tck = interpolate.splrep(emulator_k_modes, predicted_testing_spectra[0])
    model_ps = interpolate.splev(kbins_79, tck)

    model_ps = np.dot(w_mat_79, model_ps)
    return_ps = model_ps[odd_k]
    return return_ps


# calculate the power spectrum at z = 10.4
def culcPS2(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, M_TURN, L_X, NU_X_THRESH, = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}
    label_pred = np.around(myClassifier104.predict(params)[0])[0]
    if label_pred == 0:
        return np.clip(np.random.randn(int(kbins_104.shape[0]/2)) * 1 + 2, 0, 3)
    predicted_testing_spectra = nn_ps104.predict(params)
    tck = interpolate.splrep(emulator_k_modes, predicted_testing_spectra[0])
    model_ps = interpolate.splev(kbins_104, tck)

    model_ps = np.dot(w_mat_104, model_ps)
    return_ps = model_ps[even_k]
    if label_pred == 1:
        return return_ps
    return np.clip(np.random.randn(return_ps.shape[0]) * 1 + 2, 0, 3)

""" 
parameters for 21cm simulation
same as the simulations for the training set
"""


# predict luminosity function
# def predict_luminosity(theta):
#     return UV_LF_mini.predict_luminosity(theta)
#
#
# # calc the luminosity function likelihood
# def luminosity_func_lnlike(luminosity_func):
#     tot_lnlike = 0
#     redshifts = [6, 8, 10]
#     for i, func in enumerate(luminosity_func):
#         lum_data = UV_LU_data[str(redshifts[i])]['phi_k']
#         lum_err_sup = UV_LU_data[str(redshifts[i])]['err_sup']
#         lum_err_inf = UV_LU_data[str(redshifts[i])]['err_inf']
#         # print(f'data size: {len(lum_data)} err size: {len(lum_err)} func size: {len(func)}')
#
#         for j, val in enumerate(func):
#             if func[j] <= lum_data[j]:
#                 like = -(1 / 2) * (
#                         ((val - lum_data[j]) / lum_err_inf[j]) ** 2 + np.log(2 * np.pi * lum_err_inf[j] ** 2))
#             else:
#                 like = -(1 / 2) * (
#                         ((val - lum_data[j]) / lum_err_sup[j]) ** 2 + np.log(2 * np.pi * lum_err_sup[j] ** 2))
#             tot_lnlike += like
#     return tot_lnlike


# predict tau
def predict_tau(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, M_TURN, L_X, NU_X_THRESH, = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}
    predicted_tau = nn_tau.predict(params)[0]
    return predicted_tau


# predict xH
def predict_xH(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, M_TURN, L_X, NU_X_THRESH, = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}
    predicted_xH = nn_xH.predict(params)[0]
    return max(predicted_xH, 0)


# the model
def model(theta, k_modes=emulator_k_modes):
    return culcPS(theta)


# calculate the power spectrum at z = 10.4

def model2(theta, k_modes=emulator_k_modes):
    return culcPS2(theta)


def lnlike(theta):
    ps_lnLike = np.sum(np.log((1 / 2) * (1 + special.erf((delta_79 - model(theta)) / (err_79 * np.sqrt(2))))))

    ps104_lnLike = np.sum(np.log((1 / 2) * (1 + special.erf((delta_104 - model2(theta)) / (err_104 * np.sqrt(2))))))
    tau = predict_tau(theta)
    if tau > TAU_MEAN:
        tau_lnLike = (-1 / 2) * (((tau - TAU_MEAN) / TAU_STD_HIGH) ** 2 + np.log(2 * np.pi * TAU_STD_HIGH ** 2))
    else:
        tau_lnLike = (-1 / 2) * (((tau - TAU_MEAN) / TAU_STD_LOW) ** 2 + np.log(2 * np.pi * TAU_STD_LOW ** 2))
    xH = predict_xH(theta)
    if xH < 0.06:
        xH_lnLike = (-1 / 2) * np.log(2 * np.pi * XH_STD ** 2)
    else:
        xH_lnLike = (-1 / 2) * (((xH - XH_MEAN) / XH_STD) ** 2 + np.log(2 * np.pi * XH_STD ** 2))

    # UV_lum = predict_luminosity(theta)
    # luminosity_lnlike = luminosity_func_lnlike(UV_lum)
    luminosity_lnlike = likelihood(theta)
    return tau_lnLike + xH_lnLike + ps_lnLike + ps104_lnLike + luminosity_lnlike


def lnprior(theta):
    # n = np.random.rand()
    # if n > 0.9:
    #     print('theta: ', theta)
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, M_TURN, L_X, NU_X_THRESH, = theta

    if (-3.0 <= F_STAR10 <= -0.5 and -3.0 <= F_ESC10 <= 0.0 and 38 <= L_X <= 42 and 8 <= M_TURN <= 10.0
            and -0.2 <= ALPHA_STAR <= 1 and -1 <= ALPHA_ESC <= 1 and -2.5 <= F_STAR7_MINI <= -1
            and 0.1 <= NU_X_THRESH <= 1.5 and -0.5 <= ALPHA_STAR_MINI <= 0.5 and -3 <= F_ESC7_MINI <= 0):
        return 0.0
    # if (-1.62 <= F_STAR10 <= -1.04 and -1.47 <= F_ESC10 <= -0.52 and 39.29 <= L_X <= 41.52 and 8.2 <= M_TURN <= 9.17 and 300 <= NU_X_THRESH <=1210):
    #     return 0.0
    return -np.inf


def lnprob(theta, k_modes=emulator_k_modes, y_data=delta_79, data_err=err_79):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    val = lnlike(theta)
    return lp + val  # recall if lp not -inf, its 0, so this just returns likelihood


def GRforParameter(sampMatrix):
    s = np.array(sampMatrix)
    meanArr = []
    varArr = []
    n = s.shape[0]
    for samp in s:
        meanArr += [np.mean(samp)]
        varArr += [np.std(samp) ** 2]
    a = np.std(meanArr) ** 2
    b = np.mean(varArr)
    return np.sqrt((1 - 1 / n) + a / (b * n))


def main(p0, nwalkers, niter, ndim, lnprob, data):
    with MPIPool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

        print("Running burn-in...")
        p0, _, _, _ = sampler.run_mcmc(p0, 5000, progress=True)
        sampler.reset()
        flag = True
        count = 0
        while (flag):
            print("Running production...")
            pos, prob, state, _ = sampler.run_mcmc(p0, niter, progress=True)
            samples = sampler.get_chain()
            print('shape of samples: ', samples.shape)

            GR = []
            for i in range(samples.shape[2]):
                GR += [GRforParameter(samples[:, :, i])]
                tmp = np.abs((1 - np.array(GR) < 10 ** (-5)))
            count += niter
            print('position: ', pos, 'GR: ', GR, '\nnum of iterations: ', count)
            if np.all(tmp) or count >= 70000:
                flag = False
            else:
                p0 = pos
        return sampler, pos, prob, state


data = (kbins_79, delta_79, err_79)
nwalkers = 24
niter = 10000
initial = np.array([-1.24,-2.5, 0.5, 0, -1.35, -2, 0.3,  8.59, 40.0, 0.72])  # best guesses
ndim = len(initial)
p0 = [np.array(initial) + 1e-1 * np.random.randn(ndim) for i in range(nwalkers)]
sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)
samples = sampler.get_chain()

flat_samples = sampler.chain[:, :, :].reshape((-1, ndim))
pickle.dump(flat_samples, open(f'MCMC_results_{datetime.date.today()}_with_hera_updated_mini_high_prior.pk', 'wb'))

print(flat_samples.shape)
plt.ion()
labels = [r'$\log_{10}f_{\ast,10}$',
          r'$\log_{10}f_{\ast,7}$',
          r'$\alpha_{\ast}$',
          r'$\alpha_{\ast,\rm mini}$',
          r'$\log_{10}f_{{\rm esc},10}$',
          r'$\log_{10}f_{{\rm esc},7}$',
          r'$\alpha_{\rm esc}$',
          r'$\log_{10}[M_{\rm turn}/M_{\odot}]$',
          r'$\log_{10}\frac{L_{\rm X<2 \, keV/SFR}}{\rm erg\, s^{-1}\,M_{\odot}^{-1}\,yr}$',
          r'$E_0/{\rm keV}$']
fig = corner.corner(flat_samples, show_titles=True, labels=labels, plot_datapoints=True,
                    quantiles=[0.16, 0.5, 0.84])
plt.savefig(f'high_prior_mcmc_with_hera_mini_{datetime.date.today()}.png')
