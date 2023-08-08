import matplotlib.pyplot as plt
import emcee
import numpy as np
from schwimmbad import MPIPool
from scipy import interpolate
import pickle
import warnings
import copy
import hera_pspec
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

# each field is an independent file, containing Band 1 and Band 2
### update this to the correct path ('/gpfs0/elyk/users/hovavl/...')
uvp_xtk_dsq_1 = []
uvp_xtk_dsq_2 = []
for field in 'ABCDE':
    uvp = hera_pspec.UVPSpec()
    uvp.read_hdf5(f'/gpfs0/elyk/users/hovavl/python_modules/H1C_IDR3_Power_Spectra/SPOILERS/All_Epochs_Power_Spectra/results_files/Deltasq_Band_1_Field_{field}.h5')
    uvp_xtk_dsq_1.append(uvp)

    uvp = hera_pspec.UVPSpec()
    uvp.read_hdf5(f'/gpfs0/elyk/users/hovavl/python_modules/H1C_IDR3_Power_Spectra/SPOILERS/All_Epochs_Power_Spectra/results_files/Deltasq_Band_2_Field_{field}.h5')
    uvp_xtk_dsq_2.append(uvp)

# u_1C = uvp_xtk_dsq_1[2]
# u_1D = uvp_xtk_dsq_1[3]
# u_2C = uvp_xtk_dsq_2[2]
#
# spw = 0#i because we're loading only one band into each object
#
# kp_1 = u_1C.get_kparas(spw)
# ks_1 = slice(np.argmin(np.abs(kp_1 - 0.128)), None, 1)
#
# y_1C = np.real(u_1C.data_array[spw].squeeze().copy()[ks_1].copy())
# y_1C[y_1C < 0] *= 0
# yerr_1C = np.sqrt(np.diagonal(u_1C.cov_array_real[spw].squeeze()))[ks_1]
#
# y_1D = np.real(u_1D.data_array[spw].squeeze().copy()[ks_1].copy())
# y_1D[y_1D < 0] *= 0
# yerr_1D = np.sqrt(np.diagonal(u_1D.cov_array_real[spw].squeeze()))[ks_1]
#
# kbins_1 = u_1C.get_kparas(spw)
# k_1 = kbins_1[ks_1]
#
# kp_2 = u_2C.get_kparas(spw)
# ks_2 = slice(np.argmin(np.abs(kp_2 - 0.128)), None, 1)
#
# y_2C = np.real(u_2C.data_array[spw].squeeze().copy()[ks_2].copy())
# y_2C[y_2C < 0] *= 0
# yerr_2C = np.sqrt(np.diagonal(u_2C.cov_array_real[spw].squeeze()))[ks_2]
#
# kbins_2 = u_2C.get_kparas(spw)
#
# k_2 = kbins_2[ks_2]
# win1 = u_1C.window_function_array[0].squeeze()[3:,3:]
# win2 = u_2C.window_function_array[0].squeeze()[3:,3:]#[i].squeeze()[:, ks]
#
# y_1 = (y_1D+y_1C)/2
# yerr_1 = np.sqrt(yerr_1C**2 + yerr_1D**2)/np.sqrt(2)
#
# k_1 = k_1[2:]
# y_1 = y_1[2:]
# yerr_1 = yerr_1[2:]
#
# k_2 = k_2[2:]
# y_2 = y_2C[2:]
# yerr_2 = yerr_2C[2:]
#
# even_ind = np.array([i % 2 == 0 for i in range(len(y_1))])
# odd_ind = np.array([i % 2 == 1 for i in range(len(y_2))])
#
# y_1 = y_1[even_ind]
# yerr_1 = yerr_1[even_ind]
#
# y_2 = y_2[odd_ind]
# yerr_2 = yerr_2[odd_ind]



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

nn_dir = '/gpfs0/elyk/users/hovavl/21CMPSemu/mini_halos/mini_halos_NN/model_without_Mturn'


nn_ps = emulator(restore=True, use_log=False,
                 files_dir=f'{nn_dir}/retrained_model_files_7-9',
                 name='emulator_7-9_mini')
nn_ps104 = emulator(restore=True, use_log=False,
                    files_dir=f'{nn_dir}/retrained_model_files_10-4',
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


def ps_likelihood_1_for_field(y,y_pred, y_err, window_func, bins):
    model_ps = np.dot(window_func, y_pred)
    binned_ps = model_ps[bins]
    return np.sum(np.log((1 / 2) * (1 + special.erf((y - binned_ps) / (y_err * np.sqrt(2))))))


def ps_likelihood_2_for_field(y,y_pred, y_err, window_func, bins):
    model_ps = np.dot(window_func, y_pred)
    binned_ps = model_ps[bins]
    return np.sum(np.log((1 / 2) * (1 + special.erf((y - binned_ps) / (y_err * np.sqrt(2))))))


def ps_likelihood1(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, L_X, NU_X_THRESH, = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}
    u = uvp_xtk_dsq_1[0]
    spw = 0  # i because we're loading only one band into each object

    kp_1_a = u.get_kparas(spw)
    ks_1_a = slice(np.argmin(np.abs(kp_1_a - 0.128)), None, 1)
    k_a = kp_1_a[ks_1_a]
    k_a = k_a[2:]

    label_pred = np.around(myClassifier104.predict(params)[0])[0]
    if label_pred == 0:
        model_ps = np.clip(np.random.randn(int(k_a.shape[0])) * 1 + 2, 0, 3)
    else:
        predicted_testing_spectra = nn_ps104.predict(params)
        tck = interpolate.splrep(emulator_k_modes, predicted_testing_spectra[0])
        model_ps = interpolate.splev(k_a, tck)


    loglike = 0
    for i in range(5):
        u = uvp_xtk_dsq_1[i]
        spw = 0  # i because we're loading only one band into each object

        kp_1 = u.get_kparas(spw)
        ks_1 = slice(np.argmin(np.abs(kp_1 - 0.128)), None, 1)

        y = np.real(u.data_array[spw].squeeze().copy()[ks_1].copy())
        y[y < 0] *= 0
        yerr = np.sqrt(np.diagonal(u.cov_array_real[spw].squeeze()))[ks_1]

        kbins_1 = u.get_kparas(spw)
        k = kbins_1[ks_1]
        win1 = u.window_function_array[0].squeeze()[3:, 3:]

        k = k[2:]
        y = y[2:]
        yerr = yerr[2:]

        even_ind = np.array([i % 2 == 0 for i in range(len(y))])

        y = y[even_ind]
        yerr = yerr[even_ind]

        loglike += ps_likelihood_1_for_field(y, model_ps, yerr, win1, even_ind)

    return loglike

def ps_likelihood2(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, L_X, NU_X_THRESH, = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}

    u = uvp_xtk_dsq_2[0]
    spw = 0  # i because we're loading only one band into each object

    kp_2_a = u.get_kparas(spw)
    ks_2_a = slice(np.argmin(np.abs(kp_2_a - 0.128)), None, 1)
    k_a = kp_2_a[ks_2_a]
    k_a = k_a[2:]

    label_pred = np.around(myClassifier79.predict(params)[0])[0]
    if label_pred == 0:
        model_ps = np.clip(np.random.randn(int(k_a.shape[0])) * 1 + 2, 0, 3)
    else:
        predicted_testing_spectra = nn_ps.predict(params)
        tck = interpolate.splrep(emulator_k_modes, predicted_testing_spectra[0])
        model_ps = interpolate.splev(k_a, tck)

    loglike = 0
    for i in range(5):
        u = uvp_xtk_dsq_2[i]
        spw = 0  # i because we're loading only one band into each object

        kp_1 = u.get_kparas(spw)
        ks_1 = slice(np.argmin(np.abs(kp_1 - 0.128)), None, 1)

        y = np.real(u.data_array[spw].squeeze().copy()[ks_1].copy())
        y[y < 0] *= 0
        yerr = np.sqrt(np.diagonal(u.cov_array_real[spw].squeeze()))[ks_1]

        kbins_1 = u.get_kparas(spw)
        k = kbins_1[ks_1]
        win1 = u.window_function_array[0].squeeze()[3:, 3:]

        k = k[2:]
        y = y[2:]
        yerr = yerr[2:]

        odd_ind = np.array([i % 2 == 1 for i in range(len(y))])


        y = y[odd_ind]
        yerr = yerr[odd_ind]

        loglike += ps_likelihood_2_for_field(y,model_ps, yerr, win1, odd_ind)

    return loglike


# def culcPS(theta):
#     tmp = copy.deepcopy(theta)
#     F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, L_X, NU_X_THRESH, = tmp
#     params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X],
#               'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
#               'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}
#     label_pred = np.around(myClassifier79.predict(params)[0])[0]
#     if label_pred == 0:
#         return np.clip(np.random.randn(int(k_2.shape[0]/2)) * 1 + 2, 0, 3)
#     predicted_testing_spectra = nn_ps.predict(params)
#     tck = interpolate.splrep(emulator_k_modes, predicted_testing_spectra[0])
#     model_ps = interpolate.splev(k_2, tck)
#
#     model_ps = np.dot(win2, model_ps)
#     return_ps = model_ps[odd_ind]
#     return return_ps
#
#
# # calculate the power spectrum at z = 10.4
# def culcPS2(theta):
#     tmp = copy.deepcopy(theta)
#     F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, L_X, NU_X_THRESH, = tmp
#     params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X],
#               'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
#               'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}
#     label_pred = np.around(myClassifier104.predict(params)[0])[0]
#     if label_pred == 0:
#         return np.clip(np.random.randn(int(k_1.shape[0]/2)+1) * 1 + 2, 0, 3)
#     predicted_testing_spectra = nn_ps104.predict(params)
#     tck = interpolate.splrep(emulator_k_modes, predicted_testing_spectra[0])
#     model_ps = interpolate.splev(k_1, tck)
#
#     model_ps = np.dot(win1, model_ps)
#     return_ps = model_ps[even_ind]
#     return return_ps
#
# """
# parameters for 21cm simulation
# same as the simulations for the training set
# """
#
#
# # predict luminosity function
# # def predict_luminosity(theta):
# #     return UV_LF_mini.predict_luminosity(theta)
# #
# #
# # # calc the luminosity function likelihood
# # def luminosity_func_lnlike(luminosity_func):
# #     tot_lnlike = 0
# #     redshifts = [6, 8, 10]
# #     for i, func in enumerate(luminosity_func):
# #         lum_data = UV_LU_data[str(redshifts[i])]['phi_k']
# #         lum_err_sup = UV_LU_data[str(redshifts[i])]['err_sup']
# #         lum_err_inf = UV_LU_data[str(redshifts[i])]['err_inf']
# #         # print(f'data size: {len(lum_data)} err size: {len(lum_err)} func size: {len(func)}')
# #
# #         for j, val in enumerate(func):
# #             if func[j] <= lum_data[j]:
# #                 like = -(1 / 2) * (
# #                         ((val - lum_data[j]) / lum_err_inf[j]) ** 2 + np.log(2 * np.pi * lum_err_inf[j] ** 2))
# #             else:
# #                 like = -(1 / 2) * (
# #                         ((val - lum_data[j]) / lum_err_sup[j]) ** 2 + np.log(2 * np.pi * lum_err_sup[j] ** 2))
# #             tot_lnlike += like
# #     return tot_lnlike
#

# predict tau
def predict_tau(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, L_X, NU_X_THRESH, = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}
    predicted_tau = nn_tau.predict(params)[0]
    return predicted_tau


# predict xH
def predict_xH(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, L_X, NU_X_THRESH, = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}
    predicted_xH = nn_xH.predict(params)[0]
    return max(predicted_xH, 0)


# the model
# def model(theta, k_modes=emulator_k_modes):
#     return culcPS(theta)
#
#
# # calculate the power spectrum at z = 10.4
#
# def model2(theta, k_modes=emulator_k_modes):
#     return culcPS2(theta)
#

def lnlike(theta):
    # ps_lnLike = np.sum(np.log((1 / 2) * (1 + special.erf((y_2 - model(theta)) / (yerr_2 * np.sqrt(2))))))
    #
    # ps104_lnLike = np.sum(np.log((1 / 2) * (1 + special.erf((y_1 - model2(theta)) / (yerr_1 * np.sqrt(2))))))
    ps_lnLike = ps_likelihood2(theta)
    ps104_lnLike = ps_likelihood1(theta)
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
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, L_X, NU_X_THRESH, = theta

    if (-3.0 <= F_STAR10 <= -0.0 and -3.0 <= F_ESC10 <= 0.0 and 38 <= L_X <= 42
            and -0.2 <= ALPHA_STAR <= 1 and -1 <= ALPHA_ESC <= 1 and -3.5 <= F_STAR7_MINI <= -1.0
            and 0.1 <= NU_X_THRESH <= 1.5 and -0.5 <= ALPHA_STAR_MINI <= 0.5 and -3 <= F_ESC7_MINI <= 0):
        return 0.0
    # if (-1.62 <= F_STAR10 <= -1.04 and -1.47 <= F_ESC10 <= -0.52 and 39.29 <= L_X <= 41.52 and 8.2 <= M_TURN <= 9.17 and 300 <= NU_X_THRESH <=1210):
    #     return 0.0
    return -np.inf


def lnprob(theta, k_modes=emulator_k_modes):
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


def main(p0, nwalkers, niter, ndim, lnprob):
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



nwalkers = 24
niter = 10000
initial = np.array([-1.24,-2.5, 0.5, 0, -1.35, -2, 0.3, 40.0, 0.72])  # best guesses
ndim = len(initial)
p0 = [np.array(initial) + 1e-1 * np.random.randn(ndim) for i in range(nwalkers)]
sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob)
samples = sampler.get_chain()

flat_samples = sampler.chain[:, :, :].reshape((-1, ndim))
pickle.dump(flat_samples, open(f'/gpfs0/elyk/users/hovavl/jobs/21cm_mcmc_job/MCMC_results_mini/MCMC_{datetime.date.today()}_all_fields.pk', 'wb'))

print(flat_samples.shape)
plt.ion()
labels = [r'$\log_{10}f_{\ast,10}$',
          r'$\log_{10}f_{\ast,7}$',
          r'$\alpha_{\ast}$',
          r'$\alpha_{\ast,\rm mini}$',
          r'$\log_{10}f_{{\rm esc},10}$',
          r'$\log_{10}f_{{\rm esc},7}$',
          r'$\alpha_{\rm esc}$',
          r'$\log_{10}\frac{L_{\rm X<2 \, keV/SFR}}{\rm erg\, s^{-1}\,M_{\odot}^{-1}\,yr}$',
          r'$E_0/{\rm keV}$']
fig = corner.corner(flat_samples, show_titles=True, labels=labels, plot_datapoints=True,
                    quantiles=[0.16, 0.5, 0.84])
plt.savefig(f'/gpfs0/elyk/users/hovavl/jobs/21cm_mcmc_job/MCMC_results_mini/MCMC_{datetime.date.today()}_all_fields.png')
