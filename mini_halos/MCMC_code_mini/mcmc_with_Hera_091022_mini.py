import matplotlib.pyplot as plt
import emcee
import numpy as np
from schwimmbad import MPIPool
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


def interp_Wcdf(W, k, lower_perc=0.16, upper_perc=0.84):
    """
    Construct CDF from normalized window function and interpolate
    to get k at window func's 16, 50 & 84 percentile.

    Parameters
    ----------
    W : ndarray
        Normalized window function of shape (Nbandpowers, Nk)
    k : ndarray
        vector of k modes of shape (Nk,)

    Returns
    -------
    ndarray
        k of WF's 50th percentile
    ndarray
        dk of WF's 16th (default) percentile from median
    ndarray
        dk of WF's 84th (default) percentile from median
    """
    # get cdf: take sum of only abs(W)
    W = np.abs(W)
    Wcdf = np.array([np.sum(W[:, :i + 1].real, axis=1) for i in range(W.shape[1] - 1)]).T

    # get shifted k such that a symmetric window has 50th perc at max value
    kshift = k[:-1] + np.diff(k) / 2

    # interpolate each mode (xvalues are different for each mode!)
    med, low_err, hi_err = [], [], []
    for i, w in enumerate(Wcdf):
        interp = interpolate.interp1d(w, kshift, kind='linear', fill_value='extrapolate')
        m = interp(0.5)  # 50th percentile
        # m = k[np.argmax(W[i])]  # mode
        med.append(m)
        low_err.append(m - interp(lower_perc))
        hi_err.append(interp(upper_perc) - m)

    return np.array(med), np.array(low_err), np.array(hi_err)


# each field is an independent file, containing Band 1 and Band 2
# just load field 1 for now
uvp = hp.UVPSpec()
field = 1
#uvp.read_hdf5('/Users/hovavlazare/PycharmProjects/21CM Project/data/ps_files/pspec_h1c_idr2_field1.h5')
uvp.read_hdf5('/gpfs0/elyk/users/hovavl/jobs/21cm_mcmc_job/data_for_mcmc/ps_files/pspec_h1c_idr2_field1.h5')

# print the two available keys
band1_key, band2_key = uvp.get_all_keys()
keys = [band1_key, band2_key]

# get data
band2_dsq = uvp.get_data(band2_key)
band2_cov = uvp.get_cov(band2_key)
band2_wfn = uvp.get_window_function(band2_key)

band1_dsq = uvp.get_data(band1_key)
band1_cov = uvp.get_cov(band1_key)
band1_wfn = uvp.get_window_function(band1_key)

# extract data
spw = 0
kbins = uvp.get_kparas(spw)  # after spherical binning, k_perp=0 so k_mag = k_para

ks = slice(3, None)
xlim = (0, 2.0)
band2_err = np.sqrt(band2_cov[0].diagonal())
band1_err = np.sqrt(band1_cov[0].diagonal())
# data
y79 = band2_dsq[0, ks].real  # omit two first values (zeros)
y104 = band1_dsq[0, ks].real

yerr79 = band2_err[ks].real
yerr104 = band1_err[ks].real

mcmc_k_modes = kbins[ks] * h0
# should be changes to 2
smaller_2 = (mcmc_k_modes < 2)
# mcmc_k_modes[~smaller_2] = 0 # omit greater than 2

y79[~smaller_2] = 0
y104[~smaller_2] = 0

even79 = np.array([i % 2 == 0 for i in range(len(y79))])
odd_104 = ~even79

ypos = y79 > 0  # only positive values
ypos104 = y104 > 0

logical_79 = np.logical_and(even79, ypos)
logical_104 = np.logical_and(odd_104, ypos104)

ps_data79 = y79[logical_79]
ps_data104 = y104[logical_104]

yerr79 = yerr79[logical_79]
yerr104 = yerr104[logical_104]

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
                 files_dir=f'{nn_dir}/centered_model_files_7-9',
                 name='emulator_7-9_mini')
nn_ps104 = emulator(restore=True, use_log=False,
                    files_dir=f'{nn_dir}/centered_model_files_10-4',
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

    predicted_testing_spectra = nn_ps.predict(params)
    tck = interpolate.splrep(emulator_k_modes, predicted_testing_spectra[0])
    model_ps = interpolate.splev(mcmc_k_modes, tck)

    w_mat = band2_wfn[0, 3:, 3:]
    model_ps = np.dot(w_mat, model_ps)
    return_ps = model_ps[logical_79]
    if label_pred == 1:
        return return_ps
    return np.clip(np.random.randn(return_ps.shape[0]) * 0.5 + 2, 0, 5)


# calculate the power spectrum at z = 10.4
def culcPS2(theta):
    tmp = copy.deepcopy(theta)
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, M_TURN, L_X, NU_X_THRESH, = tmp
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}
    label_pred = np.around(myClassifier104.predict(params)[0])[0]

    predicted_testing_spectra = nn_ps104.predict(params)
    tck = interpolate.splrep(emulator_k_modes, predicted_testing_spectra[0])
    model_ps = interpolate.splev(mcmc_k_modes, tck)

    w_mat = band1_wfn[0, 3:, 3:]
    model_ps = np.dot(w_mat, model_ps)
    return_ps = model_ps[logical_104]
    if label_pred == 1:
        return return_ps
    return np.clip(np.random.randn(return_ps.shape[0]) * 0.5 + 2, 0, 5)

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
    ps_lnLike = np.sum(np.log((1 / 2) * (1 + special.erf((ps_data79 - model(theta)) / (yerr79 * np.sqrt(2))))))

    ps104_lnLike = np.sum(np.log((1 / 2) * (1 + special.erf((ps_data104 - model2(theta)) / (yerr104 * np.sqrt(2))))))
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
            and -0.2 <= ALPHA_STAR <= 1 and -1 <= ALPHA_ESC <= 1 and -3.5 <= F_STAR7_MINI <= -1
            and 0.1 <= NU_X_THRESH <= 1.5 and -0.5 <= ALPHA_STAR_MINI <= 0.5 and -3 <= F_ESC7_MINI <= 0):
        return 0.0
    # if (-1.62 <= F_STAR10 <= -1.04 and -1.47 <= F_ESC10 <= -0.52 and 39.29 <= L_X <= 41.52 and 8.2 <= M_TURN <= 9.17 and 300 <= NU_X_THRESH <=1210):
    #     return 0.0
    return -np.inf


def lnprob(theta, k_modes=emulator_k_modes, y_data=ps_data79, data_err=yerr79):
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
            if np.all(tmp) or count >= 60000:
                flag = False
            else:
                p0 = pos
        return sampler, pos, prob, state


data = (mcmc_k_modes, ps_data79, yerr79)
nwalkers = 24
niter = 10000
initial = np.array([-1.24,-2.5, 0.5, 0, -1.35, -1.35, -0.3,  8.59, 40.64, 0.72])  # best guesses
ndim = len(initial)
p0 = [np.array(initial) + 1e-1 * np.random.randn(ndim) for i in range(nwalkers)]
sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)
samples = sampler.get_chain()

flat_samples = sampler.chain[:, :, :].reshape((-1, ndim))
pickle.dump(flat_samples, open(f'MCMC_results_{datetime.date.today()}_with_hera_mini.pk', 'wb'))

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
plt.savefig(f'mcmc_with_hera_mini_{datetime.date.today()}.png')
