import numpy as np
import py21cmfast as p21c
import time
from scipy.interpolate import interp1d, splev, splrep
from tau import calc_tau
from power_spectrum import powerspectra
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from NN_emulator import emulator
import matplotlib.pylab as pylab

plt.rc('text', usetex=True)  # render font for tex
plt.rc('font', family='TimesNewRoman')  # use font
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.titley'] = 1.0  # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = 15  # pad is in points...
mpl.rcParams['figure.dpi'] = 500
pylab.rcParams['xtick.labelsize'] = 'xx-large'
pylab.rcParams['ytick.labelsize'] = 'xx-large'

user_params = {"DIM": 512, "HII_DIM": 128, "BOX_LEN": 256, "N_THREADS": 5, 'USE_RELATIVE_VELOCITIES': True,
               'DO_VCB_FIT': True,
               "POWER_SPECTRUM": 5}
flag_options = {"USE_MASS_DEPENDENT_ZETA": True, "USE_CMB_HEATING": False, "USE_LYA_HEATING": False,
                'USE_TS_FLUCT': True,
                'USE_MINI_HALOS': True, "INHOMO_RECO": True}

cosmo_params = {"hlittle": 0.6736, "OMb": 0.0493, "OMm": 0.3153,
                "A_s": 2.1e-9, "POWER_INDEX": 0.9649}


# print(cosmo_params['hlittle']**2 * cosmo_params['OMb'])
# print(cosmo_params['hlittle']**2 * cosmo_params['OMm'])

def run(theta):
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, M_TURN, L_X, NU_X_THRESH, = theta
    astro_params = {'F_STAR10': F_STAR10, 'F_ESC10': F_ESC10, 'L_X': L_X, 'M_TURN': M_TURN,
                    'NU_X_THRESH': NU_X_THRESH * 1000, 'ALPHA_STAR': ALPHA_STAR, 'ALPHA_ESC': ALPHA_ESC,
                    'F_STAR7_MINI': F_STAR7_MINI, 'ALPHA_STAR_MINI': ALPHA_STAR_MINI, 'F_ESC7_MINI': F_ESC7_MINI,
                    't_STAR': 0.5
                    }
    start = time.time()
    lightcone = p21c.run_lightcone(redshift=5.0,
                                   max_redshift=20.0,
                                   random_seed=1,
                                   write=False,
                                   user_params=user_params,
                                   lightcone_quantities={'brightness_temp', 'density', 'xH_box', 'Ts_box', 'dNrec_box',
                                                         'Gamma12_box', 'J_21_LW_box'},
                                   global_quantities={'brightness_temp', 'density', 'xH_box', 'Ts_box', 'dNrec_box',
                                                      'Gamma12_box', 'J_21_LW_box'},
                                   flag_options=flag_options,
                                   astro_params=astro_params,
                                   direc='_cache')
    end = time.time()
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(end - start))
    print(f'Total running time: {elapsed_time}')

    z_global = lightcone.node_redshifts
    lightcone_redshifts = lightcone.lightcone_redshifts
    T_b = lightcone.global_quantities['brightness_temp']
    xH_box = lightcone.xH_box
    global_xH = lightcone.global_xH
    density_box = lightcone.density
    tau = calc_tau(lightcone_redshifts, 1 + density_box, xH_box, cosmo_params)
    output = {'astro_params': astro_params,
              'global_xH': global_xH,
              'tau': tau,
              'T_b': T_b,
              'z_global': z_global
              }

    nchunks = 20

    data_PS, z_PS = powerspectra(lightcone_box=lightcone.brightness_temp,
                                 BOX_LEN=lightcone.user_params.BOX_LEN,
                                 HII_DIM=lightcone.user_params.HII_DIM,
                                 lightcone_redshifts=lightcone.lightcone_redshifts,
                                 nchunks=nchunks)

    output['data_PS'] = data_PS
    output['z_PS'] = z_PS
    x = 1
    with open('/Users/hovavlazare/GITs/21CMPSemu/simulations/simulation_resutls/high_likelihood_with_MCGs_lx=41_new_UVLF_Tb.pk','wb') as f:
        pickle.dump(output, f)


def extractPS(data_PS,
              z_PS,
              z_val):
    # Find the nearest points in z_PS to z_val
    z_ind_p = np.where(z_PS >= z_val)[0][0]
    z_ind_m = z_ind_p - 1
    z_p = z_PS[z_ind_p]
    z_m = z_PS[z_ind_m]
    # Extract the power spectra at the nearest points
    k_data_p = data_PS[z_ind_p]['k']
    k_data_m = data_PS[z_ind_m]['k']
    Delta_data_p = data_PS[z_ind_p]['delta']
    Delta_data_m = data_PS[z_ind_m]['delta']
    # Interpolate Delta_data_m at the values of k_data_p
    not_nan_inds = ~np.isnan(Delta_data_m)
    Delta_21_sq_interp = interp1d(k_data_m[not_nan_inds],
                                  Delta_data_m[not_nan_inds],
                                  kind='cubic', bounds_error=False)
    Delta_data_m = Delta_21_sq_interp(k_data_p)
    # Interpolate linearly between z_m and z_p
    Delta_data = (Delta_data_p * (z_val - z_m) + Delta_data_m * (z_p - z_val)) / (z_p - z_m)

    k_range = 10 ** (np.linspace(np.log10(min(k_data_p)), np.log10(max(k_data_p)), num=100))
    tck = splrep(k_data_p, Delta_data)
    delta_data_new = splev(k_range, tck)
    # plt.figure(figsize = (12,8))
    # plt.scatter(k_data_p , Delta_data, color = 'b')
    # plt.semilogx(k_range , delta_data_new, color = 'r')

    return k_range, delta_data_new


# run([-1.22, -2.72, 0.51, -0.02, -1.42, -2.32, 0.11, 8.59, 41, 0.74])
# exit(0)

# with open('/Users/hovavlazare/GITs/21CMPSemu/simulations/simulation_resutls/high_likelihood_Lx_38_no_mini.pk', 'rb') as f:
#     output_mini = pickle.load(f)

# with open('/Users/hovavlazare/GITs/21CMPSemu/simulations/single_simulation_results_high_likelihood_no_mini_halos.pk',
#           'rb') as f1:
#     output_no_mini = pickle.load(f1)
#
# with open(
#         '/Users/hovavlazare/GITs/21CMPSemu/simulations/single_simulation_results_high_likelihood_no_mini_halos_original_mcmc_params.pk',
#         'rb') as f2:
#     output_original_params = pickle.load(f2)
#
with open(
        '/Users/hovavlazare/GITs/21CMPSemu/simulations/simulation_resutls/high_likelihood_Lx_39_no_mini.pk',
        'rb') as f3:
    output_low_Lx = pickle.load(f3)
#
# with open(
#         '/Users/hovavlazare/GITs/21CMPSemu/simulations/single_simulation_results_high_likelihood_no_mini_halos_f_star_0575.pk',
#         'rb') as f4:
#     output_high_f_star = pickle.load(f4)
#
with open(
        '/Users/hovavlazare/GITs/21CMPSemu/simulations/simulation_resutls/high_likelihood_Lx_39_with_mini.pk',
        'rb') as f5:
    output_mini_halos_low_Lx = pickle.load(f5)

with open(
        '/Users/hovavlazare/GITs/21CMPSemu/simulations/simulation_resutls/high_likelihood_Lx_38_mini_with_Tb.pk',
        'rb') as f5:
    output_mini_halos_Tb = pickle.load(f5)

with open(
        '/Users/hovavlazare/GITs/21CMPSemu/simulations/simulation_resutls/high_likelihood_no_mini_Lx39_Tb_3-3-1.pk',
        'rb') as f5:
    output_no_mini_v331 = pickle.load(f5)

with open(
        '/Users/hovavlazare/GITs/21CMPSemu/simulations/simulation_resutls/high_likelihood_no_MCGs_lx=39_new_UVLF.pk',
        'rb') as f5:
    output_no_mini_UVLF20 = pickle.load(f5)

with open(
        '/Users/hovavlazare/GITs/21CMPSemu/simulations/simulation_resutls/high_likelihood_with_MCGs_lx=39_new_UVLF.pk',
        'rb') as f5:
    output_with_mini_UVLF20 = pickle.load(f5)

z_glob = output_mini_halos_Tb['z_global']
T_b = output_mini_halos_Tb['T_b']

# plt.plot(z_glob,T_b, color = 'black', label = 'Brightness temperature')
# plt.xlabel(r'redShift $z$', fontsize = 20)
# plt.ylabel(r'$\bar{T}_{21} [\rm mK]$', fontsize = 20)
# plt.tight_layout()
# plt.savefig('images/Brightness_temp.png')
# plt.show()
#
# exit(0)

nn_dir = '/Users/hovavlazare/GITs/21CMPSemu/mini_halos/mini_halos_NN'

nn_ps_centered = emulator(restore=True, use_log=False,
                          files_dir=f'{nn_dir}/retrained_model_files_7-9_v2',
                          name='emulator_7-9_mini')
nn_ps104_centered = emulator(restore=True, use_log=False,
                             files_dir=f'{nn_dir}/retrained_model_files_10-4_v2',
                             name='emulator_10-4_mini')

nn_ps = emulator(restore=True, use_log=False,
                 files_dir=f'{nn_dir}/model_files_7-9',
                 name='emulator_7-9_mini')
nn_ps104 = emulator(restore=True, use_log=False,
                    files_dir=f'{nn_dir}/model_files_10-4',
                    name='emulator_10-4_mini')

nn_dir = '/Users/hovavlazare/GITs/21CMPSemu/experimental'

nn_ps_centered_2 = emulator(restore=True, use_log=False,
                            files_dir=f'{nn_dir}/retrained_model_files_7-9',
                            name='emulator_7-9_full_range')
nn_ps104_centered_2 = emulator(restore=True, use_log=False,
                               files_dir=f'{nn_dir}/retrained_model_files_10-4',
                               name='emulator_10-4_full_range')

nn_ps_2 = emulator(restore=True, use_log=False,
                   files_dir=f'{nn_dir}/model_files_7-9',
                   name='emulator_7-9_full_range')
nn_ps104_2 = emulator(restore=True, use_log=False,
                      files_dir=f'{nn_dir}/model_files_10-4',
                      name='emulator_10-4_full_range')


def predict_ps(params_d):
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, M_TURN, L_X, NU_X_THRESH, = params_d
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}

    predicted_testing_spectra_104 = nn_ps104.predict(params)[0]
    predicted_testing_spectra_79 = nn_ps.predict(params)[0]
    return predicted_testing_spectra_104, predicted_testing_spectra_79


def predict_ps_centered(params_d):
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, M_TURN, L_X, NU_X_THRESH, = params_d
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              'F_STAR7_MINI': [F_STAR7_MINI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]}

    predicted_testing_spectra_104 = nn_ps104_centered.predict(params)[0]
    predicted_testing_spectra_79 = nn_ps_centered.predict(params)[0]
    return predicted_testing_spectra_104, predicted_testing_spectra_79


def predict_ps_no_mini(params_d):
    F_STAR10, ALPHA_STAR, F_ESC10, ALPHA_ESC, M_TURN, t_STAR, L_X, NU_X_THRESH, X_RAY_SPEC_INDEX = params_d
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              't_STAR': [t_STAR], 'X_RAY_SPEC_INDEX': [X_RAY_SPEC_INDEX]}

    predicted_testing_spectra_104 = nn_ps104_2.predict(params)[0]
    predicted_testing_spectra_79 = nn_ps_2.predict(params)[0]
    return predicted_testing_spectra_104, predicted_testing_spectra_79


def predict_ps_centered_no_mini(params_d):
    F_STAR10, ALPHA_STAR, F_ESC10, ALPHA_ESC, M_TURN, t_STAR, L_X, NU_X_THRESH, X_RAY_SPEC_INDEX = params_d
    params = {'F_STAR10': [F_STAR10], 'F_ESC10': [F_ESC10], 'L_X': [L_X], 'M_TURN': [M_TURN],
              'NU_X_THRESH': [NU_X_THRESH], 'ALPHA_STAR': [ALPHA_STAR], 'ALPHA_ESC': [ALPHA_ESC],
              't_STAR': [t_STAR], 'X_RAY_SPEC_INDEX': [X_RAY_SPEC_INDEX]}

    predicted_testing_spectra_104 = nn_ps104_centered_2.predict(params)[0]
    predicted_testing_spectra_79 = nn_ps_centered_2.predict(params)[0]
    return predicted_testing_spectra_104, predicted_testing_spectra_79


# theta = np.array([-1.15,-2.72, 0.28, -0.02, -1.44, -2.32, 0.63,  9.41, 39, 0.77])
# theta_no_mini =  np.array([-1.15, 0.28, -1.44, 0.63, 9.41, 0.5, 39, 0.77, 1])

theta = np.array([-1.22, -2.72, 0.51, -0.02, -1.42, -2.32, 0.11, 8.59, 39, 0.74])
theta_no_mini = np.array([-1.22, 0.51, -1.03, 0.11, 8.59, 0.5, 39, 0.74, 1.0])

theta_arr = np.array([np.append(np.append(theta[:8], [38 + i * 0.5]), theta[-1]) for i in range(9)])

pred_arr_79 = []
pred_arr_79_cen = []
pred_arr_104 = []
pred_arr_104_cen = []
# for theta_set in theta_arr:
#     ps_pred_104, ps_pred_79 = predict_ps(theta_set)
#     ps_pred_104_centered, ps_pred_79_centered = predict_ps_centered(theta_set)
#     pred_arr_79 += [ps_pred_79]
#     pred_arr_104 += [ps_pred_104]
#     pred_arr_79_cen += [ps_pred_79_centered]
#     pred_arr_104_cen += [ps_pred_104_centered]


ps_pred_104, ps_pred_79 = predict_ps(theta)
ps_pred_104_centered, ps_pred_79_centered = predict_ps_centered(theta)

ps_pred_104_2, ps_pred_79_2 = predict_ps_no_mini(theta_no_mini)
ps_pred_104_centered_2, ps_pred_79_centered_2 = predict_ps_centered_no_mini(theta_no_mini)

z = 7.9
# k_range, ps_mini = extractPS(output_mini['data_PS'], output_mini['z_PS'], z)

# _, ps_no_mini = extractPS(output_no_mini['data_PS'], output_no_mini['z_PS'], z)
#
# _, ps_orig = extractPS(output_original_params['data_PS'], output_original_params['z_PS'], z)
#
k_range, ps_low_lx = extractPS(output_low_Lx['data_PS'], output_low_Lx['z_PS'], z)

# _, ps_high_f_star = extractPS(output_high_f_star['data_PS'], output_high_f_star['z_PS'], z)
#
k_range, ps_mini_low_lx = extractPS(output_mini_halos_low_Lx['data_PS'], output_mini_halos_low_Lx['z_PS'], z)

k_range, ps_low_v331 = extractPS(output_no_mini_v331['data_PS'], output_no_mini_v331['z_PS'], z)

_, ps_no_mini_UVLF20 = extractPS(output_no_mini_UVLF20['data_PS'], output_no_mini_UVLF20['z_PS'], z)
_, ps_with_mini_UVLF20 = extractPS(output_with_mini_UVLF20['data_PS'], output_with_mini_UVLF20['z_PS'], z)

data_ps_104 = [1.70873566e+03, 6.82980246e+03, 2.79538804e+04, 6.74402430e+04, 5.98603267e+05, 4.84372271e+05,
               5.57589947e+05, 4.78555004e+05, 9.54421269e+05, 1.69729069e+06, 1.96517410e+06, 5.76079699e+06]
err_104 = [3729.04031526, 12467.41852979, 29918.66211936, 99099.22157678, 334553.08212765, 458190.6437129,
           1002020.81323529, 1247477.80919347, 1538374.89326397, 2294729.33778806, 2718250.51877183, 3207425.58760767]
data_ps_79 = [2.11680647e+02, 5.97405500e+03, 7.40958144e+03, 1.13768124e+04, 4.91534277e+04, 2.77321081e+04,
              1.54496271e+05, 2.85826555e+05]
err_79 = [3.67337588e+02, 3.44113928e+03, 7.37309171e+03, 1.32903481e+04, 2.18422511e+04, 3.33362502e+04,
          2.38151896e+05, 4.19391594e+05]
k_range = k_range[30:]

k_const = 0.134

# fig1, ax1 = plt.subplots(nrows=3, ncols=3, figsize=(30, 30))
# x=1
# for i in range(3):
#     for j in range(3):
#         ax1[i,j].loglog(k_range, pred_arr_79[i*3+j], color = 'black', ls = 'solid', label = f'prediction Lx = {38+(i*3+j)*0.5}')
#         ax1[i, j].loglog(k_range, pred_arr_79_cen[i*3+j], color='salmon', ls='dashed', label=f'prediction Lx = {38 + (i*3 + j) * 0.5} \n centered model')
#         ax1[i,j].legend(frameon=False, loc='lower right', fontsize = 30)
#         eb = ax1[i,j].errorbar(k_const, data_ps_79[0], err_79[0], color='gray', capsize=4, fmt='o',
#                           label=f'HERA most constraining \n data point at k = {k_const} ' + '$\mathrm{Mpc}^{-1}$')
#         eb[-1][0].set_linestyle('dotted')
#
# fig1.supxlabel('$k\,[1/\mathrm{Mpc}]$', fontsize=60)
# fig1.supylabel('$\Delta_{21}^2\,[\mathrm{mK}^2]\,\,(z=$' + str(7.9) + ')', fontsize=60)
# fig1.tight_layout()
# plt.savefig('/Users/hovavlazare/GITs/21CMPSemu/images/emulators_compression_z=79.png')
#
# fig2, ax2 = plt.subplots(nrows=3, ncols=3, figsize=(30, 30))
# k_const = 0.179
# for i in range(3):
#     for j in range(3):
#         ax2[i,j].loglog(k_range, pred_arr_104[i*3+j], color = 'black', ls = 'solid', label = f'prediction Lx = {38+(i*3 +j)*0.5}')
#         ax2[i,j].loglog(k_range, pred_arr_104_cen[i*3+j], color='salmon', ls='dashed', label=f'prediction Lx = {38 + (i*3+j) * 0.5} \n centered model')
#         ax2[i,j].legend(frameon=False, loc='lower right',fontsize = 30)
#         eb = ax2[i,j].errorbar(k_const, data_ps_104[0], err_104[0], color='gray', capsize=4, fmt='o',
#                           label=f'HERA most constraining \n data point at k = {k_const} ' + '$\mathrm{Mpc}^{-1}$')
#         eb[-1][0].set_linestyle('dotted')
#
#
# fig2.supxlabel('$k\,[1/\mathrm{Mpc}]$', fontsize=60)
# fig2.supylabel('$\Delta_{21}^2\,[\mathrm{mK}^2]\,\,(z=$' + str(10.4) + ')', fontsize=60)
# fig2.tight_layout()
# plt.savefig('/Users/hovavlazare/GITs/21CMPSemu/images/emulators_compression_z=104.png')
#
# exit(0)

fig = plt.figure(figsize=(10, 8))
# plt.loglog(k_range, ps_mini[30:], color='black', ls='solid', label='Including mini halos')
# plt.loglog(k_range, ps_no_mini[30:], color='black', ls='dashed', label='Not including mini halos')
# plt.loglog(k_range, ps_low_lx[30:], color='black', ls='solid', label=r'$L_{\rm X} = 10^{39}$ without MCGs', linewidth=3)
# plt.loglog(k_range, ps_mini_low_lx[30:], color='salmon', ls='solid', label=r'$L_{\rm X} = 10^{39}$' + ' with MCGs',
#            linewidth=3)
plt.loglog(k_range, ps_no_mini_UVLF20[30:], color='black', ls='solid',
           label=r'$L_{\rm X} = 10^{39}$' + ' without MCGs $M_{\\rm UV} >-20$', linewidth=3)
plt.loglog(k_range, ps_with_mini_UVLF20[30:], color='salmon', ls='solid',
           label=r'$L_{\rm X} = 10^{39}$' + ' with MCGs $M_{\\rm UV} >-20$', linewidth=3)
# plt.loglog(k_range, ps_pred_79, color='salmon', ls='dashed', label=' Not including mini halos- NN prediction \n')
plt.loglog(k_range, ps_pred_79_centered, color='salmon', ls='dotted',
           label='With MCGs - NN prediction')#\n Retrained model')
plt.loglog(k_range, ps_pred_79_centered_2, color='black', ls='dotted',
           label='Without MCGs - NN prediction')#\n Retrained model')
# plt.loglog(k_range, ps_orig[30:], color='black', ls='dashdot', label='highest likelihood parameters \n without mini halos')
k_const = 0.134

# plt.vlines(0.179, 0, 10**3, color = 'r', ls = 'dotted', label='k = 0.179')
eb = plt.errorbar(k_const, data_ps_79[0], err_79[0], color='gray', capsize=6, fmt='o',
                  label=f'HERA most constraining \n data point at k = {k_const} ' + '$\mathrm{Mpc}^{-1}$')

plt.scatter(k_const, data_ps_79[0], color='gray', s=300)
eb[-1][0].set_linestyle('dotted')
plt.xlabel('$k\,[1/\mathrm{Mpc}]$', fontsize=30)
plt.ylabel('$\Delta_{21}^2\,[\mathrm{mK}^2]\,\,(z=$' + str(z) + ')', fontsize=30)

plt.ylim(20, 20000)
plt.xlim(np.min(k_range), np.max(k_range))
plt.xticks(ticks=[0.134, 1], fontsize=32)
plt.yticks(fontsize=32)
plt.legend(frameon=False, loc='upper left', fontsize=26)
plt.tight_layout()
plt.savefig('/Users/hovavlazare/GITs/21CMPSemu/images/PS_compare_Lx39_UVLF>-20_v2.png')

# run([-1.15,-2.78, 0.29, 0.03, -0.86, -2.31, 0.01,  9.41, 40.50, 0.7])
