import warnings

warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import py21cmfast as p21c

user_params = {"DIM": 512, "HII_DIM": 128, "BOX_LEN": 256, "N_THREADS": 5, 'USE_RELATIVE_VELOCITIES': False,
               "POWER_SPECTRUM": 0}
flag_options = {"USE_MASS_DEPENDENT_ZETA": True,  # "USE_CMB_HEATING": False, "USE_LYA_HEATING": False,
                'USE_TS_FLUCT': True,
                'USE_MINI_HALOS': False, "INHOMO_RECO": True}
redshifts = [6, 7, 8, 10]
dir_path = '/gpfs0/elyk/users/hovavl/21CMPSemu/LF_data'
#dir_path = '/Users/hovavlazare/GITs/21CMPSemu/LF_data'


def likelihood(theta):
    lnlike_tot = 0
    Lf = np.array(predict_luminosity(theta))

    for i, z in enumerate(redshifts):
        lnlike_tot += likelihood_for_z(z, Lf[:, i, :], dir_path)
    return lnlike_tot


def likelihood_for_z(z, Lf_for_z, dir_path):
    Muv = Lf_for_z[0]
    lf_sim = Lf_for_z[2]
    logical_lf = np.isnan(lf_sim)
    lf_sim = 10 ** (np.flip(lf_sim[~logical_lf]))
    Muv = np.flip(Muv[~logical_lf])

    path = f'{dir_path}/LF_lfuncs_z{z}.npz'
    err_path = f'{dir_path}/LF_sigmas_z{z}.npz'

    LF_data = np.load(path)
    LF_err = np.load(err_path)
    Muv_data = LF_data['Muv']
    lf_data = LF_data['lfunc'][Muv_data > -20]
    err = LF_err['sigma'][Muv_data > -20]
    Muv_data = Muv_data[Muv_data > -20]

    y_model = np.interp(Muv_data, Muv, lf_sim)
    # print(f'data: {lf_data} err: {err} prediction: {y_model}')

    lnlike = np.sum(-(1 / 2) * (
            ((lf_data - y_model) / err) ** 2 + np.log(2 * np.pi * err ** 2)))
    return lnlike


def predict_luminosity(theta):
    F_STAR10, ALPHA_STAR, F_ESC10, ALPHA_ESC, M_TURN, t_STAR, L_X, NU_X_THRESH, X_RAY_SPEC_INDEX = theta
    astro_params = {'F_STAR10': F_STAR10, 'F_ESC10': F_ESC10, 'L_X': L_X, 'M_TURN': M_TURN,
                    'NU_X_THRESH': NU_X_THRESH * 1000, 'ALPHA_STAR': ALPHA_STAR, 'ALPHA_ESC': ALPHA_ESC,
                    't_STAR': t_STAR, 'X_RAY_SPEC_INDEX': X_RAY_SPEC_INDEX}

    Lf = p21c.compute_luminosity_function(
        redshifts=redshifts,
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
        nbins=100,
        component=0
    )

    return Lf


# theta_test = np.array([-1.24, 0.5, -1.11, 0.02, 8.59, 0.64, 40.64, 0.72, 0.8])
# lnlike = likelihood(theta_test)
# print(lnlike)
# LF_10_data = np.load('/Users/hovavlazare/GITs/21CMPSemu/LF_data/LF_lfuncs_z10.npz')
# lf_err = np.load('/Users/hovavlazare/GITs/21CMPSemu/LF_data/LF_sigmas_z10.npz')
# lf_err_keys = lf_err.files
# LF_keys = LF_10_data.files
# err = lf_err[lf_err_keys[0]]
# Muv = LF_10_data[LF_keys[0]]
# Lfunc = LF_10_data[LF_keys[1]]
# x = 1
