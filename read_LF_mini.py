import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import py21cmfast as p21c

user_params = {"DIM": 512, "HII_DIM": 128, "BOX_LEN": 256, "N_THREADS": 1, 'USE_RELATIVE_VELOCITIES': False,
               "POWER_SPECTRUM": 0}
flag_options = {"USE_MASS_DEPENDENT_ZETA": True, "USE_CMB_HEATING": False, "USE_LYA_HEATING": False,
                'USE_TS_FLUCT': True,
                'USE_MINI_HALOS': False, "INHOMO_RECO": True}
redshifts = [6, 7, 8, 10]
dir_path = '/gpfs0/elyk/users/hovavl/21CMPSemu/LF_data'


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

    lnlike = np.sum(-(1 / 2) * (
            ((lf_data - y_model) / err) ** 2 + np.log(2 * np.pi * err ** 2)))
    return lnlike


def predict_luminosity(theta):
    F_STAR10, F_STAR7_MINI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, L_X, NU_X_THRESH, = theta
    astro_params = {'F_STAR10': F_STAR10, 'F_ESC10': F_ESC10, 'L_X': L_X, 'M_TURN': 8.5,
                    'NU_X_THRESH': NU_X_THRESH * 1000, 'ALPHA_STAR': ALPHA_STAR, 'ALPHA_ESC': ALPHA_ESC
                    # 'F_STAR7_MIMI': [F_STAR7_MIMI], 'ALPHA_STAR_MINI': [ALPHA_STAR_MINI], 'F_ESC7_MINI': [F_ESC7_MINI]
                    }

    Lf = p21c.compute_luminosity_function(
        redshifts=redshifts,
        user_params=user_params,
        astro_params=astro_params,
        flag_options=flag_options,
        nbins=100,
        component=0
    )
    return Lf
