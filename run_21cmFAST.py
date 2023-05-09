import numpy as np
import py21cmfast as p21c
import time

user_params = {"DIM": 512, "HII_DIM": 128, "BOX_LEN": 256, "N_THREADS":5, 'USE_RELATIVE_VELOCITIES': False,
               "POWER_SPECTRUM": 0}
flag_options = {"USE_MASS_DEPENDENT_ZETA": True, "USE_CMB_HEATING": False, "USE_LYA_HEATING": False,
                'USE_TS_FLUCT': True,
                'USE_MINI_HALOS': True, "INHOMO_RECO": True}

def run(theta):
    F_STAR10, F_STAR7_MIMI, ALPHA_STAR, ALPHA_STAR_MINI, F_ESC10, F_ESC7_MINI, ALPHA_ESC, M_TURN, L_X, NU_X_THRESH, = theta
    astro_params = {'F_STAR10': F_STAR10, 'F_ESC10': F_ESC10, 'L_X': L_X, 'M_TURN': M_TURN,
                'NU_X_THRESH': NU_X_THRESH * 1000, 'ALPHA_STAR': ALPHA_STAR, 'ALPHA_ESC': ALPHA_ESC,
                 'F_STAR7_MIMI': F_STAR7_MIMI, 'ALPHA_STAR_MINI': ALPHA_STAR_MINI, 'F_ESC7_MINI': F_ESC7_MINI
                }
    start = time.time()
    lightcone = p21c.run_lightcone(redshift=5.0,
                                   max_redshift=20.0,
                                   random_seed = 1,
                                   write=False,
                                   user_params=user_params,
                                   lightcone_quantities={'brightness_temp', 'density', 'xH_box', 'Ts_box', 'dNrec_box', 'Gamma12_box', 'J_21_LW_box'},
                                   global_quantities={'brightness_temp', 'density', 'xH_box', 'Ts_box', 'dNrec_box', 'Gamma12_box', 'J_21_LW_box'},
                                   flag_options=flag_options,
                                   astro_params=astro_params,
                                   direc='_cache')
    end = time.time()
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(end - start))
    print(f'Total running time: {elapsed_time}')

    z_global = lightcone.node_redshifts
    lightcone_redshifts = lightcone.lightcone_redshifts
    xH_box = lightcone.xH_box
    global_xH = lightcone.global_xH
    density_box = lightcone.density
    x=1

run([-1.24,-2.5, 0.5, 0, -1.35, -1.35, -0.3,  8.59, 40.64, 0.72])
