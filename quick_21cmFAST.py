import sys
import py21cmfast as p21c
import time
import pickle

sys.path.insert(1, '/gpfs0/elyk/users/hovavl/21CMPSemu')
from tau import calc_tau
from power_spectrum import powerspectra


# params = sys.argv[1:]
# m_FDM = float(params[0])
# f_FDM = float(params[1])

user_params = {"DIM": 512, "HII_DIM": 128, "BOX_LEN": 256, "N_THREADS": 1, 'USE_RELATIVE_VELOCITIES': True,
               'DO_VCB_FIT': True, 'RUN_CLASS': True, 
               "POWER_SPECTRUM": 5}
flag_options = {"USE_MASS_DEPENDENT_ZETA": True, "USE_CMB_HEATING": False, "USE_LYA_HEATING": True,
                'USE_TS_FLUCT': True,
                'USE_MINI_HALOS':False, "INHOMO_RECO": True}

cosmo_params = {"hlittle": 0.6736, "OMb": 0.0493, "OMm": 0.3153,
                "A_s": 2.1e-9, "POWER_INDEX": 0.9665}
                
#cosmo_params['m_FDM'] = m_FDM
#cosmo_params['f_FDM'] = f_FDM

# print(cosmo_params)


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
                                   random_seed=1,
                                   write=False,
                                   user_params=user_params,
                                   lightcone_quantities={'brightness_temp', 'density', 'xH_box', 'Ts_box', 'dNrec_box','Tk_box',
                                                         'Gamma12_box', 'J_21_LW_box'},
                                   global_quantities={'brightness_temp', 'density', 'xH_box', 'Ts_box', 'dNrec_box','Tk_box',
                                                      'Gamma12_box', 'J_21_LW_box'},
                                   flag_options=flag_options,
                                   astro_params=astro_params,
                                   cosmo_params=cosmo_params,
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
              'T_b':T_b,
              'z_global': z_global,
              'T_k': lightcone.global_quantities['Tk_box']
              }

    nchunks = 64

    data_PS, z_PS = powerspectra(lightcone_box=lightcone.brightness_temp,
                                 BOX_LEN=lightcone.user_params.BOX_LEN,
                                 HII_DIM=lightcone.user_params.HII_DIM,
                                 lightcone_redshifts=lightcone.lightcone_redshifts,
                                 nchunks=nchunks)

    output['data_PS'] = data_PS
    output['z_PS'] = z_PS
    x = 1
    # f_FDM = 10**(-f_FDM)
    with open(f'/gpfs0/elyk/users/hovavl/jobs/FDM_jobs/psResults/21cmFAST_outputs_without_MCGs_example.pk',
              'wb') as f:
        pickle.dump(output, f)

run([-1.25,-2.5, 0.5, 0.0, -1.35, -2.32, -0.3,  8.59, 40.5, 0.5])


