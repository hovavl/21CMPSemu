# -*- coding: utf-8 -*-

import numpy as np
import pickle
import time
import os
import sys
from scipy import integrate
import matplotlib.pyplot as plt
from astropy import constants as const

#%% Set constants for calculations
G = const.G #m^3 kg^-1 s^-2
c = const.c #m/s
m_p = const.m_p #kg
sigma_T = const.sigma_T
#OMb = 0.0493
#OMm = 0.3153
#OMc = 1 - OMm
m_h = 1.6735575e-27 #kg
m_he = 6.6464731e-27 #kg
Y_P_BBN = 0.243 #from Planck 2018
#H_0 = 2.2e-18 #from Planck 2018

#%% Create function for integral
def f(x, OMm, OMc):
    return (1 + x)**2/np.sqrt(OMc+OMm*(1 + x)**3)
    
#%% Calculate tau for lightcone
def calc_tau(z_index, density_box, xH_box, cosmo_params):
    h = cosmo_params['hlittle']
    H_0 = h * 10**(-17) / 3.09
    OMb = cosmo_params['OMb']
    OMm = cosmo_params['OMm']
    OMc = 1 - OMm
    xHII_box = 1 - xH_box
    multiplied_box = density_box * xHII_box
    average = []
    func = []
    for z,val in enumerate(z_index):
    	#take redshift slice
    	spatial_mat = multiplied_box[:,:,z]
    	#calculate matrix average:
    	average += [np.mean(spatial_mat)]
    	#create integrand function
    	func += [f(val, OMm, OMc)]
    integrand_end = np.multiply(func, average)
    # Calculate first integral - only function for z = 0 to z = 5
    # Define bounds of integral
    a = 0
    b = 5
    # Generate function values
    x_range = np.arange(a,b+0.0001,.0001)
    fx = f(x_range, OMm, OMc)
    # Calculate integral
    I1, err = integrate.quad(f, a, b, args=(OMm, OMc))
    #Calculate second integral - function times the average for z = 5 to z = 35
    I2 = integrate.simpson(integrand_end, z_index)
    integral = I1 + I2
    #calculate tau
    mu = 1 + Y_P_BBN/4*(m_he/m_h - 1)
    tau = 3*H_0*OMb*sigma_T*c*integral/(8*np.pi*G*m_p*mu)
    return tau

