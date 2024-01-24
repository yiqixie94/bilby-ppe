#!/usr/bin/env/ python

import numpy as np 
import bilby
import lal 
# import lalsimulation as lalsim 

MSUN_KM = lal.MSUN_SI * lal.G_SI / lal.C_SI ** 2 / 1e3
MSUN_S  = MSUN_KM / lal.C_SI * 1e3

# ppE corrections to IMRPhenomD phase implementation

# Functions defining constant fitting factors to ensure continuity of the phase 
# see IMRPhenomD paper II (arxiv 1508.07253)

def alpha_0_correction(beta_ppE, mass_1, mass_2, b):
    mtot = mass_1 + mass_2
    mc = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1, mass_2)
    f1 = 0.018/(mtot*MSUN_S)
    return -((b-3)/3.)*beta_ppE*(np.pi*mc*MSUN_S*f1)**(b/3.)

def alpha_1_correction(beta_ppE, mass_1, mass_2, b):
    mtot = mass_1 + mass_2
    mc = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1, mass_2)
    f1 = 0.018/(mtot*MSUN_S)
    return (b/3.)*beta_ppE*((np.pi*mc*MSUN_S)**(b/3.))*f1**((b-3)/3.)

# Compute ppE correction to GW phase 

def get_phi_ppe(frequency_array, mass_1, mass_2, b, beta):
    frequency_array = np.asarray(frequency_array, dtype=np.float64)
    mtot = mass_1 + mass_2
    f1 = 0.018/(mtot*MSUN_S)
    f_rd = 0.071/(mtot*MSUN_S)
    f2 = 0.5*f_rd  
    mask = frequency_array > 0.
    freq_insp = np.logical_and(frequency_array > 0, frequency_array < f1)
    freq_intm = np.logical_and(frequency_array >= f1, frequency_array < f2)
    freq_mr = frequency_array >= f2

    mc = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1, mass_2)
    mtot = mass_1 + mass_2
    v = (np.pi*frequency_array[freq_insp]*mc*MSUN_S)**(1./3.)
    phi = np.zeros_like(frequency_array)
    phi[freq_insp] += beta*v**b

    # corrections to intermediate phase 
    phi[freq_intm] += alpha_0_correction(beta, mass_1, mass_2, b) + alpha_1_correction(beta, mass_1, mass_2, b)*frequency_array[freq_intm]

    # corrections to merger-ringdown phase
    phi[freq_mr] += alpha_0_correction(beta, mass_1, mass_2, b) + alpha_1_correction(beta, mass_1, mass_2, b)*frequency_array[freq_mr]

    return phi

# Define ppE source model 

def source_model(frequency_array, mass_1, mass_2, luminosity_distance,
                a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                b, beta,
                **kwargs):

    freqs = np.append(frequency_array, kwargs['reference_frequency'])
    phi = get_phi_ppe(freqs, mass_1, mass_2, b, beta)
    phi = phi[:-1] - phi[-1]
    polarizations = bilby.gw.source.lal_binary_black_hole(
                    frequency_array, mass_1, mass_2, luminosity_distance,
                    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, 
                    **kwargs)
    for k in polarizations:
        polarizations[k] *= np.exp(-1j * phi)
    return polarizations
