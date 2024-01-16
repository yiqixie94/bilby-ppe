import numpy as np
from lal import MSUN_SI, G_SI, C_SI
from bilby.gw.conversion import (
    component_masses_to_total_mass, 
    component_masses_to_chirp_mass,
)
from bilby.gw.source import (
    lal_binary_black_hole,
    lal_binary_neutron_star,
)

MSUN_S  = MSUN_SI * G_SI / C_SI ** 3
default_ppe_inspiral_cutoff = 1.8e-2 # from IMRPhenomD, see arxiv:1508.07253

def get_phi_ppe(f, m1, m2, b, beta, fcut_geom=default_ppe_inspiral_cutoff):
    """
    ppE phase modification.
    The post-inspiral part linearly extrapolates the inspiral part.
    `fcut_geom` defines the starting point of extrapolation.
    """
    mtot = component_masses_to_total_mass(m1, m2)
    mc = component_masses_to_chirp_mass(m1, m2)
    xcut = np.pi * fcut_geom * mc / mtot
    phicut = beta * xcut ** (b / 3.)
    dphicut = phicut / xcut * (b / 3.)
    x = np.pi * np.asarray(f) * mc * MSUN_S
    mask_low = x <= 0
    mask_high = x > xcut
    mask_mid = ~(mask_low|mask_high)
    phi = np.zeros_like(x)
    phi[mask_mid] = beta * x[mask_mid] ** (b / 3.)
    phi[mask_high] = phicut + dphicut * (x[mask_high] - xcut)
    return phi

def get_source_model_ppe(gr_source_model):
    """
    Wrapper to get ppE modified bilby source model.
    When creating the bilby `WaveformGenerator`, 
    add `ppe_inspiral_cutoff` in the `waveform_arguments`,
    which will be passed to `get_phi_ppe` as `fcut_geom`.
    """
    def model(frequency_array, mass_1, mass_2, luminosity_distance, 
              a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, 
              b, beta, **kwargs):
        fref = kwargs.get('reference_frequency')
        fcut_geom = kwargs.get('ppe_inspiral_cutoff', default_ppe_inspiral_cutoff)
        phi_ppe = get_phi_ppe(
            np.append(frequency_array, fref), 
            mass_1, mass_2, b, beta, fcut_geom=fcut_geom)
        phi_ppe = phi_ppe[:-1] - phi_ppe[-1] # forces zero modification at the reference frequency
        polarizations = gr_source_model(
            frequency_array, mass_1, mass_2, luminosity_distance, 
            a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
        for k in polarizations:
            polarizations[k] *= np.exp(-1j * phi_ppe)
        return polarizations
    return model

lal_binary_black_hole_ppe = get_source_model_ppe(lal_binary_black_hole)
lal_binary_neutron_star_ppe = get_source_model_ppe(lal_binary_neutron_star)