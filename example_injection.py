import numpy as np 
import bilby
import lal
import matplotlib.pyplot as plt
from bilby.core.prior import Uniform, PowerLaw
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters

MSUN_KM = lal.MSUN_SI * lal.G_SI / lal.C_SI ** 2 / 1e3
MSUN_S  = MSUN_KM / lal.C_SI * 1e3

# ppE corrections to phase implementation
# define fit parameters to ensure phase continuity 

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


# compute ppE correction to GW phase 

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
    v = (np.pi*frequency_array[freq_insp]*mc*MSUN_S) ** (1./3.)
    phi = np.zeros_like(frequency_array)
    phi[freq_insp] += beta * v**b
        
    # corrections to intermediate phase 
    phi[freq_intm] += alpha_0_correction(beta, mass_1, mass_2, b) + alpha_1_correction(beta, mass_1, mass_2, b)*frequency_array[freq_intm]

    # corrections to merger-ringdown phase
    phi[freq_mr] += alpha_0_correction(beta, mass_1, mass_2, b) + alpha_1_correction(beta, mass_1, mass_2, b)*frequency_array[freq_mr]
    
    return phi

# define ppE source model 

def source_model(frequency_array, mass_1, mass_2, luminosity_distance,
                 a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
                 b, beta,
                 **kwargs):
    freqs = np.append(frequency_array, kwargs['reference_frequency'])
    phi = get_phi_ppe(freqs, mass_1, mass_2, b, beta)
    phi = phi[:-1] - phi[-1]
    polarizations = bilby.gw.source.lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance,
        a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
    for k in polarizations:
        polarizations[k] *= np.exp(-1j * phi)
    return polarizations

# injection run 

duration = 8.0
sampling_frequency = 2048.0
minimum_frequency = 20

outdir = 'quick_test'
label = 'quick_test'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

np.random.seed(190521)

injection_parameters = dict(
    mass_1=12.,
    mass_2=8.,
    chi_1=0.,
    chi_2=0.,
    luminosity_distance= 440,
    theta_jn=0.52,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    b=-7,
    beta=0.3)

injection_waveform_arguments = dict(waveform_approximant='IMRPhenomD',
                             reference_frequency = 20.,
                             minimum_frequency = minimum_frequency)

injection_waveform_generator= bilby.gw.WaveformGenerator(
                  duration = duration,
                  sampling_frequency = sampling_frequency,
                  frequency_domain_source_model = source_model, 
                  parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                  waveform_arguments = injection_waveform_arguments)

# interferometers
ifos = bilby.gw.detector.InterferometerList(['L1', 'H1', 'V1'])
ifos.set_strain_data_from_zero_noise(
      sampling_frequency = sampling_frequency,
      duration=duration,start_time = injection_parameters['geocent_time'] - 3)
  
ifos.inject_signal(waveform_generator = injection_waveform_generator,
                      parameters = injection_parameters)

# priors - set narrow range for quick run 
priors = bilby.gw.prior.BBHPriorDict(aligned_spin = True)

priors['chirp_mass'] = Uniform(minimum=5, maximum=15, name='chirp_mass',
                     latex_label='$\\mathcal{M}$', unit=None, boundary=None)

priors['beta'] = Uniform(minimum=-1e-2, maximum=1e-2, name='beta',
                     latex_label='$\\beta_{ppE}$', unit=None, boundary=None)

for key in ['luminosity_distance', 'geocent_time','psi', 'ra', 'dec','theta_jn', 'chi_1', 'chi_2', 'b']:
        priors[key] = injection_parameters[key]

# likelihood
likelihood = bilby.gw.GravitationalWaveTransient(
     interferometers = ifos, 
     waveform_generator = injection_waveform_generator,
     priors= priors)

# convert m1 m2 to chirp mass and mass ratio
injection_parameters['mass_ratio'] = bilby.gw.conversion.component_masses_to_mass_ratio(injection_parameters['mass_1'],injection_parameters['mass_2'])
injection_parameters['chirp_mass'] = bilby.gw.conversion.component_masses_to_chirp_mass(injection_parameters['mass_1'],injection_parameters['mass_2'])
injection_parameters.pop('mass_1')
injection_parameters.pop('mass_2')

# run samples
result = bilby.run_sampler(likelihood=likelihood, priors=priors,
                            sampler='dynesty', npoints = 150, nact = 5, npool=32, 
                            injection_parameters=injection_parameters, 
                            outdir=outdir, label=label, resume=False)


result.plot_corner(truth=dict(chirp_mass= 8.49,                             
                            mass_ratio= 0.66,
                            beta = 0.3))
