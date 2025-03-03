import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from scipy.special import erf
import corner
import emcee
import h5py
from scipy.integrate import simpson
from tqdm import tqdm
from Zhai17HOD import Zhai17HOD
from Zheng07HOD import Zheng07HOD
from ELGHOD import ELGHOD
from Yuan22LRG_HOD import Yuan22LRG_HOD
from profile_2pt import Profile2ptHOD

def load_dict(filename):
    return np.load(filename, allow_pickle=True).item()

# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams.update({'figure.autolayout': True})


# plt.rc('text', usetex=True)
# plt.rc('font', family='serif',size=15)
# plt.rc('axes', linewidth=1.5) # change back to 1.5
# plt.rc('axes', labelsize=20) # change back to 10
# plt.rc('xtick', labelsize=22, direction='in')
# plt.rc('ytick', labelsize=22, direction='in')
# plt.rc('legend', fontsize=15) # change back to 7

# # setting xtick parameters:

# plt.rc('xtick.major',size=10,pad=4)
# plt.rc('xtick.minor',size=5,pad=4)

# plt.rc('ytick.major',size=10)
# plt.rc('ytick.minor',size=5)

# Planck 2018 cosmology
little_h = 0.67
cosmo = ccl.Cosmology(Omega_c = 0.1198/little_h**2, Omega_b = 0.02233/little_h**2, h = little_h, sigma8 =  0.8101, n_s = 0.9652, matter_power_spectrum='linear')

# Tinker et al. (2010; 1001.3162) virial overdensity definition
tinker_Dv = 200.

# Wavenumbers and scale factors
k_arr = np.geomspace(1E-4,1E3,512)
a_arr = np.linspace(0.1,1,64)
rp_array = np.logspace(-2, 1.99, 128)

# We will use a mass definition with Delta = 200 times the matter density
hmd_200m = ccl.halos.MassDef200m

# The Duffy 2008 concentration-mass relation
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)

# The Tinker 2008 mass function
nM = ccl.halos.MassFuncTinker10(mass_def=hmd_200m)

# The Tinker 2010 halo bias
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)

# The NFW profile to characterize the matter density around halos
pM = ccl.halos.HaloProfileNFW(mass_def=hmd_200m, concentration=cM, fourier_analytic=True)

hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m)

HOD2pt = Profile2ptHOD()
General2pt= ccl.halos.Profile2pt()

def zheng_dict(params):
    return {'M_min': params[0], 'sig_logm': params[1], 'M_0': params[2], 'M_1': params[3], 'alpha': params[4]}

def yuanlrg_dict(params):
    return {'M_cut': params[0], 'sig_logm': params[1], 'kappa': params[2], 'M_1': params[3], 'alpha': params[4]}

def zhai_dict(params):
    return {'M_min': params[0], 'sig_logm': params[1], 'M_sat': params[2], 'M_cut': params[3], 'alpha': params[4]}

def elg_dict(params):
    return {'pmax': params[0], 'Q': params[1], 'M_cut': params[2], 'sig_logm': params[3], 'gamma': params[4], 'M_1': params[5], 'alpha': params[6], 'k': params[7]}
#pmax = 0.075, Q = 95, log Mcut = 11.9, σ = 0.5, γ = 5, log M1 = 14.2, α = 0.65, and κ = 1.35. Yuan 2022



def CCL_hod_model(params, hod_str = 'nicola20', pass_hod_base_bool = False, pass_hod_base = None, ns_independent = False):
    if hod_str != 'nicola20' and hod_str != 'zheng07' and hod_str != 'zhai2017' and hod_str != 'elg' and hod_str != 'yuan22':
        print("For HODs I expect zheng07, zhai17, elg, yuan22, or nicola20")
        

    if pass_hod_base_bool == False:

        if hod_str == 'zhai2017':
            params_dict = zhai_dict(params)
            pg = Zhai17HOD(mass_def=hmd_200m, concentration=cM)
            pg.update_parameters(   log10Mmin_0=params_dict['M_min'],
                                    siglog10M_0=params_dict['sig_logm'],
                                    log10Msat_0=params_dict['M_sat'],
                                    log10Mcut_0=params_dict['M_cut'],
                                    alpha_0=params_dict['alpha'], ns_independent = ns_independent)
        if hod_str == 'nicola20':
            params_dict = zheng_dict(params)
            pg = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM)
            pg.update_parameters(   log10Mmin_0=params_dict['M_min'],
                                    siglnM_0=params_dict['sig_logm'],
                                    log10M0_0=params_dict['M_0'],
                                    log10M1_0=params_dict['M_1'],
                                    alpha_0=params_dict['alpha'], ns_independent = ns_independent)
        
        if hod_str == 'zheng07':
            params_dict = zheng_dict(params)
            pg = Zheng07HOD(mass_def=hmd_200m, concentration=cM)
            pg.update_parameters(   log10Mmin_0=params_dict['M_min'],
                                    siglog10M_0=params_dict['sig_logm'],
                                    log10M0_0=params_dict['M_0'],
                                    log10M1_0=params_dict['M_1'],
                                    alpha_0=params_dict['alpha'], ns_independent = ns_independent)

        if hod_str == 'yuan22':
            params_dict = yuanlrg_dict(params)
            pg = Yuan22LRG_HOD(mass_def=hmd_200m, concentration=cM)
            pg.update_parameters(   log10Mcut_0=params_dict['M_cut'],
                                    siglog10M_0=params_dict['sig_logm'],
                                    kappa_0=params_dict['kappa'],
                                    log10M1_0=params_dict['M_1'],
                                    alpha_0=params_dict['alpha'], ns_independent = ns_independent)
        if hod_str == 'elg':
            params_dict = elg_dict(params)
            pg = ELGHOD(mass_def=hmd_200m, concentration=cM)
            pg.update_parameters(   Q_0=params_dict['Q'],
                                    pmax_0=params_dict['pmax'],
                                    gamma_0=params_dict['gamma'],
                                    k_0=params_dict['k'],
                                    log10Mcut_0=params_dict['M_cut'],
                                    log10M1_0=params_dict['M_1'],
                                    siglog10M_0=params_dict['sig_logm'],
                                    alpha_0=params_dict['alpha'], ns_independent = ns_independent)

    if pass_hod_base_bool == True: # Speedup by passing a previously calculated HOD profile and just updating the parameters, currently only set up for nicola20 HOD
        if hod_str == 'nicola20':
            params_dict = zheng_dict(params)
            pass_hod_base.update_parameters(log10Mmin_0=params_dict['M_min'],
                                        siglnM_0=params_dict['sig_logm'],
                                        log10M0_0=params_dict['M_0'],
                                        log10M1_0=params_dict['M_1'],
                                        alpha_0=params_dict['alpha'], ns_independent = ns_independent)
        if hod_str == 'zheng07':
            params_dict = zheng_dict(params)
            pass_hod_base.update_parameters(log10Mmin_0=params_dict['M_min'],
                                        siglog10M_0=params_dict['sig_logm'],
                                        log10M0_0=params_dict['M_0'],
                                        log10M1_0=params_dict['M_1'],
                                        alpha_0=params_dict['alpha'], ns_independent = ns_independent)
        if hod_str == 'zhai2017':
            params_dict = zhai_dict(params)
            pass_hod_base.update_parameters(mass_def=hmd_200m, concentration=cM, 
                                    log10Mmin_0=params_dict['M_min'],
                                    siglog10M_0=params_dict['sig_logm'],
                                    log10Msat_0=params_dict['M_sat'],
                                    log10Mcut_0=params_dict['M_cut'],
                                    alpha_0=params_dict['alpha'], ns_independent = ns_independent)
            
        if hod_str == 'yuan22':
            params_dict = yuanlrg_dict(params)
            pass_hod_base.update_parameters(   log10Mcut_0=params_dict['M_cut'],
                                    siglog10M_0=params_dict['sig_logm'],
                                    kappa_0=params_dict['kappa'],
                                    log10M1_0=params_dict['M_1'],
                                    alpha_0=params_dict['alpha'], ns_independent = ns_independent)
            
        if hod_str == 'elg':
            params_dict = elg_dict(params)
            pass_hod_base.update_parameters(   Q_0=params_dict['Q'],
                                    pmax_0=params_dict['pmax'],
                                    gamma_0=params_dict['gamma'],
                                    k_0=params_dict['k'],
                                    log10Mcut_0=params_dict['M_cut'],
                                    log10M1_0=params_dict['M_1'],
                                    siglog10M_0=params_dict['sig_logm'],
                                    alpha_0=params_dict['alpha'], ns_independent = ns_independent)
        
        pg = pass_hod_base
        

    return pg


# Define the HOD model
def PofK(params, hod_str = 'nicola20', pass_hod_base_bool = False, pass_hod_base = None, ns_independent = False,
         params2 = None, hod_str2 = 'nicola20', pass_hod_base_bool2 = False, pass_hod_base2 = None, ns_independent2 = False):

    pg = CCL_hod_model(params, hod_str = hod_str, pass_hod_base_bool = pass_hod_base_bool, pass_hod_base = pass_hod_base, ns_independent = ns_independent)
    
    if params2 is not None:
        pg2 = CCL_hod_model(params2, hod_str = hod_str2, pass_hod_base_bool = pass_hod_base_bool2, pass_hod_base = pass_hod_base2, ns_independent = ns_independent2)
    
    if params2 is None:
        pg2 = None

    pk_gg_Pk2D = ccl.halos.halomod_Pk2D(cosmo, hmc, pg, prof_2pt=HOD2pt, prof2 = pg2,
                                lk_arr=np.log(k_arr), a_arr=a_arr)

    return pk_gg_Pk2D



def wprp_model_Pk_CCL(cosmo, rps, a, Pk2D):
    
    xi = ccl.correlations.correlation_3d(cosmo=cosmo, r=rps, a=a, p_of_k_a=Pk2D)

    model = np.zeros(len(rps))

    def wprp_integrand_Pk(r_, rp_):
        return 2*(np.interp(r_, rps,xi) * r_)/(np.sqrt(r_**2 - rp_**2))

    for i in np.arange(len(rps)):

        rpmin = rps[i]

        rs = np.logspace(np.log10(rpmin),2, 10000)[1:] #integrate from rp to ~infinity, 
                                        #have checked that this is large enough to converge

        def wprp_integrand_simpson(r):
            return wprp_integrand_Pk(r, rpmin)

        model[i] = simpson(wprp_integrand_simpson(rs), x = rs)

    return model

def wprp_model_HOD_CCL(cosmo, params, rps, a, hod_str = 'nicola20', pass_hod_base_bool = False, pass_hod_base = None,ns_independent = False,
                       params2 = None, hod_str2 = 'nicola20', pass_hod_base_bool2 = False, pass_hod_base2 = None, ns_independent2 = False):

    pk_gg_Pk2D = PofK(params, hod_str = hod_str, pass_hod_base_bool = pass_hod_base_bool, pass_hod_base = pass_hod_base, ns_independent = ns_independent, 
                      params2 = params2, hod_str2 = hod_str2, pass_hod_base_bool2 = pass_hod_base_bool2, pass_hod_base2 = pass_hod_base2, ns_independent2 = ns_independent2)

    model = wprp_model_Pk_CCL(cosmo, rps, a, pk_gg_Pk2D)

    return model


#MCMC functions
def log_likelihood(sample, inputrs, inputwps, inputerrs, a, hod_str = 'nicola20', pass_hod_base_bool = False, pass_hod_base = None, ns_independent = False,
                   sample2=None, hod_str2 = 'nicola20', pass_hod_base_bool2 = False, pass_hod_base2 = None, ns_independent2 = False):
    wp_rps=wprp_model_HOD_CCL(cosmo, sample, rp_array, a, hod_str = hod_str, pass_hod_base_bool = pass_hod_base_bool, pass_hod_base = pass_hod_base, ns_independent = ns_independent, 
                              params2 = sample2, hod_str2 = hod_str2, pass_hod_base_bool2 = pass_hod_base_bool2, pass_hod_base2 = pass_hod_base2, ns_independent2 = ns_independent2)

    wp_rps = np.interp(inputrs, rp_array, wp_rps) #Interpolate the theoretical PCF to the same distances as our observed PCF
    if np.ndim(inputerrs) == 1:
        return -0.5 * np.sum(((inputwps-wp_rps)/inputerrs)**2) #The function's output is the square of the "distance" between our observed and theoretical PCF
    if np.ndim(inputerrs) == 2 and inputerrs.shape[0] == inputerrs.shape[1]:
        #inverse = np.linalg.inv(inputerrs) # assume inverse covariance matrix is being passed 
        r = inputwps-wp_rps
        return -0.5 * np.matmul(np.matmul(r.T, inputerrs), r)
    else: 
        print("Error in log_likelihood: inputerrs must be 1D or 2D")


#MCMC functions
def log_likelihood_LRG_fixed(LRG_solution, sample, inputrs, inputwps, inputerrs, a, hod_str = 'nicola20', pass_hod_base_bool = False, pass_hod_base = None, ns_independent = False,
                    hod_str2 = 'nicola20', pass_hod_base_bool2 = False, pass_hod_base2 = None, ns_independent2 = False):
    wp_rps=wprp_model_HOD_CCL(cosmo, LRG_solution, rp_array, a, hod_str = hod_str, pass_hod_base_bool = pass_hod_base_bool, pass_hod_base = pass_hod_base, ns_independent = ns_independent, 
                              params2 = sample, hod_str2 = hod_str2, pass_hod_base_bool2 = pass_hod_base_bool2, pass_hod_base2 = pass_hod_base2, ns_independent2 = ns_independent2)
    #TODO clean up this function to reduce the overhead of building the HOD code every time 

    wp_rps = np.interp(inputrs, rp_array, wp_rps) #Interpolate the theoretical PCF to the same distances as our observed PCF
    if np.ndim(inputerrs) == 1:
        return -0.5 * np.sum(((inputwps-wp_rps)/inputerrs)**2) #The function's output is the square of the "distance" between our observed and theoretical PCF
    if np.ndim(inputerrs) == 2 and inputerrs.shape[0] == inputerrs.shape[1]:
        #inverse = np.linalg.inv(inputerrs) # assume inverse covariance matrix is being passed 
        r = inputwps-wp_rps
        return -0.5 * np.matmul(np.matmul(r.T, inputerrs), r)
    else: 
        print("Error in log_likelihood: inputerrs must be 1D or 2D")



def log_prior(sample, hod_str = 'nicola20'):
    if hod_str=="zhai17":
        M_min, M_sat, alpha, M_cut, sig_logm = sample 
        if 9.0 < M_min < 16.0 and 8.0 < M_sat < 16.0 and 0.0 < alpha < 2.0 and 0 < sig_logm < 2 and 8.0 < M_cut < 16.0:
            return 0.0
        else:
            return -np.inf
        
    if hod_str=="nicola20" or hod_str=="zheng07":
        M_min, sig_logm, M_0, M_1, alpha = sample 
        if 9.0 < M_min < 16.0 and 9.0 < M_0 < 16.0 and 0.0 < alpha < 2.0 and 0 < sig_logm < 2 and 10.0 < M_1 < 16.0 and (((M_0/M_min) -1) >= 0):
            return 0.0
        else:
            return -np.inf

        
    else:
        print("for HODs I expect zheng07, zhai17 or nicola20")



def ACF_model_Pk_CCL(cosmo, thetas, Pk2D, zs, dNdz, zs2= None, dNdz2 = None):
    if zs2 is None:
        zs2 = zs
    if dNdz2 is None:
        dNdz2 = dNdz
    
    b = np.ones_like(zs)
    ell = np.logspace(np.log10(0.01), np.log10(200000), 1000)

    clu1 = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(zs,dNdz), bias=(zs,b))
    clu2 = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(zs2,dNdz2), bias=(zs,b))
    cls_clu = ccl.angular_cl(cosmo, clu1, clu2, ell,  p_of_k_a=Pk2D) 
    xi_clu1 = ccl.correlation(cosmo = cosmo, ell = ell, C_ell = cls_clu, theta = thetas, type='NN', method='FFTLog') 
    
    return xi_clu1



def ACF_model_HOD_CCL(cosmo, params, thetas, zs, dNdz,  hod_str = 'nicola20', pass_hod_base_bool = False, pass_hod_base = None, ns_independent = False,
                      zs2= None, dNdz2 = None,
                      params2 = None, hod_str2 = 'nicola20', pass_hod_base_bool2 = False, pass_hod_base2 = None, ns_independent2 = False):


    pk_gg_Pk2D = PofK(params, hod_str = hod_str, pass_hod_base_bool = pass_hod_base_bool, pass_hod_base = pass_hod_base, ns_independent = ns_independent,
                      params2 = params2, hod_str2 = hod_str2, pass_hod_base_bool2 = pass_hod_base_bool2, pass_hod_base2 = pass_hod_base2, ns_independent2 = ns_independent2)

    model = ACF_model_Pk_CCL(cosmo, thetas, pk_gg_Pk2D, zs, dNdz, zs2= zs2, dNdz2 = dNdz2)

    return model

def log_likelihood_ACF(sample, thetas, inputACF, inputerrs, zs, dNdz, hod_str = 'nicola20', pass_hod_base_bool = False, pass_hod_base = None):
    
    xi_clu1=ACF_model_HOD_CCL(cosmo, sample, thetas, zs, dNdz, hod_str = hod_str, pass_hod_base_bool = pass_hod_base_bool, pass_hod_base = pass_hod_base)

    if np.ndim(inputerrs) == 1:
        return -0.5 * np.sum(((inputACF-xi_clu1)/inputerrs)**2) 
    if np.ndim(inputerrs) == 2 and inputerrs.shape[0] == inputerrs.shape[1]:
        inverse = np.linalg.inv(inputerrs)
        r = inputACF-xi_clu1
        return -0.5 * np.matmul(np.matmul(r.T, inverse), r)
    else: 
        print("Error in log_likelihood_ACF: inputerrs must be 1D or 2D")


##### Analysis functions 

#Bias function from CCL HOD profile
def bias_from_ccl_HaloProfileHOD(HODProfile_obj, cosmo, a):
    hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m)
    ng = HODProfile_obj.get_normalization(cosmo, a, hmc= hmc)

    def integ(M):
        Nc = HODProfile_obj._Nc(M, a)
        Ns = HODProfile_obj._Ns(M, a)

        return bM(cosmo, M, a) * Nc * (1+Ns)
    
    integral = hmc.integrate_over_massfunc(integ, cosmo, a)
        
    return integral/ng

def bias_from_ccl_HaloProfile_general(HODProfile_obj, cosmo, a):
    hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m)
    ng = HODProfile_obj.get_normalization(cosmo, a, hmc= hmc)

    def integ(M):
        Nc = HODProfile_obj._Nc(M, a)
        Ns = HODProfile_obj._Ns(M, a)

        return bM(cosmo, M, a) * (Nc+Ns)
    
    integral = hmc.integrate_over_massfunc(integ, cosmo, a)
        
    return integral/ng