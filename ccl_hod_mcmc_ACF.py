from ccl_hod_modeling import *
#from multiprocess import Pool, cpu_count
import sys 
import corner
import emcee
import h5py
from multiprocessing import Process, freeze_support, set_start_method
import multiprocessing
import os

os.environ["OMP_NUM_THREADS"] = "1"


steps = int(sys.argv[1])
PrepvRun = str(sys.argv[2])
data_path = str(sys.argv[3])
outfile_subject = str(sys.argv[4]) # numpy array file with x, y, yerr
#whichHOD=str(sys.argv[5]) # Zhai17 or Zh05 for now
#whichHOD="nicola20"
whichHOD="zehavi08"

data = load_dict(data_path)
x = data['theta']
y = data['omega']
err = data['covmat']

dNdz = data['dNdz']
zgrid = data['zgrid']

err1d = data['error']
idxs = np.where(np.isnan(err1d) == False)[0]

x = x[idxs]
y = y[idxs]
err = err[idxs[0]:, idxs[0]:]
err1d = err1d[idxs] # Trying out the max likelihood with just 1d errors to see if it converges

inverse_covmat= np.linalg.inv(err)

nwalk = 64

np.random.seed(42)

if whichHOD=="zhai17":
    soln = np.array([13.6, 14.9, 0.43, 11.6 , 0.8])
    pos = soln + 0.01 * np.random.randn(nwalk, 5)

elif whichHOD=="zh05":
    soln = np.array([12.6, 0.9,12.6, 11.5, 0.7])
    pos = soln + 0.01 * np.random.randn(nwalk, 5)


elif whichHOD=="nicola20":
    soln = np.array([12.6, 0.9,12.6, 11.5, 0.7])
    pos = soln + 0.01 * np.random.randn(nwalk, 5)

elif whichHOD=="zehavi08":
    soln = np.array([12.6, 12.6, 0.7])
    pos = soln + 0.01 * np.random.randn(nwalk, 3)

else:
    print("for HODs I expect zhai17, zh05, zehavi08, or nicola20")



nwalkers, ndim = pos.shape #For each point on our parameter space, we set a little walker a-wandering, to find the best fit parameters

# Currently only set up for Nicola 2020 HOD, needs to be generalized for other HODs
#initial_params = zheng_dict(soln)

# initial_model = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM)
# initial_model.update_parameters(    log10Mmin_0=initial_params['M_min'],
#                                     siglnM_0=initial_params['sig_logm'],
#                                     log10M0_0=initial_params['M_0'],
#                                     log10M1_0=initial_params['M_1'],
#                                     alpha_0=initial_params['alpha'])



def log_probability(sample, thetas, inputACF, inputerrs):
    lp = log_prior(sample, hod_str = whichHOD)
    if not np.isfinite(lp):
        return -np.inf
    
    try:
        return lp + log_likelihood_ACF(sample, thetas, inputACF, inputerrs, zgrid, dNdz, hod_str = whichHOD )#, pass_hod_base_bool = False, pass_hod_base = initial_model)
    except ccl.errors.CCLError:
        return -np.inf # Ideally just changing the Mmin prior to be limited to 10**15. Msun will reduce the chance of creating an integration error, but just in case
    
def main():
    if PrepvRun=='Prep':
        print('Prepping')
        filename = f"../{outfile_subject}_ACF_HOD_"+whichHOD+"_MCMC_0226_"+str(steps)+"_steps.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)

    elif PrepvRun=='Run':
        Pool = multiprocessing.get_context("spawn").Pool()

        print('Running!')
        # Set up the backend
        filename = f"../{outfile_subject}_ACF_HOD_"+whichHOD+"_MCMC_0226_"+str(steps)+"_steps.h5"
        backend = emcee.backends.HDFBackend(filename)
                
        nsteps = steps #Number of steps we want our walkers to take

        #sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=Pool, args=(x, y, err1d), backend=backend) # Trying out the max likelihood with just 1d errors to see if it converges
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=Pool, args=(x, y, inverse_covmat), backend=backend) # Trying out the max likelihood with just 1d errors to see if it converges

        sampler.run_mcmc(pos, nsteps, progress=True, store=True)

    else:
        print('PrepvRun should be Prep or Run')

if __name__ == "__main__":
    freeze_support()
    set_start_method('spawn')
    main()