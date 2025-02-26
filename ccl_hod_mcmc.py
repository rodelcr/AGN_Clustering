from ccl_hod_modeling import *
#from multiprocess import Pool, cpu_count
import sys 
import corner
import emcee
import h5py
from multiprocessing import Process, freeze_support, set_start_method
import multiprocessing

steps = int(sys.argv[1])
PrepvRun = str(sys.argv[2])
redshift = float(sys.argv[3])
data_path = str(sys.argv[4])
outfile_subject = str(sys.argv[5]) # numpy array file with x, y, yerr
#whichHOD=str(sys.argv[5]) # Zhai17 or Zh05 for now
whichHOD="nicola20"
filename = f"../{outfile_subject}_HOD_"+whichHOD+"_MCMC_0226_"+str(steps)+"_steps.h5"


data = load_dict(data_path)
x = data['rp']/little_h
y = data['wp']/little_h
err = data['covmat']/(little_h**2)
err = np.linalg.inv(err)

a = 1/(1.+redshift)

np.random.seed(42)

if whichHOD=="zhai17":
    soln = np.array([13.6, 14.9, 0.43, 11.6 , 0.8])
    pos = soln + 0.01 * np.random.randn(64, 5)

elif whichHOD=="zh05":
    soln = np.array([12.6, 0.9,12, 11.5, 0.7])
    pos = soln + 0.01 * np.random.randn(64, 5)


elif whichHOD=="nicola20":
    soln = np.array([12.6, 0.9,12, 11.5, 0.7])
    pos = soln + 0.01 * np.random.randn(64, 5)

else:
    print("for HODs I expect zhai17, zh05, or nicola20")



nwalkers, ndim = pos.shape #For each point on our parameter space, we set a little walker a-wandering, to find the best fit parameters

# Currently only set up for Nicola 2020 HOD, needs to be generalized for other HODs
initial_params = zheng_dict(soln)

initial_model = ccl.halos.HaloProfileHOD(mass_def=hmd_200m, concentration=cM)
initial_model.update_parameters(    log10Mmin_0=initial_params['M_min'],
                                    siglnM_0=initial_params['sig_logm'],
                                    log10M0_0=initial_params['M_0'],
                                    log10M1_0=initial_params['M_1'],
                                    alpha_0=initial_params['alpha'])
            

def log_probability(sample, inputrs, inputwps, inputerrs):
    lp = log_prior(sample, hod_str = whichHOD)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(sample, inputrs, inputwps, inputerrs, a, hod_str = whichHOD, pass_hod_base_bool = False, pass_hod_base = initial_model)




def main():
    if PrepvRun=='Prep':
        print('Prepping')
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
        #from multiprocessing import cpu_count
        #ncpu = cpu_count()
        #print("{0} CPUs".format(ncpu))

    elif PrepvRun=='Run':

        #Pool = multiprocessing.Pool()

        print('Running!')
        # Set up the backend
        backend = emcee.backends.HDFBackend(filename)

        # from schwimmbad import MPIPool

        # with MPIPool() as pool:
        #     if not pool.is_master():
        #         pool.wait()
        #         sys.exit(0)
            
            # soln = (11.6222, 12.851, 1.049) #Our initial guess for parameters. These are just the halomod defaults
            # # pos = soln + 0.01 * np.random.randn(32, 3) #We generate initial guesses in a little Guassian ball in parameter-space
            # pos = soln + 0.01 * np.random.randn(16, 3) #lucia testing
            # print(pos)
            # nwalkers, ndim = pos.shape #For each point on our parameter space, we set a little walker a-wandering, to find the best fit parameters
            # initial = np.random.randn(32, 3)
            # nwalkers, ndim = initial.shape
                
        nsteps = steps #Number of steps we want our walkers to take
        with multiprocessing.get_context("spawn").Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, args=(x, y, err), backend=backend)
            #sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=Pool, args=(x, y, err), backend=backend)


            # sampler = emcee.EnsembleSampler(
            #     nwalkers=64,
            #     ndim=5,
            #     log_prob_fn=log_prob,
            #     kwargs={
            #         "param_names": ["hod_params.M_min", "hod_params.M_sat", "hod_params.alpha", "hod_params.M_cut", "hod_params.sig_logm"],
            #         "data": (zhaiy, mock_ngal),
            #         "model": model,
            #         "derived": [
            #             "satellite_fraction",
            #             "mean_tracer_den",
            #             "bias_effective_tracer",
            #             "corr_auto_tracer",
            #         ],
            #     },
            #     pool=Pool(cpu_count()),
            #     blobs_dtype=blobs_dtype,
            #     backend=backend,
            # )



            # sampler.run_mcmc(initial, nsteps)
            sampler.run_mcmc(pos, nsteps, progress=True, store=True)

    else:
        print('PrepvRun should be Prep or Run')

if __name__ == "__main__":
    freeze_support()
    set_start_method('spawn')
    main()