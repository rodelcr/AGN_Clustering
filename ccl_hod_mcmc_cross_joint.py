from ccl_hod_modeling import *
import sys 
import corner
import emcee
import h5py
from multiprocessing import Process, freeze_support, set_start_method
import multiprocessing
import os

os.environ["OMP_NUM_THREADS"] = "1"


def file_prep(path):
    data = load_dict(path)
    x = data['rp']/little_h
    y = data['wp']/little_h
    err = data['covmat']/(little_h**2)
    err1d = data['error']
    idxs = np.where(np.isnan(err1d) == False)[0]
    x = x[idxs]
    y = y[idxs]
    err = err[idxs[0]:, idxs[0]:]
    inverse_covmat= np.linalg.inv(err)
    return x, y, inverse_covmat


steps = int(sys.argv[1])
PrepvRun = str(sys.argv[2])
redshift = float(sys.argv[3])
data_path1 = str(sys.argv[4])
data_path2 = str(sys.argv[5])
data_path3 = str(sys.argv[6])

LRG_solution = np.load(str(sys.argv[7])) #  NEED TO SET UP THE INCLUSION OF THE SOLUTION 
LRG_solution2 = np.load(str(sys.argv[8]))
LRG_solution3 = np.load(str(sys.argv[9]))

# data_path2= str(sys.argv[6]) 
outfile_subject = str(sys.argv[10])# numpy array file with x, y, yerr
#whichHOD=str(sys.argv[5]) # Zhai17 or Zh05 for now
whichHOD="nicola20"
whichHOD2="nicola20"


x,y,inverse_covmat = file_prep(data_path1)
x2,y2,inverse_covmat2 = file_prep(data_path2)
x3,y3,inverse_covmat3 = file_prep(data_path3)


LRG_solution_list = [LRG_solution, LRG_solution2, LRG_solution3]
x_list = [x, x2, x3]
y_list = [y, y2, y3]
inverse_covmat_list = [inverse_covmat, inverse_covmat2, inverse_covmat3]



a = 1/(1.+redshift) # TODO Need to allow two redshifts, and allow multiple redshifts in joint fit

np.random.seed(42)

if whichHOD=="zhai17":
    soln = np.array([13.6, 14.9, 0.43, 11.6 , 0.8])
    pos = soln + 0.01 * np.random.randn(64, 5)

elif whichHOD=="zh05":
    soln = np.array([12.6, 0.9,12.61, 11.5, 0.7])
    pos = soln + 0.01 * np.random.randn(64, 5)


elif whichHOD=="nicola20":
    soln = np.array([12.6, 0.9,12.7, 11.5, 0.7])
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
            

def log_probability_i(sample, inputrs, inputwps, inputerrs, LRG_solution = LRG_solution_list[0]):
    lp = log_prior(sample, hod_str = whichHOD)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_LRG_fixed(LRG_solution, sample, inputrs, inputwps, inputerrs, a, hod_str = whichHOD, pass_hod_base_bool = False, pass_hod_base = initial_model, 
                                         hod_str2=whichHOD2)

def log_probability(sample, inputrs_list, inputwps_list, inputerrs_list):
    logprob=0
    for i in range(len(inputrs_list)):
        logprob+=log_probability_i(sample, inputrs_list[i], inputwps_list[i], inputerrs_list[i], LRG_solution = LRG_solution_list[i])
    return logprob



def main():
    if PrepvRun=='Prep':
        print('Prepping')
        filename = f"../{outfile_subject}_HOD_"+whichHOD+"_MCMC_0226_"+str(steps)+"_steps.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)

    elif PrepvRun=='Run':
        Pool = multiprocessing.get_context("spawn").Pool()

        print('Running!')
        # Set up the backend
        filename = f"../{outfile_subject}_HOD_"+whichHOD+"_MCMC_0226_"+str(steps)+"_steps.h5"
        backend = emcee.backends.HDFBackend(filename)
                
        nsteps = steps #Number of steps we want our walkers to take

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=Pool, args=(x_list, y_list, inverse_covmat_list), backend=backend)

        sampler.run_mcmc(pos, nsteps, progress=True, store=True)

    else:
        print('PrepvRun should be Prep or Run')

if __name__ == "__main__":
    freeze_support()
    set_start_method('spawn')
    main()