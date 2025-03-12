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
idx_override = 6

def file_prep_ACF(path, idx_override = idx_override):
    data = load_dict(path)
    x = data['theta']
    y = data['omega']
    err = data['covmat']
    err1d = data['error']
    idxs = np.where(np.isnan(err1d) == False)[0]
    x = x[idxs]
    y = y[idxs]
    err1d = err1d[idxs]
    err = err[idxs[0]:, idxs[0]:]

    if idx_override is not None:
        x= x[idx_override:]
        y = y[idx_override:]
        err1d = err1d[idx_override:]
        err = err[idx_override:, idx_override:]


    inverse_covmat= np.linalg.inv(err)
    zgrid = data['zgrid']
    dNdz = data['dNdz']
    return x, y, err1d, zgrid, dNdz


steps = int(sys.argv[1])
PrepvRun = str(sys.argv[2])
data_path1 = str(sys.argv[3])
data_path2 = str(sys.argv[4])
data_path3 = str(sys.argv[5])
whichHOD="zehavi08"

# data_path2= str(sys.argv[6]) 
outfile_subject = str(sys.argv[6])# numpy array file with x, y, yerr
#whichHOD=str(sys.argv[5]) # Zhai17 or Zh05 for now
#whichHOD="nicola20"



x,y,err1d, zgrid, dNdz = file_prep_ACF(data_path1)
x2,y2,err1d2, zgrid2, dNdz2 = file_prep_ACF(data_path2)
x3,y3,err1d3, zgrid3, dNdz3 = file_prep_ACF(data_path3)


x_list = [x, x2, x3]
y_list = [y, y2, y3]
err1d_list = [err1d, err1d2, err1d3]
zgrid_list = [zgrid, zgrid2, zgrid3]
dNdz_list = [dNdz, dNdz2, dNdz3]



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
    soln = np.array([13.2, 14., 1.0])
    pos = soln + 0.01 * np.random.randn(nwalk, 3)

else:
    print("for HODs I expect zhai17, zh05, zehavi08, or nicola20")



nwalkers, ndim = pos.shape #For each point on our parameter space, we set a little walker a-wandering, to find the best fit parameters


def log_probability_i(sample, thetas, inputACF, inputerrs, zgrid = zgrid_list[0], dNdz = dNdz_list[0]):
    lp = log_prior(sample, hod_str = whichHOD)
    if not np.isfinite(lp):
        return -np.inf

    try:
        return lp + log_likelihood_ACF(sample, thetas, inputACF, inputerrs, zgrid, dNdz, hod_str = whichHOD )
    except ccl.errors.CCLError:
        return -np.inf # Ideally just changing the Mmin prior to be limited to 10**15. Msun will reduce the chance of creating an integration error, but just in case
    
def log_probability(sample, inputx_list, inputy_list, inputerrs_list):
    logprob=0
    for i in range(len(inputx_list)):
        logprob+=log_probability_i(sample, inputx_list[i], inputy_list[i], inputerrs_list[i], zgrid = zgrid_list[i], dNdz = dNdz_list[i])
    return logprob




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

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=Pool, args=(x_list, y_list, err1d_list), backend=backend)

        sampler.run_mcmc(pos, nsteps, progress=True, store=True)

    else:
        print('PrepvRun should be Prep or Run')

if __name__ == "__main__":
    freeze_support()
    set_start_method('spawn')
    main()