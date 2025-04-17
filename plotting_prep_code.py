
from AGN_Clustering.ccl_hod_modeling import *



plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'figure.autolayout': True})


plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=15)
plt.rc('axes', linewidth=1.5) # change back to 1.5
plt.rc('axes', labelsize=20) # change back to 10
plt.rc('xtick', labelsize=22, direction='in')
plt.rc('ytick', labelsize=22, direction='in')
plt.rc('legend', fontsize=15) # change back to 7

# setting xtick parameters:

plt.rc('xtick.major',size=10,pad=4)
plt.rc('xtick.minor',size=5,pad=4)

plt.rc('ytick.major',size=10)
plt.rc('ytick.minor',size=5)



def log_mass_add_littleh(mass):
    return np.log10((10**mass)*little_h)

def log_mass_remove_littleh(mass):
    return np.log10((10**mass)/little_h)



def read_chains(chain_file):
    reader = emcee.backends.HDFBackend(chain_file, read_only=True)
    samples = reader.get_chain(discard=0, thin=1, flat=True)
    logli = reader.get_log_prob(discard=0, thin=1, flat=True)

    return samples, logli

def plot_params_nicola20(chains, z = 0.8, params2 = None):
    mmin_, sig_logm_, m_0, m_1,alpha_ = (
        (v[1], v[2] - v[1], v[1] - v[0]) for v in zip(*np.percentile((chains), [16, 50, 84], axis=0))
    )
    a = 1/(1.+z)

    params= np.percentile((chains), 50, axis=0)

    rp_array_ = rp_array[rp_array>0.06]

    return rp_array_, wprp_model_HOD_CCL(cosmo, params, rp_array_, a, params2 =params2)


from scipy.ndimage import gaussian_filter1d

def gaussian_smoother(y):
    return gaussian_filter1d(y, sigma=1.5)


def params_from_chains(chains):
    mmin_, sig_logm_, m_0, m_1,alpha_ = (
        (v[1], v[2] - v[1], v[1] - v[0]) for v in zip(*np.percentile((chains), [16, 50, 84], axis=0))
    )
    params= np.percentile((chains), 50, axis=0)
    mmin_, sig_logm_, m_0, m_1,alpha_ = params

    return np.array([mmin_, sig_logm_, m_0, m_1,alpha_])

def chi2(modelx, modely, x, y, err):
    _modely = np.interp(x, modelx, modely)
    return np.sum(((_modely - y)/err)**2)



def plot_params_mixed_nicola20_zehavi08_ACF(chains, thetas, zs, dndz, chains2 = None, zs2 = None, dndz2 = None, ns_independent = False, ns_independent2 = False, pass_params = False, get_1h = True, get_2h = True):
    if pass_params:
        params = chains
    else:
        params= np.percentile((chains), 50, axis=0)

    if chains2 is not None:
        params2 = np.percentile((chains2), 50, axis=0)

    if chains2 is None:
        params2 = None

    return thetas, ACF_model_HOD_CCL(cosmo, params, thetas, zs, dndz, params2 =params2, zs2= zs2, dNdz2 = dndz2, ns_independent=ns_independent, ns_independent2=ns_independent2, hod_str2='zehavi08', hod_str='nicola20', get_1h = get_1h, get_2h = get_2h)



def plot_params_nicola20_ACF(chains, thetas, zs, dndz, params2 = None, zs2 = None, dndz2 = None, ns_independent = False, ns_independent2 = False, get_1h = True, get_2h = True):
    mmin_, sig_logm_, m_0, m_1,alpha_ = (
        (v[1], v[2] - v[1], v[1] - v[0]) for v in zip(*np.percentile((chains), [16, 50, 84], axis=0))
    )

    params= np.percentile((chains), 50, axis=0)

    return thetas, ACF_model_HOD_CCL(cosmo, params, thetas, zs, dndz, params2 =params2, zs2= zs2, dNdz2 = dndz2, ns_independent=ns_independent, ns_independent2=ns_independent2, get_1h = get_1h, get_2h = get_2h)




def plot_params_zehavi08_ACF(chains, thetas, zs, dndz, chains2 = None, zs2 = None, dndz2 = None, ns_independent = False, ns_independent2 = False, pass_params = False, get_1h = True, get_2h = True):

    if pass_params:
        params = chains
    else:
        params= np.percentile((chains), 50, axis=0)

    if chains2 is not None:
        params2 = np.percentile((chains2), 50, axis=0)
    if chains2 is None:
        params2 = None


    return thetas, ACF_model_HOD_CCL(cosmo, params, thetas, zs, dndz, params2 =params2, zs2= zs2, dNdz2 = dndz2, ns_independent=ns_independent, ns_independent2=ns_independent2, hod_str='zehavi08', hod_str2='zehavi08', get_1h = get_1h, get_2h = get_2h)





def params_from_chains(chains):
    mmin_, sig_logm_, m_0, m_1,alpha_ = (
        (v[1], v[2] - v[1], v[1] - v[0]) for v in zip(*np.percentile((chains), [16, 50, 84], axis=0))
    )
    params= np.percentile((chains), 50, axis=0)
    mmin_, sig_logm_, m_0, m_1,alpha_ = params

    return np.array([mmin_, sig_logm_, m_0, m_1,alpha_])

def chi2(modelx, modely, x, y, err):
    _modely = np.interp(x, modelx, modely)
    return np.sum(((_modely - y)/err)**2)




def file_prep_ACF(path):
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
    inverse_covmat= np.linalg.inv(err)
    zgrid = data['zgrid']
    dNdz = data['dNdz']
    return x, y, inverse_covmat, zgrid, dNdz


def file_prep_ACF_plot(path):
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
    zgrid = data['zgrid']
    dNdz = data['dNdz']
    return x, y, err1d, zgrid, dNdz

def corner_plot_zehavi08(samples):

    corner.corner(
        samples,
        labels=[
            r"$M_{\rm min}$",
            r"$M_1$",
            r"$\alpha$",

        ],
        quantiles=(0.16, 0.84),
        show_titles=True,
        title_quantiles=(0.16, 0.5, 0.84),
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4)),
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        color="blue",
        hist_kwargs={"color": "black"},
        smooth=0.5,
        smooth1d=0.5,
        #truths=[13.67, 14.93, 0.43, 11.62, 0.81],#, None, None, None],


        truth_color="darkgray",
    )


def corner_plot_nicola20(samples):

    corner.corner(
        samples,
        labels=[
            r"$M_{\rm min}$",
            r"$\sigma_{\log M}$",
            r"$M_0$",
            r"$M_1$",
            r"$\alpha$",

        ],
        quantiles=(0.16, 0.84),
        show_titles=True,
        title_quantiles=(0.16, 0.5, 0.84),
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4)),
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        color="blue",
        hist_kwargs={"color": "black"},
        smooth=0.5,
        smooth1d=0.5,
        #truths=[13.67, 14.93, 0.43, 11.62, 0.81],#, None, None, None],


        truth_color="darkgray",
    )







def all_fields_plot_zehavi08(dat1, dat2, dat3, model_obj, chains, subset_name = 'default', z = 0.7, satellitef = None, ms = 8, marker = '^', legend = True, x = 0.35, y = 0.35):

    colors= ['blue', 'red', 'orange']
    label = ['GAMA', 'VVDS', 'XMM']
    idx =0


    fig, ax = plt.subplots(1,1, figsize = (8,8))

    dA = ccl.background.angular_diameter_distance(cosmo, 1/(1+z)) #output in Mpc


    def forward_arcmin(x):    
        return np.deg2rad(x/60.)*dA


    def inverse_arcmin(x):
        return np.rad2deg(x/dA)*60.


    ax.errorbar(dat1[0][idx:]*60, dat1[1][idx:], yerr=dat1[2][idx:], fmt=marker, label='GAMA', color=colors[0], markeredgecolor='black', capsize=5, markersize= ms)
    ax.errorbar(dat2[0][idx:]*60, dat2[1][idx:], yerr=dat2[2][idx:], fmt=marker, label='VVDS', color=colors[1], markeredgecolor='black', capsize=5, markersize= ms)
    ax.errorbar(dat3[0][idx:]*60, dat3[1][idx:], yerr=dat3[2][idx:], fmt=marker, label='XMM', color=colors[2], markeredgecolor='black', capsize=5, markersize= ms)

    ax.plot(model_obj[0]*60, model_obj[1], color = 'k', ls = '--', label = 'Joint MCMC Model Fit')

    ax.loglog()
    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(r'$\omega(\theta)$')
    ax.axvline(dat1[0][6]*60*0.87, ls = ':', color = 'k', label = 'Min. Fitting Scale')

    ax.set_xlim([np.float64(0.016584082131810143), np.float64(276.0425777447446)])
    ax.set_ylim([np.float64(4.5343868126402534e-05), np.float64(19.09733875098371)])

    if legend:
        ax.legend()

    # Extracting parameter fits and errors
    qso_params_fit = np.percentile(chains, 50, axis=0)
    qso_params_err_lower = qso_params_fit - np.percentile(chains, 16, axis=0)
    qso_params_err_upper = np.percentile(chains, 84, axis=0) - qso_params_fit

    # Creating the text box content
    if satellitef is None:
        textstr = '\n'.join((
            r'$\mathrm{QSO\ Parameter\ Fits:}$',
            r'$M_{\mathrm{min}}=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[0], qso_params_err_lower[0], qso_params_err_upper[0]),
            r'$M_{1}=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[1], qso_params_err_lower[1], qso_params_err_upper[1]),
            r'$\alpha=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[2], qso_params_err_lower[2], qso_params_err_upper[2]),
        ))

    if satellitef is not None:
        textstr = '\n'.join((
            r'$\mathrm{QSO\ Parameter\ Fits:}$',
            r'$M_{\mathrm{min}}=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[0], qso_params_err_lower[0], qso_params_err_upper[0]),
            r'$M_{1}=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[1], qso_params_err_lower[1], qso_params_err_upper[1]),
            r'$\alpha=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[2], qso_params_err_lower[2], qso_params_err_upper[2]),
            rf'$f_{{sat}} = {np.round(satellitef, 2)}$'
        ))

    # Adding the text box to the plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(x,y, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)


    secax = ax.secondary_xaxis('top', functions=(forward_arcmin, inverse_arcmin))
    #secax.set_xlabel(r'Physical Scale [$h^{-1}$ Mpc]', labelpad=12)
    secax.set_xlabel(r'Physical Scale [Mpc]', labelpad=12)


    plt.suptitle(f'LRG x {subset_name} Cross Correlation, Zehavi 08 HOD Fit', y = 0.98, x=0.54 ) 

    return fig,ax




def all_fields_plot_mixed_nicola_zehavi(dat1, dat2, dat3, model_obj, chains, subset_name = 'default', z = 0.7, satellitef = None, ms = 8, marker = '^', legend = True, x = 0.35, y = 0.35):

    colors= ['blue', 'red', 'orange']
    label = ['GAMA', 'VVDS', 'XMM']
    idx =0


    fig, ax = plt.subplots(1,1, figsize = (8,8))

    dA = ccl.background.angular_diameter_distance(cosmo, 1/(1+z)) #output in Mpc


    def forward_arcmin(x):    
        return np.deg2rad(x/60.)*dA


    def inverse_arcmin(x):
        return np.rad2deg(x/dA)*60.


    ax.errorbar(dat1[0][idx:]*60, dat1[1][idx:], yerr=dat1[2][idx:], fmt=marker, label='GAMA', color=colors[0], markeredgecolor='black', capsize=5, markersize= ms)
    ax.errorbar(dat2[0][idx:]*60, dat2[1][idx:], yerr=dat2[2][idx:], fmt=marker, label='VVDS', color=colors[1], markeredgecolor='black', capsize=5, markersize= ms)
    ax.errorbar(dat3[0][idx:]*60, dat3[1][idx:], yerr=dat3[2][idx:], fmt=marker, label='XMM', color=colors[2], markeredgecolor='black', capsize=5, markersize= ms)

    ax.plot(model_obj[0]*60, model_obj[1], color = 'k', ls = '--', label = 'Joint MCMC Model Fit')

    ax.loglog()
    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(r'$\omega(\theta)$')
    ax.axvline(dat1[0][6]*60*0.87, ls = ':', color = 'k', label = 'Min. Fitting Scale')

    ax.set_xlim([np.float64(0.016584082131810143), np.float64(276.0425777447446)])
    ax.set_ylim([np.float64(4.5343868126402534e-05), np.float64(19.09733875098371)])

    if legend:
        ax.legend()

    # Extracting parameter fits and errors
    qso_params_fit = np.percentile(chains, 50, axis=0)
    qso_params_err_lower = qso_params_fit - np.percentile(chains, 16, axis=0)
    qso_params_err_upper = np.percentile(chains, 84, axis=0) - qso_params_fit

    # Creating the text box content
    if satellitef is None:
        textstr = '\n'.join((
            r'$\mathrm{QSO\ Parameter\ Fits:}$',
            r'$M_{\mathrm{min}}=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[0], qso_params_err_lower[0], qso_params_err_upper[0]),
            r'$\sigma_{\log M}=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[1], qso_params_err_lower[1], qso_params_err_upper[1]),
            r'$M_{0}=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[2], qso_params_err_lower[2], qso_params_err_upper[2]),
            r'$M_{1}=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[3], qso_params_err_lower[3], qso_params_err_upper[3]),
            r'$\alpha=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[4], qso_params_err_lower[4], qso_params_err_upper[4]),
        ))

    if satellitef is not None:
        textstr = '\n'.join((
            r'$\mathrm{QSO\ Parameter\ Fits:}$',
            r'$M_{\mathrm{min}}=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[0], qso_params_err_lower[0], qso_params_err_upper[0]),
            r'$\sigma_{\log M}=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[1], qso_params_err_lower[1], qso_params_err_upper[1]),
            r'$M_{0}=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[2], qso_params_err_lower[2], qso_params_err_upper[2]),
            r'$M_{1}=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[3], qso_params_err_lower[3], qso_params_err_upper[3]),
            r'$\alpha=%.2f_{-%.2f}^{+%.2f}$' % (qso_params_fit[4], qso_params_err_lower[4], qso_params_err_upper[4]),
            rf'$f_{{sat}} = {np.round(satellitef, 2)}$'
        ))

    # Adding the text box to the plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(x,y, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)


    secax = ax.secondary_xaxis('top', functions=(forward_arcmin, inverse_arcmin))
    #secax.set_xlabel(r'Physical Scale [$h^{-1}$ Mpc]', labelpad=12)
    secax.set_xlabel(r'Physical Scale [Mpc]', labelpad=12)


    plt.suptitle(f'LRG x {subset_name} Cross Correlation, Zehavi 08 HOD Fit', y = 0.98, x=0.54 ) 

    return fig,ax

def inverse_variance_weighted_mean(data1, err1, data2, err2, data3, err3):
    """
    Compute the inverse variance weighted mean of three datasets with errors.

    Paramet ers:
    - data1, data2, data3: Arrays of data points for the three datasets.
    - err1, err2, err3: Arrays of errors corresponding to the datasets.

    Returns:
    - weighted_mean: The inverse variance weighted mean of the datasets.
    - weighted_error: The error of the weighted mean.
    """
    # Compute weights as the inverse of variance (1 / error^2)
    weights1 = 1 / (err1**2)
    weights2 = 1 / (err2**2)
    weights3 = 1 / (err3**2)

    # Compute the weighted mean
    weighted_mean = ((data1 * weights1) + (data2 * weights2) + (data3 * weights3)) / (weights1 + weights2 + weights3)

    # Compute the error of the weighted mean
    weighted_error = 1 / np.sqrt(weights1 + weights2 + weights3)

    return weighted_mean, weighted_error

