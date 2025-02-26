__all__ = ("ELGHOD",)

import numpy as np
from scipy.special import sici, erf

from pyccl.halos import HaloProfileHOD




class ELGHOD(HaloProfileHOD):
    """ A generic halo occupation distribution (HOD)
    profile describing the number density of galaxies
    as a function of halo mass.

    The parametrization for the mean profile is:

    See :class:`~pyccl.halos.profiles_2pt.Profile2ptHOD` for a
    description of the Fourier-space two-point correlator of the
    HOD profile.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        concentration (:class:`~pyccl.halos.halo_model_base.Concentration`):
            concentration-mass relation to use with this profile.
        Q_0 (:obj:`float`): offset parameter for
            :math:`Q`.
        Q_p (:obj:`float`): tilt parameter for
            :math:`Q`.
        pmax_0 (:obj:`float`): offset parameter for
            :math:`p_{\\rm max}`.
        pmax_p (:obj:`float`): tilt parameter for
            :math:`p_{\\rm max}`.
        gamma_0 (:obj:`float`): offset parameter for
            :math:`\\gamma`.
        gamma_p (:obj:`float`): tilt parameter for  
            :math:`\\gamma`.
        k_0 (:obj:`float`): offset parameter for
            :math:`k`.
        k_p (:obj:`float`): tilt parameter for
            :math:`k`.
        siglog10M_0 (:obj:`float`): offset parameter for
            :math:`\\sigma_{{\\rm ln}M}`.
        siglog10M_p (:obj:`float`): tilt parameter for
            :math:`\\sigma_{{\\rm ln}M}`.
        log10Mcut_0 (:obj:`float`): offset parameter for
            :math:`\\log_{10}M_0`.
        log10Mcut_p (:obj:`float`): tilt parameter for
            :math:`\\log_{10}M_0`.
        log10M1_0 (:obj:`float`): offset parameter for
            :math:`\\log_{10}M_1`.
        log10M1_p (:obj:`float`): tilt parameter for
            :math:`\\log_{10}M_1`.
        alpha_0 (:obj:`float`): offset parameter for
            :math:`\\alpha`.
        alpha_p (:obj:`float`): tilt parameter for
            :math:`\\alpha`.
        fc_0 (:obj:`float`): offset parameter for
            :math:`f_c`.
        fc_p (:obj:`float`): tilt parameter for
            :math:`f_c`.
        bg_0 (:obj:`float`): offset parameter for
            :math:`\\beta_g`.
        bg_p (:obj:`float`): tilt parameter for
            :math:`\\beta_g`.
        bmax_0 (:obj:`float`): offset parameter for
            :math:`\\beta_{\\rm max}`.
        bmax_p (:obj:`float`): tilt parameter for
            :math:`\\beta_{\\rm max}`.
        a_pivot (:obj:`float`): pivot scale factor :math:`a_*`.
        ns_independent (:obj:`bool`): drop requirement to only form
            satellites when centrals are present.
        is_number_counts (:obj:`bool`): set to ``True`` if this profile
            is meant to represent galaxy overdensity.
    """
    __repr_attrs__ = __eq_attrs__ = (
        "Q_0", "Q_p", "pmax_0", "pmax_p",
        "gamma_0", "gamma_p", "k_0", "k_p",
        "siglog10M_0", "siglog10M_p", "log10Mcut_0",
        "log10Mcut_p", "log10M1_0", "log10M1_p", "alpha_0", "alpha_p", "fc_0",
        "fc_p", "bg_0", "bg_p", "bmax_0", "bmax_p", "a_pivot",
        "_is_number_counts", "ns_independent", "mass_def", "concentration",
        "precision_fftlog",)

    def __init__(self, *, mass_def, concentration,
                 Q_0=12., Q_p=0., 
                 pmax_0=0.5, pmax_p=0., 
                 gamma_0=0.5, gamma_p=0., 
                 k_0=0.5, k_p=0.,
                 siglog10M_0=0.4,
                 siglog10M_p=0., log10Mcut_0=7., log10Mcut_p=0.,
                 log10M1_0=13.3, log10M1_p=0., alpha_0=1.,
                 alpha_p=0., fc_0=1., fc_p=0.,
                 bg_0=1., bg_p=0., bmax_0=1., bmax_p=0.,
                 a_pivot=1., ns_independent=False,
                 is_number_counts=True):
        self.Q_0 = Q_0
        self.Q_p = Q_p
        self.pmax_0 = pmax_0
        self.pmax_p = pmax_p
        self.gamma_0 = gamma_0
        self.gamma_p = gamma_p
        self.k_0 = k_0
        self.k_p = k_p
        self.log10Mcut_0 = log10Mcut_0
        self.log10Mcut_p = log10Mcut_p
        self.log10M1_0 = log10M1_0
        self.log10M1_p = log10M1_p
        self.siglog10M_0 = siglog10M_0
        self.siglog10M_p = siglog10M_p
        self.alpha_0 = alpha_0
        self.alpha_p = alpha_p
        self.fc_0 = fc_0
        self.fc_p = fc_p
        self.bg_0 = bg_0
        self.bg_p = bg_p
        self.bmax_0 = bmax_0
        self.bmax_p = bmax_p
        self.a_pivot = a_pivot
        self.ns_independent = ns_independent
        super().__init__(mass_def=mass_def, concentration=concentration,
                         is_number_counts=is_number_counts)



    def update_parameters(self, *, Q_0=None, Q_p=None, 
                          pmax_0=None, pmax_p=None, 
                          gamma_0=None, gamma_p=None, 
                          k_0=None, k_p=None,
                          siglog10M_0=None, siglog10M_p=None,
                          log10Mcut_0=None, log10Mcut_p=None,
                          log10M1_0=None, log10M1_p=None,
                          alpha_0=None, alpha_p=None,
                          fc_0=None, fc_p=None,
                          bg_0=None, bg_p=None,
                          bmax_0=None, bmax_p=None,
                          a_pivot=None,
                          ns_independent=None):
        """ Update any of the parameters associated with
        this profile. Any parameter set to ``None`` won't be updated.

        Args:
            Q_0 (:obj:`float`): offset parameter for
                :math:`Q`.
            Q_p (:obj:`float`): tilt parameter for
                :math:`Q`.
            pmax_0 (:obj:`float`): offset parameter for
                :math:`p_{\\rm max}`.
            pmax_p (:obj:`float`): tilt parameter for
                :math:`p_{\\rm max}`.
            gamma_0 (:obj:`float`): offset parameter for
                :math:`\\gamma`.
            gamma_p (:obj:`float`): tilt parameter for  
                :math:`\\gamma`.
            k_0 (:obj:`float`): offset parameter for
                :math:`k`.
            k_p (:obj:`float`): tilt parameter for
                :math:`k`.
            siglog10M_0 (:obj:`float`): offset parameter for
                :math:`\\sigma_{{\\rm log10}M}`.
            siglog10M_p (:obj:`float`): tilt parameter for
                :math:`\\sigma_{{\\rm ln}M}`.
            log10Mcut_0 (:obj:`float`): offset parameter for
                :math:`\\log_{10}M_0`.
            log10Mcut_p (:obj:`float`): tilt parameter for
                :math:`\\log_{10}M_0`.
            log10M1_0 (:obj:`float`): offset parameter for
                :math:`\\log_{10}M_1`.
            log10M1_p (:obj:`float`): tilt parameter for
                :math:`\\log_{10}M_1`.
            alpha_0 (:obj:`float`): offset parameter for
                :math:`\\alpha`.
            alpha_p (:obj:`float`): tilt parameter for
                :math:`\\alpha`.
            fc_0 (:obj:`float`): offset parameter for
                :math:`f_c`.
            fc_p (:obj:`float`): tilt parameter for
                :math:`f_c`.
            bg_0 (:obj:`float`): offset parameter for
                :math:`\\beta_g`.
            bg_p (:obj:`float`): tilt parameter for
                :math:`\\beta_g`.
            bmax_0 (:obj:`float`): offset parameter for
                :math:`\\beta_{\\rm max}`.
            bmax_p (:obj:`float`): tilt parameter for
                :math:`\\beta_{\\rm max}`.
            a_pivot (:obj:`float`): pivot scale factor :math:`a_*`.
            ns_independent (:obj:`bool`): drop requirement to only form
                satellites when centrals are present
        """
        if Q_0 is not None:
            self.Q_0 = Q_0
        if Q_p is not None:
            self.Q_p = Q_p
        if pmax_0 is not None:
            self.pmax_0 = pmax_0
        if pmax_p is not None:
            self.pmax_p = pmax_p
        if gamma_0 is not None:
            self.gamma_0 = gamma_0
        if gamma_p is not None:
            self.gamma_p = gamma_p
        if k_0 is not None:
            self.k_0 = k_0
        if k_p is not None:
            self.k_p = k_p
        if log10Mcut_0 is not None:
            self.log10Mcut_0 = log10Mcut_0
        if log10Mcut_p is not None:
            self.log10Mcut_p = log10Mcut_p
        if log10M1_0 is not None:
            self.log10M1_0 = log10M1_0
        if log10M1_p is not None:
            self.log10M1_p = log10M1_p
        if siglog10M_0 is not None:
            self.siglog10M_0 = siglog10M_0
        if siglog10M_p is not None:
            self.siglog10M_p = siglog10M_p
        if alpha_0 is not None:
            self.alpha_0 = alpha_0
        if alpha_p is not None:
            self.alpha_p = alpha_p
        if fc_0 is not None:
            self.fc_0 = fc_0
        if fc_p is not None:
            self.fc_p = fc_p
        if bg_0 is not None:
            self.bg_0 = bg_0
        if bg_p is not None:
            self.bg_p = bg_p
        if bmax_0 is not None:
            self.bmax_0 = bmax_0
        if bmax_p is not None:
            self.bmax_p = bmax_p
        if a_pivot is not None:
            self.a_pivot = a_pivot
        if ns_independent is not None:
            self.ns_independent = ns_independent



    def _usat_real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        bg = self.bg_0 + self.bg_p * (a - self.a_pivot)
        bmax = self.bmax_0 + self.bmax_p * (a - self.a_pivot)
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M
        c_M *= bmax / bg

        x = r_use[None, :] / (R_s[:, None] * bg)
        prof = 1./(x * (1 + x)**2)
        # Truncate
        prof[r_use[None, :] > R_M[:, None]*bmax] = 0

        norm = 1. / (4 * np.pi * (bg*R_s)**3 * (np.log(1+c_M) - c_M/(1+c_M)))
        prof = prof[:, :] * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _usat_fourier(self, cosmo, k, M, a):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # Comoving virial radius
        bg = self.bg_0 + self.bg_p * (a - self.a_pivot)
        bmax = self.bmax_0 + self.bmax_p * (a - self.a_pivot)
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M
        c_M *= bmax / bg

        x = k_use[None, :] * R_s[:, None] * bg
        Si1, Ci1 = sici((1 + c_M[:, None]) * x)
        Si2, Ci2 = sici(x)

        P1 = 1. / (np.log(1+c_M) - c_M/(1+c_M))
        P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
        P3 = np.sin(c_M[:, None] * x) / ((1 + c_M[:, None]) * x)
        prof = P1[:, None] * (P2 - P3)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        fc = self._fc(a)
        # NFW profile
        ur = self._usat_real(cosmo, r_use, M_use, a)

        if self.ns_independent:
            prof = Nc[:, None] * fc + Ns[:, None] * ur
        else:
            prof = Nc[:, None] * (fc + Ns[:, None] * ur)

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


    def get_normalization(self, cosmo, a, *, hmc):
        """Returns the normalization of this profile, which is the
        mean galaxy number density.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology
                object.
            a (:obj:`float`): scale factor.
            hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo
                model calculator object.

        Returns:
            :obj:`float`: normalization factor of this profile.
        """
        def integ(M):
            Nc = self._Nc(M, a)
            Ns = self._Ns(M, a)
            fc = self._fc(a)
            if self.ns_independent:
                return Nc*fc + Ns
            return Nc*(fc + Ns)
        return hmc.integrate_over_massfunc(integ, cosmo, a)



    def _fourier(self, cosmo, k, M, a):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        fc = self._fc(a)
        # NFW profile
        uk = self._usat_fourier(cosmo, k_use, M_use, a)

        if self.ns_independent:
            prof = Nc[:, None] * fc + Ns[:, None] * uk
        else:
            prof = Nc[:, None] * (fc + Ns[:, None] * uk)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a):
        # Fourier-space variance of the HOD profile
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        Nc = self._Nc(M_use, a)
        Ns = self._Ns(M_use, a)
        fc = self._fc(a)
        # NFW profile
        uk = self._usat_fourier(cosmo, k_use, M_use, a)

        prof = Ns[:, None] * uk
        if self.ns_independent:
            prof = 2 * Nc[:, None] * fc * prof + prof**2
        else:
            prof = Nc[:, None] * (2 * fc * prof + prof**2)

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fc(self, a):
        # Observed fraction of centrals
        return self.fc_0 + self.fc_p * (a - self.a_pivot)

    # def _Nc(self, M, a):
    #     # Number of centrals
    #     Mmin = 10.**(self.Q_0 + self.Q_p * (a - self.a_pivot))
    #     siglog10M = self.siglog10M_0 + self.siglog10M_p * (a - self.a_pivot)    
    #     return 0.5 * (1 + erf(np.log10(M/Mmin)/siglog10M))

    # def _Ns(self, M, a):
    #     # Number of satellites
    #     M0 = 10.**(self.log10Mcut_0 + self.log10Mcut_p * (a - self.a_pivot))
    #     M1 = 10.**(self.log10M1_0 + self.log10M1_p * (a - self.a_pivot))
    #     alpha = self.alpha_0 + self.alpha_p * (a - self.a_pivot)
    #     result =  np.heaviside(M-M0, 1) * (((M-M0)/ M1)**alpha) #* self._Nc(M, a) # 0 \times nan != 0 # don't include the additional Nc factor bc P1h is defined with N = Nc(1 + Ns ) 
    #     result[np.isnan(result) == True] = 0
    #     return result #np.heaviside(M-M0, 1) * self._Nc(M, a) * ((M-M0)/ M1)**alpha
    




    def _Nc(self, M, a): # Yuan 2022 ELG Nc parametrization
        def gaussian_Ms(Ms_, Mcut_, sigma_):
            """
            Compute the normalized Gaussian distribution function.

            Parameters:
            x (array-like): The input values.
            mu (float): The mean of the distribution.
            sigma (float): The standard deviation of the distribution.

            Returns:
            array-like: The values of the Gaussian function at x.
            """
            Mcut = np.log10(Mcut_)
            Ms = np.log10(Ms_)
            y = (1/(sigma_ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (((Ms - Mcut)) / sigma_)**2)
            return y

        Mcut = 10.**(self.log10Mcut_0 + self.log10Mcut_p * (a - self.a_pivot))
        sigma = self.siglog10M_0 + self.siglog10M_p * (a - self.a_pivot)
        gamma = self.gamma_0 + self.gamma_p * (a - self.a_pivot)
        pmax = self.pmax_0 + self.pmax_p * (a - self.a_pivot)
        Q = self.Q_0 + self.Q_p * (a - self.a_pivot)

        phi_ = gaussian_Ms(M, Mcut, sigma)

        _PHI = 0.5 * (    1. + erf(  gamma*(np.log10(M/Mcut))/(np.sqrt(2.) * sigma)   )   )

        A_ = pmax - (1./Q)

        return 2 * A_ * phi_ * _PHI + ( (1/(2.*Q)) * (   1. + erf( np.log10(M/Mcut) / 0.01 )    ))

    def _Ns(self, M, a): # Yuan 2022 ELG Ns parametrization
        Mcut = 10.**(self.log10Mcut_0 + self.log10Mcut_p * (a - self.a_pivot))
        M1 = 10.**(self.log10M1_0 + self.log10M1_p * (a - self.a_pivot))
        alpha = self.alpha_0 + self.alpha_p * (a - self.a_pivot)
        k = self.k_0 + self.k_p * (a - self.a_pivot)

        #result = (1/self._Nc(M,a))*(  (M - (k*Mcut)) / M1  ) **alpha
        result = (self._Nc(M,a))*(  (M - (k*Mcut)) / M1  ) **alpha
        result[np.isnan(result) == True] = 0

        return result