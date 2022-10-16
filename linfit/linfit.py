import astropy.units as u
from astropy.table import Table
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# GENERAL FUNCTIONS
# -----------------------

def flux_at(wavelength, slope, intercept):
    '''
    Computes the flux at the specified wavelength for :math:`\\log F_\\nu = \\text{slope}\\times\\log\\nu + \\text{intercept}`
    Args:
        wavelength: Wavelength in microns
        slope: slope of the power-law
        intercept:  intercept of the power-law

    Returns: 
        **wav_flux** (float), flux in Jansky at the specified wavelength

    ''' 
    wav = (wavelength*u.um).to(u.Hz, equivalencies=u.spectral()).value
    wav_flux_log = slope*np.log10(wav) + intercept
    wav_flux = (10**(wav_flux_log))*u.Jy
    return wav_flux

# -----------------------
# CLASS
# -----------------------


class mirspec(object):
    '''
    Create a new mid-infrared spectrum object

    Args:
        wavelength: Wavelength in microns  or specified unit
        flux: Flux in Jansky or specified unit
        flux_error:  Flux error in Jansky  or specified unit

    :rtype: Attributes

    Returns: 
        **spectrum** variable with wavelength in microns and flux and flux error in Jansky.
    '''

    def __init__(self, wavelength, flux, flux_error):

        #Remove problematic datapoints
        wavelength = wavelength[flux > 0]
        flux_error = flux_error[flux > 0]
        flux = flux[flux > 0]


        # Check units conversion
        if u.Quantity(wavelength).unit == '':
            wavelength = wavelength*u.um
        else: 
            wavelength = wavelength.to(u.um)
        if u.Quantity(flux).unit == '':
            flux = flux*u.Jy
        else: 
            flux = flux.to(u.Jy)
        if u.Quantity(flux_error).unit == '':
            flux_error = flux_error*u.Jy
        else: 
            flux_error = flux_error.to(u.Jy)

        self.wavelength = wavelength
        self.flux = flux
        self.flux_error = flux_error

        spectrum = Table()
        spectrum["wavelength"] = wavelength
        spectrum["flux"] = flux
        spectrum["sigma"] = flux_error
        
        self.spectrum = spectrum

    def linfit(self, sigma=1.5, l_min=5, l_max=15,  max_iterations=50):
        '''
        Executes the fit of the equation 
        :math:`\\log F_\\nu = \\text{slope}\\times\\log\\nu + \\text{intercept}`
        by clipping the emission and absorption features where
        :math:`\\text{residual} > \\sigma\\times\\text{standard deviation}_\\text{residual}`
        
        **How to use:**

        Initiate a spectrum object
        
        ``spec = mirspec(wavelength, flux, flux_error)``

        run the method
        
        ``spec.linfit()``
        
        and call any of the resulting attributes

        ``spec.ten_flux``


        Args:
            sigma: Multiplicative number that determines how many standard deviations of the
                    residuals are needed to remove a data point before the next iteration
            l_min: Minimum wavelength to use
            l_max: Maximum wavelength to use
            max_iterations: Maximum number of times to iterate the fit

        :rtype: Attributes

        Returns: 
            Results of the fit **slope**, **intercept**, the absorption corrected 10.5 microns 
            flux (**ten_flux**) and the arrays containing the data of each iteration for 
            **frequencies**, **fluxes**, **slopes**, **intercepts**, **residuals**, **residual_stds**.
        '''

        # Select only data in the wavelength range between l_min and l_max microns
        wave_filter = (l_min < self.spectrum['wavelength']) & (self.spectrum['wavelength'] < l_max)
        self.wave_filter = wave_filter
        # Convert wavelength to frequency and remove units for the fits
        frequency =  self.wavelength[wave_filter].to(u.Hz, equivalencies=u.spectral()).value

        # Remove units for the fits
        flux = self.flux[wave_filter].value
        flux_unc = self.flux_error[wave_filter].value
        
        # Lists to save the variables between iterations of the fit
        frequencies = []
        fluxes = []
        slopes = []
        intercepts = []
        residuals = []
        residual_stds = []

        residual_std = 1000
        iteration = 0

        # Start the fit and keep iterating until maximum number of iterations is reached
        # or the standard deviation of the residuals is small
        while residual_std > 0.015 and iteration < max_iterations:
            frequencies = frequencies + [frequency]
            fluxes = fluxes + [flux]
            
            # Execute the fit in the log-log plane
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(frequency),np.log10(flux))

            # Save the parameters
            slopes = slopes + [slope]
            intercepts = intercepts + [intercept]
            
            # Compute the residuals and its standard deviation
            residual = np.log10(flux) - ((slope * np.log10(frequency)) + intercept)
            residuals = residuals + [residual]
            residual_std = residual.std()
            residual_stds = residual_stds + [residual_std]
            
            # Remove the points for wich the residual is greater than sigma*standard deviation
            # and filter the current arrays for the next iteration
            frequency = frequency[abs(residual) < sigma*residual_std]
            flux = flux[abs(residual) < sigma*residual_std]
        
            iteration = iteration + 1


        # Save the arrays as attributes
        self.frequencies = frequencies
        self.fluxes = fluxes
        self.slopes = slopes
        self.intercepts = intercepts
        self.residuals = residuals
        self.residual_stds = residual_stds
        # Save the final results as attributes
        self.slope = slope
        self.intercept = intercept
        self.ten_flux = flux_at(10.5, slope, intercept)


    def plot_linfit(self):
        """
        Plots LINFIT first 3 runs and last run in a grid
        
        :raises Assertion Error: if LINFIT is not run before
        """

        #Verifies that LINFIT has been run and attributes are in place
        assert hasattr(self,'frequencies'),"Please run LINFIT first."

        mosaic = """
            AB
            AB
            EF
            ..
            CD
            CD
            GH
            """
        fig = plt.figure(figsize=(10,8))
        ax_dict = fig.subplot_mosaic(mosaic, gridspec_kw={
                "wspace": 0.,
                "hspace": 0.,})

        xt=0.5
        yt=0.8

        ax_dict["B"].plot((self.frequencies[0]*u.Hz).to(u.um, equivalencies=u.spectral()),self.fluxes[0], color='silver')
        ax_dict["C"].plot((self.frequencies[0]*u.Hz).to(u.um, equivalencies=u.spectral()),self.fluxes[0], color='silver')
        ax_dict["D"].plot((self.frequencies[0]*u.Hz).to(u.um, equivalencies=u.spectral()),self.fluxes[0], color='silver')

        ax_dict["A"].plot((self.frequencies[0]*u.Hz).to(u.um, equivalencies=u.spectral()),self.fluxes[0])
        ax_dict["A"].plot((self.frequencies[0]*u.Hz).to(u.um, equivalencies=u.spectral()), 10**(self.slopes[0]*np.log10(self.frequencies[0]) + self.intercepts[0]), color=colors[1], zorder=-1, ls="--")
        ax_dict["A"].text(xt,yt,r"$\alpha = $ "+"{:.3f}".format((-1)*self.slopes[0]),transform=ax_dict["A"].transAxes, fontsize=12,ha="center")


        ax_dict["E"].scaselfer((self.frequencies[0]*u.Hz).to(u.um, equivalencies=u.spectral()), self.residuals[0], s=10, color='dimgray')
        ax_dict["E"].axhline(self.residual_stds[0], lw=1, color='black')
        ax_dict["E"].axhline(-self.residual_stds[0], lw=1, color='black')

        ax_dict["B"].plot((self.frequencies[1]*u.Hz).to(u.um, equivalencies=u.spectral()),self.fluxes[1])
        ax_dict["B"].plot((self.frequencies[0]*u.Hz).to(u.um, equivalencies=u.spectral()), 10**(self.slopes[1]*np.log10(self.frequencies[0]) + self.intercepts[1]), color=colors[1], zorder=-1, ls="--")
        ax_dict["B"].text(xt,yt,r"$\alpha = $ "+"{:.3f}".format((-1)*self.slopes[1]),transform=ax_dict["B"].transAxes, fontsize=12,ha="center")


        ax_dict["F"].scaselfer((self.frequencies[1]*u.Hz).to(u.um, equivalencies=u.spectral()), self.residuals[1], s=10, color='dimgray')
        ax_dict["F"].axhline(self.residual_stds[1], lw=1, color='black')
        ax_dict["F"].axhline(-self.residual_stds[1], lw=1, color='black')

        ax_dict["C"].plot((self.frequencies[2]*u.Hz).to(u.um, equivalencies=u.spectral()),self.fluxes[2])
        ax_dict["C"].plot((self.frequencies[0]*u.Hz).to(u.um, equivalencies=u.spectral()), 10**(self.slopes[2]*np.log10(self.frequencies[0]) + self.intercepts[2]), color=colors[1], zorder=-1, ls="--")
        ax_dict["C"].text(xt,yt,r"$\alpha = $ "+"{:.3f}".format((-1)*self.slopes[2]),transform=ax_dict["C"].transAxes, fontsize=12,ha="center")


        ax_dict["G"].scaselfer((self.frequencies[2]*u.Hz).to(u.um, equivalencies=u.spectral()), self.residuals[2], s=10, color='dimgray')
        ax_dict["G"].axhline(self.residual_stds[2], lw=1, color='black')
        ax_dict["G"].axhline(-self.residual_stds[2], lw=1, color='black')

        ax_dict["D"].plot((self.frequencies[-1]*u.Hz).to(u.um, equivalencies=u.spectral()),self.fluxes[-1])
        ax_dict["D"].plot((self.frequencies[0]*u.Hz).to(u.um, equivalencies=u.spectral()), 10**(self.slopes[-1]*np.log10(self.frequencies[0]) + self.intercepts[-1]), color=colors[1], zorder=-1, ls="--")
        ax_dict["D"].text(xt,yt,r"$\alpha = $ "+"{:.3f}".format((-1)*self.slopes[-1]),transform=ax_dict["D"].transAxes, fontsize=12,ha="center")

        ax_dict["H"].scaselfer((self.frequencies[-1]*u.Hz).to(u.um, equivalencies=u.spectral()), self.residuals[-1], s=10, color='dimgray')
        ax_dict["H"].axhline(self.residual_stds[-1], lw=1, color='black')
        ax_dict["H"].axhline(-self.residual_stds[-1], lw=1, color='black')

        xr,yr = 0.9,0.2
        ax_dict["A"].text(xr,yr,"1st run",transform=ax_dict["A"].transAxes, fontsize=12,ha="right")
        ax_dict["B"].text(xr,yr,"2nd run",transform=ax_dict["B"].transAxes, fontsize=12,ha="right")
        ax_dict["C"].text(xr,yr,"3rd run",transform=ax_dict["C"].transAxes, fontsize=12,ha="right")
        ax_dict["D"].text(xr,yr,"Last run",transform=ax_dict["D"].transAxes, fontsize=12,ha="right")

        ax_dict["B"].yaxis.set_ticklabels([])
        ax_dict["F"].yaxis.set_ticklabels([])
        ax_dict["D"].yaxis.set_ticklabels([])
        ax_dict["H"].yaxis.set_ticklabels([])

        ax_dict["A"].set_xlim(5,15)
        ax_dict["B"].set_xlim(5,15)
        ax_dict["C"].set_xlim(5,15)
        ax_dict["D"].set_xlim(5,15)
        ax_dict["E"].set_xlim(5,15)
        ax_dict["F"].set_xlim(5,15)
        ax_dict["G"].set_xlim(5,15)
        ax_dict["H"].set_xlim(5,15)

        ax_dict["E"].set_ylim(-0.75,0.75)
        ax_dict["F"].set_ylim(-0.75,0.75)
        ax_dict["G"].set_ylim(-0.75,0.75)
        ax_dict["H"].set_ylim(-0.75,0.75)

        ax_dict["E"].set_xlabel("Wavelength (μm)")
        ax_dict["F"].set_xlabel("Wavelength (μm)")
        ax_dict["G"].set_xlabel("Wavelength (μm)")
        ax_dict["H"].set_xlabel("Wavelength (μm)")

        ax_dict["A"].set_ylabel("Flux (Jy)")
        ax_dict["C"].set_ylabel("Flux (Jy)")
        ax_dict["E"].set_ylabel("Residuals")
        ax_dict["G"].set_ylabel("Residuals")

        return fig


