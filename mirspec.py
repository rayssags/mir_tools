import astropy.units as u
from astropy.table import Table
from scipy import stats
import numpy as np

class mirspec(object):
    '''
    Create a new mid-infrared spectrum object

    Args:
        wavelength: Wavelength in microns  or specified unit
        flux: Flux in Jansky or specified unit
        flux_error:  Flux error in Jansky  or specified unit

    :rtype: Instance variable

    Returns: 
        **spectrum** variable with wavelength in microns and flux and flux error in Jansky.

    '''

    def __init__(self, wavelength, flux, flux_error):

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
        
        and call any of the resulting instance variables

        ``spec.ten_flux``


        Args:
            sigma: Multiplicative number that determines how many standard deviations of the
                    residuals are needed to remove a data point before the next iteration
            l_min: Minimum wavelength to use
            l_max: Maximum wavelength to use
            max_iterations: Maximum number of times to iterate the fit

        :rtype: Instance variables

        Returns: 
            Results of the fit **slope**, **intercept**, the absorption corrected 10.5 microns 
            flux (**ten_flux**) and the arrays containing the data of each iteration for 
            **frequencies**, **fluxes**, **slopes**, **intercepts**, **residuals**, **residual_stds**.
        '''

        # Select only data in the wavelength range between l_min and l_max microns
        wave_filter = (l_min < self.spectrum['wavelength']) & (self.spectrum['wavelength'] < l_max)
        
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

        # Compute the 10.5 microns flux
        ten = (10.5*u.um).to(u.Hz, equivalencies=u.spectral()).value
        ten_flux_log = slope*np.log10(ten) + intercept
        ten_flux = (10**(ten_flux_log))

        # Save the arrays as instance attributes
        self.frequencies = frequencies
        self.fluxes = fluxes
        self.slopes = slopes
        self.intercepts = intercepts
        self.residuals = residuals
        self.residual_stds = residual_stds

        # Save the final results as instance attributes
        self.slope = slope
        self.intercept = intercept
        self.ten_flux = ten_flux*u.Jy
