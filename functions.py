## import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
import scipy
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.optimize import curve_fit
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d, savgol_filter
from scipy.special import voigt_profile
from scipy.interpolate import interp1d
from cmap import Colormap

## define functions

def gaussian(x, x0, fwhm):
    # resolution = FWHM/0.82 # Rayleigh criterion
    sigma = fwhm / 2.35482
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1 * (x - x0) ** 2 / (2 * sigma ** 2))

# rebin 2D function
def rebin(a, shape):
    '''usage: rebin(a, [256,256])'''
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

# despike spectrum
def z_score(intensity):
    mean_int = np.mean(intensity)
    std_int = np.std(intensity)
    z_scores = (intensity - mean_int) / std_int # standard Z-score
    return z_scores
    
def modified_z_score(intensity):
    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    modified_z_scores = 0.6745 * (intensity - median_int) / mad_int # modified Z-score (Whitaker and Hayes’)
    return modified_z_scores
    
def fixer(y,m):
    threshold = 6 # binarization threshold (8)
    spikes = abs(np.array(modified_z_score(np.diff(y)))) > threshold
    y_out = y.copy() # So we don’t overwrite y
    for i in np.arange(len(spikes)):
        if spikes[i] != 0: # If we have an spike in position i
            w = np.arange(i-m,i+1+m) # we select 2 m + 1 points around our spike
            w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
            y_out[i] = np.mean(y[w2]) # and we average their values
    return y_out

def pre_processZL(ZL_raw,m):
    # binning
    ZL_raw = rebin(ZL_raw, [256//bin1,1024//bin2])
    eneZL = np.linspace(-4.205-0.03,26.515-0.03,1024//bin2)
    
    # perform the skew correction on the Zero-loss 2D spectrum
    ZL1 = ZL_raw.copy()
    pp = 0.065
    for i in range(m):
        ZL1[i,:] = scipy.ndimage.shift(ZL1[i,:], (-pp*i + pp*m//2))
    ZL_fit = ZL1[cut1//bin1:m-cut1//bin1,:]
    
    # prepare the averaged ZL
    ZL_avg = (ZL_fit + np.flip(ZL_fit,0))/2
    
    # Prepare the symmetrized ZL
    ZL_sym = ZL_fit.copy()
    ZL_sym[0:np.size(ZL_fit,0)//2,:] = np.flip(ZL_fit[np.size(ZL_fit,0)//2:],0)
    ZL_sym = ZL_sym/np.max(ZL_sym)
    return ZL_sym, eneZL
    
def pre_process(BG_raw, despike = True, skew = True, denoise = True, symmetrize = True):    
    M1,N1 = BG_raw.shape

    # set NaN points to 0
    Bsub = BG_raw.copy()
    Bsub[np.isnan(Bsub)] = 0

    if despike:
        BG_despiked = np.empty_like(Bsub)
        for i in range (M1):
            BG_despiked[i,:] = fixer(fixer(BG_raw[i,:], m=3), m=3) # 2-passes of the fixer function
        
        ## normalize the dataset
        Bsub = BG_despiked/np.max(BG_despiked)
    
    ## rebin the dataset
    m = M1//bin1
    n = N1//bin2
    Bsub = rebin(Bsub, [m,n])

    if skew:
        # skew correction of the 2D spectra
        D1 = Bsub.copy()
    
        # perform the correction on D1 dataset
        pp = 0.065
        for i in range(m):
            D1[i,:] = scipy.ndimage.shift(D1[i,:], (-pp*i + pp*m//2))
    
        # normalize the dataset again after the skew correction
        Bsub = D1/np.max(D1)

    if denoise:
        x_raw = Bsub.copy()
        # filter data to reduce noise
        x_raw = savgol_filter(x_raw, 5, 1, axis = 1, mode = 'nearest')
        Bsub = x_raw/x_raw.max()
        m, n = x_raw.shape
    
    # experimental calibrations    
    xmin = 172.4                       # spectrum start energy
    xmax = xmin + n*0.03*bin2          # 0.03 eV per pixel
    x_ene = np.linspace(xmin, xmax, n) # experimental energy range (eV)
    
    ymax = m/2/(10/bin1)                # 0.1 hbar per pixel 
    y_OAM = np.linspace(-ymax, ymax, m) # experimental OAM range (hbar)

    if symmetrize:       
        # Reduce the dataset to the interval [-12, 12] in OAM         
        X_fit = Bsub[cut1//bin1:m-cut1//bin1,:]
        y_OAM = y_OAM[cut1//bin1:m-cut1//bin1]
        
        # Prepare the symmetrized dataset
        X_sym = X_fit.copy()
        X_sym[0:np.size(X_fit,0)//2,:] = np.flip(X_fit[np.size(X_fit,0)//2:],0)
        Bsub = X_sym/np.max(X_sym)
    
    return Bsub, x_ene, y_OAM

## custom palettes and colors

# set the colormap of 2D figures
cmap = Colormap('crameri:batlow').to_matplotlib(256, 1.0)
cmap5 = Colormap('crameri:batlow').to_matplotlib(5, 1.0)

# Take colors at regular intervals spanning the colormap
palette1 = cmap5(np.linspace(0, 1, 5))

bin1 = 2 # OAM binning
bin2 = 4 # Energy binning
cut1 = 8