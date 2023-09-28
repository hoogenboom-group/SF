import numpy as np
import logging
from skimage import io
from pathlib import Path
from tqdm.notebook import tqdm
import tifffile
import matplotlib.pyplot as plt

__all__ = ['rescale_stack',
           'plot_stacks_overlay',
           'eight_bit_as',
           'save_stack',
           'natural_sort',
           'Lyakin', 
           'stallinga_high', 
           'diel_mean', 
           'diel_median', 
           'scaling_factor', 
           'load_stack']

def rescale_stack(stack, NA, n1, n2, lam_0, ps_z):
    # make list of apparent focal positions (NFP) of stack
    shape_stack = np.shape(stack)
    nfp_stack = np.arange(shape_stack[0])*ps_z
    nfp_stack[0]=0.00000001 # prevent function from blowing up
    
    # calculate depth-dependent scaling factor
    sf_stack = scaling_factor(nfp_stack, NA,n1,n2,lam_0)
    nfp_stack[0]=0.000000 # prevent function from blowing up
    sf_stack[0]=sf_stack[1] # equation get unphysical when z -> 0, therefore replace with next number (cheating, I know)
    
    # calculate actual focal position (AFP) of stack
    afp_stack = nfp_stack*sf_stack

    # make new stack rescaled data will be added to
    # use step size of original stack and range calculated from depth-dependent scaling for last slide
    final_sf = sf_stack[-1]
    afp_new_stack = np.arange(0,afp_stack[-1],ps_z)

    # make empty array we will fill with intensities of rescaled stack
    stack_rescaled=np.empty([len(afp_new_stack),shape_stack[1],shape_stack[2]])
    
    # put intensities of rescaled stack into new evenly spaced stack:
    for i in range(len(afp_new_stack)):
        if i == 0: #first slide, no rescaling
            stack_rescaled[i] = stack[0]
        else:
            afp_slide = afp_new_stack[i] #get depth of new stack
            #find two nearest slides in air stack and their z position (AFP)
            index, value =min(enumerate(afp_stack), key=lambda x: abs(x[1]-afp_slide))
            if afp_slide > value: indices = [index, index+1]
            else: indices = [index-1, index]
            value_under, value_upper = afp_stack[indices[0]],afp_stack[indices[1]]
            dz_under, dz_upper = afp_slide - value_under, value_upper - afp_slide
            dz_under_inv, dz_upper_inv=1/dz_under, 1/dz_upper
            dz_sum = dz_under + dz_upper
            dz_inv_sum = dz_under_inv + dz_upper_inv
            # make new slide from two closest slides
            new_slide = np.divide(np.multiply(stack[indices[0],:,:],dz_under_inv)+np.multiply(stack[indices[1],:,:],dz_upper_inv),dz_inv_sum)
            #this used inverse distance weighting with power of 1: https://en.wikipedia.org/wiki/Inverse_distance_weighting
            stack_rescaled[i] = new_slide
    return stack_rescaled, afp_new_stack, afp_stack, nfp_stack

def plot_stacks_overlay(stack_rescaled,stack, afp_new_stack,afp_stack,nfp_stack,ps_xy,ps_z):
    fig,axs=plt.subplots(1,1)
    fig.set_figheight(5)
    fig.set_figwidth(15)
    shape_stack = np.shape(stack)
    im = axs.imshow(np.max(stack_rescaled,axis=1),extent = [0, shape_stack[2]*ps_xy, 0,len(afp_new_stack)*ps_z],alpha=0.5, cmap='Greys_r')
    im2 = axs.imshow(np.max(stack,axis=1),extent = [0, shape_stack[2]*ps_xy, 0, shape_stack[0]*ps_z],alpha = 0.5, cmap = 'Purples')
    axs.set_ylim(0,np.max([afp_stack[-1],nfp_stack[-1]]))
    fig.colorbar(im, ax=axs,label = 'Rescaled stack')
    fig.colorbar(im2, ax=axs, label = 'Original stack')
    axs.set_xlabel(r'X ($\mu$m)')
    axs.set_ylabel(r'Z ($\mu$m)')
    plt.tight_layout()
    plt.show()

def eight_bit_as(arr, dtype=np.float32):
    """Convert array to 8 bit integer array.
    
    Parameters
    ----------
    arr: array-like
        Array of shape (L, M, N)
        
    dtype : data type
        Data type of existing array.
        
    Returns
    -------
    arr.astype(dtype) : array-like
        Array formatted to 8 bit integer array
    
    Notes
    -----
    ...
    """
    
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr -= arr.min()
        arr *= 255./arr.max()
    else: 
        arr = arr.astype(np.float32)
    return arr.astype(dtype)

def save_stack(stack, location,filename,psx,psy,psz):
    """Save stack to file, along with metadata of stack.
    
    Parameters
    ----------
    stack: array-like
        Image stack of shape (L, M, N)
        
    file_pattern : str
        A string that is the individual filename of e.g. a tiff stack
        
    Returns
    -------
    ...
    
    Notes
    -----
    ...
    """
    
    if np.sum(stack) == 0:
        print('Empty PSF: only zeros...')
        return

    #save stack to file
    tifffile.imwrite(location+filename,eight_bit_as(stack,np.uint8),photometric='minisblack')

    #save meta data
    with open(location + '/parameters.txt','w') as f:
        f.write('stack parameters:\n')
        f.write('\n')
        f.write('X: '+str(psx)+' nm\n')
        f.write('Y: '+str(psy)+' nm\n')
        f.write('Z: '+str(psz)+' nm\n')

    print("Succesfully saved stack and parameters to file.")
    return

    
def natural_sort(l):
    """A more natural sorting algorithm

    Parameters
    ----------
    l : list
        List of strings in need of sorting

    Examples
    --------
    >>> l = ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']
    >>> sorted(l)
    ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']
    >>> natural_sort(l)
    ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']

    References
    ----------
    [1] https://stackoverflow.com/a/4836734/5285918
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
    

def Lyakin(z,n_sample,n_im,NA):
    d = 1
    top = np.add(n_im,np.sqrt(np.subtract(np.power(n_im,2),np.power(NA,2))))
    bottom_1 = np.multiply(4,np.subtract(np.power(n_sample,2),np.power(n_im,2)))
    bottom_2 = np.add(n_im,np.emath.sqrt(np.subtract(np.power(n_im,2),np.power(NA,2))))
    bottom = np.real(np.emath.sqrt(np.add(bottom_1,np.power(bottom_2,2))))
    if bottom == 0: bottom=0.000000000000001
    dz = np.multiply(d,np.divide(top,bottom))
    scaling_factor = np.divide(1,dz)    
    return np.zeros(len(z)) + scaling_factor

def stallinga_high(z,nn1,nn2,NA):
    if nn1==nn2: 
        return np.ones(len(z))
    alphas=[]
    for i in range(len(NA)):
        alphas.append(-1*(f1f2_av(nn1,nn2,NA[i]) 
                          - f_av(nn1,NA[i]) * f_av(nn2,NA[i])) 
                      / (ff_av(nn1,NA[i]) - f_av(nn1,NA[i])**2))
    d=1
    dz = np.multiply((np.add(alphas,1)),d) #we take delta d as 1
    scaling_factor = np.divide((d-dz),d)
    
    sf = np.divide(-1,alphas)
    return np.zeros(len(z)) + sf

def f_av(nn,NA):
    f=2*(nn**3-(nn**2-NA**2)**(3/2))/(3*NA**2)
    return f

def ff_av(nn,NA):
    ff=nn**2-(NA**2)/2
    return ff

def f1f2_av(nn1,nn2,NA):
    f1f2 = ( (nn1*nn2**3+nn2*nn1**3 - (nn1**2+nn2**2-2*NA**2)*np.sqrt(nn1**2-NA**2)*np.sqrt(nn2**2-NA**2) 
            - ((nn1**2-nn2**2)**2)*np.log( ( np.sqrt(nn1**2-NA**2) - np.sqrt(nn2**2-NA**2) )/ (nn1-nn2) ) )/(4*NA**2) )
    return f1f2

def diel_mean(z,n_im,n_sample,NA):
    sum=0
    number_of_rays=10000 # paper uses 100, but this is still doable.
    for i in range(number_of_rays):
        k=i+1
        top     =  np.tan(np.arcsin(np.divide((NA*k),(np.multiply(number_of_rays,n_im)))))
        bottom  =  np.tan(np.arcsin(np.divide((NA*k),(np.multiply(number_of_rays,n_sample)))))
        sum +=np.divide(top,bottom)
    return np.zeros(len(z)) + np.divide(sum,number_of_rays)

def diel_median(z,n_im,n_sample,NA):
    top = np.tan(np.arcsin(np.divide(0.5*NA,n_im)))
    bottom = np.tan(np.arcsin(np.divide(0.5*NA,n_sample)))
    return np.zeros(len(z)) + np.divide(top,bottom)

def scaling_factor(z, NA,n1,n2,lam_0):
    n2overn1 = np.divide(n2,n1)
    
    if n2overn1 < 1: eps = np.multiply(-1,np.divide(np.divide(lam_0,4),(np.multiply(z,n2))))
    else: eps = np.divide(np.divide(lam_0,4),(np.multiply(z,n2)))
    eps_term = np.multiply(eps, np.subtract(2,eps))    
    
    m = np.emath.sqrt(np.subtract(np.power(n2,2),np.power(n1,2)))
    
    sf_univ = np.multiply(np.divide(n2,n1),
                          np.divide(1-eps+np.divide(m,n1)*np.emath.sqrt(eps_term),
                                    1-np.multiply(np.divide(n2,n1)**2,eps_term)))
#     sf_crit = np.divide(n1-np.emath.sqrt(np.power(n1,2)-np.power(NA,2)),
#                         n2-np.emath.sqrt(np.power(n2,2)-np.power(NA,2)))
    
    sf_crit = Lyakin([0],n2,n1,NA)[0]
    
    sf = np.zeros(len(z))
    for i in range(len(sf)):
        if n2overn1 < 1: sf[i] = np.max([np.real(sf_univ[i]),np.real(sf_crit)])
        elif n2overn1 > 1:sf[i] = np.min([np.real(sf_univ[i]),np.real(sf_crit)])
        else: sf[i]=1
    return sf

def load_stack(file_pattern):
    
    # originally from PSF-extractor: https://github.com/hoogenboom-group/PSF-Extractor
    
    """Loads image stack into dask array allowing manipulation
    of large datasets.

    Parameters
    ----------
    file_pattern : list or str
        Either a list of filenames or a string that is either
        a) the individual filename of e.g. a tiff stack or
        b) a directory from which all images will be loaded into the stack

    Returns
    -------
    stack : dask array-like
        Image stack as 32bit float with (0, 1) range in intensity

    Examples
    --------
    * `file_pattern` is a list
    >>> file_pattern = ['/path/to/data/image1.tif',
                        '/path/to/data/image2.tif',
                        '/path/to/data/image3.tif']
    >>> get_stack(file_pattern)

    * `file_pattern` is a directory
    >>> file_pattern = '/path/to/data/'
    >>> get_stack(file_pattern)

    * `file_pattern is a tiff stack
    >>> file_pattern = '/path/to/tiff/stack/multipage.tif'
    >>> get_stack(file_pattern)
    """

    # If a list of file names is provided
    if isinstance(file_pattern, list):
        logging.info("Creating stack from list of filenames.")
        stack = []
        for i, fp in tqdm(enumerate(file_pattern),
                          total=len(file_pattern)):
            logging.debug(f"Reading image file ({i+1}/{len(file_pattern)}) : {fp}")
            image = io.imread(fp, plugin='pil')
            stack.append(image)
            
        # Create 3D image stack (Length, Height, Width)
        stack = np.stack(stack, axis=0)

    # If a directory or individual filename
    elif isinstance(file_pattern, str):
        # Directory
        if Path(file_pattern).is_dir():
            logging.info("Creating stack from directory.")
            # Collect every png/tif/tiff image in directory
            filepaths = list(Path(file_pattern).glob('*.png')) + \
                        list(Path(file_pattern).glob('*.tif')) + \
                        list(Path(file_pattern).glob('*.tiff'))
            # Sort filepaths
            filepaths = natural_sort([fp.as_posix() for fp in filepaths])
            # Load images
            stack = []
            for i, fp in tqdm(enumerate(filepaths),
                              total=len(filepaths)):
                logging.debug(f"Reading image file ({i+1}/{len(filepaths)}) : {fp}")
                image = io.imread(fp, plugin='pil')
                stack.append(image)
            # Create 3D image stack (Length, Height, Width)
            stack = np.stack(stack, axis=0)

        # Tiff stack or gif
        elif (Path(file_pattern).suffix == '.tif') or \
             (Path(file_pattern).suffix == '.tiff') or \
             (Path(file_pattern).suffix == '.gif'):
            logging.info("Creating stack from tiff stack")
            # Create 3D image stack (Length, Height, Width)
            stack = io.imread(file_pattern, plugin='pil')

        # ?
        else:
            if Path(file_pattern).exists():
                raise ValueError(f"Not sure what to do with `{file_pattern}`.")
            else:
                raise ValueError(f"`{file_pattern}` cannot be located or "
                                  "does not exist.")

    else:
        raise TypeError("Must provide a directory, list of filenames, or the "
                        "filename of an image stack as either a <list> or <str>, "
                        f"not {type(file_pattern)}.")

    # Intensity rescaling (0 to 1 in float32)
    # Based on https://github.com/scikit-image/scikit-image/blob/main/skimage/exposure/exposure.py
    # Avoids additional float64 memmory allocation for data
    stack = stack.astype(np.float32)
    imin, imax = np.min(stack), np.max(stack)
    stack = np.clip(stack, imin, imax)
    stack -= imin
    stack /= (imax - imin)
    # Return stack
    logging.info(f"{stack.shape} image stack created succesfully.")
    return stack

