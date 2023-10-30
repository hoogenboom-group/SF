import numpy as np
import logging
from skimage import io
from pathlib import Path
from tqdm.notebook import tqdm
from scipy.optimize import curve_fit
import tifffile
import matplotlib.pyplot as plt
import psf_extra as psfe

__all__ = ['plot_int_histograms',
           'exp_func',
           'max_rolling2',
           'compensate_int_loss',
           'plot_stacks_MIPs',
           'rescale_stack',
           'plot_stacks_overlay',
           'eight_bit_as',
           'save_stack',
           'natural_sort',
           'Lyakin', 
           'stallinga_high', 
           'diel_mean', 
           'diel_median', 
           'scaling_factor', 
           'scaling_factor_from_nfp',
           'load_stack']

fire = psfe.get_Daans_special_cmap()

def plot_int_histograms(stack1,stack2,names):
    stack1_list = stack1.flatten()
    stack2_list = stack2.flatten()
    fig,axs=plt.subplots(1,2)
    axs[0].hist(stack1_list,bins=200)
    axs[0].set_yscale('log')
    axs[0].set_title(names[0])
    axs[0].set_xlabel('Intensity (a.u.)')
    axs[0].set_ylabel('Counts')
    axs[0].set_xlim(0,1)
    axs[1].hist(stack2_list,bins=200)
    axs[1].set_yscale('log')
    axs[1].set_title(names[1])
    axs[1].set_xlabel('Intensity (a.u.)')
    axs[1].set_ylabel('Counts')
    axs[1].set_xlim(0,1)
    plt.tight_layout()
    plt.show()
    return

def exp_func(x,a,b,c):
    return c*a**(-b*x)

def max_rolling2(A,K):
    rollingmax = np.array([max(A[j:j+K]) for j in range(len(A)-K)])
    return rollingmax

def compensate_int_loss(stack,kernel_size=25,fit=False,skip_slides=10, plot=False):
    intensity=[]
    for slice in stack:
        intensity.append(np.mean(slice))
    zs = np.arange(1,len(intensity)+1,1)
    rol_max_intensity=max_rolling2(intensity,kernel_size)
    difference = len(intensity)-len(rol_max_intensity)
    list0=np.zeros(difference) + rol_max_intensity[0]
    rol_max_intensity_padded = np.append(list0,rol_max_intensity)
    #fit rolling maximum with exponential:
    if fit==True:
        
        popt,pcov=curve_fit(exp_func,zs,rol_max_intensity_padded)#,bounds=([0,0,0],[1,1,np.inf]))
        if plot==True: plt.plot(exp_func(zs,*popt),label='exp fit')

        fit_intensities = exp_func(zs,*popt)
    else:
        fit_intensities = rol_max_intensity_padded
    if plot==True:
        plt.plot(zs,rol_max_intensity_padded,label='rolling maximum')
        plt.plot(zs,intensity,label='intensity')
        plt.legend()
        #plt.yscale('log')
        plt.show()
    stack_int=np.zeros(np.shape(stack))
    for i in range(len(stack)):
        if i < len(stack)-skip_slides:
            stack_int[i,:,:] = np.divide(stack[i,:,:],fit_intensities[i])
        else: 
            stack_int[i,:,:] = np.divide(stack[i,:,:],fit_intensities[len(stack)-skip_slides])
    
    return stack_int, intensity,fit_intensities

def plot_stacks_MIPs(stack_one, stack_two, stack_names, planes, ps_xy_one, ps_z_one, ps_xy_two, ps_z_two):

    shape_one = np.shape(stack_one)
    shape_two = np.shape(stack_two)

    fig,axs=plt.subplots(2,2)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    for i in range(len(stack_names)):
        for j in range(len(planes)):
            if planes[j] == 'xy': 
                mip_axis = 0 
                if i==1: extent = [0, shape_two[2]*ps_xy_two, 0, shape_two[1]*ps_xy_two]
                else: extent = [0, shape_one[2]*ps_xy_one, 0, shape_one[1]*ps_xy_one]
            else: 
                mip_axis = 1
                if i==1: extent = [0, shape_two[2]*ps_xy_two, 0, shape_two[0]*ps_z_two]
                else: extent = [0, shape_one[2]*ps_xy_one, 0, shape_one[0]*ps_z_one]

            axs[j,i].imshow(np.max([stack_one,stack_two][i], axis=mip_axis),cmap=fire,extent=extent)
            axs[j,i].set_title(stack_names[i])
            axs[j,i].set_xlabel(planes[j][:1]+r' ($\mu$m)')
            axs[j,i].set_ylabel(planes[j][1:]+r' ($\mu$m)')
    plt.tight_layout()
    plt.show()
    return

def rescale_stack(stack, NA, n1, n2, lam_0, ps_z,crit = 'Lyakin'):
    # make list of apparent focal positions (NFP) of  input stack (with ref index mismatch)
    shape_stack = np.shape(stack)
    nfp_stack = np.arange(shape_stack[0])*ps_z
    nfp_stack[0]=0.00000001 # prevent SF function from blowing up
    
    # calculate depth-dependent scaling factor for each NFP
    sf_stack = scaling_factor_from_nfp(nfp_stack, NA,n1,n2,lam_0,crit=crit)
    nfp_stack[0]=0.0 # make first value of NFP array zero again (funky, I know)
    
    # calculate actual focal position (AFP) of stack using scaling factor and NFP
    afp_stack = np.multiply(nfp_stack,sf_stack)

    # make new stack rescaled data will be added to
    # use step size of original stack and range calculated from depth-dependent scaling of last slide in input stack
    afp_new_stack = np.arange(0,afp_stack[-1],ps_z)
    
    # make empty array we will fill with intensities of rescaled stack
    stack_rescaled=np.empty([len(afp_new_stack),shape_stack[1],shape_stack[2]])

    # put intensities of rescaled stack into new evenly spaced stack:
    for i in range(len(afp_new_stack)):
        if i == 0: #first slide, no rescaling
            stack_rescaled[i] = stack[0]
        else:
            afp_slide = afp_new_stack[i] #get AFP of slice in new stack
            
            #find two nearest slices in mismatched stack and their AFPs
            index, value =min(enumerate(afp_stack), key=lambda x: abs(x[1]-afp_slide)) #get the index of the closest AFP value in the AFP list of the mismatched stack
            #get the indices of the slices in the mismatch stack surrounding the AFP value in the new stack
            if afp_slide > value: indices = [index, index+1]
            else: indices = [index-1, index]
            #get the corresponding AFP values in the mismatched stack
            value_under, value_upper = afp_stack[indices[0]],afp_stack[indices[1]]
            #calculate the AFP distance between the slice in the new stack and the surrounding slices in the mismatched stack
            dz_under, dz_upper = afp_slide - value_under, value_upper - afp_slide
            
            # we will use inverse distance weighting with power of 1 to interpolate the intensity in the new stack: https://en.wikipedia.org/wiki/Inverse_distance_weighting :
            dz_under_inv, dz_upper_inv=1/dz_under, 1/dz_upper
            dz_sum = dz_under + dz_upper
            dz_inv_sum = dz_under_inv + dz_upper_inv
            # make interpolate intensities in new slide from the two closest slices in the mismatched stack:
            new_slide = np.divide(np.multiply(stack[indices[0],:,:],dz_under_inv)+np.multiply(stack[indices[1],:,:],dz_upper_inv),dz_inv_sum)
            
            #save to new stack
            stack_rescaled[i] = new_slide
            
            #debug
            # print('AFP new stack (um) ', afp_slide,# 'Mean intensity slide: ', np.mean(new_slide)*10000,
            #   '\nAFP under old stack', value_under,# 'Mean intensity slide under: ', np.mean(stack[indices[0],:,:])*10000,
            #       '\nAFP upper old stack',value_upper,# 'Mean intensity slide upper: ', np.mean(stack[indices[0],:,:])*10000,
            #       '\n' )
            
            #plot new slices and surrounding slices
            # fig,axs=plt.subplots(1,3)
            # fig.set_figheight(10)
            # fig.set_figwidth(20)
            # axs[0].imshow(new_slide)
            # axs[0].set_title('New stack index: '+str(i))
            # axs[1].set_title('Old stack index under: '+str(indices[0]))
            # axs[2].set_title('Old stack index upper: '+str(indices[1]))
            # axs[1].imshow(stack[indices[0],:,:])
            # axs[2].imshow(stack[indices[1],:,:])
            # plt.show()
            
            #print indices in both stacks
            # print('\nNew stack index: '+str(i))
            # print('Old stack index under: '+str(indices[0]))
            # print('Old stack index upper: '+str(indices[1]))
            
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
    if NA > n_sample: 
        print("Numerical aperture larger than sample refractive index, Diel mean cannot be computed.")
        return
    sum=0
    number_of_rays=10 # paper uses 100, but this is still doable.
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

def scaling_factor_from_nfp(z, NA,n1,n2,lam_0,crit = 'Lyakin'):
    #dzeta vs NFP
    n2overn1 = np.divide(n2,n1)
    m = np.emath.sqrt(np.subtract(np.power(n2,2),np.power(n1,2)))
    
    if n2overn1 < 1: delta = np.multiply(-1,np.divide(lam_0/4, np.multiply(n1,z)))
    else: delta = np.divide(lam_0/4, np.multiply(n1,z))
    one_plus_delta = np.add(1,delta)
    
    first_term = np.multiply(n2overn1,one_plus_delta)
    sec_term = np.multiply(np.divide(m,n1),np.emath.sqrt(np.multiply(delta,np.add(2,delta))))
    sf_univ = np.add(first_term, sec_term)
    
    sf = np.zeros(len(z))
    if crit != 'None': #cap off exploding SF for small depths
        if crit == 'Loginov': #use Loginov's critical value
            sf_crit = np.divide(n1-np.emath.sqrt(np.power(n1,2)-np.power(NA,2)),
                                    n2-np.emath.sqrt(np.power(n2,2)-np.power(NA,2)))
        elif crit == 'Lyakin': # use Lyakin/Stallinga's value
            sf_crit = Lyakin([0],n2,n1,NA)[0]
        for i in range(len(sf)):
            if n2overn1 < 1: sf[i] = np.max([np.real(sf_univ[i]),np.real(sf_crit)])
            elif n2overn1 > 1:sf[i] = np.min([np.real(sf_univ[i]),np.real(sf_crit)])
            else: sf[i]=1
    else: sf = np.real(sf_univ) #when no capping is performed
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

