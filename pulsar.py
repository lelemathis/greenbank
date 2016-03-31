import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
from astropy.io import fits
import numpy as np
import os
import sys
import multiprocessing as mp

def get_file_list(d):
    files = []
    
    os.chdir(d)
    for f in os.listdir('.'):
        if os.path.isfile(f):
            print('    -using: '+f)
            files.append(d+f)
                
    return sorted(files)


def show_image(showme, axis_y=None, axis_x=None):
    if axis_x is None:
        axis_x = [i for i in range(len(showme[0]))]
    if axis_y is None:
        axis_y = [i for i in range(len(showme))]
    (x_min,x_max) = (min(axis_x),max(axis_x))
    (y_min,y_max) = (min(axis_y),max(axis_y))
    plt.figure()
    plt.imshow(showme, aspect='auto',extent=[x_min,x_max,y_min,y_max])
    plt.colorbar()
    return

def overplot_parabola(sec, a, hand=None):
    
    axis_x = sec.get_x_axis()
    axis_y = sec.get_y_axis()
    left_parab = []
    right_parab = []
    
    if type(a) is int or type(a) is float:
        a = [a]
    for this_a in a:
        if hand=='left' or hand is None:
            left_x = axis_x[:int(len(axis_x)/2)]
            left_y = []
            for x in left_x:
                y = this_a*x**2
                if y>max(axis_y):
                    left_y.append(None)
                else:
                    left_y.append(y)
            left_parab.append(left_y)
        if hand=='right' or hand is None:
            right_x = axis_x[int(len(axis_x)/2):]
            right_y = []
            for x in right_x:
                y = this_a*x**2
                if y>max(axis_y):
                    right_y.append(None)
                else:
                    right_y.append(y)
            right_parab.append(right_y)
    
    for i in range(len(a)):
        plt.plot(axis_x[:int(len(axis_x)/2)], left_parab[i], 'b-')
        plt.plot(axis_x[int(len(axis_x)/2):], right_parab[i], 'b-')

def gaussian(x, mu, sig):
    return np.exp(-(x - mu)**2 / (2. * sig**2))

"""
def weight_function(eta,px,py,sigma=1):
    x,dist = dist_from_parabola(eta,px,py)
    #print("dist: " + str(dist))
    dist_from_origin = np.sqrt(px**2+py**2)
    #ret_y = gaussian(dist,0,
    ret = dist_from_origin*gaussian(dist,0,sigma)
    #print("ret: " + str(ret))
    if ret < (1/np.e**3):
        #print("ret is None")
        ret = None
    return ret
    #return gaussian(dist,0,sigma)
"""

def weight_function(eta,px,py,sigma=[1,1]):
    x = closest_point_on_the_parabola(eta,px,py)
    y = eta*x**2
    dist_from_origin = np.sqrt(px**2+py**2)
    ret_y = gaussian(py-y,0,sigma[0])
    ret_x = gaussian(px-x,0,sigma[1])
    ret = dist_from_origin*np.sqrt(ret_x*ret_y)
    if ret < (1/np.e**3):
        ret = None
    return ret

def weight_function2(y,x,py,px,sigma):
    ret_y = gaussian(py-y,0,sigma[0])
    ret_x = gaussian(px-x,0,sigma[1])
    ret = np.sqrt(ret_x*ret_y)
    if ret < (1/np.e**3):
        ret = None
    return ret

def weight_function3(eta,orig_y,orig_x,py,px,sigma):
    x = closest_point_on_the_parabola(eta,px,py)
    y = eta*x**2
    dist_from_origin = np.sqrt(orig_x**2+orig_y**2)
    ret_y = gaussian(py-y,0,sigma[0])
    ret_x = gaussian(px-x,0,sigma[1])
    ret = dist_from_origin*np.sqrt(ret_x*ret_y)
    if ret < (1/np.e**3):
        ret = None
    return ret

def crunchy(eta,sec,hand=None,sigma=None):
    powers = []
    powers_norm = []

    y_axis = sec.get_y_axis()
    x_axis = sec.get_x_axis()
    if sigma==None:
        sigma = [np.absolute(y_axis[1]-y_axis[0]),
                 np.absolute(x_axis[1]-x_axis[0])]

    for yi in range(len(y_axis)):
        y = y_axis[yi]
        for xi in range(len(x_axis)):
            x = x_axis[xi]
            this_weight = weight_function(eta,x,y,sigma)
            if this_weight is None:
                powers.append(None)
                powers_norm.append(None)
            else:
                variance = 1/this_weight
                powers.append(sec.get([yi,xi])/variance)
                powers_norm.append(1/variance)
    p = np.nansum(list(filter(None,powers)))
    pn = np.nansum(list(filter(None,powers_norm)))
    #print("eta: " + str(eta))
    #print(p)
    #print(pn)
    #print("p/pn: " + str(p/pn))
    return (eta,p/pn)

def crunchy2(pt_and_sigma,sec,hand=None):
    pt,sigma = pt_and_sigma
    py,px = pt

    powers = []
    powers_norm = []

    y_axis = sec.get_y_axis()
    x_axis = sec.get_x_axis()
    px_y = np.absolute(y_axis[1]-y_axis[0])
    px_x = np.absolute(x_axis[1]-x_axis[0])

    if sigma==None:
        sigma = [px_y,px_x]
    if sigma[0]<px_y:
        sigma = [px_y,sigma[1]]
    if sigma[1]<px_x:
        sigma = [sigma[0],px_x]

    for yi in range(len(y_axis)):
        y = y_axis[yi]
        for xi in range(len(x_axis)):
            x = x_axis[xi]
            this_weight = weight_function2(y,x,py,px,sigma)
            if this_weight is None:
                powers.append(None)
                powers_norm.append(None)
            else:
                variance = 1/this_weight
                powers.append(sec.get([yi,xi])/variance)
                powers_norm.append(1/variance)
    p = np.nansum(list(filter(None,powers)))
    pn = np.nansum(list(filter(None,powers_norm)))
    return (pt,p/pn)

def crunchy3(offset, eta, sec, sigma=None):

    powers = []
    powers_norm = []

    y_axis = sec.get_y_axis()
    x_axis = sec.get_x_axis()
    px_y = np.absolute(y_axis[1]-y_axis[0])
    px_x = np.absolute(x_axis[1]-x_axis[0])

    if sigma==None:
        sigma = [px_y,px_x]
    if sigma[0]<px_y:
        sigma = [px_y,sigma[1]]
    if sigma[1]<px_x:
        sigma = [sigma[0],px_x]

    for yi in range(len(y_axis)):
        y = y_axis[yi]
        for xi in range(len(x_axis)):
            x = x_axis[xi]
            y_eff = y + eta*offset**2
            x_eff = x - offset
            this_weight = weight_function3(eta,y,x,y_eff,x_eff,sigma)
            if this_weight is None:
                powers.append(None)
                powers_norm.append(None)
            else:
                variance = 1/this_weight
                powers.append(sec.get([yi,xi])/variance)
                powers_norm.append(1/variance)
    p = np.nansum(list(filter(None,powers)))
    pn = np.nansum(list(filter(None,powers_norm)))
    return (offset,p/pn)

def get_dynamic_spectrum(filename):
    return np.rot90(fits.open(filename)[0].data)

def get_secondary_spectrum(dyn,subtract_secondary_background=True,normalize_frequency=True,normalize_time=True,cut_off_bottom=True,xscale=1.,yscale=1.):
    dynamic = dyn - np.mean(dyn)
    secondary = np.fft.fftn(dynamic)
    #secondary /= secondary.max()
    secondary = 10.*np.log10(np.abs(np.fft.fftshift(secondary))**2) # in decibels?
    
    if normalize_frequency:
        for i in range(len(secondary)):
            norm_const_f = (np.mean(secondary[i,:len(secondary[i])/4.])
                            + np.mean(secondary[i,3.*len(secondary[i])/4.:]))/2.
            secondary[i] = secondary[i]/norm_const_f
            
    if normalize_time:
        for i in range(len(secondary[0])):
            norm_const_t = np.mean(secondary[:len(secondary)/4.,i])
            secondary[:,i] = secondary[:,i]/norm_const_t
    
    if subtract_secondary_background:
        secondary_background = np.mean(secondary[:len(secondary)/4.][:len(secondary[0])/4.])
        secondary = secondary - secondary_background
    
    ysize = secondary.shape[0]
    xsize = secondary.shape[1]
    
    xmin = int(xsize/2. - xsize/(2.*xscale))
    xmax = int(xsize/2. + xsize/(2.*xscale))
    
    ymin = int(ysize/2. - ysize/(2.*yscale))
    
    if cut_off_bottom:
        ymax = int(ysize/2.)
    else:
        ymax = int(ysize/2. + ysize/(2.*yscale))
    
    return secondary[ymin:ymax,xmin:xmax]

def get_dyn_and_sec(files):
    dyn = [get_dynamic_spectrum(f) for f in files]
    sec = [get_secondary_spectrum(d) for d in dyn]
    return (dyn,sec)

def sort_dict_by_key(dictionary):
    keys = []
    values = []
    for k,v in sorted(dictionary.items()):
        keys.append(k)
        values.append(v)
    return (keys,values)

def get_sec_axes(filename):
    """
    Returns lists containing the values of the axis elements.
    Parameters: takes a list of filenames pointing to the relevant FITS files.
    Returns: a list of tuples ([conjugate frequency axis values],[conjugate time axis values])
    """

    hdulist = fits.open(filename)
    
    t_int = hdulist[0].header["T_INT"] #Gets time interval (delta t) from header
    nchunks = hdulist[0].header["NCHUNKS"] #number of time subintegrations
    BW = hdulist[0].header["BW"] #Gets bandwidth from header 
    nchans = hdulist[0].header["NCHANS"] #number of channels 
    
    nyq_t = 1000. / (2. * t_int) #nyquist frequency for the delay axis of the secondary spectrum
    nyq_f = nchans / (2. * BW) #nyquist frequency for the fringe frequency axis of the secondary spectrum
    #print("nchunks:"+str(nchunks))
    #print("nchans:"+str(nchans))
    #print("nyq_t:"+str(nyq_t))
    #print("nyq_f:"+str(nyq_f))
    fringe = list(np.linspace(-nyq_t,nyq_t,nchunks))
    delay = list(reversed(np.linspace(0,nyq_f,nchans/2.)))
    #print("t: " + str(len(t_temp)))
    #print("f: " + str(len(f_temp)))
    return (delay,fringe)

# user specifies a parabola starting at the origin with formula y=ax**2 , 
# and also a point with x=px and y=py, and this function returns
# the x-coordinate of the closest point on the parabola.
def closest_point_on_the_parabola(a,px,py):
    
    thingy1 = 2.*a*py
    thingy2 = np.sqrt(-3+0j)
    thingy3 = 2**(1/3.)
    thingy4 = 2**(2/3.)
    thingy = (-108.*a**4*px + np.sqrt(11664.*a**8*px**2 - 864.*a**6*(-1 + thingy1)**3 + 0j))**(1/3.)
    Aone = (thingy3*(-1. + thingy1))
    Atwo = thingy
    Athree = thingy
    Afour = (6.*thingy3*a**2)
    Bone = ((1. + thingy2)*(-1. + thingy1))
    Btwo = (thingy4*thingy)
    Bthree = ((1. - thingy2)*thingy)
    Bfour = (12.*thingy3*a**2)
    Cone = (1. - thingy2)*(-1 + thingy1)
    Ctwo = thingy4*thingy
    Cthree = (1. + thingy2)*thingy
    Cfour = 12.*thingy3*a**2
    
    A = -np.real(Aone/Atwo + Athree/Afour)
    B =  np.real(Bone/Btwo + Bthree/Bfour)
    C =  np.real(Cone/Ctwo + Cthree/Cfour)
    
    solns = [A,B,C]
    solns_temp = []
    for soln in solns:
        solns_temp.append(np.abs(soln-px))
    
    val, idx = min((val, idx) for (idx, val) in enumerate(solns_temp))
    return solns[idx]

def dist_from_parabola(a,px,py):
    X = closest_point_on_the_parabola(a,px,py)
    return (X,np.sqrt(a**2*X**4-2.*a*X**2*py+py**2+X**2-2.*px*X+px**2))


"""
EVERYTHING IS INDEXED FIRST WITH RESPECT TO Y, THEN WITH RESPECT TO X
EVERYTHING IS INDEXED FIRST WITH RESPECT TO Y, THEN WITH RESPECT TO X
EVERYTHING IS INDEXED FIRST WITH RESPECT TO Y, THEN WITH RESPECT TO X
"""


class Indexed2D:
    def __init__(self,data=None,axes=None,dtype=float):
        self._is_data_set = False
        self._are_axes_set = False
        if data==None:
            self.data = np.array([[]])
        else:
            self.set_data(data,dtype)
        if axes==None:
            self.axes = ([],[])
            self.y_axis = []
            self.x_axis = []
        else:
            self.set_axes(axes)
        return 
    
    def __getitem__(self,tup):
        y = tup[0]
        x = tup[1]
        y_index = self.__get_y_index(y)
        x_index = self.__get_x_index(x)
        return Indexed2D(data=self.data[y_index,x_index],axes=(self.y_axis[y_index],self.x_axis[x_index]))
    
    def __get_y_index(self,value):
        if type(value)==slice:
            if value.start is not None:
                if value.start<min(self.y_axis) or value.start>max(self.y_axis):
                    raise IndexError('y axis index out of bounds: ' + str(value.start))
                start_index = list(np.absolute([p - value.start for p in self.y_axis]))
                start_index = start_index.index(min(start_index))
            else:
                start_index = 0
            if value.stop is not None:
                if value.stop<min(self.y_axis) or value.stop>max(self.y_axis):
                    raise IndexError('y axis index out of bounds: ' + str(value.stop))
                stop_index = list(np.absolute([p - value.stop for p in self.y_axis]))
                stop_index = stop_index.index(min(stop_index))
            else:
                stop_index = len(self.y_axis)-1
            return slice(start_index,stop_index+1)
        else:
            index = [p - value for p in self.y_axis]
            index = index.index(min(index))
            return index
    
    def __get_x_index(self,value):
        if type(value)==slice:
            if value.start is not None:
                if value.start<min(self.x_axis) or value.start>max(self.x_axis):
                    raise IndexError('x axis index out of bounds: ' + str(value.start))
                start_index = list(np.absolute([p - value.start for p in self.x_axis]))
                start_index = start_index.index(min(start_index))
            else:
                start_index = 0
            if value.stop is not None:
                if value.stop<min(self.x_axis) or value.stop>max(self.x_axis):
                    raise IndexError('x axis index out of bounds: ' + str(value.stop))
                stop_index = list(np.absolute([p - value.stop for p in self.x_axis]))
                stop_index = stop_index.index(min(stop_index))
            else:
                stop_index = len(self.x_axis)-1
            return slice(start_index,stop_index+1)
        else:
            index = [p - value for p in self.x_axis]
            index = index.index(min(index))
            return index
    
    def set_data(self,data,dtype=float):
        if type(data) is not list and type(data) is not np.ndarray:
            raise TypeError('Data does not have the right type.')
        for d in data:
            if len(d)!=len(data[0]):
                raise IndexError('Data must be rectangular in shape.')
        for d in data:
            if type(d) is not list and type(d) is not np.ndarray:
                raise TypeError('Data does not have the right type.')
            for i in range(len(d)):
                if d[i] is not dtype:
                    try:
                        d[i] = dtype(d[i])
                    except Exception as e:
                        print('your data could not be casted to '+str(dtype)+': ')
                        raise e
        if self._are_axes_set:
            y_axis_matching = len(data) == len(self.y_axis)
            x_axis_matching = len(data[0]) == len(self.x_axis)
            if y_axis_matching and x_axis_matching:
                self._is_data_set = True
                self.data = np.array(data)
            else:
                raise IndexError('Data must have dimensions as axes')
        else:
            self._is_data_set = True
            self.data = np.array(data)
        return
    
    def set_axes(self,axes):
        if type(axes) is not tuple:
            raise TypeError('Axes argument should be a tuple (y_axis,x_axis)')
        y_axis = axes[0]
        x_axis = axes[1]
        if type(y_axis) is not list and type(y_axis) is not np.ndarray:
            raise TypeError('The axes should be specified with a list or numpy array')
        if type(x_axis) is not list and type(x_axis) is not np.ndarray:
            raise TypeError('The axes should be specified with a list or numpy array')
        for i in range(len(y_axis)):
            if type(y_axis[i]) is not float:
                y_axis[i] = float(y_axis[i])
        for i in range(len(x_axis)):
            if type(x_axis[i]) is not float:
                x_axis[i] = float(x_axis[i])
        if self._is_data_set:
            y_axis_matching = len(y_axis) == len(self.data)
            x_axis_matching = len(x_axis) == len(self.data[0])
            if y_axis_matching and x_axis_matching:
                self._are_axes_set = True
                self.axes = (y_axis,x_axis)
                self.x_axis = x_axis
                self.y_axis = y_axis
            else:
                raise IndexError('Axes must have dimensions as data')
        else:
            self._are_axes_set = True
            self.x_axis = x_axis
            self.y_axis = y_axis
            self.axes = (y_axis,x_axis)
        return
    
    def get_data(self):
        return np.array(self.data)
    
    def get_axes(self):
        return self.axes
    
    def get_x_axis(self):
        return self.x_axis
    
    def get_y_axis(self):
        return self.y_axis
