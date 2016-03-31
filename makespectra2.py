#Lele Mathis and Adam Jussila
#This program is used to take a fits file of a selected directory and
#print out a dynamic and secondary spectra of the given pular data.
#3/8/16

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def main():
    psrName = raw_input('Which pulsar do you want to use? ')
    #psrdir =raw_input('Give the Full directory (Eg. Desktop/Research/B0145+22dyn.fits): ')
    
    #hdulist = fits.open(psrdir) #reads fits file into an HDUlist
    hdulist = fits.open('/Users/lelemathis/Spectra/PSR'+psrName+'_dyn.fits') #For my file system -Lele
    
    #header = hdulist[0].header #gets header of 1st file in hdulist; not used
    astrodata = hdulist[0].data #gets array of data of 1st file in hdulist
    median = np.median(astrodata)
    std = np.std(astrodata-median)
    
    #removes values outside 3 standard deviations from the mean of astrodata
    #for y in range(len(astrodata)):
    #    for x in range(len(astrodata[y])):
    #        if astrodata[y][x] >= median+(std*5) or astrodata[y][x]<= median-(std*5):
    #            astrodata[y][x] = median
    
    #sets values 9 SDs above the mean and values less than 0. to 0.
    index = np.where(np.logical_or(astrodata >= median+(9.*std), astrodata < 0.))
    astrodata[index] = 0.
        
    # indices are FIRST Y THEN X
    # smooths out time slices of astrodata (not working)
    #for x in range(len(astrodata)):
    #   bin_median = np.median(astrodata[x])
    #   astrodata[x] = astrodata[x]*(median/bin_median)
            
    
    #set parameters for fixing axes
    t_int=hdulist[0].header['T_INT']#time interval in seconds
    BW=hdulist[0].header['BW'] #bandwidth
    nchans=hdulist[0].header['NAXIS1']#number of channels
    naxis2=hdulist[0].header['NAXIS2'] #dimension of array
    nyq_t=1000./(2.*t_int)
    nyq_f=nchans/(2.*BW) 
    freq = hdulist[0].header['FREQ']
    
    #execute the commands to make spectra
    dyn_xmax,dyn_xmin,dyn_ymax,dyn_ymin,dynamic = makeDynSpec(astrodata,freq,BW)
    fig,secondary,x_min,x_max,y_min,y_max = makeSecSpec(dynamic,astrodata,naxis2,nyq_t,nchans,nyq_f)
    name = psrName
    
    colorMap = raw_input("What color map do you want? (YlOrRd,Greys,hot,Blues,Greens) ")
    
    #makes dynamic graph subplot
    sub=fig.add_subplot(2,1,1) #add plot 1 (dynamic)
    plt.imshow(dynamic,cmap=colorMap,interpolation='none',aspect='auto',extent=[dyn_xmin,dyn_xmax,dyn_ymin,dyn_ymax]) #make image of dynamic spectra
    sub.set_title('Dynamic Spectrum of Pulsar {pulsar}'.format(pulsar=name),fontsize='smaller')
    plt.xlabel('Time (min)')
    plt.ylabel('Frequency (MHz)')
    plt.colorbar()

    #makes secondary graph subplot
    sub=fig.add_subplot(2,1,2) #add plot 2 (secondary)
    plt.imshow(secondary,cmap=colorMap,aspect='auto',interpolation='none',extent=[x_min,x_max,y_min,y_max]) #make image of secondary spectra
    sub.set_title('Secondary Spectrum of Pulsar {pulsar}'.format(pulsar=name),fontsize='smaller')
    plt.xlabel('Fringe Frequency ($10^{-3}$ Hz)')
    plt.ylabel('Delay ($\mu$s)')
    plt.colorbar()
    
    plt.tight_layout() #fixes spacing issues

    plt.show() #display figure 
    #fig.savefig('/Users/lelemathis/Spectra/'+psrName+'.png') #save as png, use instead of plt.show(); for my file system -Lele 
    #plt.close(fig) #closes figure, use when saving figure
    #hdulist.close() #closes fits file
    
def makeDynSpec(astrodata,freq,BW):
    dynamic=np.rot90(astrodata) #rotates 90 degrees
    
    #time normalize
    for i in range(len(dynamic[0])):
        norm_const_t = np.mean(dynamic[:len(dynamic)/4.,i])
        dynamic[:,i] = dynamic[:,i]/norm_const_t
    
    #fix dynamic axes
    dyn_xmax = dynamic.shape[1]/6.#converts dynamic x axis to minutes
    dyn_xmin = 0.
    dyn_ymax = freq+BW/2
    dyn_ymin = freq-BW/2
    return dyn_xmax,dyn_xmin,dyn_ymax,dyn_ymin,dynamic


def makeSecSpec(dynamic,astrodata,naxis2,nyq_t,nchans,nyq_f):
    #dynamic=dynamic-np.mean(dynamic) #cleans up data by subtracting out avg values (appears to not affect spectra)

    sec_init = np.fft.fftn(dynamic) #does fast n-dimensional fourier transform
    secondary = 10*np.log10(abs(np.fft.fftshift(sec_init))**2) #shifts fourier transform, makes log plot

    #secondary_background=np.mean(secondary[:len(secondary)/4.][:len(secondary[0])/4.]) # (appears to not affect spectra)

    #gets dimensions of data array
    ysize=secondary.shape[0]
    xsize=secondary.shape[1]

    #sets min and max of data array to trim it (cuts off the bottom)
    xmin=0
    ymin=0
    xmax=xsize #sets max x value to x dimension of secondary
    ymax=ysize/2 #sets max y value to half of y dimension of secondary

    secondary=secondary[ymin:ymax,xmin:xmax] #sets size of secondary array to specified trimmed dimensions

    #fix secondary axes
    x_axis=np.linspace(-nyq_t,nyq_t,naxis2) #fringe intervals
    x_axis_e4=[int(10000*x) for x in x_axis] #x_axis times 10,000
    axis_x=[x/10000. for x in x_axis_e4]

    y_axis=np.linspace(0,nyq_f,nchans/2.) #delay intervals
    y_axis_e4=[int(10000*y) for y in y_axis][::-1] #[::-1] reverses elements
    axis_y=np.abs([y/10000. for y in y_axis_e4]) #absolute value of y_axis times 10,000 (cuts off the bottom)

    #set min and max of secondary axes
    (x_min,x_max)=(min(axis_x),max(axis_x))
    (y_min,y_max)=(min(axis_y),max(axis_y))

    fig = plt.figure() #make figure
    return fig,secondary,x_min,x_max,y_min,y_max
    
main()