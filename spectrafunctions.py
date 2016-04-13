#Lele Mathis and Adam Jussila
#This program is used to take multiple fits files of a selected directory and
#save the a dynamic and secondary spectra of each pulsar as a png.
#4/12/16

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


#takes one pulsar name and (optional) number of bins for secondary background removal
#displays dynamic and secondary spectra figure
#returns dynamic and secondary spectra arrays
def single(name,bins=25,save=False):
    psrName = name
    colorMap = 'Greys'
    hdulist = fits.open('/Users/lelemathis/Spectra/PSR'+psrName+'_dyn.fits') #For my file system -Lele
    dyn,sec = makeSpectra(hdulist,psrName, colorMap,bins=25,save = save)
    return dyn,sec

#takes (optional) number of bins
#saves all spectra with names in pulsarnames.txt file to specified file location as pngs    
def multiple(bins=25):
    cmap = raw_input("What color map do you want? (YlOrRd,Greys,hot,Blues,Greens) ") #asks for colormap for all spectra
    namefile = open("/Users/lelemathis/greenbank/pulsarnames.txt", "r")#open file with list of pulsar names
    psrNames = [] #make array to put pulsar names in
    for name in namefile.readlines():
        psrNames.append(name.strip()) #removes whitespace from each line of file
    
    #for each pulsar name, calls saveSpectra to make the dyn and sec spectra figure and save them as a png
    for i in range(0,len(psrNames)):
        makeSpectra(fits.open('/Users/lelemathis/Spectra/PSR'+psrNames[i]+'_dyn.fits'),psrNames[i], cmap,bins,save=True) #passes the data in as astrodata, pulsar name as psrName
    
def hist(secondary, nbins=25):
    plt.hist(secondary, bins=nbins)
    plt.show()

#takes the HDUlist, pulsar name, colormap, number of bins, and (optional) whether the spectrum is saved or shown
def makeSpectra(hdulist, psrName, colorMap, bins, save=False): #the main function of makespectra2
    
    astrodata = hdulist[0].data
    median = np.median(astrodata)
    std = np.std(astrodata-median)
    
    #sets values 9 SDs above the mean and values less than 0 to 0
    index = np.where(np.logical_or(astrodata >= median+(6.*std), astrodata < 0.))
    astrodata[index] = 0.
    
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
    
    ##histDyn = np.histogram(dynamic,bins)
    #binsize = (np.max(dynamic)-np.min(dynamic))/bins
    #threshold = np.min(dynamic)+binsize
    #index2 = np.where(dynamic>threshold)
    #dynamic[index2]=0
    
    secondary,x_min,x_max,y_min,y_max = makeSecSpec(dynamic,naxis2,nyq_t,nchans,nyq_f,bins)
    name = psrName
    
    #time normalize dynamic -- from psr.get_secondary_spectra
    for i in range(len(dynamic[0])):
      norm_const_t = np.mean(dynamic[:len(dynamic),i])
      dynamic[:,i] = dynamic[:,i]/norm_const_t  

    #remove secondary background -- from psr.get_secondary_spectra
    #secondary_background = np.mean(secondary[:len(secondary)/4.][:len(secondary[0])/4.])
    #secondary = secondary - secondary_background
    
    fig = plt.figure()
    
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
    
    if save:
        fig.savefig('/Users/lelemathis/Spectra/'+psrName+'.png') #save as png, use instead of plt.show(); for my file system -Lele 
        plt.close(fig) #closes figure, use when saving figure
    #hdulist.close() #closes fits file
    else:
        plt.show()
        return dynamic,secondary
    
    #hdulist.close() #closes fits file
    return dynamic, secondary
    
#makes dynamic spectra, time normalizes it, and sets bounds for axes    
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


def makeSecSpec(dynamic,naxis2,nyq_t,nchans,nyq_f,bins):
    #dynamic=dynamic-np.mean(dynamic) #cleans up data by subtracting out avg values (appears to not affect spectra)

    sec_init = np.fft.fftn(dynamic) #does fast n-dimensional fourier transform
    sec_init = abs(np.fft.fftshift(sec_init))**2
    secondary = 10*np.log10(sec_init/np.max(sec_init)) #shifts fourier transform, makes log plot

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
    
    secondary = remove_sec_background(secondary,bins)
    
    return secondary,x_min,x_max,y_min,y_max

#sets all values below the peak of the noise (in dB) to the peak of the noise
def remove_sec_background(secondary,nbins):
    histSec = np.histogram(secondary,bins=nbins)
  
    binsize = (np.max(secondary)-np.min(secondary))//nbins
    
    maxindex = np.where(histSec[0]==np.max(histSec[0])) #where frequency of occurences is highest
    
    xVal = int(maxindex[0]) #position of peak in noise
    #print xVal
    
    xValDb = np.min(secondary)+binsize*xVal+3. #value of peak in noise, offsetting by 3dB above
    #print xValDb
    
    index = np.where(secondary<xValDb) 
    
    if index!=-1: #if there are values less than threshold value
        secondary[index] = xValDb #set bottom of color table instead?
        
    return secondary
