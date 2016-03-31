import matplotlib.pyplot as plt
#import pylab
import matplotlib.pyplot as pyplot
from astropy.io import fits
import numpy as np
import os
from functools import partial
import pickle
import sys
import time
import logging
import multiprocessing as mp
import pulsar
import pulsar as psr
#logger = mp.log_to_stderr(level=mp.SUBDEBUG)

class Secondary():
    def __init__(self,filename,hand=None):
        data = psr.get_secondary_spectrum(psr.get_dynamic_spectrum(filename))
        axes = pulsar.get_sec_axes(filename)
        #print(type(axes[0]),type(axes[1]))
        self.sec = psr.Indexed2D(data=data,axes=axes)
        self.hand=hand
        self.made_1D=False
        self.parabola_power = {}
        self.observation_name = os.path.basename(filename)
        self.band = self.observation_name.split("_")[1].split("M")[0]
    
    def __getitem__(self,value):
        return self.sec[value]
    
    def get(self,value):
        return self.sec.get_data().item(tuple(value))
    
    def get_y_axis(self):
        return self.sec.y_axis
    
    def get_x_axis(self):
        return self.sec.x_axis
    
    def crop_percent(self,y_scale,x_scale):
        y_scale = float(y_scale)
        x_scale = float(x_scale)
        if y_scale<0 or x_scale<0 or y_scale>1 or x_scale>1:
            raise ValueError('x_scale and y_scale must be between 0 and 1.')
        y_max = max(self.sec.get_y_axis())
        x_max = max(self.sec.get_x_axis())
        self.sec = self.sec[y_max*y_scale:,-x_max*x_scale:x_max*x_scale]
        return
    
    def crop(self,x_lim,y_lim):
        self.sec = self.sec[y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
    
    def get_sec(self):
        return self.sec.get_data()
    
    def show_sec(self):
        pulsar.show_image(self.get_sec(),self.get_y_axis(),self.get_x_axis())
        if self.made_1D:
            self.overplot_parabolas([min(self.etas),max(self.etas)])
        plt.title(self.observation_name)
        plt.xlabel('delay')
        plt.ylabel('fringe frequency')
        return
    
    def overplot_parabolas(self, etas, offsets = [0.]):
        for eta in etas:
            for offset in offsets:
                eta = float(eta)
                axis_x = self.get_x_axis()
                plot_x = [x+offset for x in axis_x]
                axis_y = self.get_y_axis()
                parab = []
                for x in axis_x:
                    y = eta*x**2 - eta*offset**2
                    parab.append(y)
                plt.plot(plot_x, parab, 'b-')
                plt.xlim((min(axis_x),max(axis_x)))
                plt.ylim((min(axis_y),max(axis_y)))
    
    """
    
    def overplot_parabolas(self, etas, offset = 0):
        for eta in etas:
            eta = float(eta)
            axis_x = self.get_x_axis()
            axis_y = self.get_y_axis()
            left_parab = []
            right_parab = []
            midpoint_x = int((len(axis_x)+1)/2)
            if self.hand=='left' or self.hand is None:
                left_x = axis_x[:midpoint_x]
                left_y = []
                for x in left_x:
                    y = eta*x**2
                    left_y.append(y)
                left_parab.append(left_y)
                plt.plot(axis_x[:midpoint_x], left_y, 'b-')
            if self.hand=='right' or self.hand is None:
                right_x = axis_x[midpoint_x:]
                right_y = []
                for x in right_x:
                    y = eta*x**2
                    right_y.append(y)
                right_parab.append(right_y)
                plt.plot(axis_x[midpoint_x:], right_y, 'b-')
            plt.xlim((min(axis_x),max(axis_x)))
            plt.ylim((min(axis_y),max(axis_y)))
    
    """
    
    def show_power_vs_eta(self,weird=False):
        if not self.made_1D:
            print("make_1D_by_quadratic has not been run yet")
            return

        if not weird:
            plt.plot(self.etas,self.powers)
            plt.xlabel("eta")
            plt.ylabel("Power(dB), arbitrary scaling")
            plt.title("Power vs eta, " + self.observation_name)
            self.overplot_parabolas([min(sec.etas),max(sec.etas)])
            return
        else:
            fig = plt.figure()
            fig.subplots_adjust(bottom=0.2)
            plt.plot([1/eta**2 for eta in self.etas],self.powers)
            
            def OnClick(event):
                print('eta: ',1/np.sqrt(event.xdata))
            cid_up = fig.canvas.mpl_connect('button_press_event', OnClick)
            
            x_axis_points = np.linspace(1/max(self.etas)**2,1/min(self.etas)**2,10)
            x_axis = [round(1/np.sqrt(x),4) for x in x_axis_points]
            plt.xticks(x_axis_points,x_axis,rotation=90)
            plt.xlabel("eta")
            plt.ylabel("Power(dB), arbitrary scaling")
            plt.title("Power vs eta, " + self.observation_name)
            #self.overplot_parabolas([min(self.etas),max(self.etas)])
            return
        
    def __give_eta_list(self,eta_range,num_etas,decimal_places=4):
        if num_etas is not 1:
            x_max = np.sqrt(1/min(eta_range))
            x_min = np.sqrt(1/max(eta_range))
            return [1/x**2 for x in np.linspace(x_min,x_max,num_etas)]
        else:
            return [np.average(eta_range)]
    
    def make_1D_by_quadratic(self,eta_range,num_etas,num_threads=mp.cpu_count()-1,sigma=None):
        if num_threads == 0:
            num_threads = 1
        print("num threads: " + str(num_threads))
        print(self.observation_name)
        
        etas = self.__give_eta_list(eta_range,num_etas)
        
        pool = mp.Pool(processes=num_threads)
        output = pool.map(partial(pulsar.crunchy, sec=self, hand=self.hand, sigma=sigma), etas)
        
        powers = {}
        for item in output:
            powers[item[0]] = item[1]
        
        ret = psr.sort_dict_by_key(powers)
        self.made_1D = True
        self.etas = ret[0]
        self.powers = ret[1]
        return ret
    
    def power_along_parabola(self,eta,num_arclets = 100,num_threads=mp.cpu_count()-1,sigma_px=3):
        if num_threads == 0:
            num_threads = 1
        print("num threads: " + str(num_threads))
        eta = float(eta)
        max_x = np.sqrt(max(self.sec.get_y_axis())/eta)
        max_possible_x = np.absolute(max(self.sec.get_x_axis()))
        if max_x>max_possible_x:
            max_x = max_possible_x
        
        y_axis = self.get_y_axis()
        x_axis = self.get_x_axis()
        
        px_y = np.absolute(y_axis[1]-y_axis[0])
        px_x = np.absolute(x_axis[1]-x_axis[0])
        
        def dist_bw_pts(pt1,pt2):
            y1 = pt1[0]
            y2 = pt2[0]
            x1 = pt1[1]
            x2 = pt2[1]
            return np.sqrt( np.absolute(y1-y2)**2 + np.absolute(x1-x2)**2 )

        
        temp = [max_x*x**2 for x in np.linspace(0,1,num_arclets/2)]
        x_list = [-x for x in list(reversed(temp))[:-1]]
        x_list.extend(temp)
        y_list = [eta*x**2 for x in x_list]
        pts = [(y_list[i],x_list[i]) for i in range(len(x_list))]
        
        sigmas = []
        for i in range(len(pts)):
            if i == 0:
                sigmas.append( [np.absolute(pts[1][0]-pts[0][0]),np.absolute(pts[1][1]-pts[0][1])] )
            elif i == len(pts)-1:
                sigmas.append( [np.absolute(pts[-1][0]-pts[-2][0]),np.absolute(pts[-1][1]-pts[-2][1])] )
            else:
                sigma_y = px_y*sigma_px
                sigma_x = px_x*sigma_px
                sigmas.append( [sigma_y,sigma_x] )
        
        
        pts_and_sigmas = []
        for i in range(len(sigmas)):
            pts_and_sigmas.append( (pts[i],sigmas[i]) )
        
        pool = mp.Pool(processes=num_threads)
        output = pool.map(partial(pulsar.crunchy2, sec=self, hand=self.hand), pts_and_sigmas)
        
        powers = {}
        for item in output:
            powers[item[0]] = item[1]
        
        self.parabola_power[eta] = powers
        return powers
        
    def parabola_width(self,eta,max_width,num_offsets,num_threads=mp.cpu_count()-1):
        if num_threads == 0:
            num_threads = 1
        print("num threads: " + str(num_threads))
        print(self.observation_name)
        
        temp = [max_width*np.sqrt(x) for x in np.linspace(0,1,num_offsets/2)]
        offsets = [-x for x in list(reversed(temp))[:-1]]
        offsets.extend(temp)
        
        print(offsets)
        
        pool = mp.Pool(processes=num_threads)
        output = pool.map(partial(pulsar.crunchy3, sec=self, eta=eta), offsets)
        
        powers = {}
        for item in output:
            powers[item[0]] = item[1]
        
        ret = psr.sort_dict_by_key(powers)
        self.offsets = ret[0]
        self.offset_powers = ret[1]
        return ret