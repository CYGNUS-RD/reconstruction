#!/usr/bin/env python

import numpy as np
import math,itertools
import ROOT
from array import array
from cameraChannel import cameraGeometry

import utilities
utilities = utilities.utils()

class Cluster:
    def __init__(self,hits,rebin,img_fr,img_fr_zs,geometry,debug=False,fullinfo=False,clID=0):
        self.hits = hits
        self.rebin = rebin
        self.debug = debug
        self.x = hits[:, 0]; self.y = hits[:, 1]
        if img_fr.any() and img_fr_zs.any():
            self.hits_fr,self.hits_fr_zs = self.fullResHits(img_fr,img_fr_zs)
        else:
            print("WARNING! Cluster created without underlying image... Are you using it standalone?")

        if fullinfo:			#savin he pixel for the scfullinfo 

            self.nclu = clID
            self.ID=[]
            self.IDall=[]
            #if self.integral()>0 and self.hits_fr_zs != [] and self.size()<1000000:		#tries to avoid to save cluster with zero integral or too big (like with afterglow of pixels)
            if self.integral()>0 and len(self.hits_fr_zs)>0 and self.size()<1000000:		#tries to avoid to save cluster with zero integral or too big (like with afterglow of pixels)
                  self.nintpixels = self.sizeActive()
                  self.nallintpixels = self.size()
                  for k in range(self.nintpixels):
                      self.ID.append(clID)
                  for k in range(self.nallintpixels):
                      self.IDall.append(clID)
                  self.xpixelcoord= self.hits_fr_zs[:,0]
                  self.ypixelcoord= self.hits_fr_zs[:,1]
                  self.zpixel= self.hits_fr_zs[:,2]
                  self.xallpixelcoord= self.hits_fr[:,0]
                  self.yallpixelcoord= self.hits_fr[:,1]
                  self.zallpixel= self.hits_fr[:,2]
            else:								#clusters with ID =-1 can be avoided during analysis of the pixels
                  self.ID.append(-1)
                  self.nintpixels = 1
                  self.IDall.append(-1)
                  self.nallintpixels = 1
                  #if self.hits_fr_zs != []:
                  if len(self.hits_fr_zs)>0:
                      self.xpixelcoord= self.hits_fr_zs[int(self.sizeActive()/2):int(self.sizeActive()/2)+1,0]
                      self.ypixelcoord= self.hits_fr_zs[int(self.sizeActive()/2):int(self.sizeActive()/2)+1,1]
                      self.zpixel= self.hits_fr_zs[int(self.sizeActive()/2):int(self.sizeActive()/2)+1,2]
                  else:
                      self.xpixelcoord= self.hits_fr[int(self.size()/2):int(self.size()/2)+1,0]
                      self.ypixelcoord= self.hits_fr[int(self.size()/2):int(self.size()/2)+1,1]
                      self.zpixel= self.hits_fr[int(self.size()/2):int(self.size()/2)+1,2]
 
                  self.xallpixelcoord= self.hits_fr[int(self.size()/2):int(self.size()/2)+1,0]
                  self.yallpixelcoord= self.hits_fr[int(self.size()/2):int(self.size()/2)+1,1]   
                  self.zallpixel= self.hits_fr[int(self.size()/2):int(self.size()/2)+1,2]
        self.mean_point = np.array([np.mean(self.x),np.mean(self.y)])
        self.EVs,self.theta = self.eigenvectors()
        self.widths = {}
        self.profiles = {}
        self.shapes = {}
        geometryPSet   = open('modules_config/geometry_{det}.txt'.format(det=geometry),'r')
        geometryParams = eval(geometryPSet.read())
        self.cg = cameraGeometry(geometryParams)
        self.minDistKiller = self.cg.npixx
        self.nMatchKiller = 0
        self.nMatchKillerWeak = 0
        
    def integral(self):
        if hasattr(self,'hits_fr'):
            return sum([z for (x,y,z) in self.hits_fr])
        else:
            print("WARNING: Hits with full resolution map not available. Returning 0 integral!")
            return 0
            
    def corr_integral(self):
        e = 1.60217662e-7
        d2 = 0.015625
        omega = 0.00018 
        alpha = 0.08
        sigma0 = 2.5
        a0 = 0.1855
        a = a0*e/(d2*alpha*omega)
        b = (1.- 2*a0*sigma0)
        c = a0*sigma0*sigma0*(d2*alpha*omega)/e
        if hasattr(self,'hits_fr'):
            return sum([a*z*z + b*z +c for (x,y,z) in self.hits_fr])
        else:
            print("WARNING: Hits with full resolution map not available. Returning 0 corr_integral!")
            return 0

    def getSize(self,name='long'):
        if len(self.profiles)==0:
            self.calcProfiles(name)
        if name in self.widths: return self.widths[name]
        else:
            print("ERROR! You can only get 'long' or 'lat' sizes!")
            return -999

    def size(self):
        if hasattr(self,'hits_fr'):
            return len(self.hits_fr)
        else: return 0

    def sizeActive(self):
        if hasattr(self,'hits_fr_zs'):
            return len(self.hits_fr_zs)
        else: return 0

    def iterations(self):
        if hasattr(self,'iteration'):
            return self.iteration
        else: return 0

    def rms(self):
        return np.std(np.array([z for (x,y,z) in self.hits_fr]))
            
    def getXmax(self):
        if hasattr(self,'xmax'):
            return self.xmax
        else: return 0
        
    def getXmin(self):
        if hasattr(self,'xmin'):
            return self.xmin
        else: return 0
        
    def getYmax(self):
        if hasattr(self,'ymax'):
            return self.ymax
        else: return 0
       
    def getYmin(self):
        if hasattr(self,'ymin'):
            return self.ymin
        else: return 0
        
    def getNclu(self):
        if hasattr(self,'nclu'):
            return self.nclu
        else: return 0
        
    def getPearson(self):
        if hasattr(self,'pearson'):
            return self.pearson
        else: return 0
        
    def dump(self):
        if hasattr(self,'hits_fr'):
            print("DUMPING fullres hits")
            print(self.hits_fr)            
        else:
            print("DUMPING rebinned hits in absence of full res ones")
            print(self.hits)

    def dumpToFile(self,filename,zero_suppressed=False):
        if zero_suppressed and hasattr(self,'hits_fr_zs'):
            print("DUMPING zero-suppressed fullres hits to a numpy file: ",filename)
            np.save(filename, self.hits_fr_zs)
        elif hasattr(self,'hits_fr'):
            print("DUMPING fullres hits to a numpy file: ",filename)
            np.save(filename, self.hits_fr)
        else:
            print("DUMPING rebinned hits to a numpy file: ",'rebinned_hits_'+filename)
            np.save(filename, self.hits)

    def eigenvectors(self):
        mat = np.array([self.x,self.y])
        covmat = np.cov(mat.astype(float))
        eig_values, eig_vecs = np.linalg.eig(covmat)
        indexes = (np.argmax(eig_values),np.argmin(eig_values))
        eig_vec_vals = (eig_vecs[:, indexes[0]], eig_vecs[:, indexes[-1]])
        theta = np.degrees(np.arctan2(*eig_vecs[:,0][::-1]))
        return eig_vec_vals,theta

    def plotAxes(self,plot):
        def plot_line(center, dir, num_steps=400, step_size=0.5):
            line_x = []
            line_y = []
            for i in range(num_steps):
                dist_from_center = step_size * (i - num_steps / 2)
                point_on_line = center + dist_from_center * dir
                line_x.append(point_on_line[0])
                line_y.append(point_on_line[1])
            return (line_x, line_y)
        eigen_vectors = self.EVs
        lines = [plot_line(self.mean_point, ev) for ev in eigen_vectors]
        for line in lines:
            plot.plot(line[0], line[1], c="r")


    def fitProfile(self,hist):
        mean = hist.GetMean()
        rms  = hist.GetRMS()

        if hist.Integral()==0:
            ret = {'amp': 0, 'mean': 0, 'sigma': 0, 'chi2': 999, 'status': -1}
            return ret

        f = ROOT.TF1('f','gaus',mean-5*rms,mean+5*rms)
        f.SetParameter(1,mean);
        f.SetParLimits(1,mean-rms,mean+rms);
        f.SetParameter(2,rms);
        f.SetParLimits(2,0.5*rms,1.5*rms);
        fitRe = hist.Fit(f,'SQ')
        rInt   = f.GetParameter(0)
        rMean  = f.GetParameter(1)
        rSigma = f.GetParameter(2)
        if fitRe:
            chi2 = fitRe.Chi2()
            status = fitRe.CovMatrixStatus()
        else:
            chi2 = 999
            status = -1
            rInt = -999
            rMean = -999
            rSigma = -999

        ret = {'amp': rInt, 'mean': rMean, 'sigma': rSigma, 'chi2': chi2, 'status': status}
        del f
        return ret
        
    def calcProfiles(self,name='prof',plot=None):
        # if they have been attached to the cluster, do not recompute them
        if len(self.profiles)>0:
            return

        # rotate the hits of the cluster along the major axis
        rot_hits=[]
        # this is in case one wants to make the profile with a different resolution wrt the clustering
        for h in self.hits_fr:
            rx,ry = utilities.rotate_around_point(h,self.EVs[0],self.mean_point)
            rh_major_axis = (rx,ry,h[-1])
            rot_hits.append(rh_major_axis)

        # now compute the length along major axis, long profile, etc
        rxmin = min([h[0] for h in rot_hits]); rxmax = max([h[0] for h in rot_hits])
        rymin = min([h[1] for h in rot_hits]); rymax = max([h[1] for h in rot_hits])
        xedg = utilities.dynamicProfileBins_v2(rot_hits,'x',relError=0.2)
        yedg = utilities.dynamicProfileBins_v2(rot_hits,'y',relError=0.3)
        xedg = [(x-int(rxmin)) for x in xedg]
        yedg = [(y-int(rymin)) for y in yedg]

        length=(rxmax-rxmin); width=(rymax-rymin)
        if len(xedg)>1:
            longprof = ROOT.TH1F(name+'_long','longitudinal profile',len(xedg)-1,array('f',[x for x in xedg]))
            longprof.SetDirectory(0)
        else: longprof = 0
        if len(yedg)>1:
            latprof = ROOT.TH1F(name+'_lat','lateral profile',len(yedg)-1,array('f',[y for y in yedg]))
            latprof.SetDirectory(0)
        else: latprof = 0
        
        cluth2d = ROOT.TH2D('cluth2d','',int(length)+2,0,int(length)+2, int(width)+2,0,int(width)+2)
        cluth2d.SetDirectory(0)
        for h in rot_hits:
            x,y,z=h[0],h[1],h[2]
            if longprof: longprof.Fill(x-rxmin,z)
            if latprof: latprof.Fill(y-rymin,z)
            # if a neighbor (rounded) has 0 or little, do not kill a good illuminated pixel for that, at a cost of a little shape bias
            cluth2d.SetBinContent(int(np.round(x-rxmin))+1,int(np.round(y-rymin))+1,z)

        profiles = [longprof,latprof]
        titles = ['longitudinal','transverse']
        fitResults = {}
        for ip,p in enumerate(profiles):
            if p:
                #print ("profile entries = ",p.GetEntries())
                p.GetXaxis().SetTitle('X_{%s} (mm)' % titles[ip])
                p.GetYaxis().SetTitle('Number of photons per slice')
                self.applyProfileStyle(p)
                if self.iteration<3:
                    fitResults[titles[ip]] = self.fitProfile(p)
                else:
                    fitResults[titles[ip]] = {'amp': -999, 'mean': -999, 'sigma': -999, 'chi2': -999, 'status': -999}
                    
        # now set the cluster shapes and profiles
        self.profiles['long'] = longprof
        self.profiles['lat'] = latprof
        # those are not used, since they include the "margins" at 0
        # just used as the starting values in clusterShapes()
        self.widths['long'] = length
        self.widths['lat'] = width
        # variances along major/minor axis
        self.shapes['longrms'] = cluth2d.ProjectionX().GetRMS()
        self.shapes['latrms'] = cluth2d.ProjectionY().GetRMS()
        # inclination wrt the vertical
        self.shapes['theta'] = self.theta
        
        if self.integral()<10:
              self.shapes['xmean'] = 0
              self.shapes['ymean'] = 0
              self.shapes['xmin'] = 0
              self.shapes['ymin'] = 0
              self.shapes['xmax'] = 0
              self.shapes['ymax'] = 0
        else:
              self.shapes['xmean'] = np.average(np.array(self.hits_fr[:,0]),weights=np.array([max(0,z) for z in self.hits_fr[:,2]]) )
              self.shapes['ymean'] = np.average(np.array(self.hits_fr[:,1]),weights=np.array([max(0,z) for z in self.hits_fr[:,2]]) )
              self.shapes['xmin'] = np.min(np.array(self.hits_fr[:,0]))
              self.shapes['ymin'] = np.min(np.array(self.hits_fr[:,1]))
              self.shapes['xmax'] = np.max(np.array(self.hits_fr[:,0]))
              self.shapes['ymax'] = np.max(np.array(self.hits_fr[:,1]))
        for direction in titles:
            self.shapes['{direction}gaussamp'.format(direction=direction[0])] = (fitResults[direction])['amp']
            self.shapes['{direction}gaussmean'.format(direction=direction[0])] = (fitResults[direction])['mean']
            self.shapes['{direction}gausssigma'.format(direction=direction[0])] = (fitResults[direction])['sigma']
            self.shapes['{direction}chi2'.format(direction=direction[0])] = (fitResults[direction])['chi2']
            self.shapes['{direction}status'.format(direction=direction[0])] = (fitResults[direction])['status']

        # get the peaks inside the profile
        for direction in ['lat','long']:
            self.clusterShapes(direction,plot)

        del cluth2d, latprof, longprof
        
    def getProfile(self,name='long'):
        if len(self.profiles)==0:
            self.calcProfiles(name=name)
        return self.profiles[name] if name in self.profiles else None

    def clusterShapes(self,name='long',plot=False):
        # ensure the cluster profiles are ready
        if name not in ['lat','long']:
            print("ERROR! Requested profile along the ",name," direction. Should be either 'long' or 'lat'. Exiting clusterShapes().")
            return
        self.getProfile(name)

        from waveform import PeakFinder,simplePeak

        # find first the length/width with intersection of the base of the large peak
        # threshold = 3
        # min_distance_peaks = 5 # number of bins of the profile, to be converted in mm later... TO DO
        # prominence = 2 # noise seems <1
        # width = 10  # find only 1 big peak
        # pf = PeakFinder(self.profiles[name])        
        # pf.findPeaks(threshold,min_distance_peaks,prominence,width)
        # self.widths[name] = pf.getFWHMs()[0] if len(pf.getFWHMs()) else 0 # first should be the only big peak
        self.shapes['%s_width' % name] = self.widths[name]
        
        # find the peaks and store their properties
        # thresholds on the light. Should be configurable...
        threshold = 3
        min_distance_peaks = 6 # number of bins of the profile, to be converted in mm later... TO DO
        prominence = 2 # noise seems <1
        width = 2 # minimal width of the signal
        xmin = 0 # the profile always starts from 0
        xmax = self.profiles[name].GetBinLowEdge(self.profiles[name].GetNbinsX()+1) # low edge of the overflow bin
        pf = PeakFinder(self.profiles[name],xmin=0,xmax=xmax,negative=False)        
        pf.findPeaks(threshold,min_distance_peaks,prominence,width)
        if plot:
            pf.plotpy(xlabel='$X_{%s} (pixels)$' % name, ylabel='Photons / bin')
        
        amplitudes = pf.getAmplitudes()
        prominences = pf.getProminences()
        fwhms = pf.getFWHMs()
        peakPositions = pf.getPeakTimes()
        
        peaksInProfile = [simplePeak(amplitudes[i],prominences[i],peakPositions[i],fwhms[i]) for i in range(len(amplitudes))]
        peaksInProfile = sorted(peaksInProfile, key = lambda x: x.mean, reverse=True)

        self.shapes[name+'_fullrms']          = self.profiles[name].GetRMS()
        if len(peaksInProfile):
            mainPeak = peaksInProfile[0]
            self.shapes[name+'_p0amplitude']  = mainPeak.amplitude
            self.shapes[name+'_p0prominence'] = mainPeak.prominence
            self.shapes[name+'_p0mean']       = mainPeak.mean
            self.shapes[name+'_p0fwhm']       = mainPeak.fwhm
        else:
            self.shapes[name+'_p0amplitude']  = -999
            self.shapes[name+'_p0prominence'] = -999
            self.shapes[name+'_p0mean']       = -999
            self.shapes[name+'_p0fwhm']       = -999
            
    def applyProfileStyle(self,prof):
        prof.SetMarkerStyle(ROOT.kFullCircle)
        prof.SetMarkerSize(1)
        prof.SetMarkerColor(ROOT.kBlack)
        prof.SetLineColor(ROOT.kGray)
        prof.SetLineWidth(1)        
        
    def fullResHits(self,img_fullres,img_fullres_zs):
        if hasattr(self,'hits_fr') and  hasattr(self,'hits_fr_zs'):
            return self.hits_fr,self.hits_fr_zs
        allhits = []
        activehits = []
        if self.debug: print("X rebinned by ",self.rebin," = ",self.hits)
        for X in self.hits:
            for rxf in range(int(X[0]*self.rebin), int((X[0]+1)*self.rebin)):
                for ryf in range(int(X[1]*self.rebin), int((X[1]+1)*self.rebin)):
                    allhits.append((rxf,ryf,img_fullres[rxf,ryf]))
                    # this has the zero-suppression done with the right pixel n*sigma
                    if img_fullres_zs[rxf,ryf]>0:
                        activehits.append((rxf,ryf,img_fullres_zs[rxf,ryf]))
        hits_fr    = np.array(allhits)
        hits_fr_zs = np.array(activehits)
        if self.debug: print("X fullres = ",hits_fr)
        return hits_fr,hits_fr_zs
    
    def plotFullResolution(self,name,option='colz'):

        border = 15
        xmin,xmax = (min(self.hits_fr[:,0])-border, max(self.hits_fr[:,0])+border)
        ymin,ymax = (min(self.hits_fr[:,1])-border, max(self.hits_fr[:,1])+border)
        zmax = max(self.hits_fr[:,2])
        nbinsx = int(xmax-xmin)
        nbinsy = int(ymax-ymin)
        snake_fr = ROOT.TH2D(name,'',nbinsx,xmin,xmax,nbinsy,ymin,ymax)
        for (x,y,z) in self.hits_fr:
            xb = snake_fr.GetXaxis().FindBin(x)
            yb = snake_fr.GetYaxis().FindBin(y)
            snake_fr.SetBinContent(xb,yb,z)
            
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPalette(ROOT.kRainBow)

        cFR = ROOT.TCanvas("cfr","",600,600)
        snake_fr.GetXaxis().SetTitle('x (pixels)')
        snake_fr.GetYaxis().SetTitle('y (pixels)')
        snake_fr.GetZaxis().SetTitle('counts')
        snake_fr.GetXaxis().SetNdivisions(505,ROOT.kTRUE)
        snake_fr.GetYaxis().SetNdivisions(505,ROOT.kTRUE)
        # just for the 2D plotting, cut at 1.5 (mean of the RMS of all the pixels)
        snake_fr.GetZaxis().SetRangeUser(.0,(zmax*1.05))
        snake_fr.Draw(option)
        #print "cluster integral = ",snake_fr.Integral()
        #cFR.SetRightMargin(0.2); cFR.SetLeftMargin(0.1); cFR.SetBottomMargin(0.1);
        cFR.SetBottomMargin(0.3); cFR.SetLeftMargin(0.2); cFR.SetRightMargin(0.2); 
        for ext in ['pdf']:
            cFR.SaveAs('{name}.{ext}'.format(name=name,ext=ext))


    def qualityLevel(self):
        # result: 1=loose, 2=medium, 3=tight, 4=very tight
        kGood = 0
        # sanity (they are not sparse points clustered)
        if self.shapes['lat_p0fwhm']>-999: kGood += 1
        # sphericity
        if self.shapes['lat_width']>0 and self.shapes['long_width']/self.shapes['lat_width']>2.0: kGood += 1
        # minimal length (1 cm for neutrons is good)
        if self.shapes['long_width']>10: kGood += 1
        return kGood
