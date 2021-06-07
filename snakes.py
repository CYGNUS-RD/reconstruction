#!/usr/bin/env python

import numpy as np
import ROOT,math,os,sys,time
import pickle

from scipy.ndimage import gaussian_filter, median_filter
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage import measure

from morphsnakes import(morphological_chan_vese,
                        morphological_geodesic_active_contour,
                        inverse_gaussian_gradient,
                        checkerboard_level_set)

from clusterTools import Cluster
from cameraChannel import cameraTools
from cluster.ddbscan_ import DDBSCAN
from energyCalibrator import EnergyCalibrator

import debug_code.tools_lib as tl

class SnakesFactory:
    def __init__(self,img,img_fr,img_fr_zs,img_ori,vignette,name,options,geometry):
        self.name = name
        self.options = options
        self.rebin = options.rebin
        self.geometry = geometry
        self.ct = cameraTools(geometry)
        self.image = img
        self.img_ori = img_ori
        self.imagelog = np.zeros((self.image.shape[0],self.image.shape[1]))
        for (x,y),value in np.ndenumerate(self.image):
            if value > 3.0/math.sqrt(self.rebin): # tresholding needed for tracking
                self.imagelog[x,y] = math.log(value)
        self.image_fr    = img_fr
        self.image_fr_zs = img_fr_zs
        self.vignette = vignette
        self.contours = []
        
    def getClusters(self,plot=False):

        from sklearn.cluster import DBSCAN
        from sklearn import metrics
        from scipy.spatial import distance
        from scipy.stats import pearsonr
        from random import random

        outname = self.options.plotDir
        if outname and not os.path.exists(outname):
            os.system("mkdir -p "+outname)
            os.system("cp utils/index.php "+outname)
        
        #   Plot parameters  #
        
        vmin=1
        vmax=5
        
        tip = self.options.tip
        
        #-----Pre-Processing----------------#
        rescale=int(self.geometry.npixx/self.rebin)

        t0 = time.perf_counter()
        filtimage = median_filter(self.image_fr_zs, size=2)
        edges = self.ct.arrrebin(filtimage,self.rebin)
        edcopy = edges.copy()
        edcopyTight = tl.noisereductor(edcopy,rescale,self.options.min_neighbors_average)

        # make the clustering with DBSCAN algo
        # this kills all macrobins with N photons < 1
        points = np.array(np.nonzero(np.round(edcopyTight))).astype(int).T
        lp = points.shape[0]

        ## apply vignetting (if not applied, vignette map is all ones)
        ## this is done only for energy calculation, not for clustering (would make it crazy)
        image_fr_vignetted = self.ct.vignette_corr(self.image_fr,self.vignette)
        image_fr_zs_vignetted = self.ct.vignette_corr(self.image_fr_zs,self.vignette)    
        if tip=='3D':
            sample_weight = np.take(self.image, self.image.shape[0]*points[:,0]+points[:,1]).astype(int)
            sample_weight[sample_weight==0] = 1
            X = points.copy()

            
        else:
            X = points.copy()
            sample_weight = np.full(X.shape[0], 1, dtype=np.int)

        # returned collections
        superclusters = []

        # clustering will crash if the vector of pixels is empty (it may happen after the zero-suppression + noise filtering)
        if len(X)==0:
            return superclusters

        if self.options.debug_mode:
            if self.options.flag_dbscan_seeds:
                clusters_seeds = DBSCAN(eps=5.5,min_samples=40).fit(X, sample_weight = sample_weight)
                print('[Plotting 1st iteration]')
     
                import matplotlib.pyplot as plt            
                clu = [X[clusters_seeds.labels_ == i] for i in range(len(set(clusters_seeds.labels_)) - (1 if -1 in clusters_seeds.labels_ else 0))]
                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(self.image,cmap=self.options.cmapcolor, vmin=vmin,vmax=vmax,origin='lower' )
                plt.title("Clusters found DDBSCAN")             
                for j in range(0,np.shape(clu)[0]):
                    ybox = clu[j][:,0]
                    xbox = clu[j][:,1]
                    if (len(ybox) > 0) and (len(xbox) > 0):
                        contours = tl.findedges(ybox,xbox,self.geometry.npixx,self.rebin)
                        for n, contour in enumerate(contours):
                            plt.plot(contour[:, 1],contour[:, 0], '-r',linewidth=2.5)
     
     
                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}_{tip}.{ext}'.format(pdir=outname, name=self.name, esp='1st', ext=ext, tip=self.options.tip), bbox_inches='tight', pad_inches=0)
                    

        # - - - - - - - - - - - - - -
        t1 = time.perf_counter()
        ddb = DDBSCAN('modules_config/clustering.txt').fit(X, sample_weight = sample_weight)
        if self.options.debug_mode: print(f"basic clustering in {t1 - t0:0.4f} seconds")
        t2 = time.perf_counter()
        if self.options.debug_mode: print(f"ddbscan clustering in {t2 - t1:0.4f} seconds")
        # Black removed and is used for noise instead.
        unique_labels = set(ddb.labels_[:,0])

        # Number of polynomial clusters in labels, ignoring noise if present.
        n_superclusters = len(unique_labels) - (1 if -1 in ddb.labels_[:,0] else 0)
        
        for k in unique_labels:
            if k == -1:
                break # noise: the unclustered

            class_member_mask = (ddb.labels_[:,0] == k)
            xy = np.unique(X[class_member_mask],axis=0)
            x = xy[:, 0]; y = xy[:, 1]
            
            # both core and neighbor samples are saved in the cluster in the event
            if k>-1 and len(x)>1:
                cl = Cluster(xy,self.rebin,image_fr_vignetted,image_fr_zs_vignetted,self.options.geometry,debug=False)
                cl.iteration = 0
                cl.nclu = k
                cl.pearson = 999#p_value
                superclusters.append(cl)
                
        t2 = time.perf_counter()
        if self.options.debug_mode: print(f"label basic clusters in {t2 - t1:0.4f} seconds")

        ## DEBUG MODE
        if self.options.debug_mode == 1:
            print('[DEBUG-MODE ON]')
            print('[%s Method]' % (self.options.tip))

            #if self.options.flag_full_image or self.options.flag_rebin_image or self.options.flag_edges_image or self.options.flag_first_it or self.options.flag_second_it or self.options.flag_third_it or self.options.flag_all_it or self.options.flag_supercluster :
            import matplotlib.pyplot as plt

            if self.options.flag_full_image == 1:
                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(np.flipud(self.image_fr_zs),cmap=self.options.cmapcolor, vmin=vmin, vmax=vmax,origin='upper' )
                plt.title("Original Image")
                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}.{ext}'.format(pdir=outname,name=self.name,esp='oriIma',ext=ext), bbox_inches='tight', pad_inches=0)
                with open('{pdir}/{name}_{esp}.pkl'.format(pdir=outname,name=self.name,esp='oriIma',ext=ext), "wb") as fp:
                    pickle.dump(fig, fp, protocol=4)
                plt.gcf().clear()
                plt.close('all')
                
            if self.options.flag_rebin_image == 1:
                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(self.image,cmap=self.options.cmapcolor, vmin=1, vmax=vmax, origin='lower' )
                plt.title("Rebin Image")
                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}.{ext}'.format(pdir=outname,name=self.name,esp='rebinIma',ext=ext), bbox_inches='tight', pad_inches=0)
                plt.gcf().clear()
                plt.close('all')
                
            if self.options.flag_edges_image == 1:
                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(edcopyTight, cmap=self.options.cmapcolor, vmin=0, vmax=1, origin='lower' )
                plt.title('Edges after Filtering')
                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}.{ext}'.format(pdir=outname,name=self.name,esp='edgesIma',ext=ext), bbox_inches='tight', pad_inches=0)
                plt.gcf().clear()
                plt.close('all')
                
            if self.options.flag_stats == 1:
                print('[Statistics]')
                print("Polynomial clusters found: %d" % n_superclusters)
                


            if self.options.flag_polycluster == 1:
                print('[Plotting 0th iteration]')
                u,indices = np.unique(ddb.labels_,return_index = True)
                clu = [X[ddb.labels_[:,0] == i] for i in np.unique(ddb.labels_[:,0]) if i != -1]
                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(self.image,cmap=self.options.cmapcolor,vmin=vmin, vmax=vmax,origin='lower' )
                plt.title("Polynomial clusters found in iteration 0")
                colorpix = np.zeros([rescale,rescale,3])
                for j in range(0,np.shape(clu)[0]):

                    a = np.random.rand(3)
                    colorpix[clu[j][:,0],clu[j][:,1]] = a
                    
                plt.imshow(colorpix,cmap='gray',origin='lower' )
                for ext in ['png','pdf']:
                    plt.savefig('{pdir}/{name}_{esp}_{tip}.{ext}'.format(pdir=outname, name=self.name, esp='0th', ext=ext, tip=self.options.tip), bbox_inches='tight', pad_inches=0)
                with open('{pdir}/{name}_{esp}_{tip}.pkl'.format(pdir=outname,name=self.name,esp='0th',ext=ext,tip=self.options.tip), "wb") as fp:
                    pickle.dump(fig, fp, protocol=4)
                plt.gcf().clear()
                plt.close('all')
                

                plt.gcf().clear()
                plt.close('all')

        return superclusters
        
    def getTracks(self,plot=True):
        from skimage.transform import (hough_line, hough_line_peaks)
        # Classic straight-line Hough transform
        image = self.imagelog
        h, theta, d = hough_line(image)
        print("tracks found")
        
        tracks = []
        thr = 0.8 * np.amax(h)
        #######################   IMPLEMENT HERE THE SAVING OF THE TRACKS ############
        # loop over prominent tracks
        itrk = 0
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d,threshold=thr)):
            print("Track # ",itrk)
            #points_along_trk = np.zeros((self.image.shape[1],self.image.shape[0]))
            points_along_trk = []
            for x in range(self.image.shape[1]):
                y = min(self.image.shape[0],max(0,int((dist - x * np.cos(angle)) / np.sin(angle))))
                #points_along_trk[x,y] = self.image[y,x]
                #print "adding point: %d,%d,%f" % (x,y,self.image[y,x])
                # add a halo fo +/- 20 pixels to calculate the lateral profile
                for iy in range(int(y)-5,int(y)+5):
                    if iy<0 or iy>=self.image.shape[0]: continue
                    points_along_trk.append((x,iy,self.image[iy,x]))
            xy = np.array(points_along_trk)
            trk = Cluster(xy,self.rebin)
            tracks.append(trk)
            itrk += 1
        ###################################
            
        if plot:
            # Generating figure
            from matplotlib import cm
            fig, ax = plt.subplots(2, 1, figsize=(18, 6))
            #ax = axes.ravel()

            ax[0].imshow(image, cmap=cm.gray)
            ax[0].set_title('Camera image')
            #ax[0].set_axis_off()            

            ax[1].imshow(image, cmap=cm.gray)
            for _, angle, dist in zip(*hough_line_peaks(h, theta, d,threshold=thr)):
                y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
                y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
                ax[1].plot((0, image.shape[1]), (y0, y1), '-r')
            ax[1].set_xlim((0, image.shape[1]))
            ax[1].set_ylim((image.shape[0], 0))
            #ax[1].set_axis_off()
            ax[1].set_title('Fitted tracks')

            plt.tight_layout()
            #plt.show()
            outname = self.options.plotDir
            if outname and not os.path.exists(outname):
                os.system("mkdir -p "+outname)
                os.system("cp ~/cernbox/www/Cygnus/index.php "+outname)
            for ext in ['pdf']:
                plt.savefig('{pdir}/{name}.{ext}'.format(pdir=outname,name=self.name,ext=ext))
            plt.gcf().clear()

        return tracks
        
    def plotClusterFullResolution(self,clusters):
        outname = self.options.plotDir
        for k,cl in enumerate(clusters):
            cl.plotFullResolution('{pdir}/{name}_cluster{iclu}'.format(pdir=outname,name=self.name,iclu=k))

    def calcProfiles(self,clusters,plot=False):
        for k,cl in enumerate(clusters):
            profName = '{name}_cluster{iclu}'.format(name=self.name,iclu=k)
            cl.calcProfiles(name=profName,plot=plot)
                             
    def plotProfiles(self,clusters):
        print ("plot profiles...")
        outname = self.options.plotDir
        canv = ROOT.TCanvas('c1','',1200,600)
        for k,cl in enumerate(clusters):
            for dir in ['long','lat']:
                profName = '{name}_cluster{iclu}_{dir}'.format(name=self.name,iclu=k,dir=dir)
                prof = cl.getProfile(dir)
                if prof and cl.widths[dir]>0.2: # plot the profiles only of sufficiently long snakes (>200 um)
                    prof.Draw("pe1")
                    for ext in ['pdf']:
                        canv.SaveAs('{pdir}/{name}profile.{ext}'.format(pdir=outname,name=profName,ext=ext))

class SnakesProducer:
    def __init__(self,sources,params,options,geometry):
        self.picture     = sources['picture']     if 'picture' in sources else None
        self.pictureHD   = sources['pictureHD']   if 'pictureHD' in sources else None
        self.picturezsHD = sources['picturezsHD'] if 'picturezsHD' in sources else None
        self.pictureOri  = sources['pictureOri']  if 'pictureOri' in sources else None
        self.vignette    = sources['vignette']    if 'vignette' in sources else None
        self.name        = sources['name']        if 'name' in sources else None
        self.algo        = sources['algo']        if 'algo' in sources else 'DBSCAN'
        
        self.snakeQualityLevel = params['snake_qual']   if 'snake_qual' in params else 3
        self.plot2D            = params['plot2D']       if 'plot2D' in params else False
        self.plotpy            = params['plotpy']       if 'plotpy' in params else False
        self.plotprofiles      = params['plotprofiles'] if 'plotprofiles' in params else False

        self.options = options
        self.geometry = geometry
        geometryPSet   = open('modules_config/geometry_{det}.txt'.format(det=options.geometry),'r')
        geometryParams = eval(geometryPSet.read())

        self.run_cosmic_killer = self.options.cosmic_killer
        if self.run_cosmic_killer:
            from clusterMatcher import ClusterMatcher
            # cosmic killer parameters
            cosmicKillerPars = open('modules_config/clusterMatcher.txt','r')
            killer_params = eval(cosmicKillerPars.read())
            killer_params.update(geometryParams)
            self.cosmic_killer = ClusterMatcher(killer_params)

        
    def run(self):
        ret = []
        if any([x==None for x in (self.picture.any(),self.pictureHD.any(),self.picturezsHD.any(),self.name)]):
            return ret

        t0 = time.perf_counter()
        
        # Cluster reconstruction on 2D picture
        snfac = SnakesFactory(self.picture,self.pictureHD,self.picturezsHD,self.pictureOri,self.vignette,self.name,self.options,self.geometry)

        # this plotting is only the pyplot representation.
        # Doesn't work on MacOS with multithreading for some reason... 
        if self.algo=='DBSCAN':
            snakes = snfac.getClusters(plot=self.plotpy)

            # supercluster energy calibration for the saturation effect
            fileCalPar = open('modules_config/energyCalibrator.txt','r')
            params = eval(fileCalPar.read())
            calibrator = EnergyCalibrator(params,self.options.debug_mode)
            
            for sclu in snakes:
                if self.options.calibrate_clusters:
                    calEnergy,slicesCalEnergy,centers = calibrator.calibratedEnergy(sclu.hits_fr)
                else:
                    calEnergy,slicesCalEnergy,centers = -1,[],[]
                if self.options.debug_mode:
                    print ( "SUPERCLUSTER BARE INTEGRAL = {integral:.1f}".format(integral=sclu.integral()) )
                sclu.calibratedEnergy = calEnergy
                sclu.nslices = len(slicesCalEnergy)
                sclu.energyprofile = slicesCalEnergy
                sclu.centers = centers
                sclu.pathlength = -1 if self.options.calibrate_clusters==False else calibrator.clusterLength()    
            
        elif self.algo=='HOUGH':
            clusters = []
            snakes = snfac.getTracks(plot=self.plotpy)            
        t1 = time.perf_counter()
        if self.options.debug_mode: print(f"FULL RECO in {t1 - t0:0.4f} seconds")
            
        # print "Get light profiles..."
        snfac.calcProfiles(snakes,plot=self.plotpy)
        t2 = time.perf_counter()
        if self.options.debug_mode: print(f"cluster shapes in {t2 - t1:0.4f} seconds")

        # run the cosmic killer: it makes sense only on superclusters
        if self.run_cosmic_killer:
            for ik,killerCand in enumerate(snakes):
                targets = [snakes[it] for it in range(len(snakes)) if it!=ik]
                self.cosmic_killer.matchClusters(killerCand,targets)
            t3 = time.perf_counter()
            if self.options.debug_mode: print(f"cosmic killer in {t3 - t2:0.4f} seconds")

        # snfac.calcProfiles(snakes) # this is for BTF
        
        # sort snakes by light integral
        snakes = sorted(snakes, key = lambda x: x.integral(), reverse=True)
        # and reject discharges (round)
        #snakes = [x for x in snakes if x.qualityLevel()>=self.snakeQualityLevel]
        
        # plotting
        if self.plot2D:       snfac.plotClusterFullResolution(snakes)
        if self.plotprofiles: snfac.plotProfiles(snakes)

        return snakes
