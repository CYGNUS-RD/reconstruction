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
from cython_cygno import nred_cython
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

        lp, t_medianfilter, t_noisered, t_DBSCAN = 0, 0, 0, 0
        
        tip = self.options.tip
        
        #-----Pre-Processing----------------#
        rescalex=int(self.geometry.npixx/self.rebin)
        rescaley=int(self.geometry.npixy/self.rebin)

        t0 = time.perf_counter()
        filtimage = median_filter(self.image_fr_zs, size=2)
        t1_med = time.perf_counter()
        edges = self.ct.arrrebin(filtimage,self.rebin)
        edcopy = edges.copy()
        t0_noise = time.perf_counter()
        edcopyTight = nred_cython(edcopy, rescalex, rescaley, self.options.min_neighbors_average)
        t1_noise = time.perf_counter()

        t_medianfilter = t1_med - t0
        t_noisered = t1_noise - t0_noise

        
        # make the clustering with DBSCAN algo
        # this kills all macrobins with N photons < 1
        points = np.array(np.nonzero(np.round(edcopyTight))).astype(int).T
        lp = points.shape[0]

        ## apply vignetting (if not applied, vignette map is all ones)
        ## this is done only for energy calculation, not for clustering (would make it crazy)
        image_fr_vignetted = self.ct.vignette_corr(self.image_fr,self.vignette)
        image_fr_zs_vignetted = self.ct.vignette_corr(self.image_fr_zs,self.vignette)    
        if tip=='3D':
            sample_weight = self.image[points[:,0], points[:,1]].astype(int)
            sample_weight[sample_weight==0] = 1
            X = points.copy()
            save_arr = False
            if save_arr:
                import re
                items = re.split('_+', self.name)
                ev_number = re.sub('ev', '', items[2])
                np.save('../Cython-test/Events/' + items[1] + '/data' + ev_number, X)
                np.save('../Cython-test/Events/' + items[1] + '/sample' + ev_number, sample_weight)
            
        else:
            X = points.copy()
            sample_weight = np.full(X.shape[0], 1, dtype=np.int)

        # returned collections
        superclusters = []

        # clustering will crash if the vector of pixels is empty (it may happen after the zero-suppression + noise filtering)
        if len(X)==0:
            return superclusters, lp, t_medianfilter, t_noisered, t_DBSCAN 

        if self.options.debug_mode:
            if self.options.flag_dbscan_seeds:
                #reading params of dbscan seeding
                filePar = open('modules_config/clustering.txt','r')
                params = eval(filePar.read())
                seed_eps = params['dbscan_eps']
                seed_mpts = params['dbscan_minsamples']
                seed_metric = params['metric']
                seed_mp = params['metric_params']
                seed_algo = params['algorithm']
                seed_ls = params['leaf_size']
                seed_p = params['p']
                seed_njobs = params['n_jobs']
                
                #starting the seed clustering for plot
                time0 = time.perf_counter()
                clusters_seeds = DBSCAN(eps=seed_eps,min_samples=seed_mpts, metric=seed_metric, metric_params=seed_mp, algorithm=seed_algo, leaf_size=seed_ls, p=seed_p, n_jobs=seed_njobs).fit(X, sample_weight = sample_weight)
                time_seeds = time.perf_counter()
                print('DBSCAN time = ' + str(time_seeds - time0))
                print('[Plotting dbscan seeding]')
     
                import matplotlib.pyplot as plt            
                clu = [X[clusters_seeds.labels_ == i] for i in range(len(set(clusters_seeds.labels_)) - (1 if -1 in clusters_seeds.labels_ else 0))]
                if True:
                    fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                    plt.imshow(self.image,cmap=self.options.cmapcolor,vmin=vmin, vmax=vmax,origin='upper' )
                    plt.title("Clusters found in the DBSCAN seeding")
                    colorpix = np.zeros([rescaley,rescalex,3])
                    for j in range(0,len(clu)):

                        a = np.random.rand(3)
                        colorpix[clu[j][:,0],clu[j][:,1]] = a

                    plt.imshow(colorpix,cmap='gray',origin='upper' )
                    for ext in ['png']:
                        plt.savefig('{pdir}/{name}_{esp}_{tip}.{ext}'.format(pdir=outname, name=self.name, esp='seeding', ext=ext, tip=self.options.tip), bbox_inches='tight', pad_inches=0)


                    plt.gcf().clear()
                    plt.close('all')
                
                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(self.image,cmap=self.options.cmapcolor, vmin=vmin,vmax=vmax,origin='upper' )
                plt.title("Clusters found DDBSCAN")             
                for j in range(0,len(clu)):
                    ybox = clu[j][:,0]
                    xbox = clu[j][:,1]
                    if (len(ybox) > 0) and (len(xbox) > 0):
                        contours = tl.findedges(ybox,xbox,self.geometry.npixx,self.rebin)
                        for n, contour in enumerate(contours):
                            plt.plot(contour[:, 1],contour[:, 0], '-r',linewidth=2.5)
     
     
                for ext in ['png']:       #,'pdf'
                    plt.savefig('{pdir}/{name}_{esp}_{tip}.{ext}'.format(pdir=outname, name=self.name, esp='1st', ext=ext, tip=self.options.tip), bbox_inches='tight', pad_inches=0)
                    

        # - - - - - - - - - - - - - -
        if self.options.debug_mode: print ("starting DBscan")
        t1 = time.perf_counter()
        ddb = DDBSCAN('modules_config/clustering.txt').fit(X, sample_weight = sample_weight)

        if self.options.debug_mode: print(f"basic clustering in {t1 - t0:0.4f} seconds")
        t2 = time.perf_counter()
        if self.options.debug_mode: print(f"ddbscan clustering in {t2 - t1:0.4f} seconds")

        t_DBSCAN = t2-t1
        
        # Black removed and is used for noise instead.
        unique_labels = set(ddb.labels_[:,0])

        # Number of polynomial clusters in labels, ignoring noise if present.
        n_superclusters = len(unique_labels) - (1 if -1 in ddb.labels_[:,0] else 0)

        for k in unique_labels:
            if k == -1:
                break # noise: the unclustered

            class_member_mask = (ddb.labels_[:,0] == k)
            #class_member_mask = (ddb.labels_ == k)
            xy = np.unique(X[class_member_mask],axis=0)
            x = xy[:, 0]; y = xy[:, 1]
            
            
            # both core and neighbor samples are saved in the cluster in the event
            if k>-1 and len(x)>1:
                cl = Cluster(xy,self.rebin,image_fr_vignetted,image_fr_zs_vignetted,self.options.geometry,debug=False,fullinfo=self.options.scfullinfo,clID=k)
                cl.iteration = 0
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
                plt.imshow(self.image_fr_zs,cmap=self.options.cmapcolor, vmin=vmin, vmax=vmax,origin='upper' )
                plt.title("Original Image")
                for ext in ['png']:       #,'pdf'
                    plt.savefig('{pdir}/{name}_{esp}.{ext}'.format(pdir=outname,name=self.name,esp='oriIma',ext=ext), bbox_inches='tight', pad_inches=0)
                if self.options.pkl_ext == 1:
                    with open('{pdir}/{name}_{esp}.pkl'.format(pdir=outname,name=self.name,esp='oriIma',ext=ext), "wb") as fp:
                        pickle.dump(fig, fp, protocol=4)
                plt.gcf().clear()
                plt.close('all')
                
            if self.options.flag_rebin_image == 1:
                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(self.image,cmap=self.options.cmapcolor, vmin=vmin, vmax=vmax, origin='upper' )
                plt.title("Rebin Image")
                for ext in ['png']:       #,'pdf'
                    plt.savefig('{pdir}/{name}_{esp}.{ext}'.format(pdir=outname,name=self.name,esp='rebinIma',ext=ext), bbox_inches='tight', pad_inches=0)
                plt.gcf().clear()
                plt.close('all')
                
            if self.options.flag_edges_image == 1:
                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(edcopyTight, cmap=self.options.cmapcolor, vmin=0, vmax=1, origin='upper' )
                plt.title('Edges after Filtering')
                for ext in ['png']:       #,'pdf'
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
                polyclu = [X[ddb.labels_[:,1] == i] for i in np.unique(ddb.labels_[:,1]) if i != 0]
                #clu = [X[ddb.labels_ == i] for i in np.unique(ddb.labels_) if i != -1]
                fig = plt.figure(figsize=(self.options.figsizeX, self.options.figsizeY))
                plt.imshow(self.image,cmap=self.options.cmapcolor,vmin=vmin, vmax=vmax,origin='upper' )
                plt.title("Polynomial + general clusters found in iteration 0")
                colorpix = np.ones([rescaley,rescalex,3]) * [255,255,255]
                for j in range(0,len(clu)):
                    a = np.random.rand(3)
                    colorpix[clu[j][:,0],clu[j][:,1]] = a
                plt.imshow(colorpix,cmap='binary',origin='upper' )
                
                #for j in range(0,len(polyclu)):
                #    print ("covering with dark grey the polynomial cluster # ",j)
                #    black = np.array([0.0,0.0,0.0],dtype = float)
                #    colorpix[polyclu[j][:,0],polyclu[j][:,1]] = black
                #plt.imshow(colorpix,cmap='binary',origin='lower') 

                for ext in ['png']:       #,'pdf'
                    plt.savefig('{pdir}/{name}_{esp}_{tip}.{ext}'.format(pdir=outname, name=self.name, esp='0th', ext=ext, tip=self.options.tip), bbox_inches='tight', pad_inches=0)
                if self.options.pkl_ext == 1:
                    with open('{pdir}/{name}_{esp}.pkl'.format(pdir=outname,name=self.name,esp='0th'), "wb") as fp:
                        pickle.dump(fig, fp, protocol=4)

                plt.gcf().clear()
                plt.close('all')

        return superclusters,lp, t_medianfilter, t_noisered, t_DBSCAN
        
        
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
            snakes, lp_len, t_medianfilter, t_noisered, t_DBSCAN = snfac.getClusters(plot=self.plotpy)

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
            
        else:
            print('\nIt seems that the DBSCAN algorithm is not selected in SnakeProducer. ERROR.\n ANALYSIS FAILED')
            sys.exit()
        t1 = time.perf_counter()
        if self.options.debug_mode: print(f"FULL RECO in {t1 - t0:0.4f} seconds")

        if self.options.debug_mode:
            print(f"  1.1 preprocessing2 + DBSCAN in {t1 - t0:0.4f} seconds")
                
        # print "Get light profiles..."
        snfac.calcProfiles(snakes,plot=self.plotpy)
        t2 = time.perf_counter()
        if self.options.debug_mode: print(f"cluster shapes in {t2 - t1:0.4f} seconds")

        if self.options.debug_mode:
            print(f"  1.2 variable calculation in {t2 - t1:0.4f} seconds")
        t_variables = t2 - t1
        
        # sort snakes by light integral
        snakes = sorted(snakes, key = lambda x: x.integral(), reverse=True)
        # and reject discharges (round)
        #snakes = [x for x in snakes if x.qualityLevel()>=self.snakeQualityLevel]
        
        # plotting
        if self.plot2D:       snfac.plotClusterFullResolution(snakes)
        if self.plotprofiles: snfac.plotProfiles(snakes)

        return snakes, t_DBSCAN, t_variables, lp_len, t_medianfilter, t_noisered