#!/usr/bin/env python
import numpy as np
from skimage.segmentation import  inverse_gaussian_gradient,morphological_geodesic_active_contour
from clusterTools import Cluster
from scipy.stats import pearsonr
from energyCalibrator import EnergyCalibrator

class SuperClusterAlgorithm:
    def __init__(self,shape=512,neighbor_window=3,debugmode=False):
        self.shape = shape
        self.neighbor_window = neighbor_window
        self.debug = debugmode

        # supercluster energy calibration for the saturation effect
        filePar = open('modules_config/energyCalibrator.txt','r')
        params = eval(filePar.read())
        self.calibrator = EnergyCalibrator(params,self.debug)
        
    def clusters_neighborood(self,basic_clusters,raw_data):
        neighboroods = np.zeros([self.shape,self.shape],dtype=float)
        if len(basic_clusters)>0:
           Xtot_clusters = np.vstack(basic_clusters)
           for pointInCluster in Xtot_clusters:
               ix,iy = pointInCluster[0],pointInCluster[1]
               for ixn in range(-self.neighbor_window,self.neighbor_window+1):
                   for iyn in range(-self.neighbor_window,self.neighbor_window+1):
                       x = max(0,min(ix+ixn,self.shape-1))
                       y = max(0,min(iy+iyn,self.shape-1))
                       neighboroods[x,y] = raw_data[x,y]
        return neighboroods

    def store_evolution_in(self,lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """
        def _store(x):
            lst.append(np.copy(x))
        return _store

    def supercluster(self,clustered_data):
        gimage = inverse_gaussian_gradient(clustered_data)
        # Initial level set
        # this makes alternate squares active at the first iteration of 10 macro-pixels
        #init_ls = checkerboard_level_set(clustered_data.shape, 10)
        init_ls = np.zeros(clustered_data.shape, dtype=np.int8)
        init_ls[10:-10, 10:-10] = 1
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = self.store_evolution_in(evolution)
        ls = morphological_geodesic_active_contour(gimage, 300, init_ls,
                                                   smoothing=1, balloon=-1,
                                                   threshold=0.69,
                                                   iter_callback=callback)
        return ls

    def supercluster_points(self,levels):
        from scipy import ndimage as ndi
        # fill the contours of the superclusters
        fill_contours = ndi.binary_fill_holes(levels)
        # remove the smallest superclusters
        label_objects, nb_labels = ndi.label(fill_contours)
        sizes = np.bincount(label_objects.ravel())
        mask_sizes = sizes > 3
        mask_sizes[0] = 0
        contours_cleaned = mask_sizes[label_objects]
        labeled_pixels, nb_labels = ndi.label(contours_cleaned)

        superclusters = [None]*nb_labels
        for ix in range(self.shape):
            for iy in range(self.shape):
                lbl = labeled_pixels[ix,iy]
                if lbl>0: # super-clustered pixel
                    if superclusters[lbl-1]: superclusters[lbl-1].append((ix,iy))
                    else: superclusters[lbl-1] = [(ix,iy)]

        # transform to numpy array
        for isc in range(len(superclusters)):
            superclusters[isc] = np.array(superclusters[isc])
        return np.array(superclusters)

    def findSuperClusters(self,basic_clusters,raw_data,raw_data_fullreso,raw_data_fullreso_zs,iteration):
        superClusters = []; superClusterContours = np.array([])

        if len(basic_clusters):
            # use the clustered points to get "seeds" for superclustering
            # i.e. open a window to get back unclustered points with low light
            seedsAndNeighbors = self.clusters_neighborood(basic_clusters,raw_data)
          
            # run the superclustering algorithm (GAC with 300 iterations)
            superClusterContours = self.supercluster(seedsAndNeighbors)
          
            # get the superclusters with the list of points of each one
            superClusterWithPixels = self.supercluster_points(superClusterContours)
          
            # get a cluster object
            rebin = int(2048/self.shape)
            for i,scpixels in enumerate(superClusterWithPixels):
                sclu = Cluster(scpixels,rebin,raw_data_fullreso,raw_data_fullreso_zs,debug=False)
                sclu.iteration=iteration
                sclu.nclu = i
                x = scpixels[:, 0]; y = scpixels[:, 1]
                corr, p_value = pearsonr(x, y)
                sclu.pearson = p_value
                if self.debug:
                    print ( "SUPERCLUSTER BARE INTEGRAL = {integral:.1f}".format(integral=sclu.integral()) )
                sclu.calibratedEnergy = self.calibrator.calibratedEnergy(sclu.hits_fr)
                sclu.pathlength = self.calibrator.clusterLength()
                superClusters.append(sclu)

                # if i==3:
                #     print(" === hits list ")
                #     print(scpixels)
                #     sclu.dumpToFile('supercluster3')
                    
        # return the supercluster collection
        return superClusters,superClusterContours
    
