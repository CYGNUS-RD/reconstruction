import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy import ndimage
from skimage.morphology import  thin
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize,binary_closing
import mahotas as mh
import math

from utilities import bcolors

class EnergyCalibrator:
    def __init__(self,params,debugmode=False):
        self.p0 = params['p0']
        self.p1 = params['p1']
        self.p2 = params['p2']
        self.p3 = params['p3']
        self.p4 = params['p4']
        self.norm = params['norm']
        self.xscale = params['xscale']
        self.noiseThreshold = params['noiseThr']
        self.sliceRadius = params['sliceRadius']
        self.length = -1
        self.debug = debugmode
        
    def getClusterMatrix(self,hits):
        xs = [x[0] for x in hits]
        ys = [x[1] for x in hits]
        xmin = int(min(xs)); xmax = int(max(xs))
        ymin = int(min(ys)); ymax = int(max(ys))
     
        data = np.zeros((int(xmax-xmin),int(ymax-ymin)), dtype=float)
        for x,y,z in hits:
            data[int(x-xmin-1),int(y-ymin-1)] = z
        return data

    def branchedPoints(self,skel):
        branch1=np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]])
        branch2=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]])
        branch3=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])
        branch4=np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
        branch5=np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
        branch6=np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
        branch7=np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
        branch8=np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
        branch9=np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
        br1=mh.morph.hitmiss(skel,branch1)
        br2=mh.morph.hitmiss(skel,branch2)
        br3=mh.morph.hitmiss(skel,branch3)
        br4=mh.morph.hitmiss(skel,branch4)
        br5=mh.morph.hitmiss(skel,branch5)
        br6=mh.morph.hitmiss(skel,branch6)
        br7=mh.morph.hitmiss(skel,branch7)
        br8=mh.morph.hitmiss(skel,branch8)
        br9=mh.morph.hitmiss(skel,branch9)
        return br1+br2+br3+br4+br5+br6+br7+br8+br9

    def endPoints(self,skel):
        endpoint1=np.array([[0, 0, 0],
                            [0, 1, 0],
                            [2, 1, 2]])
        
        endpoint2=np.array([[0, 0, 0],
                            [0, 1, 2],
                            [0, 2, 1]])
        
        endpoint3=np.array([[0, 0, 2],
                            [0, 1, 1],
                            [0, 0, 2]])
        
        endpoint4=np.array([[0, 2, 1],
                            [0, 1, 2],
                            [0, 0, 0]])
        
        endpoint5=np.array([[2, 1, 2],
                            [0, 1, 0],
                            [0, 0, 0]])
        
        endpoint6=np.array([[1, 2, 0],
                            [2, 1, 0],
                            [0, 0, 0]])
        
        endpoint7=np.array([[2, 0, 0],
                            [1, 1, 0],
                            [2, 0, 0]])
        
        endpoint8=np.array([[0, 0, 0],
                            [2, 1, 0],
                            [1, 2, 0]])
        
        ep1=mh.morph.hitmiss(skel,endpoint1)
        ep2=mh.morph.hitmiss(skel,endpoint2)
        ep3=mh.morph.hitmiss(skel,endpoint3)
        ep4=mh.morph.hitmiss(skel,endpoint4)
        ep5=mh.morph.hitmiss(skel,endpoint5)
        ep6=mh.morph.hitmiss(skel,endpoint6)
        ep7=mh.morph.hitmiss(skel,endpoint7)
        ep8=mh.morph.hitmiss(skel,endpoint8)
        ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
        return ep

    def pruning(self,skeleton, size):
        '''remove iteratively end points "size" 
           times from the skeleton
        '''
        for i in range(0, size):
            endpoints = self.endPoints(skeleton)
            endpoints = np.logical_not(endpoints)
            skeleton = np.logical_and(skeleton,endpoints)
        return skeleton

    def points_in_circle_np(self, radius, x0=0, y0=0):
        x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
        y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
        x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius))
        points = []
        for x, y in zip(x_[x], y_[y]):
            points.append((x, y))
        return points

    def uncalibIntegral(self,hits):
        return sum([h[2] for h in hits])

    def density(self, sliceOfClu):
        nhits = len([h for h in sliceOfClu if h[2]>self.noiseThreshold])
        integral = max(sum([h[2] for h in sliceOfClu]),0)
        return integral/nhits if nhits>0 else 0

    def saturationFactorNLO(self,density):
    ## this gives eV/ph
        if density<=0: # protection for the formula below
            ret = 0.85 # seems to provide some continuity
        else:
            x = density/self.xscale
            ret = (self.p3 + self.p4*x)/(self.p0 * (1-math.exp(-1*(math.pow(x,self.p2)/self.p1))))/self.norm
        return ret
    
    def calibratedEnergy(self,hits):
        slices = self.getSlices(hits)
        integrals = [max(0.,sum([h[2] for h in sl])) for sl in slices]
        densities = [self.density(sl) for sl in slices]

        ## the energy is now in keV
        calibSlicesEnergy = [self.saturationFactorNLO(densities[sl]) * integrals[sl] / 1000. for sl in range(len(densities))]
        calibEnergy = sum(calibSlicesEnergy)

        if self.debug:
            print (bcolors.OKBLUE + "Slices bare sum = {bsum:.1f}".format(bsum=sum(integrals)) + bcolors.ENDC)
            print ("Slices integral = " + ', '.join('{:.1f}'.format(i) for i in integrals))
            print ("Slices densities = " + ', '.join('{:.1f}'.format(i) for i in densities))
            print ("Slices calib energy = " + ', '.join('{:.1f}'.format(i) for i in calibSlicesEnergy))
            print (bcolors.OKGREEN + "supercluster calibrated integral = {ene:.1f} keV".format(ene=calibEnergy) + bcolors.ENDC)
        return calibEnergy
    
    def getSlices(self,hits):
    
        cluster_matrix = self.getClusterMatrix(hits) # this has x,y,z
        cluster_img = cluster_matrix != 0 # this is the binary version to run the skeletonization
        
        skeleton = thin(cluster_img) # this is the 1-pixel wide skeleton of the cluster
        pruned =  self.pruning(skeleton,10) # remove little branches
        
        skel_points = np.column_stack(np.nonzero(pruned))
        remaining_skel_points = [(point[0],point[1]) for point in  skel_points] # simpler with an array of tuples
        remaining_cluster = cluster_img
        slices = []
        while len(remaining_skel_points):
            p = remaining_skel_points[-1]
            clu_slice = []
            circlepoints = self.points_in_circle_np(self.sliceRadius,p[0],p[1])
            for cp in circlepoints:
                ix = cp[0]; iy = cp[1];
                if ix>=cluster_matrix.shape[0] or iy>=cluster_matrix.shape[1] or ix<0 or iy<0:
                    continue
                z = cluster_matrix[ix,iy]
                if remaining_cluster[ix,iy]:
                    clu_slice.append((ix,iy,z))
                    remaining_cluster[ix,iy] = False
                # this includes the center and all the intersection of the circle with the skeleton
                if cp in remaining_skel_points:
                    remaining_skel_points.remove(cp)
            #remaining_skel_points = np.setdiff1d(remaining_skel_points,circlepoints)
            slices.append(clu_slice)
        #print ("slices ",slices)
        #print ("Found ",len(slices)," slices")
        # this is a better estimate of the length of a curved cluster (in pixels)
        self.length = len(skeleton)
        return slices

    def clusterLength(self):
        l = -999
        if self.length<0:
            print ("ERROR! You asked for EnergyCalibrator.length() before getting the calibrated energy, and this is not yet set!!")
        else:
            l = self.length
        return l
            
if __name__ == '__main__':

    # load hits
    hits = np.load('debug_code/supercluster3.npy')

    filePar = open('modules_config/energyCalibrator.txt','r')
    params = eval(filePar.read())
    calibrator = EnergyCalibrator(params)
    
    
    uncal = calibrator.uncalibIntegral(hits)
    print ("Uncalibrated integral (photons) = ",uncal)
    cal = calibrator.calibratedEnergy(hits)
    print ("Calibrated energy (keV) = ",cal)

    # skeleton = skeletonize(image)
    # thinned = thin(image)
    # pruned = pruning(thinned,10)



    # #medial_axis = medial_axis(image)

    # fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    # ax = axes.ravel()

    # ax[0].imshow(image, cmap=plt.cm.gray)
    # ax[0].set_title('original')
    # ax[0].axis('off')
    
    # ax[1].imshow(skeleton, cmap=plt.cm.gray)
    # ax[1].set_title('skeleton')
    # ax[1].axis('off')
    
    # ax[2].imshow(thinned, cmap=plt.cm.gray)
    # ax[2].set_title('thinned')
    # ax[2].axis('off')
    
    # ax[3].imshow(pruned, cmap=plt.cm.gray)
    # ax[3].set_title('pruned')
    # ax[3].axis('off')
    
    # fig.tight_layout()
    # plt.show()


