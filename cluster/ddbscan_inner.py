from __future__ import division

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import time,math

import warnings
warnings.simplefilter('ignore', np.RankWarning)

class PolynomialRegression(object):
    def __init__(self, degree=3, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

def ransac_polyfit(x,y,order,t,n=0.8,k=100,f=0.9):

    #print("\t\t*** doing polyfit with order ",order)
    besterr = np.inf
    bestfit = np.array([None])
    bestfitderi = np.array([None])
    polyderi = np.zeros(order, dtype = np.float64)
    for kk in range(k):
        maybeinliers = np.random.randint(len(x), size=int(n*len(x)))
        maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
        for i in range(order):
            polyderi[i] = maybemodel[i]*(order-i)
        res_th = t * ((1 + np.polyval(polyderi,x)**2)**0.5)
        alsoinliers = np.abs(np.polyval(maybemodel, x)-y) < res_th
        if sum(alsoinliers) > len(x)*f:
            x_in = x[alsoinliers]
            y_in = y[alsoinliers]
            bettermodel = np.polyfit(x_in, y_in, order)
            for i in range(order):
                polyderi[i] = bettermodel[i]*(order-i)
            thiserr = np.sum(np.abs(np.polyval(bettermodel, x_in)-y_in))
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                bestfitderi = polyderi

    #print("\t\t=== polyfit DONE")                     
    return bestfit, bestfitderi

#Parameters of the new ransac function:
# x, y - x and y coordinates 
# order - Order of the polynomial
# t - Thickness of the track
# n - Random fraction of the data used to find the polyfit
# k - Number of tries 
# f - Accuracy of the RANSAC to consider the fit a good one

def ddbscaninner(data, is_core, neighborhoods, neighborhoods2, labels, dir_radius, dir_min_accuracy, dir_minsamples, dir_thickness, time_threshold, max_attempts, isolation_radius, expand_noncore, debug=True):
    #Definitions
    
    #Beginning of the algorithm - DBSCAN check part
    label_num = 0
    stack = []
    clu_stra = []
    acc = []
    length = []
    clu_labels = []
    min_samples = np.inf
    
    ddbsc_t1=time.time()
    if debug:
        print("Start DBSCAN seeding...")
    #Loop 
    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:
            continue
        while True:
            if labels[i] == -1:
                labels[i] = label_num
                if is_core[i]:     #Only core points are expanded
                    neighb = neighborhoods[i]
                    for i in range(neighb.shape[0]):
                        v = neighb[i]
                        if labels[v] == -1:
                            stack.append(v)
            if len(stack) == 0:
                break
            i = stack[len(stack)-1]
            del(stack[len(stack)-1])


        #Ransac part
        moment_length = sum(labels==label_num)
        if debug:
            print("test cluster n. ",i)
        
        if moment_length < min_samples:
            min_samples = moment_length
        if moment_length > dir_minsamples:
            if debug:
                print("==> cluster ",i," has ",sum(labels==label_num)," samples")
            x = data[labels==label_num][:,0]
            y = data[labels==label_num][:,1]
            ransac = RANSACRegressor(PolynomialRegression(degree=3),
                                     residual_threshold=3 * np.std(y),
                                     random_state=0)
            ransac.fit(np.expand_dims(x, axis=1), y)
            #if (np.median(np.abs(y - np.median(y))) == 0):
            #    ransac = RANSACRegressor(min_samples=0.8, residual_threshold = 0.1)
            #    ransac.fit(np.expand_dims(x, axis=1), y)
            #else:
            #    ransac = RANSACRegressor(min_samples=0.8)
            #    ransac.fit(np.expand_dims(x, axis=1), y)
            
            accuracy = sum(ransac.inlier_mask_)/len(y)

            if accuracy > dir_min_accuracy:
                clu_stra.append(label_num)
                acc.append(accuracy)
                length.append(sum(labels==label_num))
                clu_labels.append(label_num)
        label_num += 1
    
    #End of DBSCAN loop - check if directional part is viable
    if debug:
        print("Clusters found in DBSCAN: %d" %(len(set(labels)) - (1 if -1 in labels else 0)))

    if len(clu_stra) == 0:
        #If no cluster has a good fit model, the output will be the same of the DBSCAN
        la_aux = np.copy(labels)
        labels = np.zeros([la_aux.shape[0],2], dtype=np.intp)
        labels[:,0] = la_aux
        return labels
    else:
        #If any cluster has a good fit model, it'll be marked from the worst fitted cluster to the best, each of them respecting the accuracy threshold
        auxiliar_points = []
        vet_aux = np.zeros([len(clu_stra),3])
        vet_aux[:,0] = np.asarray(clu_stra)
        vet_aux[:,1] = np.asarray(acc)
        vet_aux[:,2] = np.asarray(length)
        vet_aux = np.asarray(sorted(vet_aux,key=itemgetter(1),reverse=1))
        # in case there is more than 1 cluster with 0 energy around, sort them by accuracy
        if (sum(vet_aux[:,1]==1) > 1):
            l1 = sum(vet_aux[:,1]==1)
            vet_aux[0:l1,:] = np.asarray(sorted(vet_aux[0:l1,:],key=itemgetter(2),reverse=1))
        for u in range(len(clu_stra)):
            lt = (labels==vet_aux[u][0])*is_core
            auxiliar_points.append(np.where(lt)[0][0])
            if debug:
                print("The point %d has been assigned as part of a good fit" %(np.where(lt)[0][0]))
        
        #Now the clusterization will begin from zero with directionality enabled for the clusters that have a good fit model
        label_num = 0
        labels = np.full(data.shape[0], -1, dtype=np.intp)
        stack = []
        for i in auxiliar_points:
            if debug:
                print("Auxiliar point ",i)
            if labels[i] != -1 or not is_core[i]:
                continue
            while True:
                if labels[i] == -1:
                    labels[i] = label_num
                    if is_core[i]:
                        neighb = neighborhoods[i]
                        for i in range(neighb.shape[0]):
                            v = neighb[i]
                            if labels[v] == -1:
                                stack.append(v)

                if len(stack) == 0:
                    break
                i = stack[len(stack)-1]
                del(stack[len(stack)-1])

            moment_length = sum(labels==label_num)
            if debug:
                print("An attempt cluster fround with ",sum(labels==label_num)," 2D pix, while requested ",dir_minsamples, " min samples.  Dir search now...")
            #Now that the provisional cluster has been found, directional search begins
            if moment_length > dir_minsamples:
                #Taking unique points to use on the ransac
                x = data[labels==label_num][:,0]
                y = data[labels==label_num][:,1]

                #RANSAC fit
                order = 1
                fit_model, fit_deri = ransac_polyfit(x, y, order = order, t = dir_thickness)
                counter = 1
                #Adding new points to the cluster (If the fit_model output is None, then no model was found)
                if sum(fit_model == None) == 0:
                    #control = 1
                    pts1 = 0
                    while True:
                        if debug:
                            print ("***** attempt n ",counter)
                        dbt1 = time.time()
                        #Filling stack list with possible new points to be added (start point)
                        pts0 = pts1
                        moment_lab = np.where(labels==label_num)[0]
                        stack = []
                        #If expand_noncore is not available, then only core_points will have its neighborhoods expanded
                        for j in moment_lab:
                            if expand_noncore:
                                core_flag = 1
                            else:
                                core_flag = is_core[j]
                            if core_flag:
                                neig2 = neighborhoods2[j]
                                for j in range(neig2.shape[0]):
                                    v = neig2[j]
                                    if labels[v] != label_num:
                                        stack.append(v)
                        stack = np.unique(stack).tolist()
                        if len(stack) == 0:
                            break
                        dbt2 = time.time()
                        if debug:
                            print ("start point stack built in ",dbt2-dbt1," secs")
                        
                        res_th = dir_thickness / np.cos(np.arctan(np.polyval(fit_deri,data[:,0])))
                        inliers_bool = np.abs(np.polyval(fit_model, data[:,0])-data[:,1]) < res_th
                        inliers = np.where(inliers_bool)[0]
                        dbt3 = time.time()
                        if debug:
                            print ("inliers computed in ",dbt3-dbt2," secs")


                        i = stack[len(stack)-1]
                        #Adding the inliers points from stack list and filling stack with more possible points
                        while True:
                            if expand_noncore:
                                core_flag = 1
                            else:
                                core_flag = is_core[i]
                            if i in inliers and (labels[i] != label_num):
                                labels[i] = label_num
                                if core_flag:
                                    neig2 = neighborhoods2[i]
                                    for i in range(neig2.shape[0]):
                                        v = neig2[i]
                                        if labels[v] != label_num:
                                            stack.append(v)
                            if len(stack) == 0:
                                break
                            i = stack[len(stack)-1]
                            del(stack[len(stack)-1])
                        dbt4 = time.time()
                        if debug:
                            print("DEBUG timing: dbt1-dbt2 = ", dbt1-dbt2,"  dbt3-dbt2 = ",dbt3-dbt2,"  dbt4-dbt3 = ",dbt4-dbt3," ...")
                        
                        #Checking current cluster for possible fit model update
                        x = data[labels==label_num][:,0]
                        y = data[labels==label_num][:,1]


                        #Updating the ransac model
                        t1 = time.time()
                        fit_model, fit_deri = ransac_polyfit(x, y, order = order, t = dir_thickness)
                        pts1 = sum(labels==label_num)
                        #Stop criteria - time
                        t2 = time.time()
                        if debug:
                            print ("INFO: fit time = ",t2-t1)
                        if (t2 - t1) > time_threshold:
                            break
                        #Stop criteria - max attempts of improvements
                        if counter > max_attempts:
                            break
                        #Stop criteria - When there is no more point to be added or if the fit is not good anymore
                        counter = counter + 1
                        if (pts1 == pts0) or (sum(fit_model == None) != 0):
                            if debug:
                                print ("last polynomial order = ", order)
                            if order == 3:
                                if debug:
                                    print('The cluster %d' %(label_num) + ' needed %d attempts' %(counter))
                                break
                            else:
                                order = 3
                                fit_model, fit_deri = ransac_polyfit(x, y, order = order, t = dir_thickness)
                                if sum(fit_model == None) != 0:
                                    break
                

            label_num += 1

        for i in range(label_num):
            if sum(labels==i) < min_samples:
                labels[labels==i] = len(data)

        ddbsc_t2=time.time()
        if debug:
            print ("Polynomial clustering took ",ddbsc_t2-ddbsc_t1," secs.")
        
        # Now that the clusters with good fit models were found, the rest of the data will be clustered with the standard DBSCAN logic
        if debug:
            print("Now clustering the rest...")

        nt1 = time.time()
        # first compute distances to make isolation on each pixel of the unclustered data wrt the clustered pixels
        clustered_data = np.unique(data[labels!=-1],axis=0)
        neigh = NearestNeighbors(n_neighbors=1, radius=isolation_radius)
        neigh.fit(clustered_data)
        distances, isol_neigh_indices = neigh.kneighbors(data[labels==-1], 1, return_distance=True)
        nt2 = time.time()
        if debug:
            print ("neighbs done in ",nt2-nt1," secs")
        
        
        dbt1 = time.time()
        for i in range(labels.shape[0]):
            if labels[i] != -1 or not is_core[i]:
                continue
            while True:
                if labels[i] == -1:     
                    labels[i] = label_num
                    if is_core[i]:     #Only core points are expanded
                        neighb = neighborhoods[i]
                        for i in range(neighb.shape[0]):
                            v = neighb[i]
                            dist_closest_clustered = distances[i][0]
                            if labels[v] == -1 and dist_closest_clustered > isolation_radius:
                                stack.append(v)

                if len(stack) == 0:
                    break
                i = stack[len(stack)-1]
                del(stack[len(stack)-1])

            if sum(labels==label_num) >= min_samples:
                label_num += 1
            else:
                labels[labels==label_num] = len(data)
        dbt2 = time.time()
        if debug:
            print("last clustering took... ",dbt2-dbt1)
        
        #False clusters remotion
        labels[labels==len(data)] = -1
        

        if debug:
            print ("Clustering done")
        #Clusterization has finished, now the clusters found with ransac fit model will be marked
        la_aux = np.copy(labels)
        labels = np.zeros([la_aux.shape[0],2], dtype=np.intp)
        labels[:,0] = la_aux
        labels[auxiliar_points,1] = 1
        
        
        if debug:
            print ("Return labels")

        return labels
        
            
            
        
        
        
        
            
