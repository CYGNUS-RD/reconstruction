# cython: language_level=3

import numpy as np
cimport numpy as np

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t


def nred_cython(np.ndarray[DTYPE_t, ndim=2] edges, int escalax, int escalay, float meancut=0.35):
    cdef int rescalex = escalax 
    cdef int rescaley = escalay 
    cdef int tpx = 10
    cdef float mpx
    cdef float a,b,c,d,spx,f,g,h,i
    cdef int neighbors
    
    cdef int k, j

    for k in range(rescaley):
        for j in range(2):
            edges[k,j]=0
            edges[k,rescalex-1-j]=0

    for k in range(rescalex):
        for j in range(2):
            edges[j,k]=0
            edges[rescaley-1-j,k]=0

    for k in range(1,rescaley-2):
        for j in range(1,rescalex-2):
            a = edges[k-1,j-1]
            b = edges[k,j-1]
            c = edges[k+1,j-1]
            d = edges[k-1,j]
            spx = edges[k,j]
            f = edges[k+1,j]
            g = edges[k-1,j+1]
            h = edges[k,j+1]
            i = edges[k+1,j+1]
            mpx = (a+b+c+d+f+g+h+i)/8.
            # put very noisy pixels at the average value of the frame around
            if abs(spx - mpx) > tpx :
                edges[k,j] = mpx
            # filter the pixels with no sufficient energy around
            if (mpx < meancut):
                edges[k,j] = 0
            # require at least two neighbors above threshold
            #frame = edges[k-1:k+2,j-1:j+2]
            neighbors = 9-(not a)-(not b)-(not c)-(not d)-(not spx)-(not f)-(not g)-(not h)-(not i)
            if neighbors<3:
                edges[k,j] = 0
    return edges


def sim3d_cython(np.ndarray[np.int, ndim=2] img_rb_zs, np.ndarray[np.int, ndim=2] points):
    cdef int size = points.shape[0]
    cdef int k, nreplicas=0
    cdef np.int j,l,idx1=0,idx2=0
    for k in range(size):
        j = points[k,0]
        l = points[k,1]
        if (img_rb_zs[j,l] == 0):
            nreplicas += 1
        else:
            nreplicas += img_rb_zs[j,l] 
    cdef np.ndarray[np.int64_t, ndim=2] newpoints = np.zeros((nreplicas,2), dtype=np.int64)
    cdef int count=0
    for k in range(size):
        j = points[k,0]
        l = points[k,1]
        nreplicas = img_rb_zs[j,l]
        if (nreplicas < 1):
            newpoints[idx1,0] = j
            newpoints[idx1,1] = l
            idx1 += 1
        else:
            newpoints[(idx1),0] = j
            newpoints[(idx1),1] = l
            idx1 += 1
            for count in range(nreplicas-1):
                newpoints[(idx2+size),0] = j
                newpoints[(idx2+size),1] = l
                idx2 += 1
    return newpoints

