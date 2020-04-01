"""A Python class containing an implimentation of MPPCA denoising.
    
    By default we denoise using a 5x5x5 box kernel.
        
    Inputs are a 4D image with dimentions (X x Y x Z x N)
    
    Usage:

    from mpdenoise import MP
    denoiser = MP(img, kernel='5,5,5')
    imgdn, sigma, nparameters = denoiser.process()

    Benjamin Ades-Aron
        """

import os, sys
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

class MP(object):
    def __init__(self, dwi, kernel=None):
        self.dwi = dwi
        if kernel is None:
            kernel = np.array([[5,5,5]])
        else:
            kernel = np.array([np.fromstring(kernel, sep=',')])
        kernel = kernel+np.mod(kernel,2)-1
        self.kernel = kernel

    def boxpatch(self, coords):
        # extracts a patch of size kx x ky x kz from the padded input at specified coords
        k = (self.kernel-1)/2
        k = k.astype(int)
        kx = k[0,0]
        ky = k[0,1]
        kz = k[0,2]
        X = self.dwi_tmp[coords[0]-kx:coords[0]+kx+1, coords[1]-ky:coords[1]+ky+1, coords[2]-kz:coords[2]+kz+1 :]
        return X

    def sample(self, mask, kernel):
        # outputs a grid x, y, z of which coordinates to loop over when processing
        k = ((kernel-1)/2).astype(int)
        kx = k[0,0]; 
        ky = k[0,1]; 
        kz = k[0,2]; 
        sx, sy, sz = mask.shape
        mask[:kx,:,:] = 0
        mask[sx-kx:,:,:] = 0
        mask[:,:ky,:] = 0
        mask[:,sy-ky:,:] = 0
        mask[:,:,:kz] = 0
        mask[:,:,sz-kz:] = 0
        x, y, z = np.where(mask==1)
        return x.astype(int), y.astype(int), z.astype(int)

    def unpad(self, x, pad_width):
        slices = []
        for c in pad_width:
            e = None if c[1] == 0 else -c[1]
            slices.append(slice(c[0], e))
        return x[tuple(slices)]

    def denoise(self, coords, M, N, centering=0):
        X = self.boxpatch(coords)
        X = X.reshape((M, N))
        R = np.min((M, N)).astype(int)

        flip = False
        if M > N:  
            flip = True
            X = X.T
            M = X.shape[0]
            N = X.shape[1]
        
        if centering:
            colmean = np.mean(X, axis=0)
            X = X - np.tile(colmean, (M, 1))

        try:
            u,vals,v = np.linalg.svd(X, full_matrices=False)
            vals_orig = vals
            vals = (vals**2)/N

            csum = np.cumsum(vals[R-centering-1:None:-1])
            sigmasq_1 = csum[R-centering-1:None:-1]/range(R-centering, 0, -1)

            gamma = (M-np.array(range(0, R-centering)))/N
            rangeMP = 4*np.sqrt(gamma[:])
            rangeData = vals[0:R-centering]-vals[R-centering-1]
            sigmasq_2 = rangeData/rangeMP

            t = np.where(sigmasq_2 < sigmasq_1)
            t = t[0][0]
            sigma = np.sqrt(sigmasq_1[t])
            
            vals[t:R] = 0
            s = np.matrix(u) * np.diag(np.sqrt(N*vals)) * np.matrix(v)   

            if flip:
                s = s.T
            if centering:
                s = s + np.tile(colmean, (M, 1))
            signal = np.squeeze(s[M//2, :])

        except:
            sigma = np.nan
            if flip:
                X = X.T
            signal = np.squeeze(X[M//2, :])
            t = R
        npars = t
        return signal, sigma, npars

    def process(self):
        pwidth = ((int(self.kernel[0][0]//2), int(self.kernel[0][0]//2)), \
            (int(self.kernel[0][1]//2), int(self.kernel[0][1]//2)), \
            (int(self.kernel[0][2]//2), int(self.kernel[0][2]//2)), (0,0))
        self.dwi_tmp = np.pad(self.dwi, pwidth, 'wrap')
        sx, sy, sz, N = self.dwi_tmp.shape
        
        mask = np.ones((sx,sy,sz))
        mask.astype(bool)

        x, y, z = self.sample(mask, self.kernel)
        xsize = int(x.size)
        coords = np.vstack((x,y,z))

        centering = 0
        M = np.prod(self.kernel).astype(int)

        #print('...denoising')
        inputs = tqdm(range(0, xsize))
        num_cores = multiprocessing.cpu_count()
        
        # parallel
        signal, sigma, npars = zip(*Parallel(n_jobs=num_cores,prefer='processes')\
            (delayed(self.denoise)(coords[:,i], M, N, centering=centering) for i in inputs))
        
        # serial
        # for t in inputs:
        #     a, b, c = self.denoise(coords[:,t], patch, M, N, centering=centering, shrink=shrink)
           
        Sigma = np.zeros((sx, sy, sz))
        Npars = np.zeros((sx, sy, sz))
        Signal = np.zeros((sx, sy, sz, N))
        for nn in range(0, xsize):
            Sigma[x[nn], y[nn], z[nn]] = sigma[nn]
            Npars[x[nn], y[nn], z[nn]] = npars[nn]
            Signal[x[nn], y[nn], z[nn], :] = signal[nn]

        Signal = self.unpad(Signal, pwidth)
        Npars = self.unpad(Npars, pwidth[:][:-1])
        Sigma = self.unpad(Sigma, pwidth[:][:-1])
        return Signal, Sigma, Npars


    
