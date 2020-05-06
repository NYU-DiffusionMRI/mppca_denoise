"""A Python class containing an implimentation of MPPCA denoising.
    
    By default we denoise using a 5x5x5 box kernel.
        
    Inputs are a 4D image with dimentions (X x Y x Z x N)
    
    Usage:

    from mpdenoise import MP
    denoiser = MP(img, kernel='5,5,5')
    imgdn, sigma, nparameters = denoiser.process()
    
    LICENCE
    Authors: Benjamin Ades-Aron (Benjamin.Ades-Aron@nyulangone.org)
    Copyright (c) 2016 New York University
    
    Permission is hereby granted, free of charge, to any non-commercial entity
    ('Recipient') obtaining a copy of this software and associated
    documentation files (the 'Software'), to the Software solely for
    non-commercial research, including the rights to use, copy and modify the
    Software, subject to the following conditions:
    
    1. The above copyright notice and this permission notice shall be
    included by Recipient in all copies or substantial portions of the
    Software.
    
    2. THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIESOF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BELIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF ORIN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.
    
    3. In no event shall NYU be liable for direct, indirect, special,
    incidental or consequential damages in connection with the Software.
    Recipient will defend, indemnify and hold NYU harmless from any claims or
    liability resulting from the use of the Software by recipient.

    4. Neither anything contained herein nor the delivery of the Software to
    recipient shall be deemed to grant the Recipient any right or licenses
    under any patents or patent application owned by NYU.

    5. The Software may only be used for non-commercial research and may not
    be used for clinical care.

    6. Any publication by Recipient of research involving the Software shall
    cite the references listed below.

    REFERENCES
    Veraart, J.; Fieremans, E. & Novikov, D.S. Diffusion MRI noise mapping
    using random matrix theory Magn. Res. Med., 2016, early view, doi:
    10.1002/mrm.26059
    """

import os, sys
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

class MP(object):
    def __init__(self, dwi, kernel=None):
        self.dwi = dwi
        sx, sy, sz, N = self.dwi.shape

        if kernel is None:
            kernel = np.array([[5,5,5]])
        else:
            kernel = np.array([np.fromstring(kernel, sep=',')])
        kernel = kernel+np.mod(kernel,2)-1
        self.kernel = kernel.astype(int)
        k = self.kernel // 2

        # Jelle's padding
        # Nduplicates = (kernel[0,-1]-1)/2
        # slicesA = np.arange(kernel[0,2], kernel[0,2]-Nduplicates, -1) - 1
        # slicesB = np.arange(0, self.dwi.shape[2], 1)
        # slicesC = self.dwi.shape[2] - slicesA - 1 
        # sliceselection = np.concatenate((slicesA, slicesB, slicesC)).astype(int)

        # self.dwi_tmp = self.dwi[:,:,sliceselection,:]
        # mask = np.ones((sx,sy,sz))
        # self.mask = np.concatenate((np.zeros((sx, sy, len(slicesA))), mask, np.zeros((sx, sy, len(slicesC)))), axis=2)
        # self.origsz = sz

        pwidth = (k[0,0], k[0,0]), (k[0,1], k[0,1]), (k[0,2],k[0,2]), (0,0)
        
        self.pwidth = pwidth
        self.dwi_tmp = np.pad(self.dwi, pad_width=pwidth, mode='wrap')

    def boxpatch(self, coords):
        # extracts a patch of size kx x ky x kz from the padded input at specified coords
        k = ((self.kernel-1)/2).astype(int)
        kx = k[0,0]
        ky = k[0,1]
        kz = k[0,2]
        X = self.dwi_tmp[coords[0]-kx:coords[0]+kx+1, coords[1]-ky:coords[1]+ky+1, coords[2]-kz:coords[2]+kz+1, :]
        return X

    def unpad(self, x, pad_width):
        slices = []
        for c in pad_width:
            e = None if c[1] == 0 else -c[1]
            slices.append(slice(c[0], e))
        return x[tuple(slices)]

    def sample(self, mask):
        # outputs a grid x, y, z of which coordinates to loop over when processing
        k = ((self.kernel-1)/2).astype(int)
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
        self.mask = mask
        x, y, z = np.where(mask==1)
        return x.astype(int), y.astype(int), z.astype(int)

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
                M = X.shape[1]
                N = X.shape[0]
            if centering:
                s = s + np.tile(colmean, (M, 1))
            signal = np.squeeze(s[M//2, :])

        except:
            sigma = np.nan
            if flip:
                X = X.T
                M = X.shape[1]
                N = X.shape[0]
            signal = np.squeeze(X[M//2, :])
            t = R
        npars = t
        return signal, sigma, npars

    def process(self):
        sx, sy, sz, N = self.dwi_tmp.shape
        
        mask = np.ones((sx,sy,sz))
        x, y, z = self.sample(mask)
        xsize = int(x.size)
        coords = np.vstack((x,y,z))

        centering = 0
        M = np.prod(self.kernel).astype(int)

        #print('...denoising')
        inputs = tqdm(range(0, xsize))
        num_cores = multiprocessing.cpu_count()
        
        # parallel
        signal, sigma, npars = zip(*Parallel(n_jobs=num_cores, prefer='processes')\
            (delayed(self.denoise)(coords[:,i], M, N, centering=centering) for i in inputs))
        
        # serial
        # for t in inputs:
        #     a, b, c = self.denoise(coords[:,t], M, N, centering=centering)
           
        # reconstruct original data matrix
        Sigma = np.zeros((sx, sy, sz))
        Npars = np.zeros((sx, sy, sz))
        Signal = np.zeros((sx, sy, sz, N))
        for nn in range(0, xsize):
            Sigma[x[nn], y[nn], z[nn]] = sigma[nn]
            Npars[x[nn], y[nn], z[nn]] = npars[nn]
            Signal[x[nn], y[nn], z[nn], :] = signal[nn]

        # Signal = Signal[:,:, 2: 2+self.origsz, :]
        # Sigma =  Sigma[:,:, 2: 2+self.origsz]
        # Npars = Npars[:,:, 2: 2+self.origsz]

        Signal = self.unpad(Signal, self.pwidth)
        Npars = self.unpad(Npars, self.pwidth[:][:-1])
        Sigma = self.unpad(Sigma, self.pwidth[:][:-1])

        return Signal, Sigma, Npars

def denoise(img, kernel='5,5,5'):
    mp = MP(img, kernel)
    Signal, Sigma, Npars = mp.process()
    return Signal, Sigma, Npars

if __name__ == "__main__":
    denoise()
    
