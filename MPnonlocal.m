function [Signal, Sigma, Npars] = MPnonlocal(data, kernel, psize, nrm)
    % MPnonlocal Denoise 4d magnitude data (x, y, z, dirs) or 5d complex data 
    %   (x, y, z, coils, dirs) and estimate 3d noise maps and significant 
    %   parameter maps using nonlocal patching and eigenvalue shrinkage in
    %   the MPPCA framework
    %
    %   [Signal, Sigma, Nparams] = MPnonlocal(data, kernel, patchsize, norm)
    %   output:
    %       - Signal: [x, y, z, N] for 4d (real) [x, y, z, C, N] for 5d
    %       (complex)
    %       - Sigma: [x, y, z] noise map
    %       - Npars: [x, y, z] map of the number of signal carrying
    %       components
    %   input:
    %       - data: [x, y, z, N] for 4d (real) [x, y, z, C, N] for 5d
    %       (complex)
    %       - kernel: default [5,5,5]. Window size defining image coverage
    %       for each voxel. Must be odd.
    %       - psize: (optional) Number of voxels to include in in RMT
    %       matrix - for nonlocal denoising. Must be < prod(kernel)
    %       - nrm: (optional) default 'frob'. Can be set to 'h' for hard
    %       thresholding of eigenvalues or 'frob' to implement eigenvalue
    %       shrinkage.
    %
    %   usage: 
    %       - box patch denoising:
    %           [Signal, Sigma, Npars] = MPnonlocal(data, [5,5,5])
    %       - nonlocal denoisng:
    %           [Signal, Sigma, Npars] = MPnonlocal(data, [5,5,5], 100)
    %
    %   Authors: Benjamin Ades-Aron (Benjamin.Ades-Aron@nyulangone.org) 
    %   Jelle Veraart (jelle.veraart@nyumc.org)
    %   Gregory Lemberskiy (Gregory.Lemberskiy@nyulangone.org)
    %   Copyright (c) 2020 New York University
    %       
    %      Permission is hereby granted, free of charge, to any non-commercial entity
    %      ('Recipient') obtaining a copy of this software and associated
    %      documentation files (the 'Software'), to the Software solely for
    %      non-commercial research, including the rights to use, copy and modify the
    %      Software, subject to the following conditions: 
    %       
    %        1. The above copyright notice and this permission notice shall be
    %      included by Recipient in all copies or substantial portions of the
    %      Software. 
    %       
    %        2. THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
    %      EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIESOF
    %      MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    %      NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BELIABLE FOR ANY CLAIM,
    %      DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    %      OTHERWISE, ARISING FROM, OUT OF ORIN CONNECTION WITH THE SOFTWARE OR THE
    %      USE OR OTHER DEALINGS IN THE SOFTWARE. 
    %       
    %        3. In no event shall NYU be liable for direct, indirect, special,
    %      incidental or consequential damages in connection with the Software.
    %      Recipient will defend, indemnify and hold NYU harmless from any claims or
    %      liability resulting from the use of the Software by recipient. 
    %       
    %        4. Neither anything contained herein nor the delivery of the Software to
    %      recipient shall be deemed to grant the Recipient any right or licenses
    %      under any patents or patent application owned by NYU. 
    %       
    %        5. The Software may only be used for non-commercial research and may not
    %      be used for clinical care. 
    %       
    %        6. Any publication by Recipient of research involving the Software shall
    %      cite the references listed below.
    % 
    %  REFERENCES
    %      Veraart, J.; Fieremans, E. & Novikov, D.S. Diffusion MRI noise mapping
    %      using random matrix theory Magn. Res. Med., 2016, early view, doi:
    %      10.1002/mrm.26059


    if isreal(data) 
        data = single(data);
    else
        data = complex(single(data));
    end
    
    if ndims(data) > 4
        coil = true;
    else
        coil = false;
    end
    
    if ~exist('kernel', 'var') || isempty(kernel)
        kernel = [5 5 5];
    end
    
    if isscalar(kernel)
        kernel = [kernel, kernel, kernel];
    end
    
    if ~exist('psize', 'var') || psize > prod(kernel)
        warning('Setting M = prod(kernel)');
        nonlocal = false;
        psize = prod(kernel);
        center_idx = ceil(prod(kernel)/2);
    else
        nonlocal = true;
        center_idx = 1;
    end
    
    if ~exist('nrm', 'var') || isempty(nrm)
        nrm = 'frob';
    end
    
    kernel = kernel + (mod(kernel, 2)-1);  
    k = (kernel-1)/2; 
    kx = k(1); 
    ky = k(2); 
    kz = k(3);
    
    % pad the data in first 3 dimentions
    if coil 
        data = padarray(data, [kx, ky, kz, 0, 0], 'circular');
        [sx, sy, sz, sc, N] = size(data);
        M = psize*sc;
    else
        data = padarray(data, [kx, ky, kz, 0], 'circular');
        [sx, sy, sz, N] = size(data);
        M = psize;
        sc = 1;
    end
    
    mask = true([sx, sy, sz]);
    
    flip = false;
    if M < N
        flip = true;
    end
    
    % define a mask that excludes padded values and extract coordinates
    mask(1:kx, :, :) = 0;
    mask(:, 1:ky, :) = 0;          
    mask(:,:, 1:kz) = 0;
    mask(sx-kx+1:sx, :, :) = 0;
    mask(:, sy-ky+1:sy, :) = 0; 
    mask(:, :, sz-kz+1:sz) = 0;
    x = []; y = []; z = []; 
    for i = kz+1:sz-kz
        [x_, y_] = find(mask(:,:,i) == 1);
        x = [x; x_]; y = [y; y_];  z = [z; i*ones(size(y_))];
    end 
    x = x(:); y = y(:); z = z(:);
    
    if nonlocal
        pinds = find(ones(kernel));
        [pi,pj,pk] = ind2sub(kernel,pinds)
        patchcoords = cat(2,pi,pj,pk);
        pos_img = 1/prod(kernel) * sum((patchcoords - ceil(kernel/2)).^2, 2);
    end

    % Declare variables:
    sigma = zeros(1, numel(x), 'like', data);
    npars = zeros(1, numel(x), 'like', data);
    Sigma = zeros(sx, sy, sz, 'like', data);
    Npars = zeros(sx, sy, sz, 'like', data);

    if coil
        signal = zeros(sc, N, numel(x), 'like', data);
        Signal = zeros(sx, sy, sz, sc, N, 'like', data);
    else
        signal = zeros(1, N, numel(x), 'like', data);
        Signal = zeros(sx, sy, sz, N, 'like', data);
    end
    
    % start denoising
    parfor nn = 1:numel(x)
        X = data(x(nn)-kx:x(nn)+kx, y(nn)-ky:y(nn)+ky, z(nn)-kz:z(nn)+kz, :, :);
        
        if coil
            X = reshape(X, prod(kernel), sc, N); 
        else
            X = reshape(X, prod(kernel), N);
        end
        
        if nonlocal
            Xn = normalize(X);
            min_idx = refine_patch(Xn, kernel, psize, pos_img, coil);
            X = X(min_idx,:,:);
        end
        
        X = reshape(X,[M, N]);
        [s, sigma(nn), npars(nn)] = denoise(X, flip, nrm);
        
        if coil
            signal(:,:,nn) = s(center_idx:psize:end,:);
        else
            signal(:,:,nn) = s(center_idx,:);
        end
    end

    for nn = 1:numel(x)
        Sigma(x(nn), y(nn), z(nn)) = sigma(nn);
        Npars(x(nn), y(nn), z(nn)) = npars(nn);
        Signal(x(nn), y(nn),z(nn), :, :) = signal(:,:,nn);
    end
    Sigma = unpad(Sigma,kernel);
    Npars = unpad(Npars,kernel);
    Signal = unpad(Signal,kernel);
end

function [min_idx] = refine_patch(data, kernel, M, pos_img, coil)
    refval = data(ceil(prod(kernel)/2),:,:);
    if coil 
        refval = repmat(refval,[prod(kernel),1,1]);
        int_img = 1/(size(data,2)*size(data,3)) * sum((data - refval).^2, [2,3]);
    else
        refval = repmat(refval,[prod(kernel),1]);
        int_img = 1/size(data,2) * sum((data - refval).^2, [2]);
    end

    wdists = (pos_img .* int_img);
    [~,min_idx] = mink(wdists, M);
    
%     %%%%%% view patch (debugging) %%%%%%
%     figure;
%     a = reshape(pos_img, [kernel(1), kernel(2), kernel(3)]);
%     b = reshape(pos_img, [kernel(1), kernel(2), kernel(3)]);
%     c = reshape(wdists, [kernel(1), kernel(2), kernel(3)]);
%     subplot(1,3,1)
%     imagesc(a(:,:,ceil(kernel(3)/2))
%     subplot(1,3,2)
%     imagesc(b(:,:,ceil(kernel(3)/2))
%     subplot(1,3,3)
%     imagesc(c(:,:,ceil(kernel(3)/2))
%     keyboard

end

function data_norm = normalize(data)
    %data = abs(data);
    data_norm = abs((data - min(data(:))) * ( 1 / (max(data(:)) - min(data(:))) * 1.0));
end

function data = unpad(data,kernel)
    k = (kernel-1)/2;
    data = data(k(1)+1:end-k(1),k(2)+1:end-k(2),k(3)+1:end-k(3),:,:);
end

function s = shrink(y, gamma)
    % Frobenius norm optimal shrinkage
    % Gavish & Donoho IEEE 63, 2137 (2017)
    % DOI: 10.1109/TIT.2017.2653801
    % Eq (7)

    t = 1 + sqrt(gamma); 
    s = zeros(size(y));
    x = y(y > t); 
    s(y > t) = sqrt((x.^2-gamma-1).^2 - 4*gamma)./x;
end

function [s, sigma, npars] = denoise(X, flip, nrm)
    N = size(X,2);
    M = size(X,1);
    R = min(M,N);

    if flip
        X = X';
        M = size(X,1);
        N = size(X,2);
    end

    % compute PCA eigenvalues 
    [u, vals, v] = svd(X, 'econ');
    vale = vals;
    vals = diag(vals).^2 / N;

    csum = cumsum(vals(R:-1:1)); 
    sigmasq_1 = csum(R:-1:1)./(R:-1:1)'; 

    gamma = (M - (0:R-1)) / N;
    rangeMP = 4*sqrt(gamma(:));
    rangeData = vals(1:R) - vals(R);
    sigmasq_2 = rangeData./rangeMP;

    % sigmasq_2 > sigma_sq1 if signal-components are represented in the
    % eigenvalues
    t = find(sigmasq_2 < sigmasq_1, 1);

    if isempty(t)
        sigma = NaN;
        s = X; 
        t = R+1;
    else
        sigma = single(sqrt(sigmasq_1(t))); 
        vals(t:R) = 0;
        if strcmp(nrm,'h')
            s = u*diag(sqrt(N*vals))*v';
        elseif strcmp(nrm,'frob')
            g=gamma(1);
            vals_frob= sqrt(N)*sigma * diag(shrink(diag(vale)/(sqrt(N)*sigma), g));
            s = u*(vals_frob)*v';
        end
    end
    npars = t-1;

    if flip
        s = s';
    end
end
