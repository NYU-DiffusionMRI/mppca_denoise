function [Signal, varargout] = MPnonlocal(data, varargin)    
    % MPnonlocal Denoise 4d magnitude data (x, y, z, dirs) or 5d complex data 
    %   (x, y, z, coils, dirs) and estimate 3d noise maps and significant 
    %   parameter maps using nonlocal patching and eigenvalue shrinkage in
    %   the MPPCA framework
    %
    %   [Signal, Sigma, Nparams] = MPnonlocal(data, kernel, patchsize, norm)
    %   output:
    %       - Signal: [x, y, z, N] for 4d (real or complex) [x, y, z, C, N] for 5d
    %       (complex)
    %       - Sigma: [x, y, z] noise map
    %       - Npars: [x, y, z] map of the number of signal carrying
    %       components
    %       - Sigma_after: [x, y, z] noise map (sigma after denoiosing) estimated using Jespersen et al
    %       method
    %   input:
    %       - data: [x, y, z, N] for 4d (real or complex) [x, y, z, C, N] for 5d
    %       (complex)
    %       - kernel: (optional) default smallest isotropic box window where prod(kernel) > n volumes. Must be odd.
    %       - patchtype: (optional) default is 'box'. Can alternatively be
    %       set to 'nonlocal'
    %       - patchsize: (optional) Number of voxels to include in nonlocal
    %       patch. For nonlocal denoising only. n volumes < patch size <
    %       prod(kernel). If it is not set a default option of a nonlocal
    %       patch 20% smaller than prod(kernel) will be used.
    %       - shrink: (optional) default 'threshold'. Can be set to 'threshold' for hard
    %       thresholding of eigenvalues or 'frob' to implement eigenvalue
    %       shrinkage using the frobenius norm.
    %       - exp: (optional) default is 1. Options are 1, 2 and 3
    %       correspoinding to Veraart 2016, Cordero-Grande, and the
    %       "in-between method" respectively.
    %
    %   usage: 
    %       - box patch denoising:
    %           [Signal, Sigma, Npars] = MPnonlocal(data, [5,5,5])
    %       - shrinkage denoisng:
    %           [Signal, Sigma, Npars] = MPnonlocal(data, [5,5,5], 'patchtype','box','shrink','frob')
    %       - nonlocal denoisng:
    %           [Signal, Sigma, Npars] = MPnonlocal(data, [5,5,5], 'patchtype','nonlocal','patchsize',100)
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

    % set defaults
    
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
    
    defaultShrink = 'threshold';
    defaultExp = 1;
    nvols = size(data, ndims(data));
    p_ = (1:2:nvols);
    pf_ = find(p_.^3 >= nvols, 1);
    defaultKernel = p_(pf_);
    defaultPatchtype = 'box';
    defaultPatchsize = defaultKernel^3;
    defaultCrop = 0;

    % parse input arguments
    p = inputParser;
    addRequired(p,'data');
    addOptional(p,'kernel', defaultKernel);
    addOptional(p,'patchtype', defaultPatchtype);
    addOptional(p,'patchsize', defaultPatchsize);
    addOptional(p,'shrink', defaultShrink);
    addOptional(p,'exp', defaultExp);
    addOptional(p,'crop',defaultCrop);
    parse(p, data, varargin{:});
    
    if isscalar(p.Results.kernel)
        kernel = [p.Results.kernel, p.Results.kernel, p.Results.kernel];
    else
        kernel = p.Results.kernel;
    end
    kernel = kernel + (mod(kernel, 2)-1);  
    
    if any(kernel > [size(data,1),size(data,2),size(data,3)])
        error(['kernel size of ',num2str(kernel), ' exceeds data size along dimention ',...
            num2str(find(kernel>size(data,[1,2,3]))),', specify a smaller kernel extent']);
    end
    
    if strcmp(p.Results.patchtype,'box')
        psize = prod(kernel);
        nonlocal = false;
        center_idx = ceil(prod(kernel)/2);
        pos_img = [];
    elseif strcmp(p.Results.patchtype,'nonlocal')
        if p.Results.patchsize >= prod(kernel)
            warning('selecting sane default nonlocal patch size')
            psize = floor(prod(kernel) - 0.2*prod(kernel));
            if psize <= nvols
                psize = nvols + 1;
            end
        else
            psize = p.Results.patchsize;
        end
        nonlocal = true;
        center_idx = 1;
    else
        error('patchtype options are "box" or "nonlocal"');
    end
    
    nrm = p.Results.shrink;
    exp = p.Results.exp;
    cropdist = p.Results.crop;
    
    if p.Results.patchsize ~= prod(kernel) && strcmp(p.Results.patchtype,'box')
        warning('patchsize argument does not affect box kernel');
    end
    
    disp('Denoising data using parameters:')
    disp(['kernel     = [',num2str(kernel),']'])
    disp(['patch type = ',p.Results.patchtype]);
    disp(['patch size = ',num2str(psize)]);
    disp(['shrinkage  = ',p.Results.shrink]);
    disp(['algorithm  = ',num2str(exp)]);
    disp(['cropdist   = ',num2str(cropdist)]);
    
    % begin processing here
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
    
    % define a mask that excludes padded values and extract coordinates
    [x,y,z] = get_voxel_coords(sx,sy,sz,kx,ky,kz);
    
    if nonlocal
        [pi, pj, pk] = ind2sub(kernel, find(ones(kernel)));
        patchcoords = cat(2,pi,pj,pk);
        pos_img = 1/prod(kernel) * sum((patchcoords - ceil(kernel/2)).^2, 2);
    end

    % Declare variables:
    sigma = zeros(1, numel(x), 'like', data);
    sigma_after = zeros(1, numel(x), 'like', data);
    npars = zeros(1, numel(x), 'like', data);
    Sigma = zeros(sx, sy, sz, 'like', data);
    Sigma_after = zeros(sx, sy, sz, 'like', data);
    Npars = zeros(sx, sy, sz, 'like', data);

    if coil
        signal = zeros(sc, N, numel(x), 'like', data);
        Signal = zeros(sx, sy, sz, sc, N, 'like', data);
    else
        signal = zeros(1, N, numel(x), 'like', data);
        Signal = zeros(sx, sy, sz, N, 'like', data);
    end
    
    % start denoising
    %xi = floor(4*length(x)/7);
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
        [s, sigma(nn), npars(nn), sigma_after(nn)] = denoise(X, nrm, exp, cropdist);
        
        if coil
            signal(:,:,nn) = s(center_idx:psize:end,:);
        else
            signal(:,:,nn) = s(center_idx,:);
        end
        
    end
    
    for nn = 1:numel(x)
        Sigma(x(nn), y(nn), z(nn)) = sigma(nn);
        Sigma_after(x(nn), y(nn), z(nn)) = sigma_after(nn);
        Npars(x(nn), y(nn), z(nn)) = npars(nn);
        Signal(x(nn), y(nn),z(nn), :, :) = signal(:,:,nn);
    end
    Sigma = unpad(Sigma,kernel);
    Sigma_after = unpad(Sigma_after,kernel);
    Npars = unpad(Npars,kernel);
    Signal = unpad(Signal,kernel);
    
    varargout{1} = Sigma;
    varargout{2} = Npars;
    varargout{3} = Sigma_after;
end

function [min_idx] = refine_patch(data, kernel, M, pos_img, coil)
    refval = data(ceil(prod(kernel)/2),:,:);
    if coil 
        refval = repmat(refval,[prod(kernel),1,1]);
        int_img = 1/(size(data,2)*size(data,3)) * sum((data - refval).^2, [2,3]);
    else
        refval = repmat(refval,[prod(kernel),1]);
        %int_img = 1/size(data,2) * sum((data(:,1) - refval(1)).^2, [2]);
        int_img = 1/size(data,2) * sum((data - refval).^2, [2]);
    end

    wdists = (pos_img .* int_img);
    [~,min_idx] = mink(wdists, M);
end

function data_norm = normalize(data)
    data_norm = zeros(size(data));
    for i = 1:size(data,4)
        data_ = data(:,:,:,i);
        data_norm(:,:,:,i) = abs(data_./max(data_(:)));
    end
end

function [x,y,z] = get_voxel_coords(sx,sy,sz,kx,ky,kz)
    mask = true([sx, sy, sz]);
    mask(1:kx, :, :) = 0;
    mask(:, 1:ky, :) = 0;          
    mask(:, :, 1:kz) = 0;
    mask(sx-kx+1:sx, :, :) = 0;
    mask(:, sy-ky+1:sy, :) = 0; 
    mask(:, :, sz-kz+1:sz) = 0;
    maskinds = find(mask);
    [x,y,z] = ind2sub(size(mask),maskinds);
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

function [s, sigma, npars, sigma_after] = denoise(X, nrm, exp, tn)
    N = size(X,2);
    M = size(X,1);
    Mp = min(M,N);
    Np = max(M,N);

    if M < N
        X = X.';
    end

    % compute PCA eigenvalues 
    [u, vals, v] = svd(X, 'econ');
    vals = diag(vals).^2;
    
    [vals, order] = sort(vals,'descend');
    u = u(:,order); v = v(:,order);
    
    ptn = (0:Mp-1-tn)';
    p = (0:Mp-1)';
    csum = cumsum(vals,'reverse');
    
    
    if exp == 1 % veraart 2016
        sigmasq_1 = csum./((Mp-p).*Np); 
        rangeMP = 4*sqrt((Mp-ptn).*(Np-tn));
    elseif exp == 2 % cordero-grande
        sigmasq_1 = csum./((Mp-p).*(Np-p)); 
        rangeMP = 4*sqrt((Mp-ptn).*(Np-ptn));
    elseif exp == 3 % jespersen
        sigmasq_1 = csum./((Mp-p).*(Np-p)); 
        rangeMP = 4*sqrt((Np-tn).*(Mp));
    end
    rangeData = vals(1:Mp-tn) - vals(Mp-tn);
    sigmasq_2 = rangeData./rangeMP;
        
    t = find(sigmasq_2 < sigmasq_1(1:end-tn),1);

    if isempty(t)
        sigma = NaN;
        npars = NaN;
        s = X; 
        sigma_after = NaN;
    else
        sigma = sqrt(sigmasq_1(t)); 
        npars = t-1;
        if strcmp(nrm,'threshold')
            vals(t:end) = 0;
            s = u*diag(sqrt(vals))*v';
        elseif strcmp(nrm,'frob')
            vals_frob= sqrt(Mp)*sigma * shrink(sqrt(vals)./(sqrt(Mp)*sigma), Np/Mp);
            s = u*(diag(vals_frob))*v';
        end
        
        s2_after = sigma.^2 - csum(t)/(Mp*Np);   
        sigma_after = sqrt(s2_after);
    end

    if M < N
        s = s.';
    end
        
end
