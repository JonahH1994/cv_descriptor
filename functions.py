import numpy as np
#import utils*
from utils import *
from numpy import sqrt


def convolution(img, kernel, padding='fill'):
    """Convolve image [img] with [kernel].
    Args:
        [img]       Shape HxW, grayscale image.
        [kernel]    Shape hxw, grayscale image.
        [padding]   Please refer to utils.pad_image
    Rets:
        Shape HxW, image after convolving [kernel] over [img].
    """
    H = img.shape[0]
    W = img.shape[1]

    h = kernel.shape[0]
    w = kernel.shape[1]

    midx = int(h/2)
    midy = int(w/2)

    l = int(h/2)
    l2 = int(w/2)

    padImg = pad_image(img,l,l,l2,l2,padding)

    newImg = np.zeros((H,W))

    if h != 0 and w != 0:
        newKernel = np.zeros((h,w))
        for i in range(l):
            newKernel[i,:] = kernel[h-i-1,:]
            newKernel[h-i-1,:] = kernel[i,:]

        newKernel[l,:] = kernel[l,:]    
        kernel = newKernel
        newKernel = np.zeros((h,w))

        for j in range(l2):
            newKernel[:,j] = kernel[:,w-1-j]
            newKernel[:,w-1-j] = kernel[:,j]

        newKernel[:,l2] = kernel[:,l2]

        kernel = newKernel

    for i in range(H):
        for j in range(W):
            newImg[i,j] = np.sum(kernel*padImg[i:i+h,j:j+w])

    return newImg

def build_kernel(ksize=3,sigma=0.1):
    gaus = np.zeros((ksize,ksize))
    l = int(ksize/2) 
    for i in range(ksize):
        for j in range(ksize):
            gaus[i,j] = np.exp(-((l-i)**2+(l-j)**2)/2/sigma**2)/(2*np.pi*sigma**2)

    A = np.sum(gaus)
    gaus = gaus/A
    return gaus

def compute_Harris_response(img, patchsize, ksize=3, sigma=0.1, epsilon=1e-6):
    """ Computing the response for Harris corner detector. You need to complete
        the following steps in this function:
            1. Use Gaussian filter to smooth the image.
            2. Compute image gradients in both x-direction and y-direction.
            3. Compute M matrix for Harris Corner.
            4. Use det(M)/(trace(M) + epsilon) for the final response.
        NOTE: Though it's also a valid approach to use Sobel filter to do step-1
              and 2 together; in this assignment, please do these them separately.
        NOTE: An alternative way to compute the response is det(M)-k(trace(M))^2.
              In this assignment, we will use det(M)/(trace(M) + epsilon)
    Args:
        [img]           Shape HxW, grayscale image.
        [patchsize]     Size of the patch used for computing Harris response (i.e. M).
        [ksize]         Size of the Gaussian kernel used for smoothing.
        [sigma]         Variance of the Gaussian kernel.
    Rets:
        Shape HxW, matrix of response, where response is det/(trace+1e-6)
    """

    ## Build kernel:
    gaus = build_kernel(ksize,sigma)

    ## Part 1: Gaussian filter to smooth image:
    filtImg = convolution(img,gaus)

    ## Part 2: Compute image gradients in x and y direction:
    
    # First build the gradient vector:
    grad = np.array([-1,0,1.]).reshape((3,1))
    H,W = img.shape
    Ix = np.zeros((H,W))
    Iy = np.zeros((H,W))
    HarrisResponse = np.zeros((H,W))
    
    Ix = convolution(filtImg,grad.T) 
    Iy = convolution(filtImg,grad)
    
    patch = np.eye(patchsize)

    l = int(patchsize/2)

    Ix = pad_image(Ix,l,l,l,l)
    Iy = pad_image(Iy,l,l,l,l)

    for i in range(H):
        for j in range(W):
            # Part 3: Calculate M matrix:
            ix = Ix[i:i+patchsize,j:j+patchsize]
            iy = Iy[i:i+patchsize,j:j+patchsize]

            ixy = np.sum(ix * iy)
            ix = np.sum(ix*ix)
            iy = np.sum(iy*iy)

            M = np.array([[ix,ixy],[ixy,iy]])

            # Part 4: Calculate final response:
            HarrisResponse[i,j] = np.linalg.det(M)/(np.trace(M)+epsilon)

    return HarrisResponse


def compute_local_maxima(response, patchsize=3):
    """None-maximum suppression.
    Args:
        [response]      Shape HxW, containing the response from Harris Corner
                        detection. Assume responses >= 0
        [patchsize]     Size of the patch used to compute the local maxima.
    Rets:
        Shape HxW, value will be 0 if it's not the maximum value;
        if the value is local maxima, keep the original value
    """

    l = int(patchsize/2)
    padImg = pad_image(response,l,l,l,l,'replicate')

    H,W = response.shape
    localMax = np.zeros((H,W))

    for i in range(H):
        for j in range(W):
            neighborhood = padImg[i:i+patchsize,j:j+patchsize]
            maxVal = np.max(neighborhood)

            if response[i,j] == maxVal:
                localMax[i,j] = maxVal

    return localMax


def compute_Harris_corners(img, patchsize, thresh, ksize=3, sigma=0.1, epsilon=1e-6):
    """Harris Corner Detection Function.
    Args:
        [img]           Shape:HxW   Grayscale image.
        [patchsize]     integer     Patch-size used for compute M matrix for Harris corner
                                    detector and the non-maximum supression.
        [thresh]        float       The localtion is a corner when the response > thresh
        [ksize]         int         Kernel size of the Gaussian filter used to smooth the image.
        [sigma]         float       Variance of the Gaussian filter used to smooth the image.
    Rets:
        [corners]       Shape Nx2   Localtions for all of the [N] detected corners.
        [R]             Shape HxW   Harris corner response
    """
    R = compute_Harris_response(img, patchsize, ksize=ksize, sigma=sigma, epsilon=epsilon)
    R = compute_local_maxima(R, patchsize=patchsize)
    corners = np.where(R>thresh)
    corners = np.concatenate((corners[0].reshape((-1,1)), corners[1].reshape((-1,1))),axis=1)
    return corners, R


def compute_mini_sift_desc(img, kp_locs, orientation_norm=False,
        patch_size=32, num_spatial_bins=4, num_ori_bins=8):
    """ Compute the mini-SIFT descriptor described in the homework write-up
        NOTE : Orientation normalization is computed in image patch.
        HINT : `utils.crop_patch` and `utils.compute_histogram` will be useful.
    Args:
        [img]                   Shape:HxW   Input image (in grayscale).
        [kp_locs]               Shape:Nx2   Localtion of the keypoints: (row, col)
        [orientation_norm]      Boolean     Whether do orientation normalization.
        [patch_size]            Int         Size of the image patch.
        [num_spatial_bins]      Int         #spatial bins.
        [num_ori_bins]          Int         #bins for the orientation histogram.
    Rets:
        Shape Nxd where d = [num_spatial_bins]x[num_spatial_bins]x[num_ori_bins].
        The default settings hould produce Nx128.
    """
    padImg = pad_image(img,1,1,1,1)
    H,W = img.shape

    gv = np.array([-1.0,0,1.0]).reshape((3,1))

    Ix = convolution(img,gv.T)
    Iy = convolution(img,gv)
    theta = np.arctan2(Iy,Ix)
    grad = np.sqrt(Ix**2 + Iy**2)

    # Find patches centered at kps:
    pats = []
    num = kp_locs.shape[0]
    l = int(patch_size/2)

    # Orientation normalization if needed:
    if orientation_norm:
        # TO DO:
        for i in range(num):
            bins = np.zeros(num_ori_bins)
            x,y = kp_locs[i,:]
            thetasNew = theta[x-l:x+l,y-l:y+l]

            if thetasNew.size > 0:

                for j in range(x-l,x+l):
                    for k in range(y-l,y+l):
                        ind = int(thetasNew[j-x+l,k-y+l]/float(patch_size))
                        bins[ind] = bins[ind] + grad[j,k]

                maxInd = np.argmax(bins)

                for j in range(x-l,x+l):
                    for k in range(y-l,y+l):
                        ind = int(thetasNew[j-x+l,k-y+l]/float(patch_size))
                        if maxInd != ind:
                            if maxInd > ind:
                                thetasNew[j-x+l,k-y+l] = theta[j,k] (maxInd-ind)
                            elif maxInd < ind:
                                thetasNew[j-x+l,k-y+l] = theta[j,k] * float(maxInd/ind)

                theta[x,y] = np.sum(thetasNew)/float(patch_size*patch_size)
    
    # The number of rows/columns in each spatial bin
    bin_size = int(patch_size/num_spatial_bins)
    descriptors = np.zeros((num,num_spatial_bins**2*num_ori_bins))
    
    grad = pad_image(grad,l,l,l,l)
    theta = pad_image(theta,l,l,l,l)
    padImg = pad_image(img,l,l,l,l)
    
    for in_i in range(num):
        x,y = kp_locs[in_i,:]
        x = x
        y = y
        histogram = np.zeros((num_spatial_bins,num_spatial_bins,num_ori_bins))
        for i in range(patch_size):
            for j in range(patch_size):
                ind = int(theta[x+i,y+j]/float(num_ori_bins))
                row = int((i)/bin_size)
                col = int((j)/bin_size)
                histogram[row,col,ind] = histogram[row,col,ind] + grad[x+i,y+j]
        
        # Add the histogram into vecs:
        vec = histogram.reshape((-1,1))
        mag = np.dot(vec.T,vec)**(1/2)+1e-10
        vec = vec.T/mag
        descriptors[in_i,:] = vec.astype(int)
    
    return descriptors

def find_correspondences(pts1, pts2, desc1, desc2, match_score_type='ratio'):
    """Given two list of key-point locations and descriptions, compute the correspondences.
    Args:
        [pts1]              (N,2)   Array of (row, col) from image 1, keypoints to be matched.
        [pts2]              (M,2)   Array of (row, col) from image 2, keypoints to be matched.
        [desc1]             (N,d)   Discriptor for keypoints at location in [pts1].
        [desc2]             (M,d)   Discriptor for keypoints at location in [pts2].
        [match_score_type]  str     How to compute the match score. Options include 'ssd'|'ratio'.
                                    'ssd'   - use sum of squared distance.
                                    'ratio' - use ratio test, the score will be the ratio.
    Rets:
        Return following three things: [corr], [min_idx], and [scores]
        [corr]              (N,4)   Array of (row_1, col_1, row_2, col_2), where (row_1, col_1) is
                                    a keypoint in [pts1] and (row_2, col_2) is a keypoint in [pts2].
                                    NOTE: you need to find the best match keypoints from [pts2]
                                          for all keypoints in [pts1].
        [min_idx]           (N,)    Index of the matched keypoints in [pts2]. [min_idx[i]] is the index
                                    of the keypoint that appears in [corr[i]].
        [scores]            (N,)    Match score of the correspondences. [scores[i]] is the score for
                                    correspondences [corr[i]]. This will be either SSD or ratio from
                                    the ratio test (i.e. minimum/second_minimum).
    """
    N,d = desc1.shape
    
    corr = np.zeros((N,4))
    min_idx = np.zeros((N,))
    scores = np.zeros((N,))
    
    maxVal = np.iinfo(np.int32).max
    
    for i in range(N):
        diff = desc1[i,:] - desc2
        diff = diff * diff
        diff = np.sum(diff,axis=1)
        scores[i] = np.min(diff)
        min_idx[i] = int(np.argmin(diff))
        if match_score_type == "ratio":
            diff[int(min_idx[i])] = maxVal
            denom = np.min(diff)
            scores[i] = scores[i]/float(denom)
            
        corr[i,:] = [pts1[i,0],pts1[i,1],pts2[int(min_idx[i]),0],pts2[int(min_idx[i]),1]]
            
    return corr, min_idx, scores

def skew(ray):
    
    rets = np.zeros((3,3))
    rets[0,1] = -ray[2]
    rets[0,2] = ray[1]
    rets[1,0] = ray[2]
    rets[1,2] = -ray[0]
    rets[2,0] = -ray[1]
    rets[2,1] = ray[0]

    return rets


def estimate_3D(point1, point2, P1, P2):
    """
    Args:
        [point1]    Shape:(3,)      3D array of homogenous coordinates from image 1.
        [point2]    Shape:(3,)      3D array of homogenous coordinates from image 2.
        [P1]        Shape:(3,4)     Projection matrix for image 1.
        [P2]        Shape:(3,4)     Projection matrix for image 2.
    Rets:
        Return 3D arrary, representing the coordinate of 3D point
        X such that [point1] ~ [P1]X and [point2] ~ [P2]X
    """
    A = np.zeros((4,4))
    A[0,:] = point1[1]*P1[2,:] - P1[0,:]
    A[1,:] = point1[0]*P1[2,:] - P1[1,:]
    A[2,:] = point2[1]*P2[2,:] - P2[0,:]
    A[3,:] = point2[0]*P2[2,:] - P2[1,:]

    u,v,d = np.linalg.svd(A)
    ind = np.argmin(v)
    X = d[ind,:]
    X = X/X[-1]
    
    return X


def estimate_F(corrs):
    """ Eight Point Algorithm with Hartley Normalization.
    Args:
        [corrs]     Nx4     Correspondences between two images, organized
                            in the following way: (row1, col1, row2, col2).
                            Assume N >= 8, raise exception if N < 8.
    Rets:
        The estimated F-matrix, which is (3,3) numpy array.
    """

    n,d = corrs.shape
    if n > 7:

        X = np.zeros((n,2))
        X[:,0] = corrs[:,1]
        X[:,1] = corrs[:,0]

        mu_x = np.mean(X[:,0])
        mu_y = np.mean(X[:,1])

        sig1_x = np.std(X[:,0])
        sig1_y = np.std(X[:,1])

        T1 = np.array([[1/sig1_x,0,-mu_x/sig1_x],[0,1/sig1_y,-mu_y/sig1_y],[0,0,1]])

        X[:,0] = (X[:,0]-mu_x)/sig1_x
        X[:,1] = (X[:,1]-mu_y)/sig1_y

        Xp = np.zeros((n,2))
        Xp[:,0] = corrs[:,3]
        Xp[:,1] = corrs[:,2]

        mu_xp = np.mean(Xp[:,0])
        mu_yp = np.mean(Xp[:,1])

        sig2_y = np.std(Xp[:,1])
        sig2_x = np.std(Xp[:,0])

        T2 = np.array([[1/sig2_x,0,-mu_xp/sig2_x],[0,1/sig2_y,-mu_yp/sig2_y],[0,0,1]])

        Xp[:,0] = (Xp[:,0]-mu_xp)/sig2_x
        Xp[:,1] = (Xp[:,1]-mu_yp)/sig2_y

        one = np.array([1])

        A = np.zeros((n,9))

        for i in range(0,n):
            x = np.concatenate((X[i,:],one)).reshape((3,1))
            xp = np.concatenate((Xp[i,:],one)).reshape((3,1))
            
            row = np.dot(xp,x.T)
            row = np.concatenate((row[0,:],row[1,:],row[2,:]))
            
            A[i,:] = row
            
        AA = np.dot(A.T,A)
        u,w,v = np.linalg.svd(A)

        ind = np.argmin(w)
        F = v[ind,:]

        F = F.reshape((3,3))

        U, D, V = np.linalg.svd(F)
        D[-1] = 0
        Fp = np.dot(U,np.dot(np.diag(D),V))

        F_denorm = np.dot(T2.T,np.dot(Fp,T1))

        mag = np.sum(F_denorm*F_denorm)**(1/2)
        F_denorm /= mag

        return F_denorm

    else:
        raise ValueError("Need more than 8 data points")
        return []

def sym_epipolar_dist(corr, F):
    """Compute the Symmetrical Epipolar Distance.
    Args:
        [corr]  (4,)    (row_1, col_1, row_2, col_2), where row_1, col_1 are points
                        from image 1, and row_2, col_2 are points from image 2.
        [F]     (3,3)   Fundamental matrix from image 1 to image 2.
    Rets:
        Return the symetrical epipolar distance (float)
    """
    x1 = np.array([corr[1],corr[0],1])
    x2 = np.array([corr[3],corr[2],1])
    
    Fp1 = np.dot(F,x1)
    Fp2 = np.dot(F.T,x2)
    
    distance = np.dot(x2.T,Fp1)**2*(1/np.dot(Fp1[0:2],Fp1[0:2])+1/np.dot(Fp2[0:2],Fp2[0:2]))
    
    return distance

def evaluate_metric(metric, F, data, N,inlier):
    output = np.zeros((N,))
    for i in range(N):
        output[i] = metric(data[i,:].T,F)
        
    output = output < inlier
    return output


def ransac(data, hypothesis, metric, sample_size, num_iter, inlier_thresh):
    """ Implement the general RANSAC framework.
    Args:
        [data]          (N,d) numpy array, representing the data to fit.
        [hypothesis]    Function that takes a (m,d) numpy array, return a model
                        (represented as a numpy array). For the case of F-matrix
                        estimation, hypothesis takes Nx4 data (i.e. the
                        correspondences) and return the 3x3 F-matrix.
        [metric]        Function that take an entry from [data] and an output
                        of [hypothesis]; it returns a score (float) mesuring how
                        well the data entry fits the output hypothesis.
                        ex. metric(data[i], hypothesis(data)) -> score (float).
        [sample_size]   Number of entries to sample for each iteration.
        [num_iter]      Number of iterations we run RANSAC.
        [inlier_thres]  The threshold to decide whether a data point is inliner.
    Rets:
        Returning the best fit model [model] and the inliner mask [mask].
        [model]         The best fit model (i.e. having fewest outliner ratio).
        [mask]          Mask for inliners. [mask[i]] is 1 if data[i] is an inliner
                        for the output model [model], 0 otherwise.
    """
    N,d = data.shape
    
    thresh = np.zeros((N,))
    max_thresh = np.zeros((N,))
    
    ind = np.random.randint(0,high=N,size=(sample_size,))
    F_max = hypothesis(data[ind,:])
    max_thresh = evaluate_metric(metric, F_max, data, N,inlier_thresh)
    max_score = np.sum(max_thresh)
    
    for i in range(num_iter-1):
        ind = np.random.randint(0,high=N,size=(sample_size,))
        F = hypothesis(data[ind,:])
        thresh = evaluate_metric(metric,F,data,N,inlier_thresh)
        score = np.sum(thresh)
        
        if score > max_score:
            max_score = score
            F_max = F
            max_thresh = thresh
    return F_max, max_thresh

def estimate_F_ransac(corr, num_iter, inlier_thresh):
    """Use normalized 8-point algorithm, symetrical epipolar distance, and
       RANSAC to estimate F-matrix.
       NOTE: Please reuse the `ransac`, `sym_epipolar_dist`, and `estimate_F`
             functions implemented above.
    Args:
        [corrs]         Nx4     Correspondences between two images, organized
                                in the following way: (row1, col1, row2, col2).
        [num_iter]      Number of iterations we run RANSAC.
        [inlier_thres]  The threshold to determine whether the data point is inliner.
    Rets:
        The estimated F-matrix, which is (3,3) numpy array.
    """
    sample_size = 8
    _, mask = ransac(corr,estimate_F,sym_epipolar_dist,sample_size,num_iter,inlier_thresh)
    ind = [i for i in range(len(mask)) if mask[i] == 1]
    F = estimate_F(corr[ind,:])
    
    return F


