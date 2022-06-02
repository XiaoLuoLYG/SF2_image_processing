import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import rowdec,rowdec2
from cued_sf2_lab.simple_image_filtering import halfcos, convse
from cued_sf2_lab.laplacian_pyramid import rowint, rowint2
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dwt import dwt,idwt
from cued_sf2_lab.dct import regroup

"""
Predefined function for LBT
"""
#step optimisation
def get_step_size_LBT(X, C, s, N, acc,ratios=[1,1,1,1,1]):
    #prefilter
    Pf, Pr = pot_ii(N,s)
    t = np.s_[N//2:-N//2]
    Xp = X.copy()  # copy the non-transformed edges directly from X
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    Y = colxfm(colxfm(Xp, C).T, C).T #dct
    
    X_quantised = quantise(X, 17)
    direct_rms = np.std(X - X_quantised)
    steps= np.arange(1, 30.0, acc)
    rms_diffenences = [] #initiate
    rms=[]
    #print(f'Direct RMS Error: {direct_rms}')
    for step in steps:
        Yq = quantise(Y, step*ratios[0])
        Zq = colxfm(colxfm(Yq.T, C.T).T, C.T)#reverse dct
        #post filtering
        Zqp = Zq.copy()  #copy the non-transformed edges directly from Z
        Zqp[:,t] = colxfm(Zqp[:,t].T, Pr.T).T
        Zqp[t,:] = colxfm(Zqp[t,:], Pr.T)
        rms.append(np.std(Zqp - X))
        rms_diffenences.append(abs(direct_rms-np.std(Zqp - X)))
    #print(f'Index of optimum: {np.argmin(rms_diffenences)}')
    #print(f'Optimum Step Size: {steps[np.argmin(rms_diffenences)]}')
    #print(f'Optimum RMS Error: {rms[np.argmin(rms_diffenences)]}')
    #print(f'Optimum RMS Error difference (with original): {rms_diffenences[np.argmin(rms_diffenences)]}')
    return steps[np.argmin(rms_diffenences)]

def dctbpp(Yr, N):
    gaps = np.linspace(0,256,N+1)
    BYr = 0
    for i in range(N):
        for j in range(N):
            Ys = Yr[int(gaps[i]):int(gaps[i+1]),int(gaps[j]):int(gaps[j+1])]
            BYs = bpp(Ys) * Ys.shape[0] * Ys.shape[1]
            BYr += BYs
    return BYr

def LBT_compression(X, rise_ratio = 0.5):
    #Direct quantised
    X_quantised = quantise(X, 17)
    bppXq=bpp(X_quantised)
    TbeXq = bppXq * X_quantised.shape[0] * X_quantised.shape[1]

    s_values=[]
    CRs=[] 
    C = dct_ii(8)
    #s_optimisation using 8x8 dct
    for i in np.arange(1, 2, 0.1):
        s_values.append(i)
        Pf, Pr = pot_ii(8,i) #filter with different s

        #pre filtering
        t = np.s_[8//2:-8//2]  # N is the DCT size, I is the image size
        Xp = X.copy()  # copy the non-transformed edges directly from X
        Xp[t,:] = colxfm(Xp[t,:], Pf)
        Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
        #dct
        Y = colxfm(colxfm(Xp, C).T, C).T
        step_size = get_step_size_LBT(X, C, i,8, 0.1)
        Yq = quantise(Y, step_size, step_size*rise_ratio)
        Yr = regroup(Yq, 8)/8

        CRs.append(TbeXq/dctbpp(Yr, 8))

    Optimum_S = s_values[np.argmax(CRs)]
    Optimum_S = round(Optimum_S,3)
    Optimum_CR = CRs[np.argmax(CRs)]
    Optimum_CR = round(Optimum_CR,3)
    Pf, Pr = pot_ii(8,Optimum_S) #filter with different s

    #pre filtering
    t = np.s_[8//2:-8//2]  # N is the DCT size, I is the image size
    Xp = X.copy()  # copy the non-transformed edges directly from X
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    #dct
    Y = colxfm(colxfm(Xp, C).T, C).T
    step_size = get_step_size_LBT(X, C, i,8, 0.1)
    Yq = quantise(Y, step_size, step_size*rise_ratio)
    Yr = regroup(Yq, 8)/8
    #inverse dct
    Zq = colxfm(colxfm(Yq.T, C.T).T, C.T)
    #post filtering
    Zqp = Zq.copy()  #copy the non-transformed edges directly from Z
    Zqp[:,t] = colxfm(Zqp[:,t].T, Pr.T).T
    Zqp[t,:] = colxfm(Zqp[t,:], Pr.T)
    print(f'Optimum CR after LBT is:{Optimum_CR} at s = {Optimum_S}' )
    #std_error = np.std(Zqp - X)
    print(f"coded bits needed after LBT: {dctbpp(Yr, 8)}")
    return Optimum_CR, Zqp, Yr

"""
Predefined function for DWT
"""
def nlevdwt(X, n):
    # your code here
    if n == 0:
        Y = X
    for level in np.linspace(1,n,n):
        if int(level) == 1:
            m=32
            Y=dwt(X)
        else:
            m = m//2
            Y[:m,:m] = dwt(Y[:m,:m])
    return Y

def nlevidwt(Y, n):
    Yc = Y.copy()
    if n == 0:
        Xr = Y
    for level in np.linspace(n,1,n):
        if int(level) == n and int(level != 1):
            m = 32//(2**int(level-1))
            Yc[:m,:m] = idwt(Yc[:m,:m])
        elif int(level) == 1:
            Xr = idwt(Yc)
        else:
            m *= 2
            Yc[:m,:m] = idwt(Yc[:m,:m])
    return Xr


def get_step_size_DWT(X, n, acc,ratios=[1,1,1,1,1]):
    #prefilter
    Y = nlevdwt(X,n) #dwt
    X_quantised = quantise(X, 17)
    direct_rms = np.std(X - X_quantised)
    steps= np.arange(1, 30.0, acc)
    rms_diffenences = [] #initiate
    rms=[]
    #print(f'Direct RMS Error: {direct_rms}')
    for step in steps:
        Yq = quantise(Y, step*ratios[0])
        Zq = nlevidwt(Yq,n)
        rms.append(np.std(Zq - X))
        rms_diffenences.append(abs(direct_rms-np.std(Zq - X)))
    #print(f'Index of optimum: {np.argmin(rms_diffenences)}')
    #print(f'Optimum Step Size: {steps[np.argmin(rms_diffenences)]}')
    #print(f'Optimum RMS Error: {rms[np.argmin(rms_diffenences)]}')
    #print(f'Optimum RMS Error difference (with original): {rms_diffenences[np.argmin(rms_diffenences)]}')
    return steps[np.argmin(rms_diffenences)]

def quantdwt(Y: np.ndarray, dwtstep: np.ndarray):
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    # your code here
    row_num = dwtstep.shape[0]
    col_num = dwtstep.shape[1]
    dwtent = np.empty((row_num,col_num))
    Yq = Y
    bits=0
    for i in range(col_num - 1):
        m = Y.shape[0] // (2**int(i))
        half_m = m//2
        for k in range(row_num):
            if int(k) == 0:
                Yq[:half_m,half_m:] = quantise(Y[:half_m,half_m:],dwtstep[int(k),int(i)])
                dwtent[k,i] = bpp(Yq[:half_m,half_m:])
                bits += bpp(Yq[:half_m,half_m:]) * m/2 * m/2
            if int(k) == 1:
                Yq[half_m:,:half_m] = quantise(Y[half_m:,:half_m],dwtstep[int(k),int(i)])
                dwtent[k,i] = bpp(Yq[half_m:,:half_m])
                bits += bpp(Yq[half_m:,:half_m]) * m/2 * m/2

            if int(k) == 2:
                Yq[half_m:,half_m:] = quantise(Y[half_m:,half_m:],dwtstep[int(k),int(i)])
                dwtent[k,i] = bpp(Yq[half_m:,half_m:])
                bits += bpp(Yq[half_m:,half_m:]) * m/2 * m/2
    
    # quantise the final low-pass
    w = Y.shape[0]/2**(col_num-1)
    n = int(col_num-1)
    half_w = int(w//2)
    #print(half_w)
    Yq[:half_w,:half_w] = quantise(Y[:half_w,:half_w],dwtstep[0,n])
    dwtent[0,n] =bpp(Yq[:half_w,:half_w])
    bits += bpp(Yq[:half_w,:half_w]) * w/2 *w/2
    return Yq, dwtent,bits

def DWT_compression(X, rise_ratio=0.5):
    #Direct quantised
    X_quantised = quantise(X, 17)
    bppXq=bpp(X_quantised)
    TbeXq = bppXq * X_quantised.shape[0] * X_quantised.shape[1]
    CRs = []
    #optimized n 
    n_values = [1,2]
    for n in n_values:
        Y = nlevdwt(X, n)
        dwtstep = np.full((3,n+1),get_step_size_DWT(X, n, 0.1))
        Yq, dwtent,bits = quantdwt(Y,dwtstep)
        if bits != 0:
            CRs.append(TbeXq/bits)
        else:
            CRs.append(0)
    Optimum_n = n_values[np.argmax(CRs)]
    Optimum_CR = CRs[np.argmax(CRs)]
    Y = nlevdwt(X, Optimum_n)
    dwtstep = np.full((3,Optimum_n + 1),get_step_size_DWT(X, Optimum_n, 0.1))
    Yq, dwtent, bits = quantdwt(Y, dwtstep)
    Zq = nlevidwt(Yq, Optimum_n)
    #print(f'optimum CR for DWT is:{Optimum_CR}')
    #print(f'optimum level for DWT is:{Optimum_n}')
    return Optimum_CR, Zq, Yq

'''
Main Compression Strategy
INPUT: original image and rise_ratio (rise1 = rise_ratio*step)
OUTPUT: LBT+DWT regrouped image
Will display the coded bits and compression ratio after each step
'''
def compression_scheme(X, rise_ratio = 0.5):
    print(f"******Rise ratio = {rise_ratio} (default = 0.5)******")
    '''
    Reference Scheme
    '''
    X_quantised = quantise(X, 17)
    bppXq=bpp(X_quantised)
    TbeXq = bppXq * X_quantised.shape[0] * X_quantised.shape[1]
    
    '''
    LBT with 8x8 DCT
    Return the Optimum CR for LBT, the reconstructed LBT image (Zqp)
    and the regrouped image for the following DWT (Yr)
    '''
    
    Optimum_CR, Zqp, Yr = LBT_compression(X, rise_ratio = rise_ratio)
    
    '''
    For each 32x32 block carry out DWT (up to 2 level)
    Return Optimum CR for each block, the reconstructed image for each block (Zq)
    and the DWT image for each block (Yrdwt)
    '''
    gaps = np.linspace(0,256,9) # [  0.,  32.,  64.,  96., 128., 160., 192., 224., 256.]
    for i in range(8):
        for j in range(8):
            Yr_sub = Yr[int(gaps[i]):int(gaps[i+1]),int(gaps[j]):int(gaps[j+1])] # each sub image
            Optimum_CR_DWT, Zq, Yrdwt = DWT_compression(Yr_sub, rise_ratio = rise_ratio) # DWT for each sub image
            Yr[int(gaps[i]):int(gaps[i+1]),int(gaps[j]):int(gaps[j+1])] = Yrdwt # substitude each block with the DWT image
    print(f"coded bits needed after DWT: {dctbpp(Yr, 8)}")
    print(f'Optimum Overall CR is:{round(TbeXq/dctbpp(Yr, 8),3)}' )
    return Yr


def main():
    X, cmaps_dict = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})
    X = X - 128.0
    Yr = compression_scheme(X,rise_ratio = 1)

    fig, ax = plt.subplots()
    plot_image(Yr, ax=ax)
    ax.set(title=f"compressed_lighthouse.mat")
    print("compressed image saved")

    #fig, axs = plt.subplots(1, 2)
    #plot_image(X, ax=axs[0])
    #axs[0].set(title="Y")
    #plot_image(Yr, ax=axs[1])
    #axs[1].set(title="Z")

    plt.savefig(f"compressed_lighthouse.png")



if __name__ == "__main__":
    main()

