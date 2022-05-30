
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


#LP
#h_lp= = np.array([1, 4, 6, 4, 1]) / 16.0

def get_step_size_LP(X, h, layers, acc,ratios=[1,1,1,1,1]):
    X_quantised = quantise(X, 17)
    direct_rms = np.std(X - X_quantised)
    steps= np.arange(1, 30.0, acc)
    rms_diffenences = [] #initiate
    rms=[]
    print(f'Direct RMS Error: {direct_rms}')

    Y0, Y1, Y2, Y3, X4 = py4enc(X, h) #encoding
    for step in steps:
        Y0_q = quantise(Y0, step*ratios[0])
        Y1_q = quantise(Y1, step*ratios[1])
        Y2_q = quantise(Y2, step*ratios[2])
        Y3_q = quantise(Y3, step*ratios[3])
        X4_q = quantise(X4, step*ratios[4])
        Z3_q, Z2_q, Z1_q, Z0_q = py4dec(Y0_q, Y1_q, Y2_q, Y3_q, X4_q, h)
        rms.append(np.std(X - Z0_q))
        rms_diffenences.append(abs(direct_rms-np.std(X - Z0_q)))
    #print(f'Index of optimum: {np.argmin(rms_diffenences)}')
    #print(f'Optimum Step Size: {steps[np.argmin(rms_diffenences)]}')
    #print(f'Optimum RMS Error: {rms[np.argmin(rms_diffenences)]}')
    #print(f'Optimum RMS Error difference (with original): {rms_diffenences[np.argmin(rms_diffenences)]}')
    return steps[np.argmin(rms_diffenences)]

def LP_compression(X, h):
    #Direct quantised
    X_quantised = quantise(X, 17)
    bppXq=bpp(X_quantised)
    TbeXq = bppXq * X_quantised.shape[0] * X_quantised.shape[1]

    #optimum step size (CSS)
    step_size=get_step_size_LP(X, h, 4, 0.1)

    #LP 4 layer encoder
    Y0, Y1, Y2, Y3, X4 = py4enc(X, h)
    Z3, Z2, Z1, Z0 = py4dec(Y0, Y1, Y2, Y3, X4, h)

    #quantised bits
    bppY0 = bpp(quantise(Y0, step_size))
    TbeY0 = bppY0 * Y0.shape[0] * Y0.shape[1]
    bppY1 = bpp(quantise(Y1, step_size))
    TbeY1 = bppY1 * Y1.shape[0] * Y1.shape[1]
    bppY2 = bpp(quantise(Y2, step_size))
    TbeY2 = bppY2 * Y2.shape[0] * Y2.shape[1]
    bppY3 = bpp(quantise(Y3, step_size))
    TbeY3 = bppY3 * Y3.shape[0] * Y3.shape[1]
    bppX4 = bpp(quantise(X4, step_size))
    TbeX4 = bppX4 * X4.shape[0] * X4.shape[1]

    cr = TbeXq/(TbeY0+TbeY1+TbeY2+TbeY3+TbeX4)#compression ratio
    
    return cr, Z0

#DCT

def dctbpp(Yr, N):
    gaps = np.linspace(0,256,N+1)
    BYr = 0
    for i in range(N):
        for j in range(N):
            Ys = Yr[int(gaps[i]):int(gaps[i+1]),int(gaps[j]):int(gaps[j+1])]
            BYs = bpp(Ys) * Ys.shape[0] * Ys.shape[1]
            BYr += BYs
    return BYr

def get_step_size_DCT(X, C,acc,ratios=[1,1,1,1,1]):
    Y = colxfm(colxfm(X, C).T, C).T #dct
    X_quantised = quantise(X, 17)
    direct_rms = np.std(X - X_quantised)
    steps= np.arange(1, 30.0, acc)
    rms_diffenences = [] #initiate
    rms=[]
    #print(f'Direct RMS Error: {direct_rms}')
    for step in steps:
        Yq = quantise(Y, step*ratios[0])
        Zq = colxfm(colxfm(Yq.T, C.T).T, C.T)#reverse dct
        rms.append(np.std(Zq - X))
        rms_diffenences.append(abs(direct_rms-np.std(Zq - X)))
    #print(f'Index of optimum: {np.argmin(rms_diffenences)}')
    #print(f'Optimum Step Size: {steps[np.argmin(rms_diffenences)]}')
    #print(f'Optimum RMS Error: {rms[np.argmin(rms_diffenences)]}')
    #print(f'Optimum RMS Error difference (with original): {rms_diffenences[np.argmin(rms_diffenences)]}')

    return steps[np.argmin(rms_diffenences)]

def DCT_compression(X, N_values = [4, 8, 16]):
    #Direct quantised
    X_quantised = quantise(X, 17)
    bppXq=bpp(X_quantised)
    TbeXq = bppXq * X_quantised.shape[0] * X_quantised.shape[1]

    CRs = []
    for N in N_values:
        C = dct_ii(N)
        #DCT encoder
        Y = colxfm(colxfm(X, C).T, C).T
        Yq = quantise(Y, get_step_size_DCT(X, C, 0.1))
        Yr = regroup(Yq, N)/N
        CRs.append(TbeXq/dctbpp(Yr, N))

    Optimum_N = N_values[np.argmin(CRs)]
    Optimum_CR = CRs[np.argmin(CRs)]
    C = dct_ii(Optimum_N)
    Y = colxfm(colxfm(X, C).T, C).T
    Yq = quantise(Y, get_step_size_DCT(X, C, 0.1))
    Zq = colxfm(colxfm(Yq.T, C.T).T, C.T)
    return Optimum_CR, Zq


#LBT

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

def LBT_compression(X):
    #Direct quantised
    X_quantised = quantise(X, 17)
    bppXq=bpp(X_quantised)
    TbeXq = bppXq * X_quantised.shape[0] * X_quantised.shape[1]

    s_values=[]
    CRs=[] 
    C = dct_ii(8)
    #s_optimisation using 8x8 dct
    for i in np.arange(1, 2, 0.01):
        s_values.append(i)
        Pf, Pr = pot_ii(8,i) #filter with different s

        #pre filtering
        t = np.s_[8//2:-8//2]  # N is the DCT size, I is the image size
        Xp = X.copy()  # copy the non-transformed edges directly from X
        Xp[t,:] = colxfm(Xp[t,:], Pf)
        Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
        #dct
        Y = colxfm(colxfm(Xp, C).T, C).T
        Yq = quantise(Y, get_step_size_LBT(X, C, i,8, 0.1))
        Yr = regroup(Yq, 8)/8

        CRs.append(TbeXq/dctbpp(Yr, 8))

    Optimum_S = s_values[np.argmin(CRs)]
    Optimum_CR = CRs[np.argmin(CRs)]
    Pf, Pr = pot_ii(8,Optimum_S) #filter with different s

    #pre filtering
    t = np.s_[8//2:-8//2]  # N is the DCT size, I is the image size
    Xp = X.copy()  # copy the non-transformed edges directly from X
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    #dct
    Y = colxfm(colxfm(Xp, C).T, C).T
    Yq = quantise(Y, get_step_size_LBT(X, C, i,8, 0.1))
    Yr = regroup(Yq, 8)/8
    #inverse dct
    Zq = colxfm(colxfm(Yq.T, C.T).T, C.T)
    #post filtering
    Zqp = Zq.copy()  #copy the non-transformed edges directly from Z
    Zqp[:,t] = colxfm(Zqp[:,t].T, Pr.T).T
    Zqp[t,:] = colxfm(Zqp[t,:], Pr.T)

    return Optimum_CR, Zqp



#DWT

def nlevdwt(X, n):
    # your code here
    if n == 0:
        Y = X
    for level in np.linspace(1,n,n):
        if int(level) == 1:
            m=256
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
            m = 256//(2**int(level-1))
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
    print(f'Optimum Step Size: {steps[np.argmin(rms_diffenences)]}')
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

def DWT_compression(X):
    #Direct quantised
    X_quantised = quantise(X, 17)
    bppXq=bpp(X_quantised)
    TbeXq = bppXq * X_quantised.shape[0] * X_quantised.shape[1]
    CRs = []
    #optimized n 
    n_values = [2,3,4,5,6]
    for n in n_values:
        Y = nlevdwt(X, n)
        dwtstep = np.full((3,n+1),get_step_size_DWT(X, n, 0.1))
        Yq, dwtent,bits = quantdwt(Y,dwtstep)
        CRs.append(TbeXq/bits)

    Optimum_n = n_values[np.argmin(CRs)]
    Optimum_CR = CRs[np.argmin(CRs)]
    Y = nlevdwt(X, Optimum_n)
    dwtstep = np.full((3,Optimum_n + 1),get_step_size_DWT(X, Optimum_n, 0.1))
    Yq, dwtent, bits = quantdwt(Y, dwtstep)
    Zq = nlevidwt(Yq, Optimum_n)
    print(f'optimum CR for DWT is:{Optimum_CR}')
    return Optimum_CR, Zq

def main():
    X, cmaps_dict = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})
    X = X - 128.0
    fig, ax = plt.subplots()
    plot_image(X, ax=ax)
    ax.set(title="lighthouse.mat")
    print("a")

    plt.savefig("tmp.png")

if __name__ == "__main__":
    main()

























def py4enc(X, h):
    #Y0
    X1 = rowdec(X,h) 
    X1 = rowdec(X1.T,h)
    X1=X1.T
    X1_inter = rowint(X1, 2*h)
    X1_inter = rowint(X1_inter.T, 2*h)
    X1_inter = X1_inter.T
    Y0 = X - X1_inter
    #Y1
    X2 = rowdec(X1,h) 
    X2 = rowdec(X2.T,h)
    X2=X2.T
    X2_inter = rowint(X2, 2*h)
    X2_inter = rowint(X2_inter.T, 2*h)
    X2_inter = X2_inter.T
    Y1 = X1 - X2_inter
    #Y2
    X3 = rowdec(X2,h) 
    X3 = rowdec(X3.T,h)
    X3=X3.T
    X3_inter = rowint(X3, 2*h)
    X3_inter = rowint(X3_inter.T, 2*h)
    X3_inter = X3_inter.T
    Y2 = X2 - X3_inter
    #Y3
    X4 = rowdec(X3,h) 
    X4 = rowdec(X4.T,h)
    X4=X4.T
    X4_inter = rowint(X4, 2*h)
    X4_inter = rowint(X4_inter.T, 2*h)
    X4_inter = X4_inter.T
    Y3 = X3 - X4_inter
    return Y0, Y1, Y2, Y3, X4

def py4dec(Y0, Y1, Y2, Y3, X4, h):
    
    #Z3
    X4_inter = rowint(X4, 2*h)
    X4_inter = rowint(X4_inter.T, 2*h)
    X4_inter = X4_inter.T
    Z3 = Y3 + X4_inter
    
    #Z2
    X3_inter = rowint(Z3, 2*h)
    X3_inter = rowint(X3_inter.T, 2*h)
    X3_inter = X3_inter.T
    Z2 = Y2 + X3_inter
    
    #Z1
    X2_inter = rowint(Z2, 2*h)
    X2_inter = rowint(X2_inter.T, 2*h)
    X2_inter = X2_inter.T
    Z1 = Y1 + X2_inter
    
    #Z0
    X1_inter = rowint(Z1, 2*h)
    X1_inter = rowint(X1_inter.T, 2*h)
    X1_inter = X1_inter.T
    Z0 = Y0 + X1_inter
    return Z3, Z2, Z1, Z0