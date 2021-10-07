import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from sklearn.metrics import r2_score

# Vectorized implementation of two-way ANOVA
def anova_two_way(A, B, Y):
    num_cells = Y.shape[1]
    
    A_levels = np.unique(A); a = len(A_levels)
    B_levels = np.unique(B); b = len(B_levels)
    Y4D = np.array([[Y[(A==i)&(B==j)] for j in B_levels] for i in A_levels])
    
    r = Y4D.shape[2]

    Y = Y4D.reshape((-1, Y.shape[1]))
    
    # only test cells (units) that are active (gave a nonzero response to at least one stimulus) to avoid division by zero errors
    active_cells = np.where(np.abs(Y).max(axis=0)>0)[0] 
    Y4D = Y4D[:,:,:,active_cells]
    Y = Y[:, active_cells]
    
    N = Y.shape[0]
    
    Y_mean = Y.mean(axis=0)
    Y_mean_A = Y4D.mean(axis=1).mean(axis=1)
    Y_mean_B = Y4D.mean(axis=0).mean(axis=1)
    Y_mean_AB = Y4D.mean(axis=2)

    
    SSA = r*b*np.sum((Y_mean_A - Y_mean)**2, axis=0)
    SSB = r*a*np.sum((Y_mean_B - Y_mean)**2, axis=0)
    SSAB = r*((Y_mean_AB - Y_mean_A[:,None] - Y_mean_B[None,:] + Y_mean)**2).sum(axis=0).sum(axis=0)
    SSE = ((Y4D-Y_mean_AB[:,:,None])**2).sum(axis=0).sum(axis=0).sum(axis=0)
    SST = ((Y-Y_mean)**2).sum(axis=0)

    DFA = a - 1; DFB = b - 1; DFAB = DFA*DFB
    DFE = (N-a*b); DFT = N-1
    
    MSA = SSA / DFA
    MSB = SSB / DFB
    MSAB = SSAB / DFAB
    MSE = SSE / DFE
    
    FA = MSA / MSE
    FB = MSB / MSE
    FAB = MSAB / MSE
    
    pA = np.nan*np.zeros(num_cells)
    pB = np.nan*np.zeros(num_cells)
    pAB = np.nan*np.zeros(num_cells)
    
    pA[active_cells] = stats.f.sf(FA, DFA, DFE)
    pB[active_cells] = stats.f.sf(FB, DFB, DFE)
    pAB[active_cells] = stats.f.sf(FAB, DFAB, DFE)
    
    return pA, pB, pAB

def average_tuning_curves(Q, H):
    Qrange = np.unique(Q)
    tuning_curves = np.array([H[Q==j,:].mean(axis=0) for j in Qrange])
    
    return tuning_curves

def preferred_numerosity(Q, H):
    tuning_curves = average_tuning_curves(Q, H)

    pref_num = np.unique(Q)[np.argmax(tuning_curves, axis=0)]
    
    return pref_num

# Returns goodness-of-fit (r2) of the least-squares straight line fit to each tuning curve 
def fit_tuning_curves_with_lines(tuning_curves, xx):
    r2 = np.zeros(tuning_curves.shape[0])
    sigmas = np.zeros(tuning_curves.shape[0])
    for i, tcc in enumerate(tuning_curves):
        x = xx[~np.isnan(tcc)]
        tc = tcc[~np.isnan(tcc)]
        if x.shape[0] == 0: continue
        
        fitfunc  = lambda p, x: p[0] * x + p[1]
        errfunc  = lambda p, x, y: (y - fitfunc(p, x))
    
        p0 = np.array([0, 0])
        out   = optimize.least_squares(errfunc, p0, args=(x, tc), jac='3-point')
        p = out['x']
        r2[i] = r2_score(tc, fitfunc(p, x))
    
    return r2