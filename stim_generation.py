
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance

# Generic function for generating dataset
def generate_dataset(numerosities,
                     num_images_per_numerosity,
                     sample_fn, # function that returns random x,y location and radius of dot to place on stimulus 
                     sample_fn_args, # extra arguments to sample_fn
                     verify_fn, # function to return whether a given x, y location and dot radius is valid
                     draw_fn, # function to draw a dot on stimulus with arguments (img, x, y, radius)
                     img_size, # image size in pixels
                     max_iter, # maxmimum number of attempts to place valid dots on stimulus
                     n_channels=1, # number of color channels
                     background_func = None, # function to generate an image contatining only the desired background with a tuple argument (width, height, n_channels)
                     background_args = None, # extra arguments to background_func,
                     verbose=False):
    
    S, Q = [], []
    for ii, n in enumerate(numerosities):
        for k in range(num_images_per_numerosity):
            if verbose:
                print('Numerosity: %i, image: %i'%(n,k+1))
            
            if background_func is None:
                img = np.zeros((img_size,img_size,n_channels), np.uint8)
            else:
                img = background_func((img_size,img_size,n_channels), **background_args)
                
            if n>0:
                for v in range(max_iter):
                    x, y, r = sample_fn(img_size, n, **sample_fn_args)

                    if n > 1:
                        if verify_fn(x, y, r):
                            break
                    else:
                        break

                    if v == max_iter-1:
                        print('Maximum number of verification iterations reached')
                        return None

                if n==1: x, y, r = [x], [y], [r]

                for xi, yi, ri in zip(x, y, r):
                    img = draw_fn(img, xi, yi, ri)
                
            S.append(img); Q.append(n)
    
    randperm = np.random.permutation(len(Q))
    return np.array(S)[randperm], np.array(Q)[randperm]

# standard dot placement - only makes sure dots are not overlapping or clipped by image borders
def sample_xyr_standard(img_size, num_objects, 
                        nominal_radius, radius_noise_ratio=0.1, 
                        min_dist = 5,
                        dist_from_border=5, max_iter=10000):
    x, y, r = None, None, None
    
    for i in range(num_objects):
        for j in range(max_iter):
            ri = np.round(nominal_radius + radius_noise_ratio * nominal_radius * np.random.normal()).astype(np.int)
            xi = np.round(np.random.uniform(ri+dist_from_border, img_size-ri-dist_from_border)).astype(np.int)
            yi = np.round(np.random.uniform(ri+dist_from_border, img_size-ri-dist_from_border)).astype(np.int)
            
            if i==0:
                x, y, r = xi, yi, ri
                break
            else:
                dists = np.sqrt((x-xi)**2 + (y-yi)**2)
                if np.all(dists > r + ri + min_dist):
                    x, y, r = np.append(x, xi), np.append(y, yi), np.append(r, ri)
                    break

                if j == max_iter-1:
                    print('Placement error at (i=%i, j=%i)'%(i,j))
                    return None
                        
    return x, y, r

# area control dot placement - total area of dots is constant regardless of numerosity
def sample_xyr_area_control(img_size, num_objects, total_area, **kwargs):
    return sample_xyr_standard(img_size, num_objects, np.sqrt(total_area/num_objects/np.pi), **kwargs)

# convex hull control dot placement - dots are placed within a convex hull of given size
def sample_xyr_convex_hull_control(img_size, num_objects, 
                                   nominal_radius, radius_noise_ratio=0.1,
                                   dist_from_border=5, min_dist = 5,
                                   max_iter=50000,
                                   hull_radius = 85,
                                   hull_size=5):
    
    x, y, r = None, None, None
    
    cx = cy = img_size/2
    theta_hull = np.arange(0, 2*np.pi, (2*np.pi)/hull_size)
    theta_hull += np.random.uniform(high=np.pi)
    np.random.shuffle(theta_hull)
    
    for i in range(num_objects):
        for j in range(max_iter):
            if i < hull_size:
                theta = theta_hull[i]
                radius = hull_radius + 5*np.random.normal()
            else:
                theta = np.random.uniform(high=2*np.pi)
                radius = np.random.uniform(high=hull_radius-2*nominal_radius)
            
            xi = np.round(cx+radius*np.cos(theta)).astype(np.int)
            yi = np.round(cy+radius*np.sin(theta)).astype(np.int)
            
            ri = np.round(nominal_radius + radius_noise_ratio * nominal_radius * np.random.normal()).astype(np.int)
            
            if i==0:
                x, y, r = xi, yi, ri
                break
            else:
                dists = np.sqrt((x-xi)**2 + (y-yi)**2)
                if np.all(dists > r + ri + min_dist):
                    x, y, r = np.append(x, xi), np.append(y, yi), np.append(r, ri)
                    break

                if j == max_iter-1:
                    print('Placement error at (i=%i, j=%i)'%(i,j))
                    return None
                        
    return x, y, r

# returns whether average distance between dot pairs is within a given range
def verify_density_control(x, y, r, min_dist=90, max_dist=100):
    coords = np.concatenate((x[:,None], y[:,None]), axis=1)
    avg_dist = distance.cdist(coords, coords, 'euclidean')[np.triu_indices(len(x), 1)].mean()
    return (avg_dist >= min_dist) & (avg_dist <= max_dist)

# generate a uniform background with size (tuple, (width, height, n_channels)) and constant intensity A
def generate_uniform_background(size, A):        
    return (A*np.ones(size)).astype(np.uint8)

# draws a circle on an image
def circle(img, x, y, r):
    cv2.circle(img, (x,y), r, (255,), -1, cv2.LINE_AA)
    return img

# sample a shape from a set of shape (circle, rectangle, ellipse, triangle) and draws it
def random_shape(img, x, y, r):
    ss = np.random.choice(range(4))
    if ss == 0:
        cv2.circle(img, (x,y), r, (255,), -1, cv2.LINE_AA)
    elif ss == 1:
        r = r/np.sqrt(2)
        r1 = np.floor(np.random.uniform(0.7,1.0) * r).astype(np.int)
        r2 = np.floor(np.random.uniform(0.7,1.0) * r).astype(np.int)
        cv2.rectangle(img, (x-r1,y-r2), (x+r1,y+r2), (255,), -1)
    elif ss == 2:
        r1 = np.floor(np.random.uniform(0.3,1.0) * r).astype(np.int)
        r2 = np.floor(np.random.uniform(0.3,1.0) * r).astype(np.int)
        cv2.ellipse(img, (x,y), (r1, r2), np.random.uniform(0,360), 0, 360, (255,), -1, cv2.LINE_AA)
    elif ss == 3:
        r1 = np.floor(np.random.uniform(0.7,1.0) * r).astype(np.int)
        r2 = np.floor(np.random.uniform(0.7,1.0) * r).astype(np.int)
        r3 = np.floor(np.random.uniform(0.7,1.0) * r).astype(np.int)
        
        pts = np.array([[x+r1,y],[x,y-r2],[x-r3,y]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillConvexPoly(img,pts,(255,), cv2.LINE_AA)
        
    return img

# Generate three datasets: standard, area/density control/luminosity, shape/convex hull control
# returns 
#   S: array of size (num_images, 3, 224, 224) containing sampled stimuli
#   Q: numerosity depecited in each stimulus
#   C: condition under which each stimulus was generated
def generate_datasets(filepath=None, reload=False, num_reps=200, Qrange = np.array([0,1,2,3,4]), radius=18, area=1200, hull_size=3):
    if reload and (filepath is not None):
        ff = np.load(filepath)
        return ff['S'], ff['Q'], ff['C']

    Ss, Qs = generate_dataset(numerosities = Qrange, 
                            num_images_per_numerosity = num_reps, 
                            sample_fn = sample_xyr_standard, 
                            sample_fn_args = {'nominal_radius': radius},
                            verify_fn = lambda x,y,r: True,
                            draw_fn = circle,
                            img_size = 224,
                            max_iter = 1000,
                            background_func=generate_uniform_background,
                            background_args = {'A':50})
    
    Sc, Qc = generate_dataset(numerosities = Qrange, 
                            num_images_per_numerosity = num_reps, 
                            sample_fn = sample_xyr_area_control, 
                            sample_fn_args = {'total_area': area},
                            verify_fn = verify_density_control,
                            draw_fn = circle,
                            img_size = 224,
                            max_iter = 10000,
                            background_func=generate_uniform_background,
                            background_args = {'A':50})
    
    # fix average luminosity to a constant
    fixed_mean =  Sc.reshape((Sc.shape[0], -1)).mean(axis=1).min()
    Sc = fixed_mean * (Sc / Sc.reshape((Sc.shape[0], -1)).mean(axis=1)[:, None, None, None])
    
    Sss, Qss = generate_dataset(numerosities = Qrange, 
                            num_images_per_numerosity = num_reps, 
                            sample_fn = sample_xyr_convex_hull_control, 
                            sample_fn_args = {'nominal_radius': radius, 'hull_size':hull_size},
                            verify_fn = lambda x,y,r: True,
                            draw_fn = random_shape,
                            img_size = 224,
                            max_iter = 1000,
                            background_func=generate_uniform_background,
                            background_args = {'A':50})
    
    S = np.concatenate((Ss, Sc, Sss))
    Q = np.concatenate((Qs, Qc, Qss))
    C = np.concatenate((0*np.ones_like(Qs),
                        1*np.ones_like(Qc),
                        2*np.ones_like(Qc)))
    
    S = np.tile(S, (1,1,1,3))
    
    S = S.transpose((0,3,1,2))
    
    randperm = np.random.permutation(len(Q))
    S, Q, C = S[randperm], Q[randperm], C[randperm]
    
    S, Q, C = S.astype(np.float32)/255, Q.astype(np.int), C.astype(np.int)
    
    if filepath is not None:
        np.savez(filepath, S=S, Q=Q, C=C)
    
    return S, Q, C

def plot_samples(S, Q, C, figsize=(12,36)):
    plt.figure(figsize=figsize)
    Qlevels = np.unique(Q)
    Clevels = np.unique(C)

    ii = 1
    for i, c in enumerate(Clevels):
        for j, q in enumerate(Qlevels):
            subset = (C==c) & (Q==q)
            idx = np.random.randint(S[subset].shape[0])

            plt.subplot(len(Clevels), len(Qlevels), ii); ii += 1   
            plt.imshow(S[subset][idx].transpose((1,2,0)))
            plt.axis('off')