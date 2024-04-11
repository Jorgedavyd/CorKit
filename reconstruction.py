from skimage.restoration import inpaint
from scipy.fft import fft, ifft
import numpy as np

version = '@(#)reconstruction.py	0.1 , 14/03/24 (Deprecated, using reconstruction autoencoders instead)'

def image_reconstruction(img: np.array):
    map_miss_blocks = -np.fix(img > 0.1)

    # Find locs
    mask = np.zeros(img.shape, dtype = bool)

    mask[map_miss_blocks == 0] = 1
    
    # Perform restoration
    img_restored = inpaint.inpaint_biharmonic(img, mask)
    
    return img_restored

def fuzzy_image():
    side = 32
    too_many = 0


def read_zone(img, list_miss_blocks, rebindex):
    # Side of a square block
    side = 32

    # Extremes columns and rows of the missing zone
    i1 = np.min(list_miss_blocks[0, :] - 1 > 0)
    i2 = np.max(list_miss_blocks[0, :] + 1 < ((32 / rebindex) - 1))
    j1 = np.min(list_miss_blocks[1, :] - 1 > 0)
    j2 = np.max(list_miss_blocks[1, :] + 1 < ((32 / rebindex) - 1))

    # Dimensions of the zone (array of pixels)
    zone_left = side * i1
    zone_right = side * (i2 + 1) - 1
    zone_bottom = side * j1
    zone_top = side * (j2 + 1) - 1
    zone_width = zone_right - zone_left + 1
    zone_height = zone_top - zone_bottom + 1

    zone = img[zone_left:zone_right, zone_bottom:zone_top]

    return zone_width, zone_height, zone

def dct(array: np.array, inverse: bool = False):
    shape = array.shape
    dim = len(shape)

    if dim == 1:
        n = int(shape)
        k = np.arange(n)
        w4n = np.exp(k*complex(0, -2*np.pi/(4*n)))

        sub = np.concatenatate((k, np.flip(k)))
        sub = sub[::2]

        if inverse:
            vv = 0.5 / w4n * (array[k] - array[n- k[::-1]])
            v = ifft(vv)
            x = np.zeros(n)
            x[sub] = v
        else:
            v = array[sub]
            vv = fft(v)
            x = 2*np.real(w4n*vv[k])

    elif dim == 2:
        n2, n1 = shape

        k1 = np.arange(n1) @ np.ones(n2)
        k2 = np.ones(n1) @ np.arange(n2)

        n1_k1 = np.roll(np.flip(k1, 1), 1, axis = 1)
        n2_k2 = np.roll(np.flip(k2, 0), 1, axis = 0)

        sub1 = np.concatenate((k1, np.flip(k1, 1)))
        sub1 = sub1[:, 2*np.arange(n1)]
        sub2 = np.concatenate(([k2], np.flip(k2, 0)))
        sub2 = sub2[2*np.arange(n2), :]

        w4n1 = np.exp(k1*complex(0, -2*np.pi/(4*n1)))
        w4n2 = np.exp(k2*complex(0, -2*np.pi/(4*n2)))

        if inverse:
            mask1 = np.ones(n2,n1)
            mask1[:, 0] = 0
            mask2 = np.ones(n2,n1)
            mask2[0, :] = 0

            vv = 1/4/w4n1/w4n2 * complex(array[k2, k1] - array[n2_k2, n1_k1] * mask1 * mask2, -(array[k2, n1_k1] * mask1 + array[n2_k2, k1]*mask2))
            v = ifft(vv)
            x = np.zeros(n2, n1)
            x[sub2, sub1] = v
        else:
            vv = fft(array[sub2, sub1])
            x = 2*float(w4n1 * (w4n2*vv[k2, k1] + 1/w4n2*vv[n2_k2, k1]))
    
    return x


def fuzzy_block(img, i, j, rebindex):
    side = 32
    zone_width, zone_height, zone = read_zone(img, np.array([i,j]), rebindex)
    
    #given zone: (height, width)
    ny = zone.shape[0]/side 
    nx = zone.shape[1]/side

    sp = np.zeros((side, side, ny, nx), dtype=complex)


    for jj in range(ny):
        for ii in range(nx):
            block = read_block(zone, ii, jj)
            sp[:, :, jj, ii] = dct(block)

    x0 = 0
    y0 = 0
    side = 32
    xx = np.arange(side)
    yy = np.ones(side)  # Equivalent to replicate(1, side) in IDL

    sub_mask = np.zeros((side, side), dtype=np.uint8)

    # Define sub-mask values
    mean, low, horiz, vert, diag, high = 0, 1, 2, 3, 4, 5

    # Parameters for sub-mask generation
    diag_angle = np.pi / 6
    low_radius = 4
    high_radius = side * 0.9

    # Generate sub-mask
    sub_mask[:, :] = diag
    sub_mask[np.abs(yy - y0) < np.abs(xx - x0) * np.tan(diag_angle / 2)] = horiz
    sub_mask[np.abs(xx - x0) < np.abs(yy - y0) * np.tan(diag_angle / 2)] = vert
    sub_mask[np.abs(np.complex(xx - x0, yy - y0)) < low_radius] = low
    sub_mask[np.abs(np.complex(xx - x0, yy - y0)) > high_radius] = high
    sub_mask[y0, x0] = mean

    # Corresponding subspectra calculation
    sub_sp = np.zeros((side, side, 6, ny, nx), dtype=float)
    for jj in range(ny):
        for ii in range(nx):
            for s in range(6):
                sub_sp[:, :, s, jj, ii] = sp[:, :, jj, ii] * (sub_mask == s)

    # Number of elements in each sub-spectrum
    n_sub_sp = np.histogram(sub_mask.astype(int), bins=np.arange(7))[0]

    norm_en = np.zeros(6, ny, nx)

    for jj in range(ny):
        for ii in range(nx):
            for s in range(6):
                norm_en[s, jj,ii] = np.sum(np.abs(sub_sp[:,:,s,jj,ii])**2)/n_sub_sp[s]

    xcent = np.zeros(6, ny, nx)
    ycent = np.zeros(6, ny, nx)

    for jj in range(ny):
        for ii in range(nx):
            for s in range(6):
                xcent[s, jj, ii] = np.sum(xx*np.abs(sub_sp[:,:,s,jj,ii])**2)/np.sum(np.abs(sub_sp[:,:,s,jj,ii])**2)
                ycent[s, jj, ii] = np.sum(yy*np.abs(sub_sp[:,:,s,jj,ii])**2)/np.sum(np.abs(sub_sp[:,:,s,jj,ii])**2)
    
    orient = np.zeros(nx, ny)

    for jj in range(ny):
        for ii in range(nx):
            max_index = np.argmax(norm_en[:, jj, ii] * np.array([0, 0, 1, 1, 1, 0]))
            orient[jj, ii] = max_index
    
    ph_sk = sub_sp < 0

    hist, bins = np.histogram(orient.flatten(), bins=6, range=(0, 5))
    nmax = np.max(hist * np.array([0, 0, 1, 1, 1, 0]))
    smax = np.argmax(hist)

    sim = np.zeros((4, ny, nx))

    for jj in range(ny):
        for ii in range(nx):
            sim[0, jj, ii] = 1 - np.abs(norm_en[smax, jj, ii] - norm_en[smax, 1, 1]) / \
                            np.abs(np.max(norm_en[smax, :, :]) - np.min(norm_en[smax, :, :]))
            
            sim[1, ii, jj] = 1 - np.sqrt((xcent[smax, jj, ii] - xcent[smax, 1, 1])**2 +
                                        (ycent[smax, jj, ii] - ycent[smax, 1, 1])**2) / 20

    member = orient == smax

    member[1,1] = 0

    conf = np.zeros(ny, nx)
    
    for jj in range(ny):
        for ii in range(nx):
            conf[jj, ii] = 1 - np.sum(ph_sk[:, :, smax, jj, ii] ^ ph_sk[:,:, smax, 1,1]) / n_sub_sp(smax)

    all_sub_sp = sub_sp[:,:, smax, :, :]
    all_sub_sp = all_sub_sp[all_sub_sp != 0]
    min_ = np.min(all_sub_sp)
    max_ = np.max(all_sub_sp)

    fuzzy_sub_sp = num_to_fuzzy(sub_sp[:,:,smax,1,1], min_, max_, 0)

    for jj in range(ny):
        for ii in range(nx):
            if member[jj, ii]:
                fuzzy_sub_sp = inter_fuzzy(fuzzy_sub_sp, num_to_fuzzy(sub_sp[:, :, smax, jj, ii], min_, max_, 0.5))
    number_sub_sp = fuzzy_to_num(fuzzy_sub_sp)

    
    sp_block = sp[:, :, 1,1]
    sp_block[sub_mask == smax] = number_sub_sp[sub_mask == smax]
    block = dct(sp_block, inverse = True)

    return block

def num_to_fuzzy(a0, amin, amax, conf):
    # If amin is greater than amax, swap them
    if amax < amin:
        amin, amax = amax, amin
    
    # Ensure each element of a0 is within the range [amin, amax]
    a0 = np.clip(a0, amin, amax)
    
    # Create an array of fuzzy numbers with the same shape as a0
    afuzzy = np.zeros_like(a0, dtype=[('low', float), ('high', float)])
    
    # Assign low and high values for each fuzzy number
    afuzzy['low'] = amin + conf * (a0 - amin)
    afuzzy['high'] = amax - conf * (amax - a0)
    
    return afuzzy

def inter_fuzzy(afuzzy, bfuzzy): #mirar
    cfuzzy = {}
    cfuzzy['low'] = np.maximum(afuzzy['low'], bfuzzy['low'])
    cfuzzy['high'] = np.minimum(afuzzy['high'], bfuzzy['high'])
    return cfuzzy

def fuzzy_to_num(afuzzy):
    a = (afuzzy['low'] + afuzzy['high']) / 2
    return a

def read_block(image, i, j, side=32):
    # Reading a square block from the image
    block = image[side*j:side*(j+1), side*i:side*(i+1)]
    return block

def getl05hdrparam(header):
    out = {}
    out['detector'] = header['detector']
    out['sx'] = header['NAXIS1']
    out['sy'] = header['NAXIS2']
    out['fystart'] = header['R1ROW'] - 1

def get_tmax():
    pass