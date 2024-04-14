"""General util for LASCO & COR 2"""

from scipy.ndimage import affine_transform, map_coordinates
from datetime import datetime, timedelta
from collections import defaultdict
import skimage.transform as tt
from astropy.time import Time
from astropy.io import fits
from matplotlib import tri
from PIL import Image
import pandas as pd
import numpy as np 
import glob
import os

version = '1.0.4'

radeg = 180/np.pi

DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__),'data')

#done
def datetime_interval(init: datetime, last: datetime, step_size: timedelta, output_format: str = '%Y%m%d'):
    current_date = init
    date_list = []
    while current_date <= last:
        date_list.append(current_date.date().strftime(output_format))
        current_date += step_size
    return date_list
#done
def save_to_fits(img, header, filepath):
    primary_hdu = fits.PrimaryHDU(img)
    hdul = fits.HDUList([primary_hdu])
    header['COMMENT'] = 'CORKIT-LASCO-LEVEL 1'
    del header['HISTORY']
    hdul[0].header = header
    #Writing to filepath
    hdul.writeto(filepath, overwrite=True)
#done
def save_to_png(img, filepath):
    Image.fromarray(img).save(filepath, 'PNG')
#done
def save_to_jp2(img, filepath):
    Image.fromarray(img).save(filepath, quality_mode = 'lossless')
#done
def save(filepath, filetype, img, header = None):
    if filetype == 'fits':
        assert (header is not None), 'You missed the header for fits file'
        save_to_fits(img, header, filepath)
    elif filetype == 'jp2':
        save_to_jp2(img, filepath)
    elif filetype == 'png':
        save_to_png(img, filepath)
#done
def FITS(fits_file):
    with fits.open(fits_file) as hdul:
        img0 = hdul[0].data
        header = hdul[0].header
        hdul.close()

    return img0, header
#done
def get_exp_factor(header):
    tool = header['detector'].strip().lower()
    date = datetime.strptime(header['date-obs'], '%Y/%m/%d')
    time = datetime.strptime(header['time-obs'], '%H:%M:%S.%f')
    try:
        root = os.path.join(DEFAULT_SAVE_DIR, f'data_anal/data/{date.strftime("%y%m")}/')
        with open(os.path.join(root, f'{tool}_expfactor_{date.strftime("%y%m%d")}.dat')) as file:
            times = []
            lines = file.readlines()
            if '<!DOCTYPE HTML' in lines[0]: raise FileNotFoundError
            for line in lines:
                line = line.split()
                if header['filename'].strip() in line:
                    header.add_history(f'corkit/utils.py get_exp_factor: (function) 12/04/24, {float(line[1])}')
                    header.add_history(f'Bias ({float(line[2])}) from {tool}_expfactor_{date}.dat')
                    return header, float(line[1]), float(line[2])
                times.append(float(line[4])*1000)
            time = min(times, key = lambda key: abs(key-to_mil(time)))
            idx = times.index(time)
            header.add_history(f'corkit/utils.py get_exp_factor: (function) 12/04/24, {float(lines[idx][1])}')
            header.add_history(f'Bias ({float(lines[idx][2])}) from {tool}_expfactor_{date}.dat')
            return header, float(lines[idx][1]), float(lines[idx][2])
    except FileNotFoundError:
        return header, 1, header['offset']
#done
def correct_var(header, *args):
    args = list(args)

    if ((header['r1col'] != 20) or (header['r1row'] != 1) or \
        (header['r2col'] != 1043) or (header['r2row'] != 1024))\
         and header['r2col'] != 0:

        x1 = header['r1col'] - 20
        x2 = header['r2col'] - 20
        y1 = header['r1row'] - 1
        y2 = header['r2row'] - 1
        
        for i, arg in enumerate(args):
            args[i] = arg[y1:y2+1, x1:x2+1]
    
    args = apply_summing_corrections(header, *args)

    return args
#done
def apply_summing_corrections(header, *args):
    args = list(args)
    summing = np.maximum(header['sumcol'], 1)*np.maximum(header['sumrow'], 1)
    vig_size = args[0].shape
    if summing > 1:
        for _ in range(1, summing + 1, 4):
            for i, arg in enumerate(args):
                args[i] = rebin(arg, vig_size[1]/2, vig_size[0]/2)

    summing = header['lebxsum'] * header['lebysum']
    if summing > 1:
        for _ in range(1, summing + 1, 4):
            for i, arg in enumerate(args):
                args[i] = rebin(arg, vig_size[1]/2, vig_size[0]/2)
    
    return args
#done
def c2_warp(img, header):
    header.add_history(f'corkit/utils.py c2_warp: (function) {version} 12/04/24')
    gridsize = 32
    num_points = (1024 // gridsize + 1) ** 2
    w = np.arange(num_points)
    y = w // (1024 // gridsize + 1)
    x = w % (1024 // gridsize + 1)
    x = x * gridsize
    y = y * gridsize
    img = reduce_std_size(img, header, NO_REBIN = True, NOCAL = True)
    xc, yc = occltr_cntr(header)

    sumx = header['lebxsum'] * np.maximum(header['sumcol'], 1)
    sumy = header['lebysum'] * np.maximum(header['sumrow'], 1)
    
    if sumx > 0:
        x = x / float(sumx)
        xc = xc / float(sumx)
    if sumy > 0:
        y = y / float(sumy)
        yc = yc / float(sumy)

    scalef = get_sec_pixel(header)

    r = np.sqrt((sumx*(x-xc))**2+(sumy*(y-yc))**2)
    r0 = c2_distortion(r, scalef)/(sumx*scalef)
    header.add_history(f'corkit/utils.py distortion_coeff: (function) {version} 12/04/24')
    theta = np.arctan2(y - yc, x - xc)
    x0 = r0 * np.cos(theta) + xc
    y0 = r0 * np.sin(theta) + yc
    img = warp_tri(x, y, x0, y0, img)
    header.add_history(f'corkit/utils.py warp_tri: (function) {version} 12/04/24')
    return img, header
#done
def c3_warp(img, header):
    gridsize = 32
    num_points = (1024 // gridsize + 1) ** 2
    w = np.arange(num_points)
    y = w // (1024 // gridsize + 1)
    x = w % (1024 // gridsize + 1)
    x = x * gridsize
    y = y * gridsize

    img = reduce_std_size(img, header, NO_REBIN=True, SAVE_HDR=True) #imp

    xc, yc = occltr_cntr(header)

    x1 = header['r1col'] - 20
    x2 = header['r2col'] - 20
    y1 = header['r1row'] - 1
    y2 = header['r2row'] - 1
    
    sumx = header['lebxsum'] * np.maximum(header['sumcol'], 1)
    sumy = header['lebysum'] * np.maximum(header['sumrow'], 1)
    if sumx > 1:
        x /= sumx
        xc /= sumx
        x1 /= sumx
        x2 /= sumx
    if sumy > 1:
        y /= sumy
        yc /= sumy
        y1 /= sumy
        y2 /= sumy

    scalef = get_sec_pixel(header)

    r = np.sqrt((sumx * (x - xc)) ** 2 + (sumy * (y - yc)) ** 2)
    r0 = c3_distortion(r, scalef) / (sumx * scalef)  # convert from arcsec to pixel

    theta = np.arctan2((y - yc), (x - xc))
    x0 = r0 * np.cos(theta) + xc
    y0 = r0 * np.sin(theta) + yc
    
    # Distort the image by shifting locations (x,y) to (x0,y0) and return it:
    img = warp_tri(x, y, x0, y0, img) 

    return img[y1:y2+1,x1:x2+1], header
#done
def warp_tri(x0, y0, xi, yi, img):
    y_new, x_new = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    #creating the x0,y0 irregular grid
    src_tri = tri.Triangulation(x0, y0)

    x_grid= tri.LinearTriInterpolator(src_tri, xi)(x_new, y_new)
    y_grid= tri.LinearTriInterpolator(src_tri, yi)(x_new, y_new)
    
    return map_coordinates(img, [y_grid, x_grid], order = 1)   
#done
def c2_distortion(data, arcs):
    mm = data * 0.021
    cf = np.array(distortion_coeffs('C2'))  # Assuming DISTORTION_COEFFS returns a list or tuple
    f1 = mm * (cf[0] + cf[1] * (mm ** 2) + cf[2] * (mm ** 4))
    f1 = data + f1 / 0.021
    secs = subtense('c2') if arcs is None else float(arcs)
    return secs * f1
#done
def distortion_coeffs(telescope: str):
    tel = telescope.upper().strip()
    if tel == 'C2':
        cf = [0.0051344125, -0.00012233862, 1.0978595e-7]  # C2
    elif tel == 'C3':
        cf = [-0.0151657, 0.000165455, 0.0]  # C3
    else:
        cf = [0.0, 0.0, 0.0]
        print('%Distortion_coeffs: Telescope not defined')

    return cf
#done
def subtense(tool: str):
    tool = tool.strip().upper()

    if tool == 'C1':
        return 5.8
    elif tool == 'C2':
        return 11.9
    elif tool == 'C3':
        return 56.0
    elif tool == 'EIT':
        return 2.59
    elif tool == 'MK3':
        return 9.19
    elif tool == 'MK4':
        return 9.19
    else:
        return 0
#done
def get_sec_pixel(header, FULL: float = False):
    sec_pix = subtense(header['detector'])

    if FULL:
        return sec_pix * 1024/FULL

    binfac = float(header['r2col'] - header['r1col'] + 1) / header['naxis1']
    sum_check = (header['sumcol'] + 1) * header['lebxsum']
    if binfac != sum_check:
        print("Warning: Result not correct for Subfields!")
    sec_pix *= binfac

    return sec_pix    
#done
def solar_ephem(yymmdd, soho=False):
    dte = Time(datetime.strptime(yymmdd, '%y%m%d'))
    j2000 = Time(datetime.strptime('20000101', '%Y%m%d'))
    # Calculate days since J2000
    n = dte.mjd - j2000.mjd

    # Calculate solar longitude
    long = 280.460 + 0.9856474 * n
    while long < 0:
        long += 360
    while long >= 360:
        long -= 360

    # Calculate mean anomaly
    g = 357.528 + 0.9856003 * n
    g /= radeg

    # Calculate distance correction
    dist = 1.00014 - 0.01672 * np.cos(g) - 0.00014 * np.cos(2 * g)

    # Adjust for SOHO if needed
    if soho:
        dist *= 0.99

    # Calculate solar radius
    radius = 0.2666 / dist

    return radius
#done
def reduce_statistics2(img, header, **kwargs):
    if np.max(img) > .1: img[img>.00005] = .00005
    
    wmn = np.where(img > 0)
    n = len(wmn[0])
    
    if n < 1:
        return header

    shape = img.shape

    mn = np.min(img[wmn])
    mx = np.max(img[wmn])
    
    medyen = np.median(img[wmn])

    bscale: float = 1.

    if medyen > 0:
        while medyen*bscale<1000:
            bscale*=10

    header['DATAMIN'] = float(mn)

    if 'SATMAX' in kwargs: mxval = kwargs['SATMAX'] 
    else: mxval = mx

    if 'SATMIN' in kwargs: mnval = kwargs['SATMIN'] 
    else: mnval = mn
    
    wltmax = np.where(img<mxval)
    nltmx = len(wltmax[0])
    if mx == mxval: dmx = np.max(img[wltmax]) 
    else: dmx = mx

    cond = np.logical_and(img < mnval, img != 0)
    wltmn = np.where(cond)
    nsatmin = len(wltmn[0])

    header['DATAMAX'] = float(dmx)

    zeros = shape[0] - n

    header['DATAZER'] = zeros    
    header['DSATVAL'] = mxval
    header['DSATMIN'] = mnval
    header['NSATMIN'] = nsatmin

    w = np.where(img>0)
    nw = len(w[0])
    if nw < 1: return None
    else: 
        sig = np.std(img[w])
        men = np.mean(img[w])
    header['DATAAVG'] = men 
    header['DATASIG'] = sig

    # Create histogram of the scaled image values
    temparr = img[w] * bscale
    h, _ = np.histogram(temparr, bins=np.arange(temparr.min(), temparr.max() + 1))
    nh = len(h)
    tot = 0
    limits = np.array([0.01,0.10,0.25,0.5,0.75,0.90,0.95,0.98,0.99])
    nw = len(w)
    low = 0

    for limit in limits:
        tile = limit*nw
        while low < nh and tot <= tile:
            tot += h[low]
            low += 1
        s = f'DATAP{int(limit*100)}'
        header[s] = (low-1)/bscale

    return header
#done
def reduce_std_size(img0, hdr, BIAS: float = None, NOCAL: bool = False, NO_REBIN: bool = False, FULL: bool = False, SAVE_HDR: bool = False):
    sumrow = np.maximum(hdr['sumrow'], 1)
    sumcol = np.maximum(hdr['sumcol'], 1)
    lebxsum = np.maximum(hdr['lebxsum'],1)
    lebysum = np.maximum(hdr['lebysum'],1)
    naxis1 = hdr['naxis1']
    naxis2 = hdr['naxis2']
    tel= hdr['telescop'].strip()

    if naxis1 <= 0 or naxis2 <= 0:
        print(naxis1, naxis2, img0)
        print(hdr.filename + 'Invalid Image -- Returning')
        return img0

    r1col = hdr['R1COL']
    r1row = hdr['R1ROW']
    if (r1col < 1): r1col = hdr['P1COL']
    if (r1row < 1): r1row = hdr['P1ROW']

    if BIAS is not None and not NOCAL:
        if BIAS == 1:
            abias = offset_bias(hdr)
        else:
            abias = BIAS

        if sumcol > 1:
            img = (img0 - abias) / (sumcol * sumrow)
        else:
            if lebxsum > 1:
                print('Correcting for LEB summing')
            img = (img0 - abias) / (lebxsum * lebysum)
        
        print('Offset bias of ' + str(abias) + ' subtracted.')
        hdr['OFFSET'] = 0

    else:
        img = img0

    nxfac = 2 ** (sumcol + lebxsum - 2)
    nyfac = 2 ** (sumrow + lebysum - 2)

    if tel == 'SOHO':
        if hdr['R2COL'] - r1col == 1023 and hdr['R2ROW'] - r1row == 1023 and naxis1 == 512:
            nxfac = 2
            nyfac = 2

    nx = nxfac * naxis1
    ny = nyfac * naxis2

    if (nx < 1024 or ny < 1024) and tel == 'SOHO':
        sz = img.shape
        full_img = np.zeros((1024 / nyfac, 1024 / nxfac), dtype=img.dtype)
        nx = np.minimum(sz[1],1024)
        ny = np.minimum(sz[0],1024)
        naxis1 = 1024 if nxfac < 2 else 512
        naxis2 = 1024 if nyfac < 2 else 512

        offrow = 1 if r1row > 1024 else r1row

        if r1col < 20:
            startrow = (offrow - 1) / nyfac
            startrow = np.minimum(startrow, 1024 - ny)
            full_img[startrow, (r1col - 1) // nxfac] = img[0:ny-1, 0:nx-1]
        else:
            startcol = 0 if ((r1col - 20) / nxfac + (nx - 1) > 1024 / nxfac) else (r1col - 20) / nxfac
            startrow = 0 if ((offrow - 1) / nyfac + (ny - 1) > 1024 / nyfac) else (offrow - 1) / nyfac
            full_img[startrow, startcol] = img[0:nx-1, 0:ny-1]
            hdr['crpix1'] += (r1col - 20) / nxfac
            hdr['crpix2'] += (offrow - 1) / nyfac
    else:
        full_img = img

    scale_to = 512

    scale_to = naxis1 if NO_REBIN else scale_to
    scale_to = 1024 if FULL else scale_to

    if not SAVE_HDR:
        if NOCAL:
            hdr['CRPIX1'] = ((hdr['CRPIX1'] * nxfac) + r1col - 20) * scale_to / 1024
            hdr['CRPIX2'] = ((hdr['CRPIX2'] * nyfac) + r1row - 1) * scale_to / 1024
        else:
            c, crota = get_sun_center(hdr, DEGREES=True, FULL=scale_to) #return crota
            hdr['CRPIX1'] = c['xcen'] + 1
            hdr['CRPIX2'] = c['ycen'] + 1
            hdr['CROTA1'] = crota
            try:
                hdr['crota'] = crota
            except KeyError:
                pass
                
        if tel == 'SOHO':
            hdr['R1COL'] = 20
            hdr['R1ROW'] = 1
            hdr['R2COL'] = 1043
            hdr['R2ROW'] = 1024

            if BIAS is not None:
                hdr['LEBXSUM'] = 1
                hdr['LEBYSUM'] = 1
                hdr['OFFSET'] = 0

        hdr['NAXIS1'] = scale_to
        hdr['NAXIS2'] = scale_to
        hdr['CDELT1'] *= (1024 / scale_to)
        hdr['CDELT2'] *= (1024 / scale_to)

    full_img = rebin(full_img, scale_to, scale_to)

    return full_img
#done
def rebin(arr, *args):
    return tt.resize(arr, args, anti_aliasing=True)
#done
def offset_bias(hdr,SUM: bool = False):
    
    port: str = hdr['readport'].strip().upper()
    tel: str = hdr['detector'].strip().upper()
    mjd= hdr['mid_date']

    if mjd == 0:
        mjd = Time(datetime.strptime(hdr['date-obs'], '%Y/%m/%d')).mjd

    if tel == 'C1':
        if port == 'A':
            b = 364
        elif port == 'B':
            b = 331
        elif port == 'C':
            del_value = (mjd - 50395)
            bias = 351.958 + 30.739 * (1 - np.exp(-del_value / 468.308))
            b = round(bias)
        elif port == 'D':
            b = 522
        else:
            b = 0

    elif tel == 'C2':
        if port == 'A':
            b = 364
        elif port == 'B':
            b = 540
        elif port == 'C':
            firstday = 50079
            if mjd <= 51057:
                coeff = np.array([470.97732,0.12792513,-3.6621933e-05])
            elif mjd < 51819:
                coeff = np.array([551.67277,0.091365902,-0.00012637790,7.4049597e-08])
                firstday = 51099
            elif mjd<51915:
                coeff = np.array([574.5788,0.019772032])
                firstday = 51558
            elif mjd<54792:
                coeff = np.array([581.81517,0.019221920,-2.3110489e-06])
                firstday = 51915
            elif mjd<55044:
                coeff = np.array([617.70556,0.010290491,-6.0131545e-06])
                firstday = 54792
            elif mjd<56450:
                coeff = np.array([619.99733,.0059081617,-3.3932229e-7 ])
                firstday = 55044
            elif mjd<57388:
                coeff = np.array([627.61246,.0049003351,-5.3812001e-7 ])
                firstday = 56450
            elif mjd<58571:
                coeff = np.array([631.20515,0.0056815108,-1.3439596e-06])
                firstday = 57290
            elif mjd<58802:
                coeff = np.array([651.50189,-0.028926206,7.8531807e-05,-5.8964538e-08])
                firstday = 58578
            else:
                coeff = np.array([648.41904,-0.0020514176,1.8072963e-05])
                firstday = 58802
            
            nc = len(coeff)
            dd = mjd-firstday
            b = np.polynomial.polynomial.polyval(dd, coeff)

        elif port == 'D':
            b = 526
        else:
            b = 0
    elif tel == 'C3':
        if port == 'A':
            b = 314
        elif port == 'B':
            b = 346
        elif port == 'C':
            if mjd < 50072:
                b = 319
            elif mjd <= 51057:
                coeff = np.array([322.21639, 0.011775379, 4.4256968e-05, -3.167423e-08])
                firstday = 50072
            elif mjd <= 51696:
                coeff = np.array([354.50857, 0.062196067, -8.8114799e-05, 5.0505447e-08])
                firstday = 51099
            elif mjd < 51915:
                coeff = np.array([369.02719, 0.014994955, -4.0873204e-06])
                firstday = 51558
            elif mjd < 54792:
                coeff = np.array([374.11139, 0.010731823, -1.0726207e-06])
                firstday = 51915
            elif mjd < 55044:
                coeff = np.array([395.85091, 0.0079344115, -6.2530780e-06])
                firstday = 54792
            elif mjd < 56170:
                coeff = np.array([397.52040, 0.0040765192])
                firstday = 55044
            elif mjd < 56360:
                coeff = np.array([407.04606, -0.024819780, 0.00011694347])
                firstday = 56170
            elif mjd < 56478:
                coeff = np.array([406.01009, 0.0046765547, -9.912626e-7])
                firstday = 56360
            elif mjd < 56597:
                coeff = np.array([406.72179, 0.0045780623, -2.3134855e-06])
                firstday = 56478
            elif mjd < 57024:
                coeff = np.array([406.77706, 0.0040538719, -1.6571028e-06])
                firstday = 56597
            else:
                coeff = np.array([408.38117, 0.0027558157, -1.3694218e-07])
                firstday = 57024
            
            nc = len(coeff)
            dd = mjd - firstday
            b = np.polynomial.polynomial.polyval(dd, coeff)

        elif port == 'D':
            b = 283
        else:
            b = 0
    elif tel == 'EIT':
        if port == 'A':
            b = 1017
        elif port == 'B':
            b = 840
        elif port == 'C':
            b = 1041
        elif port == 'D':
            b = 844
        else:
            b = 0
    else:
        b = 0

    if SUM:
        lebsum = np.maximum(hdr['lebxsum'], 1) * np.maximum(hdr['lebysum'],1)
        b *= lebsum

    return b

date_list = [
    '1996/08/10',
    '1997/08/18',
    '1998/03/29',
    '1998/05/06',
    '1999/02/20',
    '2000/02/20',
    '2000/10/20',
]
date_list = list(map(lambda x: Time(datetime.strptime(x, '%Y/%m/%d')).mjd, date_list))
pos = [
    (516.514, 529.717),
    (516.527, 529.685),
    (516.385, 529.671),
    (516.373, 529.650),
    (516.407, 531.172),
    (516.408, 531.160),
    (516.456, 531.132)
]

c3_occult_cntr_list = dict(zip(date_list, pos))
#done
def occltr_cntr(header):
    tel = header['detector'].strip().upper()
    filt = header['filter'].strip().upper()
    datest = header['date-obs'].strip().upper()
    dte = datetime.strptime(datest, '%Y/%m/%d')
    mjd = Time(dte).mjd

    if tel == 'C3':
        key = min(date_list, key = lambda x: abs(mjd - x))
        return c3_occult_cntr_list[key]
    
    all_occ = read_occ_dat()

    try:
        x, y = all_occ[tel][filt]
        return float(x), float(y)
    except KeyError:	
        if tel  == 'C1':
            if filt == 'FE X': return 511.029,494.521	  
            elif filt == 'FE XIV': return 510.400,495.478
            else: return 511,495 
        elif tel == 'C2':
            if filt == 'ORANGE': return 512.634,505.293
            else: return 512.634,505.293
        elif tel == 'C3':
            if filt == 'ORANGE': return 516.284,529.489	
            elif filt =='CLEAR': return 516.284,529.489	
            else: return 516.284,529.489
        elif tel == 'EIT':
            if filt == '195A': return 505.500,514.300
            else: return 505.500,514.300
        else:
            raise ValueError('Telescope not found')
#done
def read_occ_dat():
    occ = defaultdict(dict)
    with open(os.path.join(DEFAULT_SAVE_DIR, 'occulter_center.dat')) as file:
        lines = file.readlines()
        for line in lines[1::2]:
            x, y, _  = line[:25].split()
            tel_filt = line[31:]
            tel = tel_filt.split()[0]
            filt = ' '.join(tel_filt.split()[1:])
            occ[tel.upper()][filt.upper()] = (x,y)
    return occ
#done
def c3_distortion(data, ARC: float = None):
    mm = data * .021
    cf = distortion_coeffs('C3')
    f1 = mm*(cf[0]+cf[1]*(mm**2))
    f1 = data + f1/.21
    secs = float(ARC) if ARC is not None else subtense('C3')
    return secs*f1	
   
def get_solar_radius(header, **kwargs):
    date_obs = datetime.strptime(header['date-obs'], "%Y/%m/%dT%H:%M:%S.%f")
    tai_obs = utc2tai(date_obs) + header['exptime']/2

    tdb_obs = tai_obs + 32.184
    tdb_utc = tai2utc(tdb_obs)

    yymmdd = datetime.strftime(tdb_utc, "%y%m%d")

    radius = solar_ephem(yymmdd)
    radius*=3600
    
    if 'PIXEL' in kwargs:
        radius /= get_sec_pixel(header)

    return radius

leap_seconds = [
        (datetime(1972, 6, 30), 10),
        (datetime(1972, 12, 31), 11),
        (datetime(1973, 12, 31), 12),
        (datetime(1974, 12, 31), 13),
        (datetime(1975, 12, 31), 14),
        (datetime(1976, 12, 31), 15),
        (datetime(1977, 12, 31), 16),
        (datetime(1978, 12, 31), 17),
        (datetime(1979, 12, 31), 18),
        (datetime(1981, 6, 30), 19),
        (datetime(1982, 6, 30), 20),
        (datetime(1983, 6, 30), 21),
        (datetime(1985, 6, 30), 22),
        (datetime(1987, 12, 31), 23),
        (datetime(1989, 12, 31), 24),
        (datetime(1990, 12, 31), 25),
        (datetime(1992, 6, 30), 26),
        (datetime(1993, 6, 30), 27),
        (datetime(1994, 6, 30), 28),
        (datetime(1995, 12, 31), 29),
        (datetime(1997, 6, 30), 30),
        (datetime(1998, 12, 31), 31),
        (datetime(2005, 12, 31), 32),
        (datetime(2008, 12, 31), 33),
        (datetime(2012, 6, 30), 34),
        (datetime(2015, 6, 30), 35),
        (datetime(2016, 12, 31), 36),
        (datetime(2018, 12, 31), 37),
    ]
#done
def utc2tai(utc_time: datetime):
    leap_seconds_dict = {date: leap_sec for date, leap_sec in leap_seconds}

    leap_seconds_list = sorted(list(leap_seconds_dict.keys()))

    # Convert UTC time to TAI
    tai_time = utc_time
    for leap_date in leap_seconds_list:
        if utc_time >= leap_date:
            tai_time += timedelta(seconds=leap_seconds_dict[leap_date])

    return date_to_seconds_since_1958(tai_time)
#done
def tai2utc(tai_time: float):
    tai_time = seconds_since_1958_to_date(tai_time)

    leap_seconds_dict = {date: leap_sec for date, leap_sec in leap_seconds}

    leap_seconds_list = sorted(leap_seconds_dict.keys())

    # Convert TAI time to UTC
    utc_time = tai_time
    for leap_date in reversed(leap_seconds_list):
        if utc_time >= leap_date:
            utc_time -= timedelta(seconds=leap_seconds_dict[leap_date])

    return utc_time
#done
def date_to_seconds_since_1958(date_time):
    # Define the reference date (1 January 1958)
    reference_date = datetime(1958, 1, 1)
    
    # Calculate the difference in days between the given date and the reference date
    days_difference = (date_time - reference_date).days
    
    # Calculate the number of seconds corresponding to the time of day
    seconds_of_day = date_time.hour * 3600 + date_time.minute * 60 + date_time.second
    
    # Convert the difference in days to seconds and add the seconds of the day
    total_seconds = days_difference * 86400 + seconds_of_day
    
    return total_seconds
#done
def seconds_since_1958_to_date(seconds):
    seconds = int(seconds)
    # Calculate the number of days and remaining seconds
    days = seconds // 86400
    remaining_seconds = seconds % 86400
    
    # Calculate the date by adding the number of days to 1 January 1958
    date = datetime(1958, 1, 1) + timedelta(days=days)
    
    # Extract the time part
    hours = remaining_seconds // 3600
    minutes = (remaining_seconds % 3600) // 60
    seconds = remaining_seconds % 60
    
    # Combine date and time to create a datetime object
    date_time = datetime(date.year, date.month, date.day, hours, minutes, seconds)
    
    return date_time
"""
------------------------------------------Final step-------------------------------------------------------
"""
#done
def fixwrap(in_val):
    max_unsigned_int = 0xffff  # Maximum unsigned 32-bit integer value
    out = in_val + (in_val < 0) * max_unsigned_int
    out = out + (out < 0) * max_unsigned_int
    return out
#done
def get_offset(utime: datetime):
    filename = os.path.join(DEFAULT_SAVE_DIR, 'data/data_anal/c2_time_offsets.dat')
    offsets = {}
    sutime = Time(utime).mjd
    to_full_date = lambda mjd, time: int(mjd) + (int(time)/86400000)
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines[2:]:
            mjd1, time1, xpos1, moved1, offset1 = line.split()
            offsets[to_full_date(mjd1, time1)] = [xpos1, moved1, offset1]
    values = list(filter(lambda mjd: mjd>= sutime, list(offsets.keys())))
    if values == []: 
        smjd = list(offsets.keys())[-1]
        xpos, moved, offset = offsets[smjd]

    else: 
        smjd = values[0]
        keys = list(offsets.keys())

        mjd2 = keys[keys.index(smjd) - 1]
        mjd1 = smjd
        
        xpos2, moved2, offset2 = offsets[keys[keys.index(smjd) - 1]]
        xpos1, moved1, offset1 = offsets[smjd]
        
        if offset1 != offset2:
            days = (mjd1 - mjd2)*86400000
            s = (offset1-offset2)/days
            days1 = sutime - mjd2
            x = days1*86400000
            offset = s*(x-abs(int(mjd2)-mjd2)*86400000) + offset2
        else:
            offset = offset1

    return offset
#done
def adjust_all_date_obs(hdr):
    adj = {'date': '', 'time': '', 'err': ''}

    date = hdr['date-obs']
    time = hdr['time-obs'].strip()

    if time == '':
        adj['date'] = f"{date[:4]}/{date[5:7]}/{date[8:10]}"  # yyyy/mm/dd
        adj['time'] = date[11:]
        return adj

    utctime = datetime.strptime(date + ' ' + time, '%Y/%m/%d %H:%M:%S.%f')
    offset = get_offset(utctime)
    utctime -= timedelta(seconds = float(offset))

    new_date = datetime.strftime(utctime, '%Y/%m/%d')
    new_time = datetime.strftime(utctime, '%H:%M:%S.%f')

    adj['date'] = new_date
    adj['time'] = new_time

    return adj

def adjust_hdr(hdr):
    adjusted = {'date': '', 'time': '', 'err': '', 'xpos': 0.0, 'ypos': 0.0, 'roll': 0.0}
    
    fits_hdr = True if isinstance(hdr, dict) else False
    tel = hdr['detector'].upper().strip()
    date = hdr['date-obs'].strip()
    time = hdr['time-obs'].strip()

    adj_dt = adjust_all_date_obs(hdr)

    adjusted['date'] = adj_dt['date']
    adjusted['time'] = adj_dt['time']
    adjusted['err'] = adj_dt['err']    

    tai = utc2tai(datetime.strptime(adj_dt['date'] + ' ' + adj_dt['time'], '%Y/%m/%d %H:%M:%S.%f')) + 32.184
    
    datafile = lambda filename: os.path.join(DEFAULT_SAVE_DIR, filename)

    if tel == 'C2':
        datafile = datafile('c2_pre_recovery_adj_xyr_medv2.sav' if adj_dt['date'] < '1998/07/01' else 'c2_post_recovery_adj_xyr_medv2.sav')
    elif tel == 'C3':
        datafile = datafile('c3_pre_recovery_adj_xyr_medv2.sav' if adj_dt['date'] < '1998/07/01' else 'c3_post_recovery_adj_xyr_medv2.sav')
    else:
        print(f'WARNING: Header is for a {tel} image. Only C2 and C3 images can be corrected for sun-center and roll.')
        print('(using roll=0.0, xpos=crpix1, ypos=crpix2).')
        adjusted['xpos'] = hdr.get('crpix1', 0.0)
        adjusted['ypos'] = hdr.get('crpix2', 0.0)
        adjusted['roll'] = 0.0
        return adjusted
    
    df = pd.read_csv(datafile)
    c_tai = df['c_tai'].values
    c_xmed = df['c_xmed'].values
    c_ymed = df['c_ymed'].values
    c_rmed = df['c_rmed'].values
    

    version2 = datafile
    ind = np.where(c_tai <= tai)[0]
    cnt = len(ind)

    if cnt == 0:
        bind = 0
        aind = 0
        print('WARNING: Header date_obs < first date_obs row in center and roll file; Using first row.')
    else:
        bind = cnt - 1
        if c_tai[bind] == tai:
            aind = bind
        else:
            aind = bind + 1
            if aind == len(c_tai):
                aind = bind
                print('WARNING: Header date_obs > last date_obs row in center and roll file; Using last row.')

    if bind != aind:
        # interpolate
        adjusted['xpos'] = linear_interp(c_tai[bind], c_xmed[bind], c_tai[aind], c_xmed[aind], tai)
        adjusted['ypos'] = linear_interp(c_tai[bind], c_ymed[bind], c_tai[aind], c_ymed[aind], tai)
        adjusted['roll'] = linear_interp(c_tai[bind], c_rmed[bind], c_tai[aind], c_rmed[aind], tai)
        version2 += ' (interpolated)'
    else:
        adjusted['xpos'] = c_xmed[bind]
        adjusted['ypos'] = c_ymed[bind]
        adjusted['roll'] = c_rmed[bind]

    return adjusted
#done
def linear_interp(x1, y1, x2, y2, x):
        s = (y2 - y1) / (x2 - x1)
        y = s * (x - x1) + y1
        return y
#done
def get_roll_or_xy(hdr, DEGREES=False):
    adjusted = {'xpos': 0.0, 'ypos': 0.0, 'roll': 0.0}
    sunroll = 0.
    interpolatedroll = 0.

    tel = hdr['TELESCOP'].strip().upper()

    date = hdr['date-obs'].strip()
    time = hdr['time-obs'].strip()

    adj_dt = adjust_all_date_obs(hdr)

    tai = utc2tai(datetime.strptime(adj_dt['date'] + ' ' + adj_dt['time'], '%Y/%m/%d %H:%M:%S.%f')) + timedelta(seconds =32.184)

    datafile = lambda filename: os.path.join(DEFAULT_SAVE_DIR, filename)
    
    if tel == 'C2':
        datafile = datafile('c2_pre_recovery_adj_xyr_medv2.sav' if adj_dt['date'] < '1998/07/01' else 'c2_post_recovery_adj_xyr_medv2.sav')
    elif tel == 'C3':
        datafile = datafile('c3_pre_recovery_adj_xyr_medv2.sav' if adj_dt['date'] < '1998/07/01' else 'c3_post_recovery_adj_xyr_medv2.sav')

    df = pd.read_csv(datafile, index = 0)
    c_tai = df['c_tai'].values
    c_xmed = df['c_xmed'].values
    c_ymed = df['c_ymed'].values
    c_rmed = df['c_rmed'].values

    ind = np.where(c_tai <= tai)
    cnt = len(ind[0])
    if cnt < 1:
        bind = 0
        aind = 1
    else:
        bind = cnt - 1
        aind = bind if abs(c_tai[bind] - tai)<60 else bind+1
        if tai > utc2tai(datetime.strptime('2001/11/14 10:00', '%Y/%m/%d %H:%M')) and tai < utc2tai(datetime.strptime('2001/11/15 15:00', '%Y/%m/%d %H:%M')):
            aind = bind + 1
        if aind == len(c_tai):
            aind = bind
            bind = aind-1
            adjusted = {'xpos': 0, 'ypos': 0, 'roll': 0}
            rollval = adjusted['roll']
            if tel != 'MLO' and rollval == 0:
                if tel == 'C2':
                    defroll=0.5
                    varianc= 0.1
                elif tel == 'C3':
                    defroll = -0.23
                    varianc = 0.07
                else:
                    defroll = 0
                    varianc = 0


                dobs = Time(datetime.strptime(date + ' ' + time, '%Y/%m/%d %H:%M:%S.%f'))
                pnt, type = get_sc_point(dobs, True)
                pntroll = pnt['sc_roll']
                
                if dobs.mjd>55501 and (pntroll == 0 or pntroll == 180): pntroll=read_so_at_roll_dat(dobs, type)
                if np.abs(pntroll)<3 and interpolatedroll!=0: rollval = interpolatedroll
                else:
                    if dobs.mjd > 52821:
                        if abs(np.abs(pntroll) - get_crota(dobs)>10): pntroll+=180
                        rollval = pntroll + defroll
            else:
                try:
                    sroll = hdr['crota1']
                except KeyError:
                    sroll = hdr['crota']
                if abs(sroll-adjusted['roll']) > 170:
                    center['xcen'] = hdr['crpix1'] - 1
                    center['ycen'] = hdr['crpix2'] - 1
            sunroll = rollval
    
    if bind!=aind: #interpolate
        val = linear_interp(c_tai[bind],c_xmed[bind],c_tai[aind],c_xmed[aind],tai)
        adjusted['xpos'] = val
        val = linear_interp(c_tai[bind],c_ymed[bind],c_tai[aind],c_ymed[aind],tai)
        adjusted['ypos'] = val
        if hdr['naxis1']!=1024 or hdr['naxis2'] != 1024:
            val = linear_interp(c_tai[bind], c_rmed[bind], c_tai[aind], c_rmed[aind], tai)
            interpolatedroll = val
        if (np.abs(c_rmed[bind]-c_rmed[aind])>5):
            adjusted['xpos'] = c_xmed[bind]
            adjusted['ypos'] = c_ymed[bind]
    else:
        adjusted['xpos'] = c_xmed[bind]
        adjusted['ypos'] = c_ymed[bind]
        adjusted['roll'] = c_rmed[bind]

    try:
        binfac = np.maximum(hdr['lebxsum'], 1)
    except KeyError:
        binfac = 1

    center = {}
    center['xcen'] = adjusted['xpos']/binfac
    center['ycen'] = adjusted['ypos']/binfac

    roll_out = sunroll if DEGREES else sunroll*np.pi/180

    return center, roll_out
#done
def get_sun_center(header, FULL: float = None, DEGREES: bool = False, RAW: bool = False):
    tel = header['detector'].upper()
    
    sun_cen = {}
    
    header['time-obs'] = '' if len(header['date-obs'].strip())>12 else header['time-obs']
    
    try: 
        factor1 = ((header['r2col'] - header['r1col'] + 1))/float(header['naxis1'])
        factor2 = ((header['r2row'] - header['r1row'] + 1)/ float(header['naxis2']))
        binfac = np.maximum(factor1, factor2)
    except KeyError: binfac = 1024/header['naxis1']

    sun_cen, roll = get_roll_or_xy(header, DEGREES) 
    
    if (sun_cen.xcen == 0 and sun_cen.ycen == 0) or (sun_cen.xcen == -1 and sun_cen.ycen == -1):
        if tel == 'C2':
            sun_cen['xcen'] = 510.2/binfac
            sun_cen['ycen'] = 506.5/binfac
        elif tel == 'C3':
            sun_cen['xcen'] = 518.2/binfac
            sun_cen['ycen'] = 532.5/binfac 
    
    if RAW: return sun_cen, roll

    if FULL is not None:
        factor = (1024/FULL)/binfac
        sun_cen['xcen'] /= factor
        sun_cen['ycen'] /= factor
        return sun_cen, roll
    
    return sun_cen, roll
#done
def rotate(Array, Direction):
    if Direction == 0 or Direction == 2:
        return Array.copy()
    elif Direction == 1:
        return np.rot90(Array, 1)
    elif Direction == 3:
        return np.rot90(Array, 2)
    elif Direction == 4:
        return np.transpose(Array)
    elif Direction == 5:
        return np.rot90(np.transpose(Array), 1)
    elif Direction == 6:
        return np.rot90(np.transpose(Array), 2)
    elif Direction == 7:
        return np.rot90(np.transpose(Array), -1)
    else:
        raise ValueError("Invalid direction. Direction should be in the range [0, 7].")
#done
def read_so_at_roll_dat(date):
    predictive_dir = os.path.join(DEFAULT_SAVE_DIR,'ancil_data/attitude/predictive')

    filename = os.path.join(predictive_dir, str(date.year), f'SO_AT_ROL_{date.strftime("%Y%m%d")}_V*.DAT')

    type = 'PredictFile'
    file = glob.glob(filename)[0]
    count = len(files)

    if count>1: files = sorted(files, reverse = True)
    
    if count == 0:
        print(f'Sorry; {filename} not found; returning 0.')
        vals = np.zeros(2)
        times = np.array([1, 83999999])
        type = 'None'
    else:
        # Get data from the first file
        fname = os.path.basename(file)
        print(f'Using {fname} for attitude info for {date}.')
        vals = np.loadtxt(file, delimiter = ' ', dtype = str)
        datetimes, vals = vals[:, 0], vals[:, 1]
        times = np.array([datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S.%f') for dt in datetimes])

    diff = np.abs(times - date)
    w = np.argmin(np.abs(diff))
    result = -float(vals[w])

    return result
#done
def get_sc_point(date, type, **kwargs):
    A = get_sc_att(date, type, **kwargs)
    RESULT = {'SC_X0': 0, 'SC_Y0': 0, 'SC_ROLL': 0}
    
    if type != 'Predictive':
        if not 'NOCORRECT' in kwargs and type != 'PredictFile' and\
        date> datetime(1996, 4, 16, 23, 25):A['SC_AVG_PITCH']+=1

        A['SC_AVG_PITCH']+=198/(radeg*3600)
        X0 = -A['SC_AVG_YAW'] * radeg * 3600
        Y0 = -A['SC_AVG_PITCH'] * radeg * 3600
        COS_R = np.cos(A['SC_AVG_ROLL'])
        SIN_R = np.sin(A['SC_AVG_ROLL'])
        RESULT['SC_X0'] = X0*COS_R + Y0*SIN_R
        RESULT['SC_Y0'] = -X0*SIN_R + Y0*COS_R
        RESULT['SC_ROLL'] = -A['SC_AVG_ROLL'] * radeg
    else:
        if os.path.exists(filepath:=os.path.join(DEFAULT_SAVE_DIR,'ancil_data/attitude/roll/nominal_roll_attitude.dat')):
            with open(filepath, 'r') as file:
                lines = file.readlines()
                roll_date = []
                roll_value = []
                for line in lines:
                    if line:= line.strip() and not line.startswith('#'):
                        date_str, value_str = line.split()
                        roll_date.append(datetime.strptime(date_str, '%Y-%m-%d'))
                        roll_value.append(float(value_str))

                w = max([i for i, rd in enumerate(roll_date) if rd <= date])
                if w > 0:
                    RESULT['SC_ROLL'] = roll_value[w]

    return RESULT	
#done
def to_mil(date: datetime):
    start = datetime(date.year, date.month, date.day)
    return (date - start).total_seconds() * 1000
#done
def get_sc_att(date):
    s_year = str(date.year).strip()
    base = os.path.join(DEFAULT_SAVE_DIR, 'ancil_data/attitude')
    filetype = 'PredictFile'
    last_file = glob.glob(f'{base}/predictive/{s_year}/SO_AT_PRE_{datetime.strftime(date, "%Y%m%d")}_V*.fts')[0]

    unit = np.asarray(fits.getdata(last_file))

    year = unit['YEAR']
    time = unit['ELLAPSED MILLISECONDS OF DAY']
    pmin = unit['SC MIN PITCH']
    rmin = unit['SC MIN ROLL']
    ymin = unit['SC MIN YAW']
    pmax = unit['SC MAX PITCH']
    rmax = unit['SC MAX ROLL']
    ymax = unit['SC MAX YAW']

    w = np.where(year!=0 and radeg*(rmax-rmin)<1 and\
         60*radeg*(ymax-ymin) < 1 and \
        60*radeg*(pmax-pmin) < 1)
    if len(w) == 0:
        print('Empty data file or no good data')
    
    time = time[w]
    diff = np.abs(time - to_mil(date))
    row = np.argmin(diff)

    # Store the result in the output structure
    result = {
        'sc_avg_pitch_eclip': unit['SC AVG PITCH ECLIPTIC (RAD)'][row],
        'sc_avg_roll_eclip': 	unit['SC AVG ROLL ECLIPTIC(RAD)'][row],
        'sc_avg_yaw_eclip': 	unit['SC AVG YAW ECLIPTIC(RAD)'][row],
        'sc_avg_pitch': 	unit['SC AVG PITCH (RAD)'][row],
        'sc_avg_roll': 	unit['SC AVG ROLL (RAD)'][row],
        'sc_avg_yaw': 	unit['SC AVG YAW (RAD)'][row],
        'gci_avg_pitch': 	unit['GCI AVG PITCH'][row],
        'gci_avg_roll': 	unit['GCI AVG ROLL'][row],
        'gci_avg_yaw': 	unit['GCI AVG YAW'][row],
        'gse_avg_pitch': 	unit['GSE AVG PITCH'][row],
        'gse_avg_roll': 	unit['GSE AVG ROLL'][row],
        'gse_avg_yaw': 	unit['GSE AVG YAW'][row],
        'gsm_avg_pitch': 	unit['GSM AVG PITCH'][row],
        'gsm_avg_roll': 	unit['GSM AVG ROLL'][row],
        'gsm_avg_yaw': 	unit['GSM AVG YAW'][row],
        'sc_std_dev_pitch': 	unit['SC STD DEV PITCH'][row],
        'sc_std_dev_roll': 	unit['SC STD DEV ROLL'][row],
        'sc_std_dev_yaw': 	unit['SC STD DEV YAW'][row],
        'sc_min_pitch': 	unit['SC MIN PITCH'][row],
        'sc_min_roll': 	unit['SC MIN ROLL'][row],
        'sc_min_yaw': 	unit['SC MIN YAW'][row],
        'sc_max_pitch': 	unit['SC MAX PITCH'][row],
        'sc_max_roll': 	unit['SC MAX ROLL'][row],
        'sc_max_yaw':	unit['SC MAX YAW'][row],
    }

    return result
#done
def get_crota(indate):
    intai = utc2tai(indate)
    datfile = os.path.join(DEFAULT_SAVE_DIR, 'ancil_data/attitude/roll/nominal_roll_attitude.dat')

    with open(datfile, 'r') as file:
        lines = list(map(
            lambda x: x.strip().split()[:3],
            filter(
                lambda x: '#' not in x,
                file.readlines()
            )
        ))
        for date, time, roll in lines:
            roll = float(roll)
            tai = utc2tai(datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S'))
            if intai<tai:
                break
            else:
                crota = roll
    return crota
#done
def rot(A, ANGLE, MAG=1.0, X0=None, Y0=None, INTERP=False, MISSING=None, PIVOT=False, CUBIC=False):
    if X0 is None:
        X0 = (A.shape[1] - 1) / 2.0
    if Y0 is None:
        Y0 = (A.shape[0] - 1) / 2.0

    if PIVOT:
        XC, YC = X0, Y0
    else:
        XC = (A.shape[1] - 1) / 2.0
        YC = (A.shape[0] - 1) / 2.0

    theta = np.deg2rad(-ANGLE)  # Angle in degrees clockwise
    c = np.cos(theta) * MAG
    s = np.sin(theta) * MAG

    kx = -XC + c * X0 - s * Y0
    ky = -YC + s * X0 + c * Y0

    affine_matrix = np.array([[c, -s, kx],
                               [s, c, ky]])

    if INTERP or CUBIC:
        order = 3 if CUBIC else 1
        output = affine_transform(A, affine_matrix, output_shape=A.shape, order=order, mode='constant', cval=MISSING)
    else:
        output = affine_transform(A, affine_matrix, output_shape=A.shape, order=0, mode='constant', cval=MISSING)

    return output




    
