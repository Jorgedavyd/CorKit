"""Utils dependencies"""
from corkit.utils import FITS, fixwrap, c2_warp, c3_warp, reduce_std_size, save, \
adjust_hdr, get_roll_or_xy, get_sun_center, rotate, rot, get_sec_pixel, get_solar_radius,\
reduce_statistics2, datetime_interval, get_exp_factor, correct_var, DEFAULT_SAVE_DIR

"""Pypi dependencies"""
from datetime import datetime, timedelta
from astropy.io import fits
from typing import Union
from io import BytesIO
import numpy as np
import aiofiles
import aiohttp
import asyncio
import os
import glob

version = '1.0.3'

#done
def level_1(
        fits_file: Union[str, BytesIO],
        target_path: str,
        **kwargs
) -> None:
    filetype = target_path.split('.')[-1]
    #Import data
    print('Importing data')
    img0, header = FITS(fits_file)
    detector = header['detector'].strip().upper()
    header.add_history(f'corkit/lasco.py level_1: (function) {version}, 12/04/24')
    assert (detector in ['C2', 'C3']), f'Not valid detector, not implemented for {detector}'
    #Applying pixelwise implemented corrections
    xsumming = np.maximum(header['sumcol'],1)*np.maximum(header['lebxsum'],1)
    ysumming = np.maximum(header['sumrow'],1)*np.maximum(header['lebysum'],1)
    summing = xsumming*ysumming
    if summing > 1:
        img0 = fixwrap(img0)
        dofull = False
    else:
        dofull = True
    if header['r2col']-header['r1col']+header['r2row']-header['r1row']-1023-1023 != 0: reduce_std_size(img0, header, FULL = dofull)
    
    print(f'LASCO-{header["detector"]}:{header["filename"]}:{header["date-obs"]}T{header["time-obs"]}...')
    match detector:
        case 'C2':
            b, header = c2_calibrate(img0, header, **kwargs)
            b, header = c2_warp(b, header)
            zz = np.where(img0<=0)
            maskall = np.ones((header['naxis1'], header['naxis2']))
            maskall[zz] = 0
            maskall, _ = c2_warp(maskall, header) 
            b*=maskall
        case 'C3':
            b, header = c3_calibrate(img0, header, **kwargs)
            bn, header = c3_warp(b, header)
            zz = np.where(img0<=0)
            maskall = np.ones((header['naxis1'], header['naxis2']))
            maskall[zz]=0
            maskallw, _ = c3_warp(maskall, header)
            _, mask = read_mask_full(header)
            b = bn*maskallw*mask

    img, header = final_step(target_path, filetype, b, header, xsumming, ysumming, **kwargs)

    save(target_path, filetype, img, header)

    return img, header
#done
def final_step(target_path, filetype, img, header, xsumming, ysumming, **kwargs):
    tcr = adjust_hdr(header)
    if header['date'] == '':
        r = get_roll_or_xy(header, 'ROLL', DEGREES=True)
        c = get_sun_center(header, FULL= 1024, MEDIAN=True) 
        cx = c['xcen']
        cy = c['ycen']
    else:
        r = tcr['roll']
        cx = tcr['xpos']
        cy = tcr['ypos']
    if np.abs(header['crota1']) > 170:
        rectify = 180
        cntr = 511.5
        x = cx - cntr
        y = cy - cntr
        cx = cntr + x*np.cos(rectify*np.pi/180.) - y*np.sin(rectify*np.pi/180.)
        cy = cntr + x*np.sin(rectify*np.pi/180.) + y*np.cos(rectify*np.pi/180.)
        img = rotate(img, 2)
        r-=180
    else: rectify=0
    xc = (cx - header['r1col']+ 20)/xsumming
    yc = (cy - header['r1row']+ 1)/ysumming

    r_hdr = r
    if r<-180: r_hdr +=360
    crpix_x = xc+1		
    crpix_y = yc+1

    if 'NOROLL_CORRECTION' in kwargs or np.abs(r)<1:
        pass
    else:
        img = rot(img, -1*r, 1, xc, yc, INTERP = True, PIVOT = True, MISSING=0) 
        rectify+=r
    if filetype == 'fits':
        #Adding final keywords and history modification
        header.add_history(f'CORKIT Level 1 calibration with python modules: to_level1.py, open source level 1 implementation.')
        header['date'] = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")
        header['filename'] = os.path.basename(target_path)
        header['CRPIX1'] = crpix_x
        header['CRPIX2'] = crpix_y
        header['CROTA'] = r_hdr
        header['CROTA1'] = r_hdr
        header['CROTA2'] = r_hdr
        header['CRVAL1'] = 0
        header['CRVAL2'] = 0
        header['CTYPE1'] = 'HPLN-TAN'
        header['CTYPE2'] = 'HPLN-TAN'
        header['CUNIT1'] = 'arcsec'
        header['CUNIT2'] = 'arcsec'
        platescl = get_sec_pixel(header)
        header['CDELT1'] = platescl
        header['CDELT2'] = platescl
        header['XCEN'] = 0 + platescl*((header['naxis1']+1)/2 - crpix_x)
        header['YCEN'] = 0 + platescl*((header['naxis2']+1)/2 - crpix_y)
        header['DATE-OBS'] = header['DATE-OBS'] + 'T' + header['TIME-OBS']
        header['TIME-OBS'] = ''
        rsun = get_solar_radius(header)
        header['RSUN'] = rsun
    if 'NOSCALE' in kwargs:
        pass
    else:
        #mirar
        if header['detector'].strip().upper() == 'C3' and header['filter'] == 'Clear':
            scalemin = 0
            scalemax = 6.5e-9
            if not 'NOSTAT' in kwargs:
                header = reduce_statistics2(img, header, satmax = scalemax)

            bscale = (scalemax-scalemin)/65536
            bzero=bscale*32769
            ind = np.where(img!=0)
            bout = np.zeros((header['naxis2'], header['naxis1']))
            bout[ind] = np.round((np.maximum(np.minimum(img[ind], scalemax), scalemin) - bzero) / bscale).astype(int)
            if filetype == 'fits':
                header['BSCALE'] = bscale
                header['BZERO'] = bzero
                header['BLANK'] = -32768
                header['COMMENT'] = f' FITS coordinate for center of full image is (512.5,512.5). Rotate image CROTA degrees CCW to correct. Data is scaled between {scalemin} and {scalemax}. Percentile values are before scaling.'
        else:
            if not 'NOSTAT' in kwargs: 
                header = reduce_statistics2(img, header)
            bout = np.float32(img)

    return bout, header
#done
class downloader:
        tools = ['c2', 'c3']
        batch_size = 2
        def __init__(self, tool: str, root: str = './SOHO/LASCO/'):
            assert (tool in self.tools), f'Not in tools: {self.tools}'
            self.tool = tool
            self.lasco_root = root
            self.fits_root = lambda day, hour: os.path.join(self.lasco_root, f'{self.tool}/{day}_{hour}.fits')
            self.url = lambda date, name: f'https://lasco-www.nrl.navy.mil/lz/level_05/{date[2:]}/{self.tool}/{name}'
            self.url_img_txt = lambda date: f'https://lasco-www.nrl.navy.mil/lz/level_05/{date[2:]}/{self.tool}/img_hdr.txt'
            os.makedirs(os.path.join(self.lasco_root, self.tool), exist_ok=True)

        def get_check_tasks(self, scrap_date: tuple[datetime, datetime]):
            scrap_date = datetime_interval(scrap_date[0], scrap_date[-1], timedelta(days = 1))
            self.new_scrap_date_list = [date for date in scrap_date if glob.glob(self.fits_root(date, '*')) == []]

        async def get_download_tasks(self):
            for i in range(0, len(self.new_scrap_date_list), self.batch_size):
                await asyncio.gather(*[self.download_day(day) for day in self.new_scrap_date_list[i:i+self.batch_size]])

        async def download_day(self, day):
            names_hours = await asyncio.gather(self.scrap_metadata(self.url_img_txt(day)))
            await asyncio.gather(*[self.download_url(name, day, hour) for c in names_hours for name, hour in c.items()])

        async def download_url(self, name, day, hour):
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url(day, name), ssl = False) as response, aiofiles.open(self.fits_root(day, hour), 'wb') as f:
                    await f.write(await response.read())

        async def scrap_metadata(self, url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url, ssl = False) as response:
                    data = await response.text()
                    return {line.split()[0]: line.split()[2].replace(':','') for line in data.split('\n')[:-1]}
            
        async def downloader_pipeline(self, scrap_date):
            self.get_check_tasks(scrap_date)
            await self.get_download_tasks()

        async def __call__(self, scrap_date_list):
            for scrap_date in scrap_date_list:    
                await self.downloader_pipeline(scrap_date)
#done
def read_bkg_full():
    bkg_path = os.path.join(DEFAULT_SAVE_DIR, '3m_clcl_all.fts')
    with fits.open(bkg_path) as hdul:
        bkg = hdul[0].data.astype(float)
        bkg *= 0.8/hdul[0].header['exptime']
    return bkg
#done
def read_ramp_full(header):
    ramp_path = os.path.join(DEFAULT_SAVE_DIR,'C3ramp.fts')
    ramp = fits.getdata(ramp_path)
    header.add_history('C3ramp.fts, 1999/03/18')
    return header, ramp
#done
def read_mask_full(header):
    msk_fn=os.path.join(DEFAULT_SAVE_DIR,'c3_cl_mask_lvl1.fts')
    mask = fits.getdata(msk_fn)
    header.add_history('c3_cl_mask_lvl1.fts 2005/08/08')
    return header, mask
#done
def read_vig_full(date, header):
    if date<51000:
        vig_path = os.path.join(DEFAULT_SAVE_DIR,'c3vig_preint_final.fts')
        header.add_history('c3vig_preint_final.fts')
    else:
        vig_path = os.path.join(DEFAULT_SAVE_DIR,'c3vig_postint_final.fts')
        header.add_history('c3vig_postint_final.fts')
    vig = fits.getdata(vig_path)
    return header, vig
#done
def c3_calibrate(
        img0: np.array, 
        header: dict,
        **kwargs):
    
    assert (header['detector'] == 'C3'), 'Not valid C3 fits file'
    header.add_history('corkit/lasco.py c3_calibrate: (function) - Python implementation')
    #returns the date
    mjd = header['mid_date']
    
    # Get exposure factor and bias
    header, expfac, bias = get_exp_factor(header) # define get_exp_factor
    
    # Correct the raw values
    header['exptime'] *= expfac
    header['offset'] = bias

    # Get calibration factor
    if not 'NO_CALFAC' in kwargs:
        header, calfac = c3_calfactor(header)
    else:
        calfac = 1.0
        header.add_history('No calibration factor: 1')

    # Get mask, ramp, bkg(fuzzy) and vignetting function
    header, vig = read_vig_full(mjd, header)
    header, mask = read_mask_full(header)
    header, ramp = read_ramp_full(header)
    bkg = read_bkg_full()

    if not 'NO_VIG' in kwargs:
        pass
    else:
        vig = np.ones_like(vig)
    
    vig, ramp, bkg, mask = correct_var(header, vig, ramp, bkg, mask)

    img = c3_calibration_forward(img0, header, calfac, vig, mask, bkg, ramp, **kwargs)
    
    header.add_history(f'corkit/lasco.py c3_calibrate: (function) {version}, 12/04/24')

    return img, header
#done
def c3_calibration_forward(
        img0: np.array,
        header,
        calfac: float,
        vig: np.array,
        mask: np.array,
        bkg: np.array,  
        ramp: np.array,
        **kwargs,
):
    
    
    if header['fileorig'] == 0:
        img = img0/header['exptime']
        img = img*calfac*vig - ramp
        if not 'NO_MASK' in kwargs.keys(): img*=mask
        return img
    
    if header['filter'] != 'Clear': ramp = 0

    header['polar'] = header['polar'].strip()

    if header['polar'] ==  'PB' or \
            header['polar'] == 'TI' or \
            header['polar'] == 'UP' or \
            header['polar'] == 'JY' or \
            header['polar'] == 'JZ' or \
            header['polar'] == 'Qs' or \
            header['polar'] == 'Us' or \
            header['polar'] == 'Qt' or \
            header['polar'] == 'Qt' or \
            header['polar'] == 'Jr' or \
            header['polar'] == 'Jt':
        
        img = img0/header['exptime']
        img = img*calfac*vig
        
        if not 'NO_MASK' in kwargs: img*=mask
        
        return img
    else:
        img = (img0-header['offset'])/header['exptime']
        ## recons
        img = img*vig*calfac - ramp
        if not 'NO_MASK' in kwargs:
            img*=mask
        return img
#done
def c3_calfactor(header,**kwargs):
    # Set calibration factor for the various filters
    filter_ = header['filter'].upper().strip()
    polarizer = header['polar'].upper().strip()
    mjd = header['mid_date']
    if filter_ == 'ORANGE':
        cal_factor = 0.0297
        polref = cal_factor / 0.25256  # absolute value of +60
        if polarizer == 'CLEAR':
            cal_factor *= 1.0
        elif polarizer == '+60DEG':
            cal_factor = polref
        elif polarizer == '0DEG':
            cal_factor = polref * 0.9648
        elif polarizer == '-60DEG':
            cal_factor = polref * 1.0798
        else:
             cal_factor*=1
    elif filter_ == 'BLUE':
        cal_factor = 0.0975
        polref = cal_factor / 0.25256  # absolute value of +60
        if polarizer == 'CLEAR':
            cal_factor *= 1.0
        elif polarizer == '+60DEG':
            cal_factor = polref
        elif polarizer == '0DEG':
            cal_factor = polref * 0.9734
        elif polarizer == '-60DEG':
            cal_factor = polref * 1.0613
        else:
             cal_factor*=1
    elif filter_ == 'CLEAR':
        cal_factor = 7.43e-8 * (mjd - 50000) + 5.96e-3
        polref = cal_factor / 0.25256  # absolute value of +60
        if polarizer == 'CLEAR':
            cal_factor *= 1.0
        elif polarizer == '+60DEG':
            cal_factor = polref
        elif polarizer == '0DEG':
            cal_factor = polref * 0.9832
        elif polarizer == '-60DEG':
            cal_factor = polref * 1.0235
        elif polarizer == 'H_ALPHA':
            cal_factor = 1.541
        else:
             cal_factor = 0.
    elif filter_ == 'DEEPRD':
        cal_factor = 0.0259
        polref = cal_factor / 0.25256  # absolute value of +60
        if polarizer == 'CLEAR':
            cal_factor *= 1.0
        elif polarizer == '+60DEG':
            cal_factor = polref
        elif polarizer == '0DEG':
            cal_factor = polref * 0.9983
        elif polarizer == '-60DEG':
            cal_factor = polref * 1.0300
        else:
             cal_factor*=1
    elif filter_ == 'IR':
        cal_factor = 0.0887
        polref = cal_factor / 0.25256  # absolute value of +60
        if polarizer == 'CLEAR':
            cal_factor *= 1.0
        elif polarizer == '+60DEG':
            cal_factor = polref
        elif polarizer == '0DEG':
            cal_factor = polref * 0.9833
        elif polarizer == '-60DEG':
            cal_factor = polref * 1.0288
        else:
             cal_factor*=1
    else:
        cal_factor = 0.0

    # Correct calibration factor for pixel summation
    if not 'NO_SUM' in kwargs:
        sumcol = header['SUMCOL']
        sumrow = header['SUMROW']
        lebxsum = header['LEBXSUM']
        lebysum = header['LEBYSUM']
        if sumcol > 0: cal_factor /= sumcol
        if sumrow > 0: cal_factor /= sumrow
        if lebxsum > 1: cal_factor /= lebxsum
        if lebysum > 1: cal_factor /= lebysum
    
    cal_factor *= 1e-10

    header.add_history(f'corkit/lasco.py c3_calfactor: (function) 12/04/24: {cal_factor}')

    return header, cal_factor
#done
def c2_calfactor(header, **kwargs):
    mjd = header['mid_date']
    filter_ = header['filter'].strip().upper()
    polarizer = header['polar'].strip().upper()

    cal_factor = 0.0
    deg = ['+60DEG', '0DEG', '-60DEG', 'ND']

    if filter_ == 'ORANGE':
        cal_factor = 4.60403e-07 * mjd + 0.0374116
        polref = cal_factor / 0.25256
        if polarizer == 'CLEAR':
             cal_factor *= 1
        elif polarizer in deg:
            cal_factor = polref
        else:
            cal_factor *= 1
    elif filter_ in ['BLUE', 'DEEPRD']:
        cal_factor = 0.1033
        polref = cal_factor / 0.25256
        if polarizer == 'CLEAR':
             cal_factor *= 1
        elif polarizer in deg:
            cal_factor = polref
        else:
            if filter_ == 'BLUE':
                cal_factor *= 1
            else:
                 cal_factor = 0
    elif filter_ in ['HALPHA', 'LENS']:
        cal_factor = 0.1033
        polref = cal_factor / 0.25256
        if polarizer == 'CLEAR':
             cal_factor *= 1
        elif polarizer in deg:
            cal_factor = polref
        else:
            cal_factor *= 1
    else:
         cal_factor = 0

    if not 'NO_SUM' in kwargs:
        if header['sumcol'] > 0:
            cal_factor /= header['sumcol']
        if header['sumrow'] > 0:
            cal_factor /= header['sumrow']
        if header['lebxsum'] > 1:
            cal_factor /= header['lebxsum']
        if header['lebysum'] > 1:
            cal_factor /= header['lebysum']

    cal_factor *= 1e-10

    header.add_history(f'corkit/lasco.py c2_calfactor: (function) 12/04/24: {cal_factor}')

    return header, cal_factor
#done
def c2_calibrate(img0, header, **kwargs):
    assert (header['detector']=='C2'), "This is not a C2 valid fits file"

    # Get exposure factor and dark current offset
    header, expfac, bias = get_exp_factor(header) #change for python imp

    header['exptime'] *= expfac
    header['offset'] = bias

    # Calculate calibration factor
    
    if not 'NO_CALFAC' in kwargs:
        header, calfac = c2_calfactor(header, **kwargs)
    else:    
        calfac = 1.0

    # Read vignetting function and mask
    vig_fn = os.path.join(DEFAULT_SAVE_DIR, 'c2vig_final.fts')
    vig_full = fits.getdata(vig_fn)
    
    if not 'NO_VIG' in kwargs:
        # Apply mask to vignetting correction
        vig_full[vig_full < 0.0] = 0.0
        vig_full[vig_full > 100.0] = 100.0
    else:
        vig_full = np.ones_like(vig_full)

    vig_full = correct_var(header, vig_full)[0]

    img = c2_calibration_forward(img0, header, calfac, vig_full)

    header.add_history(f'corkit/lasco.py c2_calibrate: (function) {version}, 12/04/24')

    return img, header
#done
def c2_calibration_forward(img0, header, calfac, vig):
    if header['polar'] in ['PB', 'TI', 'UP', 'JY', 'JZ', 'Qs', 'Us', 'Qt', 'Qt', 'Jr', 'Jt']:
        img = img0/header['exptime']
        img = img*calfac
        img = img*vig
        return img
    else:
        img = (img0 - header['offset']) * calfac / header['exptime']
        img = img*vig
        return img
