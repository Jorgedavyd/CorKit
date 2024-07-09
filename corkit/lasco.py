"""Utils dependencies"""

from .utils import (
    FITS,
    fixwrap,
    c2_warp,
    c3_warp,
    reduce_std_size,
    save,
    adjust_hdr,
    get_roll_or_xy,
    get_sun_center,
    rotate,
    rot,
    get_sec_pixel,
    get_solar_radius,
    reduce_statistics2,
    datetime_interval,
    get_exp_factor,
    correct_var,
    DEFAULT_SAVE_DIR,
    rebin,
    solar_ephem,
    subtense,
    telescope_pointing,
    sundist,
    eltheory,
    ne2mass,
    check_05,
)

from .reconstruction import dl_image, normal_model_reconstruction, transforms, cross_model_reconstruction, fourier_model_reconstruction

"""Pypi dependencies"""
from astropy.visualization import HistEqStretch, ImageNormalize
from typing import Union, Dict, Tuple, List, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from astropy.io import fits
from io import BytesIO
import numpy as np
import matplotlib
import aiofiles
import aiohttp
import asyncio
import copy
import glob
import os
from icecream import ic

from . import __version__

__all__ = ["level_1", "CME", "LASCOplot", "downloader", "c3_calibrate", "c2_calibrate"]


##############
#    LASCO   #
##############
# done
def level_1(
    fits_files: Union[str, BytesIO, List[Union[str, BytesIO]]],
    target_path: Optional[str] = None,
    format: Optional[str] = "fits",
    *args,
    **kwargs,
) -> None:
    assert format in [
        "fits",
        "fts",
        "jpg2",
        "png",
    ], "Must be fits, fts, jpg2 or png filetype"
    os.makedirs(target_path, exist_ok=True)
    if target_path is not None:
        assert format is not None, "Must define the extension of the file"
    # Import data
    if isinstance(fits_files, (BytesIO, str)):
        print("Importing data")
        try:
            img0, header = FITS(fits_files)
            filename = fits_files.split("/")[-1][:-4]
        except OSError:
            print("Corrupted file, try downloading again. Deleting...")
            os.remove(fits_files)
        if check_05(header):
            pass
        else:
            print(
                "File is already level 1.(Implemented just for CorKit derived fits files.)"
            )
            return img0, header
        if img0 is None:
            print("FITS file doesn't contain image data, avorting and deleting...")
            os.remove(fits_files)
            return None, None
        detector: str = header["detector"].strip().upper()
        header.add_history(
            f"corkit/lasco.py level_1: (function) {__version__}, 12/04/24"
        )
        if "detector" in kwargs:
            assert (
                kwargs["detector"].upper() in ["C2", "C3"]
                and detector == kwargs["detector"].upper()
            ), f'Not valid detector or file is not from LASCO {kwargs["detector"].upper()}, make sure all files in fits_files parameters are from the same detector and from LASCO.'
        # Applying pixelwise implemented corrections
        xsumming: float = np.maximum(header["sumcol"], 1) * np.maximum(
            header["lebxsum"], 1
        )
        ysumming: float = np.maximum(header["sumrow"], 1) * np.maximum(
            header["lebysum"], 1
        )
        summing: float = xsumming * ysumming
        if summing > 1:
            img0: np.array = fixwrap(img0)
            dofull: bool = False
        else:
            dofull: bool = True
        if (
            header["r2col"]
            - header["r1col"]
            + header["r2row"]
            - header["r1row"]
            - 1023
            - 1023
            != 0
        ):
            img0: np.array = reduce_std_size(img0, header, FULL=dofull)

        print(
            f'LASCO-{header["detector"]}:{header["filename"]}:{header["date-obs"]}T{header["time-obs"]}...'
        )
        match detector:
            case "C2":
                b, header = c2_calibrate(img0, header, **kwargs)
                b, header = c2_warp(b, header)
                zz: np.array = np.where(img0 <= 0)
                maskall: np.array = np.ones((header["naxis1"], header["naxis2"]))
                maskall[zz] = 0
                maskall, _ = c2_warp(maskall, header)
                b *= maskall
            case "C3":
                b, header = c3_calibrate(img0, header, *args, **kwargs)
                bn, header = c3_warp(b, header)
                zz: np.array = np.where(img0 <= 0)
                maskall: np.array = np.ones((header["naxis1"], header["naxis2"]))
                maskall[zz] = 0
                maskallw, _ = c3_warp(maskall, header)
                b = bn * maskallw * correct_var(header, args[1])[0]

        img, header = final_step(
            target_path, format, b, header, xsumming, ysumming, **kwargs
        )

        header["level_1"] = 1

        if target_path is not None:
            save(target_path, filename.replace(".", ""), format, img, header)

        print("Done!")
        return img, header

    elif isinstance(fits_files, list):
        _, sample_hdr = FITS(fits_files[0])
        detector = sample_hdr["detector"].strip().upper()
        out: List[Tuple[np.array, fits.Header]] = []

        match detector:
            case "C2":
                vig_fn = os.path.join(DEFAULT_SAVE_DIR, "c2vig_final.fts")
                vig_full = fits.getdata(vig_fn)
                for filepath in fits_files:
                    level_1(
                        filepath,
                        target_path,
                        format,
                        **kwargs,
                        detector=detector,
                        vig_full=vig_full,
                    )
            case "C3":
                vig_pre, vig_post = _read_vig_full()
                mask = _read_mask_full()
                ramp = _read_ramp_full()
                bkg = _read_bkg_full()
                forward, inverse = transforms()
                if 'model' not in kwargs:
                    model = normal_model_reconstruction()
                else:
                    match kwargs['model']:
                        case 'normal':
                            model = normal_model_reconstruction()
                        case 'fourier':
                            model = fourier_model_reconstruction()
                        case 'cross':
                            model = cross_model_reconstruction()
                args = (vig_pre, vig_post, mask, ramp, bkg, model, forward, inverse)
                for filepath in fits_files:
                    level_1(
                        filepath,
                        target_path,
                        format,
                        *args,
                        **kwargs,
                        detector=detector,
                    )

        return out


# done
def final_step(
    target_path: str,
    filetype: str,
    img: np.array,
    header: fits.Header,
    xsumming,
    ysumming,
    **kwargs,
) -> Tuple[np.array, fits.Header]:
    tcr = adjust_hdr(header)
    if header["date"] == "":
        c, r = get_sun_center(header, FULL=1024, MEDIAN=True)
        cx = c["xcen"]
        cy = c["ycen"]
    else:
        r: float = tcr["roll"]
        cx: float = tcr["xpos"]
        cy: float = tcr["ypos"]
    if np.abs(header["crota1"]) > 170:
        rectify: float = 180
        cntr: float = 511.5
        x: float = cx - cntr
        y: float = cy - cntr
        cx: float = (
            cntr
            + x * np.cos(rectify * np.pi / 180.0)
            - y * np.sin(rectify * np.pi / 180.0)
        )
        cy: float = (
            cntr
            + x * np.sin(rectify * np.pi / 180.0)
            + y * np.cos(rectify * np.pi / 180.0)
        )
        img: np.array = rotate(img, 2)
        r -= 180
    else:
        rectify = 0
    xc: float = (cx - header["r1col"] + 20) / xsumming
    yc: float = (cy - header["r1row"] + 1) / ysumming

    r_hdr: float = r
    if r < -180:
        r_hdr += 360
    crpix_x: float = xc + 1
    crpix_y: float = yc + 1

    if "NOROLL_CORRECTION" in kwargs or np.abs(r) < 1:
        pass
    else:
        img: np.array = rot(img, -1 * r, 1, xc, yc, INTERP=True, PIVOT=True, MISSING=0)
        rectify += r
    if filetype == "fits":
        # Adding final keywords and history modification
        header.add_history(
            f"CorKit Level 1 calibration with python modules: level_1.py, open source level 1 implementation."
        )
        header["date"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")
        header["filename"] = os.path.basename(target_path)
        header["CRPIX1"] = crpix_x
        header["CRPIX2"] = crpix_y
        header["CROTA"] = r_hdr
        header["CROTA1"] = r_hdr
        header["CROTA2"] = r_hdr
        header["CRVAL1"] = 0
        header["CRVAL2"] = 0
        header["CTYPE1"] = "HPLN-TAN"
        header["CTYPE2"] = "HPLN-TAN"
        header["CUNIT1"] = "arcsec"
        header["CUNIT2"] = "arcsec"
        platescl = get_sec_pixel(header)
        header["CDELT1"] = platescl
        header["CDELT2"] = platescl
        header["XCEN"] = 0 + platescl * ((header["naxis1"] + 1) / 2 - crpix_x)
        header["YCEN"] = 0 + platescl * ((header["naxis2"] + 1) / 2 - crpix_y)
        header["DATE-OBS"] = header["DATE-OBS"] + "T" + header["TIME-OBS"]
        header["TIME-OBS"] = ""
        rsun = get_solar_radius(header)
        header["RSUN"] = rsun
    if "NOSCALE" in kwargs:
        pass
    else:
        # mirar
        if header["detector"].strip().upper() == "C3" and header["filter"] == "Clear":
            scalemin = 0
            scalemax = 6.5e-9
            if not "NOSTAT" in kwargs:
                header = reduce_statistics2(img, header, satmax=scalemax)

            bscale = (scalemax - scalemin) / 65536
            bzero = bscale * 32769
            ind = np.where(img != 0)
            bout = np.zeros((header["naxis2"], header["naxis1"]))
            bout[ind] = np.round(
                (np.maximum(np.minimum(img[ind], scalemax), scalemin) - bzero) / bscale
            ).astype(int)
            if filetype == "fits":
                header["BSCALE"] = bscale
                header["BZERO"] = bzero
                header.add_comment(
                    f" FITS coordinate for center of full image is (512.5,512.5). Rotate image CROTA degrees CCW to correct. Data is scaled between {scalemin} and {scalemax}. Percentile values are before scaling."
                )
        else:
            if not "NOSTAT" in kwargs:
                header = reduce_statistics2(img, header)
            bout = np.float32(img)

    return bout, header


# done
class downloader:
    tools = ["c2", "c3"]
    batch_size = 2

    def __init__(self, tool: str, root: str = "./SOHO/LASCO/"):
        assert tool in self.tools, f"Not in tools: {self.tools}"
        self.tool = tool
        self.lasco_root = root
        self.fits_root = lambda day, hour: os.path.join(
            self.lasco_root, f"{self.tool}/{day}_{hour}.fits"
        )
        self.url = (
            lambda date, name: f"https://lasco-www.nrl.navy.mil/lz/level_05/{date[2:]}/{self.tool}/{name}"
        )
        self.url_img_txt = (
            lambda date: f"https://lasco-www.nrl.navy.mil/lz/level_05/{date[2:]}/{self.tool}/img_hdr.txt"
        )
        os.makedirs(os.path.join(self.lasco_root, self.tool), exist_ok=True)

    def get_check_tasks(self, scrap_date: tuple[datetime, datetime]):
        scrap_date = datetime_interval(scrap_date[0], scrap_date[-1], timedelta(days=1))
        self.new_scrap_date_list = [
            date for date in scrap_date if glob.glob(self.fits_root(date, "*")) == []
        ]

    async def get_download_tasks(self):
        for i in range(0, len(self.new_scrap_date_list), self.batch_size):
            await asyncio.gather(
                *[
                    self.download_day(day)
                    for day in self.new_scrap_date_list[i : i + self.batch_size]
                ]
            )

    async def download_day(self, day):
        names_hours = await asyncio.gather(self.scrap_metadata(self.url_img_txt(day)))
        await asyncio.gather(
            *[
                self.download_url(name, day, hour)
                for c in names_hours
                for name, hour in c.items()
            ]
        )

    async def download_url(self, name, day, hour):
        async with aiohttp.ClientSession() as session:
            async with (
                session.get(self.url(day, name), ssl=False) as response,
                aiofiles.open(self.fits_root(day, hour), "wb") as f,
            ):
                await f.write(await response.read())

    async def scrap_metadata(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url, ssl=False) as response:
                data = await response.text()
                return {
                    line.split()[0]: line.split()[2].replace(":", "")
                    for line in data.split("\n")[:-1]
                }

    async def downloader_pipeline(self, scrap_date):
        self.get_check_tasks(scrap_date)
        await self.get_download_tasks()

    async def __call__(self, scrap_date_list):
        for scrap_date in scrap_date_list:
            await self.downloader_pipeline(scrap_date)


# done
def _read_bkg_full():
    bkg_path = os.path.join(DEFAULT_SAVE_DIR, "3m_clcl_all.fts")
    with fits.open(bkg_path) as hdul:
        bkg = hdul[0].data.astype(float)
        bkg *= 0.8 / hdul[0].header["exptime"]
    return bkg


# done
def _read_ramp_full() -> np.array:
    ramp_path = os.path.join(DEFAULT_SAVE_DIR, "C3ramp.fts")
    ramp = fits.getdata(ramp_path)
    return ramp


# done
def _read_mask_full() -> np.array:
    msk_fn = os.path.join(DEFAULT_SAVE_DIR, "c3_cl_mask_lvl1.fts")
    mask = fits.getdata(msk_fn)
    return mask


# done
def _read_vig_full() -> Tuple[np.array, np.array]:
    vig_pre = os.path.join(DEFAULT_SAVE_DIR, "c3vig_preint_final.fts")
    vig_post = os.path.join(DEFAULT_SAVE_DIR, "c3vig_postint_final.fts")
    vig_pre = fits.getdata(vig_pre)
    vig_post = fits.getdata(vig_post)
    return vig_pre, vig_post


# done
def c3_calibrate(img0: np.array, header: fits.Header, *args, **kwargs):

    assert header["detector"] == "C3", "Not valid C3 fits file"

    if check_05(header):
        pass
    else:
        print("This file is already a Level 1 product.")
        return img0, header

    # returns the date
    mjd = header["mid_date"]

    # Get exposure factor and bias
    header, expfac, bias = get_exp_factor(header)  # define get_exp_factor

    # Correct the raw values
    header["exptime"] *= expfac
    header["offset"] = bias

    # Get calibration factor
    if not "NO_CALFAC" in kwargs:
        header, calfac = c3_calfactor(header)
    else:
        calfac = 1.0
        header.add_history("No calibration factor: 1")

    # Get mask, ramp, bkg(fuzzy) and vignetting function
    if args:
        vig_pre, vig_post, mask, ramp, bkg, model, forward, inverse = args
    else:
        vig_pre, vig_post = _read_vig_full()
        mask = _read_mask_full()
        ramp = _read_ramp_full()
        bkg = _read_bkg_full()
        forward, inverse = transforms()
        if 'model' not in kwargs:
            model = normal_model_reconstruction()

    header.add_history("C3ramp.fts 1999/03/18")
    header.add_history("c3_cl_mask_lvl1.fts 2005/08/08")

    if mjd < 51000:
        vig = vig_pre
        header.add_history("c3vig_preint_final.fts")
    else:
        vig = vig_post
        header.add_history("c3vig_postint_final.fts")

    if not "NO_VIG" in kwargs:
        pass
    else:
        vig = np.ones_like(vig)

    vig, ramp, bkg, mask = correct_var(header, vig, ramp, bkg, mask)

    img = c3_calibration_forward(img0, header, calfac, vig, mask, bkg, ramp, model, forward, inverse, **kwargs)

    header.add_history(
        f"corkit/lasco.py c3_calibrate: (function) {__version__}, 12/04/24"
    )

    return img, header


# done
def c3_calibration_forward(
    img0: np.array,
    header,
    calfac: float,
    vig: np.array,
    mask: np.array,
    bkg: np.array,
    ramp: np.array,
    model,
    forward,
    inverse,
    **kwargs,
) -> np.array:

    if header["fileorig"] == 0:
        img = img0 / header["exptime"]
        img = img * calfac * vig - ramp
        if not "NO_MASK" in kwargs.keys():
            img *= mask
        return img.T

    if header["filter"] != "Clear":
        ramp = 0

    header["polar"] = header["polar"].strip()

    if (
        header["polar"] == "PB"
        or header["polar"] == "TI"
        or header["polar"] == "UP"
        or header["polar"] == "JY"
        or header["polar"] == "JZ"
        or header["polar"] == "Qs"
        or header["polar"] == "Us"
        or header["polar"] == "Qt"
        or header["polar"] == "Qt"
        or header["polar"] == "Jr"
        or header["polar"] == "Jt"
    ):

        img = img0 / header["exptime"]
        img = img * calfac * vig

        if not "NO_MASK" in kwargs:
            img *= mask

        return img.T
    else:
        img = (img0 - header["offset"]) / header["exptime"]
        img = img * vig * calfac - ramp
        if not "NO_MASK" in kwargs:
            img *= mask
        img = dl_image(model, img.T, bkg, forward, inverse)
        return img


# done
def c3_calfactor(header: fits.Header, **kwargs) -> Tuple[fits.Header, float]:
    # Set calibration factor for the various filters
    filter_ = header["filter"].upper().strip()
    polarizer = header["polar"].upper().strip()
    mjd = header["mid_date"]
    if filter_ == "ORANGE":
        cal_factor = 0.0297
        polref = cal_factor / 0.25256  # absolute value of +60
        if polarizer == "CLEAR":
            cal_factor *= 1.0
        elif polarizer == "+60DEG":
            cal_factor = polref
        elif polarizer == "0DEG":
            cal_factor = polref * 0.9648
        elif polarizer == "-60DEG":
            cal_factor = polref * 1.0798
        else:
            cal_factor *= 1
    elif filter_ == "BLUE":
        cal_factor = 0.0975
        polref = cal_factor / 0.25256  # absolute value of +60
        if polarizer == "CLEAR":
            cal_factor *= 1.0
        elif polarizer == "+60DEG":
            cal_factor = polref
        elif polarizer == "0DEG":
            cal_factor = polref * 0.9734
        elif polarizer == "-60DEG":
            cal_factor = polref * 1.0613
        else:
            cal_factor *= 1
    elif filter_ == "CLEAR":
        cal_factor = 7.43e-8 * (mjd - 50000) + 5.96e-3
        polref = cal_factor / 0.25256  # absolute value of +60
        if polarizer == "CLEAR":
            cal_factor *= 1.0
        elif polarizer == "+60DEG":
            cal_factor = polref
        elif polarizer == "0DEG":
            cal_factor = polref * 0.9832
        elif polarizer == "-60DEG":
            cal_factor = polref * 1.0235
        elif polarizer == "H_ALPHA":
            cal_factor = 1.541
        else:
            cal_factor = 0.0
    elif filter_ == "DEEPRD":
        cal_factor = 0.0259
        polref = cal_factor / 0.25256  # absolute value of +60
        if polarizer == "CLEAR":
            cal_factor *= 1.0
        elif polarizer == "+60DEG":
            cal_factor = polref
        elif polarizer == "0DEG":
            cal_factor = polref * 0.9983
        elif polarizer == "-60DEG":
            cal_factor = polref * 1.0300
        else:
            cal_factor *= 1
    elif filter_ == "IR":
        cal_factor = 0.0887
        polref = cal_factor / 0.25256  # absolute value of +60
        if polarizer == "CLEAR":
            cal_factor *= 1.0
        elif polarizer == "+60DEG":
            cal_factor = polref
        elif polarizer == "0DEG":
            cal_factor = polref * 0.9833
        elif polarizer == "-60DEG":
            cal_factor = polref * 1.0288
        else:
            cal_factor *= 1
    else:
        cal_factor = 0.0

    # Correct calibration factor for pixel summation
    if not "NO_SUM" in kwargs:
        sumcol = header["SUMCOL"]
        sumrow = header["SUMROW"]
        lebxsum = header["LEBXSUM"]
        lebysum = header["LEBYSUM"]
        if sumcol > 0:
            cal_factor /= sumcol
        if sumrow > 0:
            cal_factor /= sumrow
        if lebxsum > 1:
            cal_factor /= lebxsum
        if lebysum > 1:
            cal_factor /= lebysum

    cal_factor *= 1e-10

    header.add_history(
        f"corkit/lasco.py c3_calfactor: (function) 12/04/24: {cal_factor}"
    )

    return header, cal_factor


# done
def c2_calfactor(header: fits.Header, **kwargs) -> Tuple[fits.Header, float]:
    mjd = header["mid_date"]
    filter_ = header["filter"].strip().upper()
    polarizer = header["polar"].strip().upper()

    cal_factor = 0.0
    deg = ["+60DEG", "0DEG", "-60DEG", "ND"]

    if filter_ == "ORANGE":
        cal_factor = 4.60403e-07 * mjd + 0.0374116
        polref = cal_factor / 0.25256
        if polarizer == "CLEAR":
            cal_factor *= 1
        elif polarizer in deg:
            cal_factor = polref
        else:
            cal_factor *= 1
    elif filter_ in ["BLUE", "DEEPRD"]:
        cal_factor = 0.1033
        polref = cal_factor / 0.25256
        if polarizer == "CLEAR":
            cal_factor *= 1
        elif polarizer in deg:
            cal_factor = polref
        else:
            if filter_ == "BLUE":
                cal_factor *= 1
            else:
                cal_factor = 0
    elif filter_ in ["HALPHA", "LENS"]:
        cal_factor = 0.1033
        polref = cal_factor / 0.25256
        if polarizer == "CLEAR":
            cal_factor *= 1
        elif polarizer in deg:
            cal_factor = polref
        else:
            cal_factor *= 1
    else:
        cal_factor = 0

    if not "NO_SUM" in kwargs:
        if header["sumcol"] > 0:
            cal_factor /= header["sumcol"]
        if header["sumrow"] > 0:
            cal_factor /= header["sumrow"]
        if header["lebxsum"] > 1:
            cal_factor /= header["lebxsum"]
        if header["lebysum"] > 1:
            cal_factor /= header["lebysum"]

    cal_factor *= 1e-10

    header.add_history(
        f"corkit/lasco.py c2_calfactor: (function) 12/04/24: {cal_factor}"
    )

    return header, cal_factor


# done
def c2_calibrate(
    img0: np.array, header: fits.Header, **kwargs
) -> Tuple[np.array, fits.Header]:
    assert header["detector"] == "C2", "This is not a C2 valid fits file"

    if check_05(header):
        pass
    else:
        print("This file is already a Level 1 product.")
        return img0, header
    
    vig_full = kwargs.get('vig_full', None)
    # Get exposure factor and dark current offset
    header, expfac, bias = get_exp_factor(header)  # change for python imp

    header["exptime"] *= expfac
    header["offset"] = bias

    # Calculate calibration factor

    if not "NO_CALFAC" in kwargs:
        header, calfac = c2_calfactor(header, **kwargs)
    else:
        calfac = 1.0

    # Read vignetting function and mask
    if vig_full is not None:
        pass
    else:
        vig_fn = os.path.join(DEFAULT_SAVE_DIR, "c2vig_final.fts")
        vig_full = fits.getdata(vig_fn)

    if not "NO_VIG" in kwargs:
        # Apply mask to vignetting correction
        vig_full[vig_full < 0.0] = 0.0
        vig_full[vig_full > 100.0] = 100.0
        header.add_history(f"c2vig_final.fts")
    else:
        vig_full = np.ones_like(vig_full)

    vig_full = correct_var(header, vig_full)[0]
    img = c2_calibration_forward(img0, header, calfac, vig_full)

    header.add_history(
        f"corkit/lasco.py c2_calibrate: (function) {__version__}, 12/04/24"
    )

    return img, header


# done
def c2_calibration_forward(img0, header, calfac, vig):
    if header["polar"] in [
        "PB",
        "TI",
        "UP",
        "JY",
        "JZ",
        "Qs",
        "Us",
        "Qt",
        "Qt",
        "Jr",
        "Jt",
    ]:
        img = img0 / header["exptime"]
        img = img * calfac
        img = img * vig
        return img.T
    else:
        img = (img0 - header["offset"]) * calfac / header["exptime"]
        img = img * vig
        return img.T


#######################
#       Plot base     #
#######################
class _Plot:
    def imshow(self, img, metadata):
        # Defining plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100, facecolor="black")
        cmap = copy.copy(matplotlib.colormaps["gray"])
        cmap.set_bad(color="red")

        # Showing the image
        ax[0].imshow(
            img, norm=self.__norm(img), interpolation="nearest", origin="lower"
        )
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # Next to the metadata
        metadata_str = "\n\n".join(
            [f"{key}: {value}" for key, value in metadata.items()]
        )
        ax[1].text(
            0, 0.85, metadata_str, fontsize=10, color="white", verticalalignment="top"
        )
        ax[1].set_title(self.name, color="white")
        ax[1].axis("off")

        plt.show()

    def __norm(self, img):
        finite_data = img[np.isfinite(img)]
        norm = ImageNormalize(
            stretch=HistEqStretch(finite_data)
        )  # from sunpy mapbase.py
        return norm


###########
#   CME   #
###########


class LASCOplot(_Plot):
    """
    # LASCOplot
    Visualiser object for LASCO C2, C3 images
    """

    def __create_meta(self, header) -> Dict[str, str]:
        self.name = header["telescop"].strip()
        return {
            "Coronagraph": header["instrume"].strip()
            + " "
            + header["detector"].strip(),
            "Date - MJD - Time": f'{header["date-obs"]} - {header["MID_DATE"]} - {header["MID_TIME"]}',
            "Level": "0.5" if check_05(header) else "1",
            "Bias": header["OFFSET"],
            "Exposure time": header["exptime"],
            "Resolution": f'({header["naxis1"]}, {header["naxis2"]})',
            "Filter": header["filter"].strip(),
            "Polar": header["polar"].strip(),
        }

    def plot(self, img: np.array, header: fits.Header) -> None:
        """
        # LASCOPlot().plot
        Creates a plot for the given image and metadata from the header.
        """
        self.imshow(img, self.__create_meta(header))


class CME(_Plot):
    """
    # CME
    Object to compute and visualize Coronal Mass Ejections (CMEs)
    """

    def __create_meta(
        self, cme_mass: np.array, fn_header: fits.Header
    ) -> Dict[str, Union[str, float]]:
        self.name = (
            fn_header["instrume"].strip()
            + " "
            + fn_header["detector"].strip()
            + " - "
            + "CME Analysis"
        )
        return {
            "Coronagraph": fn_header["instrume"].strip()
            + " "
            + fn_header["detector"].strip(),
            "CME mass": np.sum(cme_mass),
            "Date - MJD - Time": f'{fn_header["date-obs"]} - {fn_header["MID_DATE"]} - {fn_header["MID_TIME"]}',
            "Bias": fn_header["OFFSET"],
            "Exposure time": fn_header["exptime"],
            "Resolution": f'({fn_header["naxis1"]}, {fn_header["naxis2"]})',
            "Filter": fn_header["filter"].strip(),
            "Polar": fn_header["polar"].strip(),
        }

    def __calc_cme_mass(
        self, img: np.array, header: np.array, box, **kwargs
    ) -> Union[float, np.array]:
        coords = telescope_pointing(header)
        xsz = header["naxis1"]
        ysz = header["naxis2"]
        _, dist = sundist(coords, xsize=xsz, ysize=ysz)
        pos_angle = kwargs["POS"] if "POS" in kwargs else 0
        _, b, bt, br, _ = eltheory(dist, pos_angle)
        wb = np.where(b == 0)
        nb = len(wb[0])
        if nb > 0:
            b[wb] = 1
        d = header["date-obs"]
        if check_05(header):
            yymmdd = d.replace("/", "").strip()[2:]
        else:
            yymmdd = d.split("T")[0].replace("/", "").strip()[2:]
        tel = header["telescop"].strip().upper()
        solar_radius = (
            solar_ephem(yymmdd, True) if tel == "SOHO" else solar_ephem(yymmdd, False)
        )
        solar_radius *= 3600
        cm_per_arcsec = 6.96e10 / solar_radius
        cm2_per_pixel = (cm_per_arcsec * subtense(header["detector"].strip())) ** 2
        conv = cm2_per_pixel if "ONLY_NE" in kwargs else ne2mass(1.0) * cm2_per_pixel
        mass = img / (bt - br) if "PB" in kwargs else img / b
        mass *= conv
        if "ROI" in kwargs:
            mass = mass[box]
        elif "ALL" not in kwargs:
            p1col = box[0, 0]
            p1row = box[1, 0]
            p2col = box[0, 1] - 1
            p2row = box[1, 1] - 1
            mass = mass[p1row : p2row + 1, p1col : p2col + 1]
        totmass = np.sum(mass)

        return mass if "ALL" in kwargs else totmass

    def mass(
        self,
        bn: Union[str, Tuple[np.array, fits.Header]],
        fn: Union[str, List[str]],
        *args,
        **kwargs,
    ) -> Union[np.array, List[np.array]]:
        """
        # CME().mass
        Computes the mass for the Coronal Mass ejection in fn (filepath) given
        a base image (bn)
        bn (str): filepath for the base image.
        fn (str): filepath for the target image.
        """
        if isinstance(bn, tuple):
            b, hb = bn
        else:
            b, hb = FITS(bn)

        if isinstance(fn, str):
            a, ha = FITS(fn)
            assert (
                hb["detector"].strip() == ha["detector"].strip()
            ), "Base image and output final image should belong to the same detector"
            nx: int = ha["naxis1"]
            ny: int = ha["naxis2"]

            match ha["detector"].strip().upper():
                case "C3":
                    cala, ha = c3_calibrate(a, ha, *args, **kwargs)
                    calb, hb = c3_calibrate(b, hb, *args, **kwargs)
                case "C2":
                    cala, ha = c2_calibrate(a, ha, **kwargs)
                    calb, hb = c2_calibrate(b, hb, **kwargs)
            sumxb = int(np.maximum(hb["sumcol"], 1)) * hb["lebxsum"]
            sumyb = int(np.maximum(hb["sumrow"], 1)) * hb["lebysum"]
            sumxa = int(np.maximum(ha["sumcol"], 1)) * ha["lebxsum"]
            sumya = int(np.maximum(ha["sumrow"], 1)) * ha["lebysum"]

            if (sumxb != sumxa) or (sumyb != sumya):
                calb = rebin(calb, hb["naxis1"] / sumxb, hb["naxis2"] / sumyb)
                hb["lebxsum"] = sumxa // int(np.maximum(hb["sumcol"], 1))
                hb["lebysum"] = sumya // int(np.maximum(hb["sumrow"], 1))
                sb = b.shape
                hb["naxis1"] = sb[0]
                hb["naxis2"] = sb[1]
            if nx != hb["naxis1"] or ny != hb["naxis2"]:
                r1col: int = int(np.maximum(ha["r1col"], hb["r1col"]))
                r2col: int = int(np.minimum(ha["r2col"], hb["r1col"]))
                r1row: int = int(np.maximum(ha["r1row"], hb["r1row"]))
                r2row: int = int(np.minimum(ha["r2row"], hb["r1row"]))
                xa1: int = (ha["r1col"] - r1col) // sumxa
                xa2: int = (ha["r2col"] - r2col) // sumxa
                ya1: int = (ha["r1row"] - r1row) // sumya
                ya2: int = (ha["r2row"] - r2row) // sumya

                cala = cala[int(ya1) : int(ya2 + 1), int(xa1) : int(xa2 + 1)]
                calb = calb[int(ya1) : int(ya2 + 1), int(xa1) : int(xa2 + 1)]

                ha["r1col"] = r1col
                hb["r1col"] = r1col
                ha["r1row"] = r1row
                hb["r1row"] = r1row
                ha["r2col"] = r1col
                hb["r2col"] = r2col
                ha["r2row"] = r2row
                hb["r2row"] = r2row
                ha["naxis1"] = xa2 - xa1 + 1
                hb["naxis1"] = ha["naxis1"]
                ha["naxis2"] = ya2 - ya1 + 1
                hb["naxis2"] = ha["naxis2"]

            dif = cala - calb

            mass = self.__calc_cme_mass(dif, ha, None, ALL=True)

            if "target_path" in kwargs and "format" in kwargs:
                ha.add_history(
                    f'corkit/lasco.py CME (object) analysis module, returned the mass given the base {hb["fileorig"]}.'
                )
                ha["CME_MASS"] = f"{np.sum(mass):.3e}"
                save(kwargs["target_path"], kwargs["format"], mass, ha)

            return mass
        elif isinstance(fn, list):
            _, sample_hdr = FITS(fn[0])
            detector = sample_hdr["detector"].strip().upper()
            out: List[np.array] = []

            match detector:
                case "C2":
                    vig_fn = os.path.join(DEFAULT_SAVE_DIR, "c2vig_final.fts")
                    vig_full = fits.getdata(vig_fn)
                    for filepath in fn:
                        mass = self.mass(
                            bn, filepath, **kwargs, detector=detector, vig_full=vig_full
                        )
                        out.append(mass)
                case "C3":
                    vig_pre, vig_post = _read_vig_full()
                    mask = _read_mask_full()
                    ramp = _read_ramp_full()
                    bkg = _read_bkg_full()
                    args = (vig_pre, vig_post, mask, ramp, bkg)
                    for filepath in fn:
                        mass = self.mass(
                            bn, filepath, *args, **kwargs, detector=detector
                        )
                        out.append(mass)

            return out

    def plot(self, mass: np.array, fn_header: fits.Header) -> None:
        """
        # CME().plot
        Creates a visualization of the Coronal Mass ejection given the output mass from CME().mass()
        and its target fits path (fn):

        mass (np.array): Output from CME().mass(bn, fn)

        fn_header (fits.Header): Header from the fn path fits file.
        """
        metadata = self.__create_meta(mass, fn_header)
        self.imshow(mass, metadata)
