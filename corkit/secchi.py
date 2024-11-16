"""COR 2 & COR 1"""

import numpy as np
from .utils import FITS

from . import __version__

# class downloader():


"""level 0.5"""


def level_05(src_path, trg_path):
    img, header = FITS(src_path)

    return img, header


"""level 1"""


def level_1(src_path, trg_path):
    return src_path


import asyncio
import aiohttp
import aiofiles
from .utils import datetime_interval
from datetime import timedelta
from io import BytesIO
from bs4 import BeautifulSoup
import numpy as np
import glob
from itertools import chain
import os
import time
from PIL import Image


class STEREO:
    class SECCHI:
        class COR2:
            def __init__(self) -> None:
                self.url = (
                    lambda date, name: f"https://secchi.nrl.navy.mil/postflight/cor2/L0/a/img/{date}/{name}"
                )
                self.png_path = (
                    lambda date, hour: f"./data/SECCHI/COR2/{date}_{hour}.png"
                )
                self.fits_path = (
                    lambda date, hour: f"./data/SECCHI/COR2/{date}_{hour}.fits"
                )
                os.makedirs("./data/SECCHI/COR2/", exist_ok=True)

            def check_tasks(self, scrap_date):
                self.new_scrap_date_list = [
                    date
                    for date in scrap_date
                    if len(glob.glob(self.png_path(date, "*"))) == 0
                ]
                print(self.new_scrap_date_list)

            async def scrap_date_names(self, date):
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.url(date, ""), ssl=False) as response:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")
                        names = [
                            name["href"]
                            for name in soup.find_all(
                                "a", href=lambda href: href.endswith("d4c2A.fts")
                            )
                        ]
                        print(names)

            async def download_url(self, name):
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.url(name.split("_")[0], name), ssl=False
                    ) as response, aiofiles.open(
                        self.fits_path(*name.split("_")[:2]), "wb"
                    ) as f:
                        await f.write(await response.read())

            def get_days(self, scrap_date):
                return [
                    *chain.from_iterable(
                        [glob.glob(self.png_path(date, "*")) for date in scrap_date]
                    )
                ]

            def get_scrap_names_tasks(self, scrap_date):
                return [self.scrap_date_names(date) for date in scrap_date]

            async def downloader_pipeline(self, scrap_date):
                self.check_tasks(scrap_date)
                names = await asyncio.gather(*self.get_scrap_names_tasks(scrap_date))
                names = [*chain.from_iterable(names)]
                await asyncio.gather(*[self.download_url(name) for name in names])

            def data_prep(self, scrap_date):
                scrap_date = datetime_interval(
                    scrap_date[0], scrap_date[-1], timedelta(days=1)
                )
                return self.get_days(scrap_date)
