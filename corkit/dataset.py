from corkit.utils import DEFAULT_SAVE_DIR, datetime_interval
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from collections import defaultdict
from bs4 import BeautifulSoup
from scipy.io import readsav
from itertools import chain
import pandas as pd
import numpy as np
import aiofiles
import aiohttp
import asyncio
import os

#done
def clean_old(file_list):
    dat_files = filter(lambda filename: filename.lower().endswith('.dat'), file_list)
    fts_files = filter(lambda filename: not filename.lower().endswith('.fts'), file_list)
    dat_file = sorted(dat_files)[-1]
    fts_file = sorted(fts_files)[-1]
    return [dat_file, fts_file]
#done
async def get_names(url: str, href):
    async with aiohttp.ClientSession() as client:
        async with client.get(url, ssl = False) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            names = soup.find_all('a', href = href)
            names = [name['href'] for name in names]
            return names
#done	
async def download_single(url: str, filepath: str):
    if not os.path.exists(filepath):
        async with aiohttp.ClientSession() as client:
            async with client.get(url, ssl = False) as response, aiofiles.open(filepath, 'wb') as f:
                await f.write(await response.read())
#done
async def update():
    print('Checking calib datasets...')
    base_link = lambda filename: f'https://soho.nascom.nasa.gov/solarsoft/soho/lasco/lasco/data/calib/{filename}'
    base_root = lambda filename: os.path.join(DEFAULT_SAVE_DIR, filename)
    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
    names = [
        'c3vig_postint_final.fts',
        'c3vig_preint_final.fts',
        '3m_clcl_all.fts',
        'C3ramp.fts',
        'c3_cl_mask_lvl1.fts',
        'c2vig_final.fts',
    ]

    links = list(map(base_link, names))
    roots = list(map(base_root, names))

    await asyncio.gather(*[download_single(url, filepath) for url, filepath in zip(links, roots)])
    print('Cheking data_anal directory...')

    os.makedirs(os.path.join(DEFAULT_SAVE_DIR, 'data/data_anal/'), exist_ok=True)
    await download_single('https://soho.nascom.nasa.gov/solarsoft/soho/lasco/lasco/data_anal/data/c2_time_offsets.dat', os.path.join(DEFAULT_SAVE_DIR, 'data/data_anal/c2_time_offsets.dat'))

    print('Checking ancil_data/attitude/predictive dataset...')
    root_base = lambda filename: os.path.join(DEFAULT_SAVE_DIR, f'ancil_data/attitude/predictive/{filename.split("_")[-2][:4]}',filename)

    years = [date[:4] for date in datetime_interval(datetime(2010,1,1), datetime.now(), relativedelta(years = 1))]
    
    for year in years:
        os.makedirs(os.path.join(DEFAULT_SAVE_DIR, 'ancil_data/attitude/predictive', year), exist_ok=True)

    date_base = lambda year: f'{year}/' if int(year)<2022 else ''
    
    pred_link = lambda date: f'https://soho.nascom.nasa.gov/data/ancillary/attitude/predictive/{date_base(date)}'

    names = await asyncio.gather(*[get_names(pred_link(date), lambda href: ('PRE' in href and href.lower().endswith('.fts')) or ('ROL' in href and href.lower().endswith('.dat'))) for date in years])

    names = [*chain.from_iterable(names)]

    cat_str = defaultdict(list)
    
    for filename in names:
        date = filename.split('_')[-2]
        cat_str[date].append(filename)
    
    download_filenames = []
    for file_list in cat_str.values():
        download_filenames.append(clean_old(file_list))
    
    download_filenames = [*chain.from_iterable(download_filenames)]
    
    batch_size = 500
    for i in range(0, len(download_filenames), batch_size):
        await asyncio.gather(*[download_single(pred_link(name.split('_')[-2][:4]) + f'{name}', root_base(name)) for name in download_filenames[i: i+batch_size]])	

    print('Checking nominall_roll_att.dat...')
    os.makedirs(os.path.join(DEFAULT_SAVE_DIR, 'ancil_data/attitude/roll/'), exist_ok=True)
    await download_single('https://soho.nascom.nasa.gov/data/ancillary/attitude/roll/nominal_roll_attitude.dat', os.path.join(DEFAULT_SAVE_DIR, 'ancil_data/attitude/roll/nominal_roll_attitude.dat'))

    print('Checking for .sav files...')
    await medv_update()

    print('Checking for exposure factor files...')
    
    exp_link = lambda date, n_tool: f'https://hesperia.gsfc.nasa.gov/ssw/soho/lasco/lasco/expfac/data/{date[2:6]}/c{n_tool}_expfactor_{date[2:]}.dat'
    exp_root = lambda date, n_tool: os.path.join(DEFAULT_SAVE_DIR, f'data_anal/data/{date[2:6]}/c{n_tool}_expfactor_{date[2:]}.dat') #YYYYMMDD
    
    interval = datetime_interval(datetime(1996, 2, 1), datetime.now(), timedelta(days = 1))
    folders = list(set([date[2:6] for date in interval]))
    for folder in folders:
        os.makedirs(os.path.join(DEFAULT_SAVE_DIR, 'data_anal/data/', folder), exist_ok=True)
    
    batch_size = 250
    for i in range(0, len(interval), batch_size):
        await asyncio.gather(*[download_single(exp_link(date,n_tool), exp_root(date, n_tool)) for date in interval[i: i+batch_size] for n_tool in [2,3]])
    
    print('Checking for occulter_center.dat file...')
    await download_single(
        'https://hesperia.gsfc.nasa.gov/ssw/soho/lasco/idl/convert/occulter_center.dat',
        os.path.join(DEFAULT_SAVE_DIR, 'occulter_center.dat')
    )
#done
async def update_single(url: str, filepath: str):
    if os.path.exists(filepath):
        os.remove(filepath)

    async with aiohttp.ClientSession() as client:
            async with client.get(url, ssl = False) as response, aiofiles.open(filepath, 'wb') as f:
                await f.write(await response.read())
#done
async def medv_update():
    link = lambda file: f'https://soho.nascom.nasa.gov/solarsoft/soho/lasco/idl/data/calib/{file}'
    pathfile = lambda file: os.path.join(DEFAULT_SAVE_DIR, file)
    files = [
        'c2_pre_recovery_adj_xyr_medv2.sav',
        'c3_pre_recovery_adj_xyr_medv2.sav',
        'c2_post_recovery_adj_xyr_medv2.sav',
        'c3_post_recovery_adj_xyr_medv2.sav'
    ]

    await asyncio.gather(*[update_single(link(file), pathfile(file)) for file in files])  
    filepaths = [pathfile(file) for file in files]
    dfs = []
    
    for filepath in filepaths:
        sav_df = readsav(filepath)
        index = pd.DatetimeIndex([date.decode('utf-8') for date in sav_df['c_dt']])
        keys = ['c_r', 'c_rmed', 'c_rmed', 'c_tai', 'c_x', 'c_xmed', 'c_y', 'c_ymed']
        data = {key: sav_df[key] for key in keys}
        new = np.asarray(sav_df['c_utc'])
        data['c_utc_time'] = new['time']
        data['c_utc_mjd'] = new['mjd']
        df = pd.DataFrame(data, index = index, columns = data.keys())
        dfs.append(df)
    
    for filepath, df in zip(filepaths, dfs):
        df.to_csv(filepath)

if __name__ == '__main__':
    asyncio.run(update())