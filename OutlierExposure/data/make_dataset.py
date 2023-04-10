import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import h5py
from OutlierExposure.data import elia_data_loader as dl
from OutlierExposure.features import build_features
from config import settings
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
    filename='logs/generate_data.log'
)
EPS = sys.float_info.epsilon

start_time = datetime(2022, 3, 30, 0, 0, 0)
end_time = datetime(2022, 6, 25, 0, 0, 0)
psd_frame = timedelta(minutes=10)
psd_step = timedelta(minutes=8)
fs = 250
psd_toverlap = 15 
psd_tperseg = 40
psd_decimate = 2


def process_iteration(args):
    i, dt = args
    sensor = dl.Sensor(name='ACC', location='MO04', data_type='TDD', format='.tdms')
    loader = dl.DataLoader(sensor=sensor)
    data = loader.get_data(dt, dt + psd_frame)
    if data is None:
        logging.info(f'No data at {dt}')
        return None
    psds = {}
    for sensor_name, sensor_data in data.items():
        f, psd = build_features.compute_PSD(sensor_data, fs=fs,q=psd_decimate, toverlap=psd_toverlap, tperseg=psd_tperseg)
        psds[sensor_name] = psd
    return (i, dt.timestamp(), psds, f)


def main():
    path_psd = Path(settings.default['path_processed_data']) / 'PSDs1.h5'

    with h5py.File(path_psd, 'w') as f:
        psd_group = f.create_group('PSDs')
        psd_group.attrs['fs'] = fs
        psd_group.attrs['psd_frame'] = psd_frame.total_seconds()
        psd_group.attrs['psd_step'] = psd_step.total_seconds()
        psd_group.attrs['psd_toverlap'] = psd_toverlap
        psd_group.attrs['psd_tperseg'] = psd_tperseg
        psd_group.attrs['psd_decimate'] = psd_decimate

        num_iterations = int((end_time - start_time) / psd_step)
        args_list = [(i, start_time + i*psd_step) for i in range(num_iterations)]
        
        with Pool() as pool:
            results = pool.imap(process_iteration, args_list)

            for result in tqdm(results, total=num_iterations):
                if result is None:
                    continue
                i, dt, psds,freq = result
                sample_group = psd_group.create_group(str(i))
                psd_group[str(i)].attrs['time'] = dt
                for sensor_name, psd in psds.items():
                    sample_group.create_dataset(sensor_name, data=psd)

        psd_group.create_dataset('frequency', data=freq)
        

if __name__ == '__main__':
    main()
