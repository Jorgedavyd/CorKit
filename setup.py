from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys
import os

class InstallDataset(install):
    def run(self):
        install.run(self)
        subprocess.check_call([sys.executable, 'corkit/dataset.py'])

version = '1.0.4'

def find_calibration_files():
    module_root = os.path.dirname(__file__)
    calibration_data_dir = os.path.join(module_root, 'corkit', 'data')
    calibration_files = []
    for root, _, files in os.walk(calibration_data_dir):
        for file in files:
            calibration_files.append(os.path.relpath(os.path.join(root, file), module_root))
    return calibration_files

if __name__ == '__main__':
    setup(
        name='corkit',
        version=version,
        packages=find_packages(),
        author='Jorge David Enciso Martínez',
        author_email='jorged.encyso@gmail.com',
        description='Open source coronagraph data downloader and calibrator',
        url='https://github.com/Jorgedavyd/corkit',
        license='MIT',
        install_requires=[
            'astropy',
            'numpy',
            'aiofiles',
            'scipy',
            'scikit-image',
            'beautifulsoup4',
            'matplotlib',
            'pillow',
            'pandas',
        ],
        cmdclass={
            'install': InstallDataset
        }
    )



