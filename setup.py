from setuptools import setup, find_packages
from setuptools.command.install import install
from corkit.dataset import update
import asyncio
import os

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        asyncio.run(update())

version = '1.0.3'

def find_calibration_files():
    module_root = os.path.dirname(__file__)
    calibration_data_dir = os.path.join(module_root, 'corkit', 'data')
    calibration_files = []
    for root, _, files in os.walk(calibration_data_dir):
        for file in files:
            calibration_files.append(os.path.relpath(os.path.join(root, file), module_root))
    return calibration_files


setup(
    name='corkit',
    version=version,
    packages=find_packages(),
    package_data= {'corkit': find_calibration_files()},
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
        'dateutil',
        'scikit-image',
        'beautifulsoup4',
        'matplotlib',
        'pillow',
        'pandas',
    ],
    cmdclass={
        'install': CustomInstallCommand,
    }
)

