from setuptools import setup, find_packages
from setuptools.command.install import install
import os

from corkit import __version__
from corkit.dataset import update

from pathlib import Path
import asyncio

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


class InstallDataset(install):
    def run(self):
        install.run(self)
        asyncio.run(update())


def find_calibration_files():
    module_root = os.path.dirname(__file__)
    calibration_data_dir = os.path.join(module_root, "corkit", "data")
    calibration_files = []
    for root, _, files in os.walk(calibration_data_dir):
        for file in files:
            calibration_files.append(
                os.path.relpath(os.path.join(root, file), module_root)
            )
    return calibration_files


if __name__ == "__main__":
    setup(
        name="corkit",
        version=__version__,
        packages=find_packages(),
        author="Jorge David Enciso Martínez",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author_email="jorged.encyso@gmail.com",
        description="Open source coronagraph data downloader and calibrator",
        url="https://github.com/Jorgedavyd/corkit",
        license="MIT",
        install_requires=[
            "astropy",
            "numpy",
            "aiofiles",
            "scipy",
            "aiohttp",
            "scikit-image",
            "beautifulsoup4",
            "matplotlib",
            "pillow",
            "pandas",
            "torch",
            "torchvision",
            "gdown"
        ],
        cmdclass={"install": InstallDataset},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: Implementation :: CPython",
            "Programming Language :: Python :: Implementation :: PyPy",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS :: MacOS X",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Astronomy",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Visualization",
            "Topic :: Scientific/Engineering :: Image Processing",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Framework :: Matplotlib",
            "Framework :: Pytest",
            "Framework :: Sphinx",
            "Framework :: Jupyter",
            "Framework :: IPython",
            "Environment :: Console",
            "Environment :: Web Environment",
            "Natural Language :: English",
            "Typing :: Typed",
        ],
    )
