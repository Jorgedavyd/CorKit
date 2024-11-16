from setuptools import setup, find_packages
from setuptools.command.install import install
from corkit import __version__
import subprocess
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

class CustomInstall(install):
    def run(self):
        super().run()
        subprocess.run(["corkit-update", "--batch-size", "10"], check=True)

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
        entry_points = {
            "console_scripts": [
                "corkit-update=corkit.cli:main"
            ]
        },
        cmdclass={
            "install": CustomInstall
        },
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

