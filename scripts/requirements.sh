#!/bin/bash
python3 -m pip install --upgrade pip
pip3 install pytest black pipreqs pip-tools
pip install .
corkit-update --batch-size 1
