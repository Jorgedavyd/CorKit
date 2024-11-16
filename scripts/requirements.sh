#!/bin/bash
python3 -m pip install --upgrade pip
pip3 install pytest black pipreqs pip-tools
VERSION="0.0.$(date +%s)"
sed -i "s/{{VERSION_PLACEHOLDER}}/$VERSION/" corkit/__init__.py
pip install .
corkit-update --batch-size 1
