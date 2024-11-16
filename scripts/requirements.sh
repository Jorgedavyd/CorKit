#!/bin/bash
python3 -m pip install --upgrade pip
pip3 install pytest black
VERSION="0.0.$(date +%s)"
sed -i "s/{{VERSION_PLACEHOLDER}}/$VERSION/" corkit/__init__.py
pip install --no-build-isolation -e .
python3 update.py
