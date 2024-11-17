#!/bin/bash
python3 -m pip install --upgrade pip
pip3 install pytest black
pip install --no-build-isolation -e .
python3 update.py
