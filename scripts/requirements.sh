#!/bin/bash
python3 -m pip install --upgrade pip
pip3 install pytest black pipreqs pip-tools
rm -f requirements.txt
pipreqs corkit --savepath=./requirements.in
pip-compile ./requirements.in
rm -f ./requirements.in
pip install -r requirements.txt
python3 update.py