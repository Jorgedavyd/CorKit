#!/bin/bash
python3 -m pip install --upgrade pip
pip3 install pytest black pipreqs pip-tools
rm -f requirements.txt
pipreqs corkit --savepath=./requirements.in
sed -i 's/==.*$//' requirements.in
sed -i 's/skimage/scikit-image/' ./requirements.in
sort requirements.in | uniq > requirements_unique.in
pip-compile ./requirements_unique.in
rm -f *.in
pip install -r requirements.txt
python3 update.py
