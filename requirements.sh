#!/bin/bash
rm -rf requirements.txt

pipreqs corkit --savepath=requirements.in

pip-compile requirements.in 

rm -f requirements.in
