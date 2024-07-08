#!/bin/bash

pipreqs corkit --savepath=requirements.in

pip-compile requirements.in 

rm -f requirements.in
