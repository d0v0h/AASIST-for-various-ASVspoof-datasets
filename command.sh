#!/bin/bash
source .venv/bin/activate

python main_2019.py --config ./config/AASIST.conf

python main_2021.py --config ./config/AASIST.conf --track LA

python main_2021.py --config ./config/AASIST.conf --track DF