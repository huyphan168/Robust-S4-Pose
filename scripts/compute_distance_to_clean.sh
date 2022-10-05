#!/bin/bash

python reliability_scr_generator.py -s gaussian_noise
python reliability_scr_generator.py -s impulse_noise
python reliability_scr_generator.py -s temporal
python reliability_scr_generator.py -s motion_blur
python reliability_scr_generator.py -s crop
python reliability_scr_generator.py -s erase