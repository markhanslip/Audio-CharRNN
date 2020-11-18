#!/bin/bash

cd && . ./anaconda3/etc/profile.d/conda.sh && conda activate PyTorch && cd ./Audio-CharRNN \
&& python ./generate.py --model_file ./models/tenor_sax_interval_exercises.pt \
--predict_len 230 --temperature 0.99
