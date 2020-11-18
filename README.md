# Audio-CharRNN

# Character-level RNN adapted for use with audio

Model implementation based on https://github.com/spro/char-rnn.pytorch

# Installation:

It's recommended to have Anaconda installed - https://docs.anaconda.com/anaconda/install/

To install dependencies, create an Anaconda environment from the root of this repository:
```
conda create -n audio-rnn python=3.7 pytorch==1.4.0 cudatoolkit=10.1 -c pytorch 
pip install -r requirements.txt
```
The above works for Windows and Linux, Mac users should omit `cudatoolkit=10.1` from the first line 

SuperCollider can be a pain to install, look here: https://supercollider.github.io/download

# Data preprocessing:

`audio_data_prep.py` takes a folder of .wav files (`--src_path`), resamples (`--sr`), analyses their content and returns a .txt file (`--out_file`) to be used for training the model.

# Training

Run `train.py` on the .txt file outputted by `audio_data_prep.py` or use the one provided in the data folder to train and save the network:

After training the model will be saved as `[filename].pt`.

# Training options

```
Usage: train.py [filename] [options]

Options:
--model            Whether to use LSTM or GRU units        gru
--n_epochs         Number of epochs to train               2000
--print_every      Print learning progress every n epochs  100
--hidden_size      Hidden size of GRU                      100
--n_layers         Number of GRU layers                    3
--learning_rate    Learning rate                           0.01
--chunk_len        Length of training chunks               240
--batch_size       Number of examples per batch            10
--cuda             Use CUDA
```

# Generation

`generate.py` has been adapted for OSC communication - for now it takes `--model_file` (path to trained model), `--temperature` and `--predict_len` (length of generated sequence).

For now, generation depends on having cuda device available - I would like to add cpu option.

Included are two example SuperCollider patches for OSC communication with a the trained model. 

In `CharRNN_prompter_responder_synth.scd`, the model is prompted automatically, and the model outputs are mapped to a synth.

If you already have SuperCollider installed, you can run this patch and a provided model together by running `sh RNN_SC_parallel.sh` from inside this repository (might not work on Windows)

In `CharRNN_prompter_responder_samples.scd`, the model is prompted by a live input, and the model outputs are mapped to a bank of samples (you need to provide your own)

