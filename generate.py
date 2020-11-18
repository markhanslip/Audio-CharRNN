#!/usr/bin/env python

import torch
import os
import argparse
import numpy as np

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from helpers import *
from model import *

def generate(decoder, predict_len, temperature, prime_str, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str
    #print(server.get_port()))

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()
        
    return predicted

def print_handler(address, *args):
    argss = ''.join(args)
      
    output = generate(decoder=torch.load(opt.model_file), predict_len=opt.predict_len, temperature=opt.temperature, prime_str=str(argss)+', ', cuda=True)
         
    print(output)
    floats = [float(idx) for idx in output.split(', ')] 
    print(floats)
    client = SimpleUDPClient("127.0.0.1", 1337)
    client.send_message("/RNN1", floats)
    
    
# Run as standalone script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='./models/tenor_sax_interval_exercises.pt', help='path to trained model .pt file')
    parser.add_argument('--predict_len', type=int, default=230, help='length of generated phrase after prompt')
    parser.add_argument('--temperature', type=float, default=0.99, help='value between 0 and 1')
    opt = parser.parse_args()
    
    dispatcher = Dispatcher()
    dispatcher.map("/hello from SuperCollider", print_handler)

    ip = "127.0.0.1"
    port = 2019 # need to find a way to make this communicable to SC 
    server =ThreadingOSCUDPServer((ip, port), dispatcher)
    server.serve_forever()  # Blocks forever
