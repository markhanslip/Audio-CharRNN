#!/usr/bin/env python

import sys
import numpy as np
from numpy import inf
import parselmouth
import argparse
from pydub import AudioSegment

def concatenate(opt):

    import wave
    infiles = []

    import os
    for root, dirs, files in os.walk(opt.src_path):
        for file in files:
            if file.endswith(".wav"):
                infiles.append(os.path.join(root, file)) 
                
    data= []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()

    output = wave.open(opt.src_path+'/concat.wav', 'wb')
    output.setparams(data[0][0])
    for i in range(len(data)): 
        output.writeframes(data[i][1])
    print('concatenated audio files')

def resample(opt):

    print('resampling audio (this can take a while)')
    sound = AudioSegment.from_file(opt.src_path+'/concat.wav', format='wav')
    sound = sound.set_frame_rate(opt.sr)
    sound.export(opt.src_path+'/concat_{}.wav'.format(opt.sr), format='wav')
    #del(opt.src_path+'/concat.wav')
    print('resampled audio')

def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    return list(pitch_values)
   
def draw_intensity(intensity):
    intensity_values = intensity.values
    return intensity_values

# count zero values and replace with a single duration value:
def get_durations(seq):
    newseq=[]
    count_0 = 0
    count_1 = 0
    #parsed = list(parsed)
    for i in range(len(seq)-1):
        if seq[i] > 1:
            newseq.append(seq[i])
        if seq[i] == 1 and seq[i+1] == 1:
            count_1 += 1
        if seq[i] == 1 and seq[i+1] > 1:
            count_1 += 1
            newseq.append(count_1)
            count_1 = 0 
            newseq.append(0)
            continue
        if seq[i] == 1 and seq[i+1] == 0:
            count_1 += 1 
            newseq.append(count_1)
            count_1 = 0
            #newseq.append(seq[i+1])
        if seq[i] == 0 and seq[i+1] == 0:
            count_0 += 1 
        if seq[i] == 0 and seq[i+1] > 1:
            count_0 += 1 
            newseq.append(count_0)
            count_0 = 0
            continue
       # if seq[i] == 0 and seq[i+1] 
    return newseq

def wav2array_midinote_on_off_dB(opt):
        
        print('loading audio file...')
        src = parselmouth.Sound(opt.src_path+'/concat_{}.wav'.format(opt.sr))
        print('extracting pitches... (this can take a while)')  
        pitch = src.to_pitch_ac(time_step=0.01, pitch_floor=50.0, pitch_ceiling=1400.0)
        # get pitch list 
        pitch_vals = draw_pitch(pitch)
        pitch_vals = np.nan_to_num(pitch_vals) # replace nan vals with zeros 
        print('parsing pitch data')
        #convert freq to midi 
        midi = 12*np.log2(pitch_vals/440)+69
        midi[midi == -inf] = 0
        midi = np.around(midi, decimals=1)
        
        # work out which note values represent an onset and multiply the two vectors 
        d = np.ediff1d(midi) #or d = diff(midi) 

        onsets = (d <= -0.8) & (d >= -44) | (d >= 0.8) 
        onsets = onsets.astype(int)

        # replace consecutive onsets with 0: 
        onsets = list(onsets)
        new_onsets1=[]
        for i, n in enumerate(onsets):
                if n == 0:
                    new_onsets1.append(n)
                if n == 1 and onsets[i+1] == 0:
                    new_onsets1.append(n)
                if n == 1 and onsets[i+1] == 1:
                    new_onsets1.append(0)

        new_onsets1 = np.insert(new_onsets1, 0, 0)
        
        # work out which note values represent note-on (ie hold) 
        e = np.ediff1d(midi) #or d = diff(midi) 

        note_on = (e <= -0.8) | (e <= 0.8)  
        note_on = note_on.astype(int)
        note_on = np.insert(note_on, 0, 1)


        note_on = note_on * midi

        note_on_w = np.where(note_on > 0, 1, note_on)    
        
        parsed = new_onsets1 * midi
        parsed = parsed + note_on_w
        
        nz = np.flatnonzero(parsed)
        parsednz = parsed[nz[0]:]

        p = list(get_durations(list(parsednz)))
        p = p[:-1] # remove last val to preserve format  

        # insert 1 after each pitch val (each 1 will be multiplied by an intensity val in a bit )
        m=[]
        for i in range(len(p)-2):
            if isinstance(p[i], float):
                m.append(p[i])
                m.append(1)
                m.append(p[i+1])
                m.append(p[i+2])
                continue   
        
        #get intensity values: 
        intensity = src.to_intensity(time_step=0.01)
        print('extracting amplitudes... (this can take a while)')       
        intensity_vals = draw_intensity(intensity)
        intensity_vals = np.ndarray.tolist(intensity_vals)
        intensity_vals = intensity_vals[0]  
        print('combining with pitch data')
        #ensure same length as onsets so they can be multipled 
        if len(intensity_vals) > len(new_onsets1):
                intensity_vals = intensity_vals[:(len(intensity_vals)-(len(intensity_vals)-len(new_onsets1)))]
        if len(new_onsets1) > len(intensity_vals):
                new_onsets1 = new_onsets1[:(len(new_onsets1)-(len(new_onsets1)-len(intensity_vals)))]

        # multiply intensity by onsets and pass only intensity vals to a new array n: 
        intensity_parsed = new_onsets1 * intensity_vals
        n = []

        for i in range(len(intensity_parsed)):
            if intensity_parsed[i] > 0:
                n.append(intensity_parsed[i])
        n=n[:-1]

        # pad with 1s in new array f_i so can be multipled by pitch / note on / note off 
        f_i =[]
        for i in range(len(n)):
            if isinstance(n[i], float):
                f_i.append(n[i])
                f_i.append(int(1))
                f_i.append(int(1))
                f_i.append(int(1))
                continue 
        f_i = np.insert(f_i, 0, 1) # add an extra 1 at the beginning 
        f_i = np.around(f_i, decimals=1) # round to one decimal place 
        f_i = f_i[:-1] # clip last 1 

        # make sure f_i and m are same length, lop a bit off the end of one if not: 
        if len(f_i) > len(m):
            f_i = f_i[:(len(f_i)-(len(f_i)-len(m)))]
        if len(m) > len(f_i):
            m = m[:(len(m)-(len(m)-len(f_i)))]

        final = f_i * m
        final = np.clip(final, 10.0, 99.0) # clip vals to preserve format (lack of zero note-off vals can be addressed in SuperCollider)
        final = np.ndarray.tolist(final)

        print('all data parsed successfully')
        np.savetxt(opt.out_file, [final], fmt='%.1f', delimiter=', ')
        print('saved output array to {}'.format(opt.out_file))


if __name__=='__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--src_path', type=str)
        parser.add_argument('--sr', type=int)
        parser.add_argument('--out_file', type=str)
        opt = parser.parse_args()
        concatenate(opt)
        resample(opt)
        wav2array_midinote_on_off_dB(opt)
