import numpy as np
import pandas as pd
import csv
import os
import librosa
import scipy.io.wavfile as wavf

from segmentation import segment_cough
    
dataset_folder = "public_dataset"
prep_data = "prep_dataset"
seg_data = "seg_dataset"
i = 1;

def seg_file(files):
    filename = prep_data + '/' + files
    x, fs = librosa.load(filename, mono=True)
    cough_segments, cough_mask = segment_cough(x,fs,cough_padding=0)
    cough_seg=[]
    
    for i in range(0,len(cough_segments)):
        cough_seg.extend(cough_segments[i]) #concatenate segments
        i=i+1
    wav_seg = np.asarray(cough_seg)
    wavf.write(filename='seg_dataset/' + files, rate=fs, data=wav_seg) 

cough_csv = pd.read_csv('cough_classify/cough_true.csv')
cough_files = cough_csv['filename'].tolist()

for files in cough_files:
    print(i)
    i += 1
    seg_file(files)
    

df = pd.read_csv(dataset_folder + '/metadata_compiled.csv')

data='covid_dataset.csv'

#Create the header for the CSV File 
header = 'filename'
for x in range(1, 21):
    header += f" mfcc{x}"
header += ' label'
header = header.split()

#create and write to file
file_dataset = open(data, 'w', newline="")
with file_dataset: 
    writer = csv.writer(file_dataset)
    writer.writerow(header)

def parse_file(files):
    size = len(files)
    label = 'error'
    mod_files = files[:size - 4]
    test = df.status[df['uuid'] == mod_files].to_string(index=False).strip()
    
    if test in ['healthy', 'healthy ', ' healthy', ' healthy ']:
        label = 'negative'
    elif test in ['COVID-19', ' COVID-19', 'COVID-19 ', ' COVID-19 ', 'symptomatic', 'symptomatic ', ' symptomatic', ' symptomatic ']:
        label = 'positive'
    else:
        label = 'unknown'
    
    filename = seg_data + '/' + files
    x, sr = librosa.load(filename, mono=True)
    l = librosa.get_duration(y=x, sr=sr)
    if(l != 0.0):
        mfcc = librosa.feature.mfcc(y=x, sr=sr)
    
        to_append = f'{filename}'    
        for k in mfcc:
            to_append += f' {np.mean(k)}'
        to_append += f' {label}'
    
        file = open(data, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
    else:
        print(l, 'skipped' + f' {filename}')
        
i = 1
seg_list=os.listdir(seg_data)

for files in seg_list:
    print(i)
    i += 1
    parse_file(files)
