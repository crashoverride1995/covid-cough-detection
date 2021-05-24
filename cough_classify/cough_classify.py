import os
import sys
from featureclass import features
from DSP import classify_cough
from scipy.io import wavfile
import pickle
import csv

location = 'C:/Users/Hitesh/OneDrive/MS CS/Mobile Sensors and Computing/Project/MLClassification/';

data_folder = location + 'prep_dataset'
loaded_model = pickle.load(open(os.path.join(location + 'cough_classify/models', 'cough_classifier'), 'rb'))
loaded_scaler = pickle.load(open(os.path.join(location + 'cough_classify/models','cough_classification_scaler'), 'rb'))

filename = '00a2faca-e1f2-4848-9afe-058f949d3252.wav'
fs, x = wavfile.read(data_folder+'/'+filename)
probability = classify_cough(x, fs, loaded_model, loaded_scaler)
print("The file {0} has a {1}\% probability of being a cough".format(filename,round(probability*100,2)))

entries=os.listdir(data_folder)
print(entries)

temp = 1;
no=1;

filename="cough_true.csv"
header = 'filename'
header = header.split()

file_dataset = open(filename, 'w', newline="")
with file_dataset: 
    writer = csv.writer(file_dataset)
    writer.writerow(header)
  
for entry in entries:
    fs, x = wavfile.read(data_folder+'/'+ entry)
    probability = classify_cough(x, fs, loaded_model, loaded_scaler)
    #print("The file {0} has a {1}\% probability of being a cough".format(entry,round(probability*100,2)))
    
    if probability>=0.6:
        with open(filename,'a', newline='\n') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(entry.split())
            csvfile.close()
        no=no+1;
    print(temp, no)
    temp = temp + 1;