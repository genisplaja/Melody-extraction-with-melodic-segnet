import os
import csv
import mir_eval
import random
import pickle
import numpy as np
import pandas as pd

def get_split_lists():
    train_ids = []
    val_ids = []
    test_ids = []
    return train_ids, val_ids, test_ids

def get_split_lists_vocal(data_folder):
    ids = []
    for mix_path in os.listdir(os.path.join(data_folder, 'audio')):
        if '.wav' in mix_path:
            ids.append(mix_path.split('_')[-1].replace('.wav', ''))
    
    train_ids = random.sample(ids, int(len(ids)*0.65))
    val_ids = random.sample(ids, int(len(ids)*0.20))
    test_ids = random.sample(ids, int(len(ids)*0.15))
    return train_ids, val_ids, test_ids

def melody_eval(ref, est):

    ref_time = ref[:, 0]
    ref_freq = ref[:, 1]

    est_time = est[:, 0]
    est_freq = est[:, 1]

    output_eval = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)
    VR = output_eval['Voicing Recall']*100.0 
    VFA = output_eval['Voicing False Alarm']*100.0
    RPA = output_eval['Raw Pitch Accuracy']*100.0
    RCA = output_eval['Raw Chroma Accuracy']*100.0
    OA = output_eval['Overall Accuracy']*100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, OA])
    return eval_arr

def csv2ref(ypath):
    ycsv = pd.read_csv(ypath, names=["time", "freq"])
    gtt = ycsv['time'].values
    gtf = ycsv['freq'].values
    ref_arr = np.concatenate((gtt[:, None], gtf[:, None]), axis=1)
    return ref_arr

def select_vocal_track(ypath, lpath):
    ycsv = pd.read_csv(ypath, names=["time", "freq"])
    gt0 = ycsv['time'].values
    gt0 = gt0[:, np.newaxis]

    gt1 = ycsv['freq'].values
    gt1 = gt1[:, np.newaxis]

    z = np.zeros(gt1.shape)

    f = open(lpath, 'r')
    lines = f.readlines()

    for line in lines:

        if 'start_time' in line.split(',')[0]:
            continue
        st = float(line.split(',')[0])
        et = float(line.split(',')[1])
        sid = line.split(',')[2]
        for i in range(len(gt1)):
            if st < gt0[i, 0] < et and 'singer' in sid:
                z[i, 0] = gt1[i, 0]

    gt = np.concatenate((gt0, z), axis=1)
    return gt

def save_csv(data, savepath):
    with open(savepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['VR', 'VFA', 'RPA', 'RCA', 'OA'])
        for est_arr in data:
            writer.writerow(est_arr)
            
def load_list(savepath):
    with open(savepath, 'rb') as file:
        xlist = pickle.load(file)
    return xlist

