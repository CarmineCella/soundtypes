
# -*- coding: utf-8 -*-
"""
@author: Carmine-Emanuele Cella, 2016
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

N_COEFF = 14
SOURCE_FILE = 'samples/kapustin.wav'
TARGET_FILE = 'samples/Beethoven_Symph7_short.wav'
FRAME_SIZE = 4096 
HOP_SIZE = 1024
ST_RATIO = .5
CLUSTERING_ALGO = MiniBatchKMeans

if __name__ == "__main__":
    print ('[soundtypes - timbre matching]\n')
    print ('computing features...')
    [y_src, sr] = librosa.core.load(SOURCE_FILE)

    y_pad_src = np.zeros(len(y_src) + FRAME_SIZE)
    y_pad_src[1:len(y_src)+1] = y_src

    C_src = librosa.feature.mfcc(y=y_src, sr=sr, n_mfcc=N_COEFF, n_fft=FRAME_SIZE, 
                             hop_length=HOP_SIZE)
    [y_dst, sr] = librosa.core.load(TARGET_FILE)
    
    y_pad_dst = np.zeros(len(y_dst) + FRAME_SIZE)
    y_pad_dst[1:len(y_dst)+1] = y_dst
    
    C_dst = librosa.feature.mfcc(y=y_dst, sr=sr, n_mfcc=N_COEFF, n_fft=FRAME_SIZE, 
                             hop_length=HOP_SIZE)
   
    print ('computing multidimensional scaling...')
    mds = MDS(2)
    C_scaled_src = mds.fit_transform (C_src.T)
    C_scaled_dst = mds.fit_transform (C_dst.T)

    scaler = StandardScaler ()
    C_scaled_dst = scaler.fit_transform (C_scaled_dst)
    C_scaled_src = scaler.transform (C_scaled_src)    
    
    print ('computing clusters...')
    n_clusters_src = int(C_scaled_src.shape[0] * ST_RATIO)
    cl_algo_src = CLUSTERING_ALGO (n_clusters_src).fit (C_scaled_src)
    centroids_src = cl_algo_src.cluster_centers_
    labels_src = cl_algo_src.predict(C_scaled_src)
    n_clusters_dst = int(C_scaled_dst.shape[0] * ST_RATIO)
    cl_algo_dst = CLUSTERING_ALGO (n_clusters_dst).fit (C_scaled_dst)
    labels_dst = cl_algo_dst.predict(C_scaled_dst)
    # matching clusters
    labels_match = cl_algo_dst.predict(centroids_src)
    
    print ('generate hybridization...')
    soundtypes = {i:[] for i in range(n_clusters_dst)}
    for i in range(n_clusters_dst):
        for j in range(len(labels_dst)):
            if labels_dst[j] == i:
                soundtypes[i].append(j)
    
    n_frames = len(labels_match)
    gen_sound = np.zeros(n_frames * HOP_SIZE + FRAME_SIZE)
    for i in range(n_frames):
        p = soundtypes[labels_match[i]]
        atom = p[np.random.randint(len(p))]

        amp = np.sqrt(np.sum(
            np.abs(y_pad_src[atom*HOP_SIZE:atom*HOP_SIZE+FRAME_SIZE]**2)))
        
        chunk = y_pad_dst[atom*HOP_SIZE:atom*HOP_SIZE+FRAME_SIZE] \
            * np.hanning(FRAME_SIZE)
        gen_sound[i*HOP_SIZE:i*HOP_SIZE+FRAME_SIZE] += chunk * amp

    print ('saving audio data...')
    librosa.output.write_wav('generated_sound.wav', gen_sound, sr)

    plt.close ('all')

    plt.figure ()
    plt.plot (C_scaled_dst[:, 0], C_scaled_dst[:, 1], 'go')
    plt.plot (C_scaled_src[:, 0], C_scaled_src[:, 1], 'ro')
    plt.title('destination and source points')



