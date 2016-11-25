
# -*- coding: utf-8 -*-
"""
@author: Carmine-Emanuele Cella, 2016
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

N_COEFF = 14
SOURCE_FILE = 'samples/cage.wav'
TARGET_FILE = 'samples/unvisage_short.wav'
FRAME_SIZE = 2948
HOP_SIZE = 1024
ST_RATIO = .9
CLUSTERING_ALGO = KMeans

if __name__ == "__main__":
    print ('[soundtypes - timbre matching]\n')
    print ('computing features...')
    [y_src, sr] = librosa.core.load(SOURCE_FILE)

    y_pad_src = np.zeros(len(y_src) + FRAME_SIZE)
    y_pad_src[1:len(y_src)+1] = y_src

    C_src = librosa.feature.mfcc(y=y_src, sr=sr, n_mfcc=N_COEFF, 
                                 n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    [y_dst, sr] = librosa.core.load(TARGET_FILE)
    
    y_pad_dst = np.zeros(len(y_dst) + FRAME_SIZE)
    y_pad_dst[1:len(y_dst)+1] = y_dst
    
    C_dst = librosa.feature.mfcc(y=y_dst, sr=sr, n_mfcc=N_COEFF, 
                                 n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

    C_scaled_dst = C_dst.T
    C_scaled_src = C_src.T
        
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
    centroids_dst = cl_algo_dst.cluster_centers_
    labels_dst = cl_algo_dst.predict(C_scaled_dst)
  
    print ('matching clusters...')
    labels_match = np.zeros(centroids_src.shape[0])
    for i in range (centroids_src.shape[0]):
        distances = []
        for j in range (centroids_dst.shape[0]):
            distances.append (np.sum (np.abs(centroids_src[i, :] - \
                centroids_dst[j, :])))

        labels_match[i] = np.argmin (distances)
        
    
    print ('generate hybridization...')
    soundtypes = {i:[] for i in range(n_clusters_dst)}
    for i in range(n_clusters_dst):
        for j in range(len(labels_dst)):
            if labels_dst[j] == i:
                soundtypes[i].append(j)
    
    n_frames = len(labels_src)
    gen_sound = np.zeros(n_frames * HOP_SIZE + FRAME_SIZE)
    for i in range(n_frames):
        x = labels_match[labels_src[i]]
        p = soundtypes[x]
        if len(p) == 0:
            atom = 0
        else:
            atom = p[np.random.randint(len(p))]

        amp = np.sum (np.abs(y_pad_src[i * HOP_SIZE : i * HOP_SIZE + \
            FRAME_SIZE]))
        chunk = y_pad_dst[atom * HOP_SIZE : atom * HOP_SIZE + FRAME_SIZE] \
            * np.hanning(FRAME_SIZE)

        norm = np.max (np.abs(chunk))
        if norm == 0:
            norm = 1
            
        chunk /= norm
        gen_sound[i * HOP_SIZE : i * HOP_SIZE + FRAME_SIZE] += (chunk * amp)

    print ('saving audio data...')
    librosa.output.write_wav('generated_sound.wav', gen_sound, sr)


    pca = PCA(2)
    C_scaled_dst = pca.fit_transform (C_scaled_dst)
    C_scaled_src = pca.fit_transform (C_scaled_src)
        
    plt.close ('all')

    plt.figure ()
    plt.plot (C_scaled_src[:, 0], C_scaled_src[:, 1], 'go')
    plt.plot (C_scaled_dst[:, 0], C_scaled_dst[:, 1], 'ro')
    plt.title('source and destination points')
    