#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:41:13 2016

@author: Carmine
"""

# -*- coding: utf-8 -*-
"""
@author: Carmine-Emanuele Cella, 2016
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from st_tools import get_segments

N_COEFF = 20
K = 3
SOURCE_FILE = 'samples/cage.wav'
TARGET_FILE = 'samples/lachenmann.wav'
FRAME_SIZE = 1024
HOP_SIZE = 1024
WIDTH_SRC = 8
WIDTH_DST = 16
FADE_MS = 10

if __name__ == "__main__":
    print ('[direct timbre matching - experimental]\n')

    print ('computing segments...')   
    [y_src, sr] = librosa.core.load(SOURCE_FILE)            
    (segments_src, onsets_src, flux_src) = get_segments (y_src, sr, FRAME_SIZE,\
        HOP_SIZE, FADE_MS, WIDTH_SRC)

    [y_dst, sr] = librosa.core.load(TARGET_FILE)
    (segments_dst, onsets_dst, flux_dst) = get_segments (y_dst, sr, FRAME_SIZE,\
        HOP_SIZE, FADE_MS, WIDTH_DST)
    
    print ('computing features...')   
    features_src = []
    for i in range (len(segments_src)):
        C_src = librosa.feature.mfcc(y=segments_src[i], sr=sr, n_mfcc=N_COEFF, 
                                 n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        features_src.append(np.mean (C_src, axis=1))                             
    C_src = np.vstack(features_src)

    features_dst = []
    for i in range (len(segments_dst)):
        C_dst = librosa.feature.mfcc(y=segments_dst[i], sr=sr, n_mfcc=N_COEFF, 
                                 n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        features_dst.append(np.mean (C_dst, axis=1))                             
    C_dst = np.vstack(features_dst)
    
    C_scaled_dst = C_dst
    C_scaled_src = C_src
        
#    scaler = StandardScaler ()    
#    C_scaled_dst = scaler.fit_transform (C_scaled_dst)
#    C_scaled_src = scaler.transform (C_scaled_src)

    print ('fitting datasets...')
    knn = NearestNeighbors(K).fit (C_scaled_dst);
    
    print ('generate hybridization...')
    
    n_frames = onsets_src.shape[0] 
    gen_sound = np.zeros (2 * len(y_src))
    for i in range(1, n_frames):
        dist, idx = knn.kneighbors(C_scaled_src[i - 1, :].reshape(1, -1))
        atom = idx[0][np.random.randint(K)]

        gen_sound[onsets_src[i - 1] * HOP_SIZE : onsets_src[i - 1] * HOP_SIZE \
                  + len (segments_dst[atom])] += (segments_dst[atom])
           
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
    