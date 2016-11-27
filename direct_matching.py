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

N_COEFF = 20
K = 5
SOURCE_FILE = 'samples/cage.wav'
TARGET_FILE = 'samples/ciaccona.wav'
FRAME_SIZE = 2048
HOP_SIZE = 1024

if __name__ == "__main__":
    print ('[direct timbre matching]\n')
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
    C_scaled_src = scaler.fit_transform (C_scaled_src)

    print ('fitting datasets...')
    knn = NearestNeighbors(K).fit (C_scaled_dst);
    
    print ('generate hybridization...')
    
    n_frames = C_scaled_src.shape[0]
    gen_sound = np.zeros(n_frames * HOP_SIZE + FRAME_SIZE)
    for i in range(n_frames):
        dist, idx = knn.kneighbors(C_scaled_src[i, :].reshape(1, -1))
        atom = idx[0][np.random.randint(K)]
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
    