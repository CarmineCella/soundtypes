
# -*- coding: utf-8 -*-
"""
@author: Carmine-Emanuele Cella, 2016
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial import minkowski_distance

N_COEFF = 14
SOURCE_FILE = 'kapustin.wav'
TARGET_FILE = 'Beethoven_Symph7_short.wav'
FRAME_SIZE = 4096 
HOP_SIZE = 1024


if __name__ == "__main__":
    print ('[soundtypes - probabilistic generation]\n')
    print ('computing features...')
    [y_src, sr] = librosa.core.load(SOURCE_FILE)
    C_src = librosa.feature.mfcc(y=y_src, sr=sr, n_mfcc=N_COEFF, n_fft=FRAME_SIZE, 
                             hop_length=HOP_SIZE)
    [y_dst, sr] = librosa.core.load(TARGET_FILE)
    C_dst = librosa.feature.mfcc(y=y_dst, sr=sr, n_mfcc=N_COEFF, n_fft=FRAME_SIZE, 
                             hop_length=HOP_SIZE)
   
    print ('computing multidimensional scaling...')
    mds = MDS(2)
    C_src_scaled = mds.fit_transform (C_src.T)
    C_dst_scaled = mds.fit_transform (C_dst.T)

    print ('computing distances...')
    dist = minkowski_distance (C_src, C_dst) 
#    
#    plt.close ('all')
#
#    plt.figure ()
#    plt.plot (C_scaled[:, 0], C_scaled[:, 1], 'go')
#    plt.plot (centroids[:, 0], centroids[:, 1], 'ro')
#    plt.title('points and centroids')
#
#



