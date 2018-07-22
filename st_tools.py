#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:56:49 2016

@author: Carmine
"""

import numpy as np
from sklearn.cluster import KMeans
import librosa

CLUSTERING_ALGO = KMeans

def spectral_flux (M):
    flux = []
    for i in range (M.shape[1] - 1):
        flux.append (np.sum(np.abs (M[:, i] -  M[:, i + 1])))                                                

    if (np.max(flux)):
        flux /= np.max(flux)                                   
    return np.array(flux)
    
def find_peaks (data, width):
    peaks = [0]
    prev  = 0
    delay = ((width) // 2) + 1
    data = np.convolve(data, np.hanning (width))[:-width]    
    for i in range (1, len(data)-1):
        if data[i - 1] < data [i] and data [i + 1] < data[i]:
            pos = max (0, i - delay)
            if (pos != prev):
                peaks.append(pos)
                prev = pos
            
    return np.array(peaks), data[delay:]
    
def fade_segment (segment, ms, sr):
    samples = int ((ms / 1000) * sr)
    if (samples * 2 >= len(segment)):
        return segment
    ramp_up = np.linspace (0, 1, samples)
    ramp_dw = np.linspace (1, 0, samples)
    segment[0:samples] *= ramp_up
    segment[-samples:] *= ramp_dw
    return segment
    
def get_segments (y, sr, frame_size, hop_size, fade_ms, width):
    M = np.abs (librosa.spectrum.stft(y=y, n_fft=frame_size, 
                                        hop_length=hop_size))
    
    flux = spectral_flux (M)
    onsets, smoothed_flux = find_peaks (flux, width)                              

    segments = []
    for i in range (1, len(onsets)):
        chunk = y[onsets[i - 1] * hop_size : onsets[i] * hop_size]
        chunk = fade_segment (chunk, fade_ms, sr)
        segments.append (chunk)

    return (segments, onsets, smoothed_flux)
    
def make_soundtypes (C, ratio):
    n_clusters = int(C.shape[0] * ratio)
    cl_algo = CLUSTERING_ALGO (n_clusters).fit (C)
    centroids = cl_algo.cluster_centers_
    labels = cl_algo.predict(C)

    markov = {i:[] for i in labels}

    pos = 0
    while pos < len(labels) - 1:
        if labels[pos+1] != labels[pos]:
            markov[labels[pos]].append(labels[pos+1])
        pos += 1
    dictionary = {i:[] for i in range(n_clusters)}
    for i in range(n_clusters):
        for j in range(len(labels)):
            if labels[j] == i:
                dictionary[i].append(j)
                
    return (dictionary, markov, centroids, labels)

    