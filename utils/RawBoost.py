#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/RawBoost.py
=================
Low-level RawBoost signal-processing primitives for audio augmentation.

Implements three noise injection algorithms used to augment spoofed speech:
  LnL_convolutive_noise  — Linear-and-nonlinear convolutive noise (algo 1).
  ISD_additive_noise     — Impulsive signal-dependent additive noise (algo 2).
  SSI_additive_noise     — Stationary signal-independent additive noise (algo 3).
  genNotchCoeffs         — Random notch filter bank used by convolutive noise.
  normWav                — Peak-normalize a waveform; avoids divide-by-zero on silence.

These functions are called by data_utils_rawboost.process_Rawboost_feature(),
which dispatches to the correct algorithm via the algo parameter.  The augmentation
is stochastic: all random decisions are governed by numpy.random, so the data
loader must seed numpy workers independently for reproducibility.

Source: Tak et al., "RawBoost: A Raw Data Boosting and Augmentation Method
applied to Automatic Speaker Verification Anti-Spoofing", ICASSP 2022.
"""
import numpy as np
from scipy import signal
import copy


# Draws a single random value uniformly from [x1, x2]; integer=True rounds it
# to a discrete value (e.g. a filter order).
def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y

def normWav(x,always):
    max_val = np.amax(abs(x))
    if max_val == 0:
        return x  # Avoid division by zero for silent/zero signals
    if always:
        x = x / max_val
    elif max_val > 1:
        x = x / max_val
    return x



# Builds a random FIR filter by cascading nBands randomly-parameterized band
# filters (random center frequency, bandwidth, order), then scales to a random gain.
def genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF,maxF,0);
        bw = randRange(minBW,maxBW,0);
        c = randRange(minCoeff,maxCoeff,1);
          
        # firwin requires an odd filter length with this window type; bump even orders up by one.
        if c/2 == int(c/2):
            c = c + 1
        f1 = fc - bw/2
        f2 = fc + bw/2
        if f1 <= 0:
            f1 = 1/1000
        if f2 >= fs/2:
            f2 =  fs/2-1/1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),b)

    G = randRange(minG,maxG,0);
    _, h = signal.freqz(b, 1, fs=fs)
    max_h = np.amax(abs(h))
    if max_h == 0:
        return b  # Avoid division by zero
    # Converts the target gain G from dB to a linear factor and rescales the filter so
    # its peak frequency response equals that gain.
    b = pow(10, G/20)*b/max_h
    return b


def filterFIR(x,b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N/2):int(y.shape[0]-N/2)]
    return y

# Linear and non-linear convolutive noise
def LnL_convolutive_noise(x,N_f,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,minBiasLinNonLin,maxBiasLinNonLin,fs):
    y = [0] * x.shape[0]
    # Sums N_f filtered harmonics of x: the first tap models the linear path, later
    # taps model nonlinear distortion using a gain range shifted by the bias terms.
    for i in range(0, N_f):
        if i == 1:
            minG = minG-minBiasLinNonLin;
            maxG = maxG-maxBiasLinNonLin;
        b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
        y = y + filterFIR(np.power(x, (i+1)),  b)     
    y = y - np.mean(y)
    y = normWav(y,0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)
    
    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len*(beta/100))
    p = np.random.permutation(x_len)[:n]
    f_r= np.multiply(((2*np.random.rand(p.shape[0]))-1),((2*np.random.rand(p.shape[0]))-1))
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y,0)
    return y


# Stationary signal independent noise

def SSI_additive_noise(x,SNRmin,SNRmax,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise,1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise_norm = np.linalg.norm(noise, 2)
    if noise_norm == 0:
        return x  # Avoid division by zero if noise is all zeros
    noise = noise / noise_norm * np.linalg.norm(x, 2) / 10.0**(0.05 * SNR)
    x = x + noise
    return x





