# Derived from or adapted from https://github.com/TakHemlata/RawBoost-antispoofing
# Copyright (c) 2021 Hemlata. Original code licensed under the MIT License.
# Modified for Spooftral; full original license text: ../THIRD_PARTY_NOTICES.md.
"""
utils/data_utils_rawboost.py
============================
High-level RawBoost augmentation dispatcher.

Wraps the low-level RawBoost primitives (RawBoost.py) with a single entry
point — process_Rawboost_feature() — that selects the augmentation algorithm
by index:
  algo 1 : LnL convolutive noise
  algo 2 : ISD impulsive additive noise
  algo 3 : SSI stationary additive noise
  algo 4 : combination of algorithms 1+2
  algo 5 : combination of algorithms 1+3
  algo 6 : combination of algorithms 1+2+3

Authors: Hemlata Tak, Massimiliano Todisco ({tak,todisco}@eurecom.fr).
"""
try:
    # Prefer relative import to ensure the bundled RawBoost implementation is found
    from .RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav
except ImportError:
    try:
        from utils.RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav  # type: ignore
    except ImportError:  # pragma: no cover - last resort for legacy paths
        from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav

___author__ = "Hemlata Tak, Massimiliano Todisco"
__email__ = "{tak,todisco}@eurecom.fr"


#--------------RawBoost data augmentation algorithms---------------------------##

# args is expected to expose the RawBoost hyperparameters (N_f, nBands, minF, maxF,
# ... SNRmin, SNRmax, etc.) as attributes, matching the original RawBoost paper's config.
def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature


