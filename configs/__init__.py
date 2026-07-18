"""
Configuration files for Voxtral training experiments.

Classification-as-Generation (CAG) configs:
- hyperparameters_05_dif_margins21.yaml / _dif_margins21bigmargin.yaml / _dif_margins21_LLM.yaml
- hyperparameters_05_no_margin.yaml / _no_margin_realfake.yaml
- hyperparameters_2019LA.yaml
- hyperparameters_2021_LA_2.yaml / hyperparameters_2021_DF_2.yaml

Audio-Adapter + MLP classification-head configs:
- hyperparameters_5_classification_head.yaml / _classification_head_margin.yaml
- hyperparameters_2019LA_classification_head.yaml
- hyperparameters_2021LA_classification_head.yaml / hyperparameters_2021DF_classification_head.yaml

Usage:
    python scripts/train.py --config configs/hyperparameters_05_no_margin.yaml
"""
