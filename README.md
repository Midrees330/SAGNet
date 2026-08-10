# SAGNet
Shadow-Aware Disentanglement via Mask-Guided Pathway and Uncertainty Refinement for Mask-Annotation-Free Shadow Removal
- This code is directly related to the manuscript submitted to The British Machine Vision Conference (BMVC-2026): `Shadow-Aware Disentanglement via Mask-Guided Pathway and Uncertainty Refinement for Mask-Annotation-Free Shadow Removal.' 
If you use this code or models in your research, please cite the corresponding manuscript.
# Requirement
- Python 3.11
- Pytorch 2.0
- CUDA 11.7
- MATLAB R2023b
# Datasets
- AISTD (ISTD+) [link](https://github.com/cvlab-stonybrook/SID)
- SRD [Training](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view) [Testing](https://drive.google.com/file/d/1GTi4BmQ0SJ7diDMmf-b7x2VismmXtfTo/view) [Mask](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_um_edu_mo/EZ8CiIhNADlAkA4Fhim_QzgBfDeI7qdUrt6wv2EVxZSc2w?e=wSjVQT) (detected by [DHAN](https://github.com/vinthony/ghost-free-shadow-removal))
- WSRD+ [link](https://github.com/movingforward100/Shadow_R)
# Pre-trained models
The corresponding pre-trained models:
- AISTD (ISTD+) [checkpoint](https://drive.google.com/file/d/17qBepePbQROPfgjqZL7aPu3-bJ05yjYM/view?usp=drive_link)
- SRD [checkpoint](https://drive.google.com/file/d/1r3IxYgDpnhHL-LC2QHWNSe4V2MWrsFsh/view?usp=drive_link)
- WSRD+ [checkpoint](https://drive.google.com/file/d/1MDXXNVBk_mEtjunzg2BCttFFyTGgWMRH/view?usp=drive_link)
# Test the model
You can directly test the performance of the pre-trained model as follows:
Modify the paths to dataset and pre-trained model. You need to modify the following path in the `test.py` and run
- python test.py --load [checkpoint numbers, e.g 630]
# Train
1. Download datasets and set the following structure

    ```
    -- AISTD_Dataset
       |-- train
       |   |-- train_A  # shadow image
       |   |-- train_B  # shadow mask (N/U)
       |   |-- train_C  # shadow-free GT
       |
       |-- test
           |-- test_A  # shadow image
           |-- test_B  # shadow mask (N/U)
           |-- test_C  # shadow-free GT

    -- SRD_Dataset
       |-- train
       |   |-- train_A  # shadow image
       |   |-- train_B  # shadow mask (N/U)
       |   |-- train_C  # shadow-free GT
       |
       |-- test
           |-- test_A  # shadow image
           |-- test_B  # shadow mask (N/U)
           |-- test_C  # shadow-free GT

    -- WSRD+_Dataset
       |-- train
       |   |-- train_A  # shadow image
       |   |-- train_B  # shadow-free GT
       |
       |-- test
           |-- test_A  # shadow image
           |-- test_B  # shadow-free GT
# Evaluation
The results reported in the paper are calculated by the `matlab` script used in [previouse method](https://github.com/hhqweasd/G2R-ShadowNet/blob/main/evaluate.m)
# Visual results
The Visual results on dataset  AISTD (ISTD+), SRD, and WSRD+ are:
- AISTD (ISTD+) [Results](https://drive.google.com/file/d/1bGlkhT2pj1i2gE-6QKyt3zMs4ncF0qF8/view?usp=drive_link)
- SRD [Results](https://drive.google.com/file/d/1oRKT8PBlDeQFNaAr0AjogGZqDNuNylNk/view?usp=drive_link)
- WSRD+ [Results](https://drive.google.com/file/d/16dViYByhEjYr_Ki3UXcc_0DyJL7Eeh8h/view?usp=drive_link)

# Contact
If you have any questions, please contact idreeskhan045@gmail.com/ huangying@cqupt.edu.cn
