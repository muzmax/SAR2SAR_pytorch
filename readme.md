# SAR2SAR

This projet is an implementation in pytorch of SAR2SAR, a denoised deep neural network for SAR images denoising.
The network was trained on TerraSAR-X images

Authors and citation for the paper :
@article{dalsasso2021sar2sar,
  title={SAR2SAR: A semi-supervised despeckling algorithm for SAR images},
  author={Dalsasso, Emanuele and Denis, Loic and Tupin, Florence},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={14},
  pages={4321--4329},
  year={2021},
  publisher={IEEE}
}

To run this code you must have :
- Pytorch
- numpy
- opencv
- matplotlib

## Predictions 

To make prediction with this network you have to :
 - put .npy images in 'data/real'
 - in a cmd type 'python predict.py'
 - results will be stored in 'data/results/real'

 If you want to add artificial speckle on your images you have to set add_speck to True line 19 of predict.py

## Training

To train the network you have to :
 - put phase A .npy train images in data/train_A
 - put phase A .npy eval images in data/eval
 - put phase B & C train images in data/train_BC. Each image pile must be in a different folder, if you have only one pile create a single folder
 - put phase B & C eval images in data/eval_real
 - in a cmd type 'python train_ABC.py'
 - network weights will be stored in 'pipeline/out/unsupervised'
 - in data/sample you will have a denoised version of eval images for each epoch

