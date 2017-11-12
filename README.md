# CycleGAN PyTorch #

### What is this repository for? 
Implementation of CycleGan model in PyTorch ([original implementation link](https://github.com/junyanz/CycleGAN)). The implementation is used to remove rain from rainy images.

### How do I get set up ?
##### Step by step:
Install:
* PyTorch and dependencies 
* Torch vision
* visdom and dominate
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install

pip install visdom
pip install dominate
```
### How do I train CycleGAN with new images ?
you may have information on how to run ```train.py``` by:
```
python train.py --help
```
you can train your own model by running (N.B.: example):
```
python train.py --dataroot ./data --name cyclegan_custom --model cycle_gan --no_dropout
```
### How do I test CycleGAN after training?
you can test the model on a given collection, in order to transform A to B or B to A (Possible only after training).
```
python test.py --dataroot ./data --name cyclegan_custom --model cycle_gan --no_dropout --phase test --results_dir ./result_folder
```
### Contents
```
└── cyclegan
    ├── data                          # data folder contaning both A and B images
         ├── testA                    # test images belonging to class A
         ├── testB                    # test images belonging to class B
         ├── trainA                   # train images belonging to class A
         └── trainB                   # train images belonging to class B
    ├── images                        # images ... 
    ├── models
        └── ...                       # cycle gan model implementation .py
    ├── options                      
        └── ...                       # options : base, train, test .py
    ├── util    
        └── ...                       # utils .py               
    ├── deepenv.yml                   # Environment (keras 2, tensorflow 1.1, etc ...)
    ├── test.py                       # to test
    ├── train.py                      # to train
    ├── README.md                     # Readme


```
### Demonstration: De-raining images 
The example below present 12 rainy images where cycleGAN has been used to de-rain.

![](https://github.com/HagopB/cyclegan_pytorch/blob/master/images/demo.png)

### Acknowledgement
Based on two implementations:
* [https://github.com/junyanz/CycleGAN](https://github.com/junyanz/CycleGAN) by Jun-Yan Zhu and Taesung Park
* [https://github.com/EliasVansteenkiste/CycleGANwithPerceptionLoss](https://github.com/EliasVansteenkiste/CycleGANwithPerceptionLoss) by Elias Vansteenkiste



