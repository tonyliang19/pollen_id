# Pollen ID project tutorial
# Author: Tony Liang
# Date: Aug 2022


## Table of Content
This is a place holder for table of content

### Setup
There are various dependencies of this project to python packages. Hence, recommended to use virtual environments either from Python venv or conda (the first one is recommended). 

The list of packages required for the project is the following:
    - Placeholder
    - Placeholder
    - Placeholder

Specifically, the most important one is [Detectron2](https://detectron2.readthedocs.io/en/latest/index.html), the base library that does the machine learning part and allow us to extend their base modules of engines of trainer and predictor, and dataloader as well. 

Since computer vision depends a lot on gpus, make sure your device has CUDA enabled, whereas you need to refer to NVIDIA official documentation.

Note: Installing Detectron2 might be painful and loop over and over due to some compiling error, but at least the following workflow worked out for me:

1) Make sure you have CUDA globally installed in your device, to check it, prompt in the following command in your terminal:
    `nvcc -V or nvcc --version`
And result in the following output:
    ```$ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Fri_Dec_17_18:28:54_Pacific_Standard_Time_2021
Cuda compilation tools, release 11.6, V11.6.55
Build cuda_11.6.r11.6/compiler.30794723_0```

2) Follow [the official documentation of Detectron2 installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html]), install 