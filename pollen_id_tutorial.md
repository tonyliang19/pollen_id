# Pollen ID project tutorial
# Author: Tony Liang
# Date: Aug 2022


## Table of Content
This is a place holder for table of content

### Setup
There are various dependencies of this project to python packages. Hence, recommended to use virtual environments either from Python venv or conda (the first one is recommended). 

The list of important packages required for the project is the following:
    * Placeholder
    * Placeholder
    * Placeholder

For the rest of dependencies please check at [the requirements.txt file][#requirements.txt]

Specifically, the most important one is [Detectron2](https://detectron2.readthedocs.io/en/latest/index.html), the base library that does the machine learning part and allow us to extend their base modules of engines of trainer and predictor, and dataloader as well.  Since computer vision depends a lot on gpus, make sure your device has CUDA enabled, whereas you need to refer to NVIDIA official documentation.

Note: Installing Detectron2 might be painful and loop over and over due to some compiling error, but at least the following workflow worked out for me:

1) Make sure you have CUDA globally installed in your device, to check it, prompt in the following command in your terminal:
    `nvcc -V or nvcc --version`
And result in the following output:
    ```
    $ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2021 NVIDIA Corporation
    Built on Fri_Dec_17_18:28:54_Pacific_Standard_Time_2021
    Cuda compilation tools, release 11.6, V11.6.55
    Build cuda_11.6.r11.6/compiler.30794723_0
    ```

2) Follow [the official documentation of Detectron2 installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html]), install PyTorch and torchvision that matches PyTorch installation, which redirects you to [PyTorch official website](https://pytorch.org/)

And you will have a table of selections that generates the command to install pytorch and torchvision accordingly. Eg. I used the following options:

```
PyTorch Build: Stable(1.12.1)
Your OS: Windows
Package: Pip
Language: Python
Compute Platform: CUDA 11.6

------
This was generated from Pytorch:

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

```

3) After successfully completing [step 2](#2)), install these packages prior to build and install detectron2 from source:

```
pip install cython ninja opencv-python

```

The first two are required, as cython makes writing C extensions (most of detectron2 are build from and depend on) as easy as Python itself. And, ninja that builds detectron2 faster (helps a lot, as the package is huge). 

