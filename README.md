# Repro_ICLR18 : reproduction of Dynamically Expandable Networks

This project aim is to reproduce [the DEN model](https://arxiv.org/abs/1708.01547) submitted at ICLR18.
It tries to reconstruct the perfomance plots presented in the original article on a variation of the MNIST dataset. 

## To run code:

1. unzip [data archive](http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip)
2. run main.py using python 3.5

This produces a `figures/` directory and saves three plots :
- Online AUROC per task.
- Offline AUROC per task.
- Average AUROC after each task with timestamped inference (evaluation used in the original paper).

## Poster :
The `poster/` directory contains the pdf and the original latex files of the poster that illustrated this work.