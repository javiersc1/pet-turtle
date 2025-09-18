# PET-TURTLE: Unsupervised Deep SVMs for Imbalanced Data Clusters

![illustration](https://github.com/javiersc1/pet-turtle/blob/299b4e7beda5a036f375801af95274e5beb0a0d6/illustration.png)

Foundation vision, audio, and language models enable zero-shot performance on downstream tasks via their latent representations. Recently, unsupervised learning of data group structure with deep learning methods has gained popularity. TURTLE, a state of the art deep clustering algorithm, uncovers data labeling without supervision by alternating label and hyperplane updates, maximizing the hyperplane margin, in a similar fashion to support vector machines (SVM). However, TURTLE assumes clusters are balanced; when data is imbalanced, it yields non-ideal hyperplanes that causes higher clustering error. We propose PET-TURTLE, which generalizes the cost function to handle imbalanced data distributions. Additionally, by introducing sparse logits in the labeling process, PET-TURTLE optimizes a simpler search space and improves accuracy for both balanced and imbalanced datasets. Experiments on synthetic and real data show that PET-TURTLE improves accuracy for imbalanced sources, prevents over-prediction of minority clusters, and enhances overall clustering.

## Installation

### Clone repo
```
git clone https://github.com/javiersc1/pet-turtle
```

### Create conda environment
```
conda create -n turtle python=3.12
```

### Dependencies
The code is built with Pytorch 2.7 but should work on newer versions due to minimal complexity. We use openAI's CLIP model package from [github.com/openai/CLIP](https://github.com/openai/CLIP).

```
pip install torch torchvision
pip install numpy
pip install scipy
pip install scikit-learn
pip install ftfy regex tqdm
pip install matplotlib
pip install medmnist
pip install git+https://github.com/openai/CLIP.git
```


## Quick Start

1. Precompute representations and save ground truth labels for some dataset, e.g., cifar10. Possible choices: {cifar10, food101, eurosat, caltech101, dtd, DermaMNIST, OCTMNIST, BloodMNIST, OrganAMNIST, TissueMNIST}. In our experiments, we use the largest CLIP model available which is 'clipRN50x64'. 
```
python precompute_representations.py --phis clipRN50x64 --dataset cifar10
python precompute_labels.py --dataset cifar10
```

2. Run a single trial. The option '--warm_start' allows for using last known hyperplane (in previous iteration) to be used in the next one. Remove this line to randomly initialize each time the hyperplane needs to be updated.
```
python kmeans.py --phis clipRN50x64 --dataset cifar10
python linear_probe.py --phis clipRN50x64 --dataset cifar10
python turtle_original.py --phis clipRN50x64 --warm_start --dataset cifar10
python turtle_pet.py --phis clipRN50x64 --warm_start --dataset cifar10
```
