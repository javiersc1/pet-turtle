# PET-TURTLE: Unsupervised Deep SVMs for Imbalanced Data Clusters

## Quick Start

1. Precompute representations and save ground truth labels for the dataset
```
python precompute_representations.py --phis clipRN50x64 --dataset cifar10
python precompute_labels.py --dataset cifar10
```

2. Experiment with turtle and pet-turtle:
```
python kmeans.py --phis clipRN50x64 --dataset cifar10
python linear_probe.py --phis clipRN50x64 --dataset cifar10
python turtle_original.py --phis clipRN50x64 --warm_start --dataset cifar10
python turtle_pet.py --phis clipRN50x64 --warm_start --dataset cifar10
```
