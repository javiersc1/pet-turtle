# PET-TURTLE: Unsupervised Deep SVMs for Imbalanced Data Clusters

![illustration](https://github.com/javiersc1/pet-turtle/blob/299b4e7beda5a036f375801af95274e5beb0a0d6/illustration.png)

Foundation vision, audio, and language models enable zero-shot performance on downstream tasks via their latent representations. Recently, unsupervised learning of data group structure with deep learning methods has gained popularity. TURTLE, a state of the art deep clustering algorithm, uncovers data labeling without supervision by alternating label and hyperplane updates, maximizing the hyperplane margin, in a similar fashion to support vector machines (SVM). However, TURTLE assumes clusters are balanced; when data is imbalanced, it yields non-ideal hyperplanes that causes higher clustering error. We propose PET-TURTLE, which generalizes the cost function to handle imbalanced data distributions. Additionally, by introducing sparse logits in the labeling process, PET-TURTLE optimizes a simpler search space and improves accuracy for both balanced and imbalanced datasets. Experiments on synthetic and real data show that PET-TURTLE improves accuracy for imbalanced sources, prevents over-prediction of minority clusters, and enhances overall clustering.

## Method
Our method takes latent features $\mathcal{Z}$ from some foundational model $\phi$ with dataset $\mathcal{D}$, and finds a classifier $\tau$ and hyperplane $w$ such that the soft margin is maximixed via the cross entropy loss $\mathcal{L}_{\text{CE}}$. The method alternates between updating the classifier $\tau$ in the outer level and updating the hyperplane $w$ in the inner level using some optimization scheme $\Xi$ such as stochastic gradient descent. This bilevel problem requires regularization to prevent the trivial solution where the entire data belongs to one cluster which would technically minimize the cost function. We take the KL divergence between the emperical distribution of our data from the classifier given as $\bar{\tau}$ and encourage a power law distribution prior with $\Pi(\alpha)$ where $\alpha$ is the level of data imbalance. Note that $\alpha=0$ would return a uniform distribution. One can find ideal $\gamma,\alpha$ parameters by cross validation to find the right regularization that achieves the highest margin in the hyperplane problem, i.e., the lowest cost function value.

```math
\mathcal{L}_{\text{TURTLE-SSP}}(\theta) = \sum_{z \in \mathcal{\phi(D)}} \mathcal{L}_{\text{CE}} (w_{\theta}^M z ; \text{sparsemax}( \tau_{\theta} (z))) \quad \text{s.t.} \quad w_{\theta}^M = \Xi^{(M)} (w_{\theta}^M, \phi(\mathcal{D}))
```
```math
\min_{\theta} \mathcal{L}_{\text{TURTLE-SSP}}(\theta) + \gamma D_{\text{KL}}[s(\bar{\tau_{\theta}}) \| \Pi(\alpha) ]
```

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
The code is built with Pytorch 2.7 but should work on newer versions due to minimal complexity. We use openAI's CLIP model package from [github.com/openai/CLIP](https://github.com/openai/CLIP) and the MedMNIST package for easy dataloading of medical datasets from [github.com/MedMNIST/MedMNIST](https://github.com/MedMNIST/MedMNIST).

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

2. Run a single trial. The option '--warm_start' allows for using last known hyperplane (in previous iteration) to be used in the next one. Remove this line to randomly initialize each time the hyperplane needs to be updated. We recommend leaving this option on.
```
python kmeans.py --phis clipRN50x64 --dataset cifar10
python linear_probe.py --phis clipRN50x64 --dataset cifar10
python turtle_original.py --phis clipRN50x64 --warm_start --dataset cifar10
python turtle_pet.py --phis clipRN50x64 --warm_start --dataset cifar10
```
