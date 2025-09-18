import numpy as np
import sys

rootDir = "/home/javier/Desktop/turtle/data/"
trainFeatures = np.load(f"{rootDir}representations/clipRN50x64/cifar10_train.npy")
valFeatures = np.load(f"{rootDir}representations/clipRN50x64/cifar10_val.npy")
trainLabels = np.load(f"{rootDir}labels/cifar10_train.npy")
valLabels = np.load(f"{rootDir}labels/cifar10_val.npy")

def power_law_distribution(size: int, exponent: float):
    """Returns a power law distribution summing up to 1."""
    k = np.arange(1, size + 1)
    power_dist = k ** (-exponent)
    power_dist = power_dist / power_dist.sum()
    return power_dist

# each class has 5k images
alpha = float(sys.argv[1]) # feed value as first argument
trueDist = power_law_distribution(size=10, exponent=alpha)
print("Training distribution with alpha = ", alpha)
trainDist = (5000/trueDist[0]) * trueDist
trainDist = trainDist.astype(int)
trainFeaturesLT = []
trainLabelsLT = []
for i in range(10):
    data = trainFeatures[trainLabels == i ,:]
    labels = trainLabels[trainLabels == i]
    trainFeaturesLT.append(data[0:trainDist[i],:])
    trainLabelsLT.append(labels[0:trainDist[i]])
    print(f"Class {i} with {trainDist[i]} samples")

trainFeaturesLT = np.concatenate( trainFeaturesLT, axis=0 )
trainLabelsLT = np.concatenate( trainLabelsLT, axis=0 )

print("Validation distribution with alpha = ", alpha)
valDist = (1000/trueDist[0]) * trueDist
valDist = valDist.astype(int)
valFeaturesLT = []
valLabelsLT = []
for i in range(10):
    data = valFeatures[valLabels == i ,:]
    labels = valLabels[valLabels == i]
    valFeaturesLT.append(data[0:valDist[i],:])
    valLabelsLT.append(labels[0:valDist[i]])
    print(f"Class {i} with {valDist[i]} samples")


valFeaturesLT = np.concatenate( valFeaturesLT, axis=0 )
valLabelsLT = np.concatenate( valLabelsLT, axis=0 )

np.save(f"{rootDir}representations/clipRN50x64/cifar10lt_train.npy", trainFeaturesLT)
np.save(f"{rootDir}representations/clipRN50x64/cifar10lt_val.npy", valFeaturesLT)
np.save(f"{rootDir}labels/cifar10lt_train.npy", trainLabelsLT)
np.save(f"{rootDir}labels/cifar10lt_val.npy", valLabelsLT)
