import numpy as np

rootDir = "/home/javier/Desktop/turtle/data/"
trainFeatures = np.load(rootDir + "representations/clipRN50x64/food101_train.npy")
valFeatures = np.load(rootDir + "representations/clipRN50x64/food101_val.npy")
trainLabels = np.load(rootDir + "labels/food101_train.npy")
valLabels = np.load(rootDir + "labels/food101_val.npy")

decayFactor = 1.05

print("train stats:")
size = 750
trainFeaturesLT = []
trainLabelsLT = []
for i in range(101):
    labels = (trainLabels == i)
    data = trainFeatures[ labels ,:]
    labels = trainLabels[labels]
    trainFeaturesLT.append(data[0:size,:])
    trainLabelsLT.append(labels[0:size])
    print(data[0:size,:].shape)
    size = int(size // decayFactor)
    if size < 75:
        size = 75

trainFeaturesLT = np.concatenate( trainFeaturesLT, axis=0 )
trainLabelsLT = np.concatenate( trainLabelsLT, axis=0 )

print("val stats:")
size = 250
distribution = []
distribution.append(size)
valFeaturesLT = []
valLabelsLT = []
for i in range(101):
    labels = (valLabels == i)
    data = valFeatures[ labels ,:]
    labels = valLabels[labels]
    valFeaturesLT.append(data[0:size,:])
    valLabelsLT.append(labels[0:size])
    print(data[0:size,:].shape)
    size = int(size // decayFactor)
    if size < 25:
        size = 25
    distribution.append(size)

valFeaturesLT = np.concatenate( valFeaturesLT, axis=0 )
valLabelsLT = np.concatenate( valLabelsLT, axis=0 )
distribution = np.stack( distribution[0:101] )
distribution = distribution/sum(distribution)

np.save(rootDir + "representations/clipRN50x64/food101lt_train.npy", trainFeaturesLT)
np.save(rootDir + "representations/clipRN50x64/food101lt_val.npy", valFeaturesLT)
np.save(rootDir + "labels/food101lt_train.npy", trainLabelsLT)
np.save(rootDir + "labels/food101lt_val.npy", valLabelsLT)
np.save(rootDir + "labels/food101lt_distribution.npy", distribution)