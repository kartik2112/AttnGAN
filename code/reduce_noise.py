'''
The purpose of this utility is to reduce the dimensionality of a noise matrix. AttnGAN needs a 100-dim noise vector.
This utility would reduce the dimensionality to 100
'''
from sklearn.decomposition import PCA
import numpy as np

X_train_std = np.load('lafite_noise.npy')
X_train_std = X_train_std[55085:,]

# intialize pca
pca = PCA(n_components=100)

# fit and transform data
X_train_pca = pca.fit_transform(X_train_std)

np.save('Lafite_COCO_noise.npy', X_train_pca)