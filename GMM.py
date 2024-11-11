

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from collections import Counter
import tensorflow as tf


'''
1. Retrieve and load the Olivetti faces dataset
'''
faces = fetch_olivetti_faces()


X = faces.data
y = faces.target
X.shape, y.shape


min_value = np.min(X)
max_value = np.max(X)
print(min_value, max_value)


face_index = np.unique(y, return_index=True)[1]
face_index



unique_faces = X[face_index]


'''
2. Split the training set, a validation set,
    and a test set using stratified sampling to ensure that
    there are the same number of images per person in each set.
    Provide your rationale for the split ratio.
'''
train_indices = []
val_indices = []
test_indices = []

for index in face_index:
  person_indices = np.arange(index, index + 10)

  np.random.shuffle(person_indices)

  train_indices.extend(person_indices[:6])
  val_indices.extend(person_indices[6:8])
  test_indices.extend(person_indices[8:])

train_indices = np.array(train_indices)
val_indices = np.array(val_indices)
test_indices = np.array(test_indices)

X_train, y_train = X[train_indices], y[train_indices]
X_val, y_val = X[val_indices], y[val_indices]
X_test, y_test = X[test_indices], y[test_indices]



'''
3. Apply PCA on the training data, preserving 99% of the variance, to reduce the dataset's dimensionality.
'''
pca = PCA(n_components=0.99)

X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

print(f"Number of components selected to preserve 99% variance: {pca.n_components_}")



'''
4. Determine the most suitable covariance type for the dataset. 
5. Determine the minimum number of clusters that best represent the dataset using either AIC or BIC.
'''
n_components_range = range(1, 50)  
bic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=21, reg_covar=1e-6)
    gmm.fit(X_train_pca)  

    bic = gmm.bic(X_val_pca)  
    bic_scores.append(bic)  
    
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, marker='o')
plt.title('BIC Scores for Different Number of Clusters (Full Covariance)')
plt.xlabel('Number of Components (Clusters)')
plt.ylabel('BIC Score')
plt.grid(True)
plt.show()

min_bic = min(bic_scores)
optimal_n_components = n_components_range[bic_scores.index(min_bic)]
print(f"Optimal number of clusters: {optimal_n_components}, Minimum BIC: {min_bic}")



n_components_range = range(1, 50)  
bic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=21, reg_covar=1e-6)
    gmm.fit(X_train_pca)  

    bic = gmm.bic(X_val_pca)  
    bic_scores.append(bic)  
    
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, marker='o')
plt.title('BIC Scores for Different Number of Clusters (Tied Covariance)')
plt.xlabel('Number of Components (Clusters)')
plt.ylabel('BIC Score')
plt.grid(True)
plt.show()

min_bic = min(bic_scores)
optimal_n_components = n_components_range[bic_scores.index(min_bic)]
print(f"Optimal number of clusters: {optimal_n_components}, Minimum BIC: {min_bic}")



n_components_range = range(1, 50)  
bic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=21, reg_covar=1e-6)
    gmm.fit(X_train_pca)  

    bic = gmm.bic(X_val_pca)  
    bic_scores.append(bic)  
    
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, marker='o')
plt.title('BIC Scores for Different Number of Clusters (Diag Covariance)')
plt.xlabel('Number of Components (Clusters)')
plt.ylabel('BIC Score')
plt.grid(True)
plt.show()

min_bic = min(bic_scores)
optimal_n_components = n_components_range[bic_scores.index(min_bic)]
print(f"Optimal number of clusters: {optimal_n_components}, Minimum BIC: {min_bic}")



n_components_range = range(1, 50)  
bic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=21, reg_covar=1e-6)
    gmm.fit(X_train_pca)  

    bic = gmm.bic(X_val_pca)  
    bic_scores.append(bic)  
    
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, marker='o')
plt.title('BIC Scores for Different Number of Clusters (Spherical Covariance)')
plt.xlabel('Number of Components (Clusters)')
plt.ylabel('BIC Score')
plt.grid(True)
plt.show()

min_bic = min(bic_scores)
optimal_n_components = n_components_range[bic_scores.index(min_bic)]
print(f"Optimal number of clusters: {optimal_n_components}, Minimum BIC: {min_bic}")



'''
7. Output the hard clustering assignments for each instance to identify which cluster each image belongs to. [2.5 points]
'''

gmm = GaussianMixture(n_components=4, covariance_type='spherical', random_state=21, reg_covar=1e-6)
gmm.fit(X_train_pca)  
hard_assignments = gmm.predict(X_train_pca)

Counter(hard_assignments)


'''
8. Output the soft clustering probabilities for each instance to show the likelihood of each image belonging to each cluster.
'''
soft_assignments = gmm.predict_proba(X_train_pca)
print(soft_assignments)


'''
9. Use the model to generate some new faces (using the sample() method) 
    and visualize them (use the inverse_transform() method 
    to transform the data back to its original space based on the PCA method used).
'''

generated_samples, labels = gmm.sample(40)

generated_faces = pca.inverse_transform(generated_samples)

fig, axes = plt.subplots(4, 10, figsize=(20, 10))  
axes = axes.ravel()

for i in range(40):
    axes[i].imshow(generated_faces[i].reshape(64, 64), cmap='gray')
    axes[i].set_title(f'Cluster {labels[i]}')  
    axes[i].axis('off')

plt.suptitle('Generated Faces with Cluster Labels')
plt.show()



'''
10. Modify some images
'''
generated_faces_tensor = np.expand_dims(generated_faces.reshape(-1, 64, 64), axis=-1)  

def random_rotate(image):
    return tf.image.rot90(image, k=np.random.randint(1, 4))  

def random_flip(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image

def random_brightness(image, delta=0.8):
    return tf.image.random_brightness(image, max_delta=delta)  

modified_faces = []
for face in generated_faces_tensor:
    rotated = random_rotate(face)
    flipped = random_flip(face)
    brightened = random_brightness(face)

    modified_faces.append(rotated.numpy())
    modified_faces.append(flipped.numpy())
    modified_faces.append(brightened.numpy())

fig, axes = plt.subplots(4, 10, figsize=(20, 10))  
axes = axes.ravel()

for i in range(40):
    axes[i].imshow(modified_faces[i].reshape(64, 64), cmap='gray')
    axes[i].set_title(f'Cluster {labels[i]}') 
    axes[i].axis('off')

plt.suptitle('Generated and Dramatically Modified Faces with Original Cluster Labels')
plt.show()



'''
11. Determine if the model can detect the anomalies produced in step 10 
by comparing the output of the score_samples() method for normal images and for anomalies.
'''


generated_faces_flat = generated_faces.reshape(-1, 64 * 64)  
modified_faces_flat = np.array(modified_faces).reshape(-1, 64 * 64)  

generated_faces_pca = pca.transform(generated_faces_flat)  
modified_faces_pca = pca.transform(modified_faces_flat)    

normal_scores = gmm.score_samples(generated_faces_pca)  
modified_scores = gmm.score_samples(modified_faces_pca)  

avg_normal_score = np.mean(normal_scores)
avg_modified_score = np.mean(modified_scores)

print(f"\nAverage Log Likelihood for Normal Images: {avg_normal_score}")
print(f"Average Log Likelihood for Modified (Anomalous) Images: {avg_modified_score}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(normal_scores, bins=20, alpha=0.7, color='blue')
plt.title('Log Likelihood Distribution for Normal Images')
plt.xlabel('Log Likelihood Score')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(modified_scores, bins=20, alpha=0.7, color='red')
plt.title('Log Likelihood Distribution for Modified Images')
plt.xlabel('Log Likelihood Score')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()  
plt.show()