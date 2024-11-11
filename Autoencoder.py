
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt



'''
1. Use the training set, validation set, and test set 
    from Assignment 3 (Hierarchical Clustering) for this Assignment.
'''
faces = fetch_olivetti_faces()


X = faces.data
y = faces.target
X.shape, y.shape


print(X[0])

min_value = np.min(X)
max_value = np.max(X)
print(min_value, max_value)


face_index = np.unique(y, return_index=True)[1]
face_index



unique_faces = X[face_index]


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
2. Use PCA preserving 99% of the variance to reduce the dataset’s dimensionality 
    as in Assignment 4 (Gaussian Mixture Models) and use it to train the autoencoder 
'''
pca = PCA(n_components=0.99)

X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

print(f"Number of components selected to preserve 99% variance: {pca.n_components_}")


X_train_pca.shape


'''
3. Define an autoencoder with the following architecture
'''
input_dim = X_train_pca.shape[1]

def build_autoencoder(input_dim, hidden_units_1, hidden_units_2, learning_rate, reg_strength):
    input_layer = Input(shape=(input_dim,))
    
    hidden_layer_1 = Dense(hidden_units_1, activation='relu', kernel_regularizer=l2(reg_strength))(input_layer)
    
    central_layer = Dense(hidden_units_2, activation='relu', kernel_regularizer=l2(reg_strength))(hidden_layer_1)
    
    hidden_layer_3 = Dense(hidden_units_1, activation='relu', kernel_regularizer=l2(reg_strength))(central_layer)
    
    output_layer = Dense(input_dim, activation='sigmoid')(hidden_layer_3)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return autoencoder

def run_kfold_cv(X_train_pca, hidden_units_1_values, hidden_units_2_values, learning_rates, reg_strengths, k=5):
    input_dim = X_train_pca.shape[1]  
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  

    best_loss = float('inf')
    best_params = None
        
    for hidden_units_1 in hidden_units_1_values:
        for hidden_units_2 in hidden_units_2_values:
            for lr in learning_rates:
                for reg_strength in reg_strengths:
                    fold_losses = []
                    
                    for train_idx, val_idx in kf.split(X_train_pca):
                        X_train_fold, X_val_fold = X_train_pca[train_idx], X_train_pca[val_idx]
                        
                        model = build_autoencoder(input_dim, hidden_units_1, hidden_units_2, lr, reg_strength)

                        history = model.fit(
                            X_train_fold, X_train_fold,  
                            epochs=10,
                            batch_size=32,
                            validation_data=(X_val_fold, X_val_fold),
                            verbose=0
                        )
                        
                        val_loss = history.history['val_loss'][-1]
                        fold_losses.append(val_loss)
                    
                    avg_loss = np.mean(fold_losses)
                    print(f"Hidden 1: {hidden_units_1}, Hidden 2: {hidden_units_2}, LR: {lr}, Reg: {reg_strength}, Avg Loss: {avg_loss}")
                    
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_params = {
                            'hidden_units_1': hidden_units_1,
                            'hidden_units_2': hidden_units_2,
                            'learning_rate': lr,
                            'reg_strength': reg_strength
                        }
    
    print(f"Best Loss: {best_loss}")
    print(f"Best Params: {best_params}")
    return best_params


hidden_units_1_values = [128, 256]  
hidden_units_2_values = [64, 128]  
learning_rates = [0.001, 0.0001]  
reg_strengths = [0.0001, 0.001] 


best_hyperparams = run_kfold_cv(X_train_pca, hidden_units_1_values, hidden_units_2_values, learning_rates, reg_strengths, k=5)

'''
Run the best model with the test set and display the original image and the reconstructed image.
'''

final_model = build_autoencoder(
    input_dim=X_train_pca.shape[1],
    hidden_units_1=best_hyperparams['hidden_units_1'],
    hidden_units_2=best_hyperparams['hidden_units_2'],
    learning_rate=best_hyperparams['learning_rate'],
    reg_strength=best_hyperparams['reg_strength']
)



history = final_model.fit(
    X_train_pca, X_train_pca,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_pca, X_val_pca),
    verbose=1
)


reconstructed_images_pca = final_model.predict(X_test_pca)


reconstructed_images_original = pca.inverse_transform(reconstructed_images_pca)
original_images_original = pca.inverse_transform(X_test_pca)



def plot_original_vs_reconstructed(original, reconstructed, n_images=10):
    """
    Plots original images vs reconstructed images side by side.
    """
    plt.figure(figsize=(20, 4))
    for i in range(n_images):
        # Original images
        ax = plt.subplot(2, n_images, i + 1)
        plt.imshow(original[i].reshape(64, 64), cmap="gray")
        plt.title("Original")
        plt.axis("off")
        
        # Reconstructed images
        ax = plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(reconstructed[i].reshape(64, 64), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")
    
    plt.show()


plot_original_vs_reconstructed(original_images_original, reconstructed_images_original, n_images=10)

