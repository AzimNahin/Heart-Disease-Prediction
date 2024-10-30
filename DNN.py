# Import necessary libraries
import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

# Set random seeds for reproducibility
seed = 63
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Load and preprocess dataset
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets['num']

# Convert target to binary (1: disease, 0: no disease)
y = y.apply(lambda x: 1 if x > 0 else 0)  

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# One-hot encode target variable for training in the neural network
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Define the DNN model
model = Sequential([
    Dense(128, input_shape=(X.shape[1],), activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    # Output layer for binary classification
    Dense(2, activation='softmax')  
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model
model.fit(X_train, y_train_cat, epochs=50, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

# Evaluate model on the test set
y_pred_prob = model.predict(X_test)
# Convert from one-hot encoding to binary labels
y_pred = np.argmax(y_pred_prob, axis=1)  

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob[:, 1])

# Display results
print("DNN Model Performance on Heart Disease Dataset:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
