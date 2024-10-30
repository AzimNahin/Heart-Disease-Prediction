# Import Libraries
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load and preprocess dataset
heart_disease = fetch_ucirepo(id=45)

# Extract features and target
X = heart_disease.data.features
y = heart_disease.data.targets['num']

# Convert target to binary (1: disease, 0: no disease)
y = y.apply(lambda x: 1 if x > 0 else 0)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=63)

# Define models with base configurations
model_LR = LogisticRegression(max_iter=3000, solver='liblinear', random_state=63)
model_KNN = KNeighborsClassifier(n_neighbors=10)
model_DT = DecisionTreeClassifier(max_depth=10, random_state=63)
model_RF = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=63)
model_GB = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=63)
model_XGB = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=63, eval_metric='logloss')
model_SVM = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=63)
model_NN = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=63)

# Dictionary to store models
models= {
    "Logistic Regression": model_LR,
    "K-Nearest Neighbors": model_KNN,
    "Decision Tree": model_DT,
    "Random Forest": model_RF,
    "Gradient Boosting": model_GB,
    "XGBoost": model_XGB,
    "SVM (RBF Kernel)": model_SVM,
    "Neural Network (MLP)": model_NN
}

# Train and evaluate each model
performance = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    performance[name] = accuracy

# Convert performance results to DataFrame for display
performance_df = pd.DataFrame(list(performance.items()), columns=['Model', 'Accuracy'])

# Sort models by accuracy
performance_df = performance_df.sort_values(by='Accuracy', ascending=False)

# Visual comparison
plt.figure(figsize=(10, 6))
plt.barh(performance_df['Model'], performance_df['Accuracy'], color='skyblue')
plt.xlabel('Accuracy')
plt.title('Optimized Model Accuracy Comparison')
plt.gca().invert_yaxis()
plt.show()

# Display performance results
print(performance_df)

# Define parameter grids for hyperparameter tuning
param_grids = {
    "Logistic Regression": {
        'solver': ['liblinear'],
        'C': [0.1, 1, 10]
        #'solver': ['liblinear', 'lbfgs'],
        #'C': [0.1, 1, 10, 100, 1000]
    },
    "K-Nearest Neighbors": {
        'n_neighbors': list(range(5, 15))
        #'n_neighbors': list(range(1, 21)),
        #'weights': ['uniform', 'distance']
    },
    "Decision Tree": {
        'max_depth': list(range(5, 15))
        #'max_depth': list(range(3, 21)),
        #'min_samples_split': list(range(2, 11)),
    },
    "Random Forest": {
        'n_estimators': [100, 150],
        'max_depth': list(range(5, 15))
        #'n_estimators': [100, 150, 200, 250, 300],
        #'max_depth': list(range(3, 21)),
    },
    "Gradient Boosting": {
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 5, 6]
        #'n_estimators': [100, 150, 200, 250, 300],
        #'learning_rate': [0.05, 0.1, 0.2, 0.4],
        #'max_depth': list(range(3, 11))
    },
    "XGBoost": {
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 5, 6]
        #'n_estimators': [100, 150, 200, 250, 300],
        #'learning_rate': [0.05, 0.1, 0.2, 0.4],
        #'max_depth': list(range(3, 11))
    },
    "SVM (RBF Kernel)": {
        'C': [0.5, 1.0, 1.5]
        #'C': [0.5, 1.0, 1.5, 2.0],
        #'gamma': ['scale', 'auto']
    },
    "Neural Network (MLP)": {
        'hidden_layer_sizes': [(100,), (50, 50)],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500]
        #'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        #'learning_rate_init': [0.001, 0.01, 0.1],
        #'max_iter': [500, 1000, 2000]
    }
}

# Use Stratified K-Fold cross-validation to ensure balanced class splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=63)

# Dictionary to store best models after tuning
best_models = {}
best_scores = {}

# Perform Grid Search for each model
for name, model in models.items():
    print(f"Tuning {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=skf, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    best_scores[name] = grid_search.best_score_

# Convert best scores to DataFrame for better display
best_scores_df = pd.DataFrame(list(best_scores.items()), columns=['Model', 'Best CV Accuracy'])

# Sort models by best CV accuracy
best_scores_df = best_scores_df.sort_values(by='Best CV Accuracy', ascending=False)

# Visual comparison
plt.figure(figsize=(10, 6))
plt.barh(best_scores_df['Model'], best_scores_df['Best CV Accuracy'], color='salmon')
plt.xlabel('Best CV Accuracy')
plt.title('Best Model Accuracy After Hyperparameter Tuning')
plt.gca().invert_yaxis()
plt.show()

# Display best scores after tuning
print(best_scores_df)

# Combine data into a single DataFrame for easier plotting
combined_df = performance_df.merge(best_scores_df, on='Model', suffixes=('_Initial', '_Best'))

# Set up plot dimensions and bar width
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(combined_df))

# Plot initial accuracy
initial_bars = ax.bar(index, combined_df['Accuracy'], bar_width, label='Initial Accuracy', color='skyblue', alpha=0.6)

# Plot best accuracy after tuning
best_bars = ax.bar(index + bar_width, combined_df['Best CV Accuracy'], bar_width, label='Best CV Accuracy After Tuning', color='salmon', alpha=0.6)

# Add labels, title, and legend
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Initial vs. Best Accuracy After Tuning')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(combined_df['Model'], rotation=45)
ax.legend()

# Display plot
plt.tight_layout()
plt.show()
