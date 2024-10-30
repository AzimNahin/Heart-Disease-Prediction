# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
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
model_LR = LogisticRegression(max_iter=3000, solver='liblinear')
model_KNN = KNeighborsClassifier(n_neighbors=10)
model_DT = DecisionTreeClassifier(max_depth=10)
model_RF = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=63)
model_GB = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=6)
model_XGB = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, eval_metric='logloss')
model_SVM = SVC(kernel='rbf', C=1.5, gamma='auto', probability=True)
model_NN = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=63)

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

# Dictionary to store AUC scores
auc_scores = {}

# Train and evaluate each model for AUC
for name, model in models.items():
    model.fit(X_train, y_train)
    
    # Get prediction probabilities for AUC calculation
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    auc_scores[name] = auc_score

# Convert AUC scores to DataFrame for better display
auc_scores_df = pd.DataFrame(list(auc_scores.items()), columns=['Model', 'AUC Score'])
auc_scores_df = auc_scores_df.sort_values(by='AUC Score', ascending=False)

# Visual comparison
plt.figure(figsize=(10, 6))
plt.barh(auc_scores_df['Model'], auc_scores_df['AUC Score'], color='lightcoral')
plt.xlabel('AUC Score')
plt.title('Individual Model AUC Comparison')
plt.gca().invert_yaxis()
plt.show()

# Display individual AUC scores
print(auc_scores_df)

# Define parameter grids with refined ranges
param_grids = {
    "Logistic Regression": {
        'solver': ['liblinear', 'lbfgs'],
        'C': [0.1, 1, 10, 100, 1000]
    },
    "K-Nearest Neighbors": {
        'n_neighbors': list(range(1, 21)),
        'weights': ['uniform', 'distance']
    },
    "Decision Tree": {
        'max_depth': list(range(3, 21)),
        'min_samples_split': list(range(2, 11)),
    },
    "Random Forest": {
        'n_estimators': [100, 150, 200],
        'max_depth': list(range(3, 16)),
    },
    "Gradient Boosting": {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': list(range(3, 11))
    },
    "XGBoost": {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': list(range(3, 11))
    },
    "SVM (RBF Kernel)": {
        'C': [0.5, 1.0, 1.5, 2.0],
        'gamma': ['scale', 'auto']
    },
    "Neural Network (MLP)": {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500, 1000]
    }
}

# Use Stratified K-Fold cross-validation to ensure balanced class splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=63)

# Dictionary to store best models after tuning
best_models = {}
best_scores = {}

# Perform Grid Search for each model
for name, model in models.items():
    print(f"Tuning {name} with AUC scoring...")
    grid_search = GridSearchCV(model, param_grids[name], cv=skf, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    best_scores[name] = grid_search.best_score_

# Convert best scores to DataFrame for better display
best_scores_df = pd.DataFrame(list(best_scores.items()), columns=['Model', 'Best CV AUC'])

# Sort models by best CV accuracy
best_scores_df = best_scores_df.sort_values(by='Best CV AUC', ascending=False)

# Visual comparison
plt.figure(figsize=(10, 6))
plt.barh(best_scores_df['Model'], best_scores_df['Best CV AUC'], color='salmon')
plt.xlabel('Best CV AUC')
plt.title('Best Model AUC After Hyperparameter Tuning')
plt.gca().invert_yaxis()
plt.show()

# Display best scores after tuning
print(best_scores_df)

# Combine data into a single DataFrame for easier plotting
combined_df = auc_scores_df.merge(best_scores_df, on='Model', suffixes=('_Initial', '_Best'))

# Set up plot dimensions and bar width
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(combined_df))

# Plot initial accuracy
initial_bars = ax.bar(index, combined_df['AUC Score'], bar_width, label='Initial AUC', color='skyblue', alpha=0.6)

# Plot best accuracy after tuning
best_bars = ax.bar(index + bar_width, combined_df['Best CV AUC'], bar_width, label='Best CV AUC After Tuning', color='salmon', alpha=0.6)

# Add labels, title, and legend
ax.set_xlabel('Model')
ax.set_ylabel('AUC')
ax.set_title('Comparison of Initial vs. Best AUC After Tuning')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(combined_df['Model'], rotation=45)
ax.legend()

# Display plot
plt.tight_layout()
plt.show()

