# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from ucimlrepo import fetch_ucirepo

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

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Balance classes with SMOTE
smote = SMOTE(random_state=63)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=63)


# Define parameter grids
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

param_grid_gb = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

stf = StratifiedKFold(n_splits=4, shuffle=True, random_state=63)

# Run Bayesian Optimization for Random Forest
bayes_rf = BayesSearchCV(RandomForestClassifier(random_state=63), param_grid_rf, n_iter=10, scoring='roc_auc', cv=stf, n_jobs=-1)
bayes_rf.fit(X_train, y_train)
best_rf = bayes_rf.best_estimator_

# Run Bayesian Optimization for Gradient Boosting
bayes_gb = BayesSearchCV(GradientBoostingClassifier(random_state=63), param_grid_gb, n_iter=10, scoring='roc_auc', cv=stf, n_jobs=-1)
bayes_gb.fit(X_train, y_train)
best_gb = bayes_gb.best_estimator_


# Define Base Models
base_estimators = [
    ('rf', best_rf),
    ('gb', best_gb),
]

# Define the meta-model
meta_model = LogisticRegression()

# Stratified K-Fold cross-validation
stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=63)

# Stacking Classifier with cross-validation
stacking_model = StackingClassifier(estimators=base_estimators, final_estimator=meta_model, cv=stf)

# Train the Stacking Model
stacking_model.fit(X_train, y_train)

# Model Evaluation
y_pred = stacking_model.predict(X_test)
y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Display results
print("ML-HDPM Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
