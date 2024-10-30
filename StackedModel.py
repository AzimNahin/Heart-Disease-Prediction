# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
import warnings
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

# Scale features
scaler = StandardScaler()
X= scaler.fit_transform(X)

# Data Balancing with SMOTE
smote = SMOTE(random_state=63)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=63)

# Define Base Models
model_RF = RandomForestClassifier(n_estimators=100, random_state=63)
model_GB = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=63)
model_XGB = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, eval_metric='logloss', random_state=63)

# Base Estimators
base_estimators = [
    ('rf', model_RF),
    ('gb', model_GB),
    ('xgb', model_XGB)
]

# Meta-model
meta_model = LogisticRegression()

# Stratified K-Fold
stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=63)

# Stacking Classifier
stacked_model = StackingClassifier(estimators=base_estimators, final_estimator=meta_model, cv=stf, n_jobs=-1)

# Hyperparameter Search Space
param_grid = {
    'rf__n_estimators': [50, 150, 200],
    'gb__n_estimators': [50, 150],
    'gb__learning_rate': [0.05, 0.1, 0.2],
    'xgb__n_estimators': [50, 100, 150],
    'xgb__learning_rate': [0.05, 0.1, 0.2],
    # for logistic regression meta-model
    'final_estimator__C': [0.1, 1, 10]  
}

# Initialize BayesSearchCV
model_Tuned = BayesSearchCV(stacked_model, param_grid, cv=stf, n_iter=20, scoring='roc_auc', n_jobs=-1, random_state=63)

# Train the model with hyperparameter tuning
model_Tuned.fit(X_train, y_train)

# Evaluate the Tuned Model
best_model = model_Tuned.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
print("Tuned Stacked Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Calculate Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Display the metrics
print("Performance Metrics for Stacked Model:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

