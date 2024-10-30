# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.feature_selection import SelectFromModel
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

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=63)

# Feature Selection with Initial Random Forest
initial_rf = RandomForestClassifier(random_state=63, class_weight='balanced')
initial_rf.fit(X_train, y_train)

# Select features based on importance
selector = SelectFromModel(initial_rf, threshold="mean", prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Define hyperparameter search space
param_grid = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(10, 30),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 5),
    'max_features': Real(0.5, 1.0, 'uniform')
}

# Use StratifiedKFold for balanced cross-validation
stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=63)

# Initialize BayesSearchCV with RandomForestClassifier
model_RF = RandomForestClassifier(random_state=63, class_weight='balanced')
bayes_search = BayesSearchCV(estimator=model_RF, search_spaces=param_grid, cv=stf, n_iter=30, scoring='roc_auc', n_jobs=-1, random_state=63)

# Train the model with hyperparameter tuning
bayes_search.fit(X_train_selected, y_train)

# Get the best model from Bayesian Optimization
best_model_RF = bayes_search.best_estimator_

# Make Predictions
y_pred = best_model_RF.predict(X_test_selected)
y_pred_proba = best_model_RF.predict_proba(X_test_selected)[:, 1]

# Calculate Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Display the metrics
print("Performance Metrics for Random Forest Model:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
