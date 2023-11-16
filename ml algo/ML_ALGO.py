import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

# Load the dataset
file_url = "D:/Courses/Python for datascience/ml algo/pima-indians-diabetes.csv"

# Define column names
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# Read the dataset with the specified column names
df = pd.read_csv(file_url, header=None, names=column_names)

# Assuming your target variable is named 'Outcome'
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Binarize the labels for binary classification
y_bin = label_binarize(y, classes=np.unique(y))

# Flatten the target variable to a 1D array
y_bin_flat = y_bin.ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_bin_flat, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models dictionary with hyperparameter tuning
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),  # Example of hyperparameter tuning
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5),  # Example of hyperparameter tuning
    'SVM': SVC(C=1.0, kernel='rbf'),
    'k-NN': KNeighborsClassifier(n_neighbors=5),
}

# Get user input for the model to train
print("Choose a model to train:")
for i, model_name in enumerate(models.keys(), 1):
    print(f"{i}. {model_name}")

selected_model_index = int(input("Enter the number corresponding to the model: ")) - 1

# Get the selected model
selected_model_name = list(models.keys())[selected_model_index]
selected_model = models[selected_model_name]

# Train the selected model with cross-validation
print(f"Training and evaluating {selected_model_name}...")
cross_val_scores = cross_val_score(selected_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy: {np.mean(cross_val_scores):.2f} (+/- {np.std(cross_val_scores):.2f})')

# Fit the model on the entire training set
selected_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = selected_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f'{selected_model_name} Model:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Compute ROC curve and ROC area
if selected_model_name != 'SVM':  # ROC curve is not applicable for SVM
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {selected_model_name}')
    plt.legend(loc='lower right')
    plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix for {selected_model_name}')
plt.colorbar()
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show()
