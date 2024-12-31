import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load training and testing datasets
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Display the first few rows of the datasets
print("Training Data:")
print(train_data.head())

print("\nTesting Data:")
print(test_data.head())

# Check for missing values
print("\nMissing values in training data:", train_data.isnull().sum().sum())
print("Missing values in testing data:", test_data.isnull().sum().sum())

# Separate features (X) and labels (y)
# Assuming 'Activity' is the target column and the rest are features
X_train = train_data.drop(columns=['Activity'])
y_train = train_data['Activity']

X_test = test_data.drop(columns=['Activity'])
y_test = test_data['Activity']

# Encode labels (if necessary)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train a Random Forest Classifier
print("\nTraining the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split)

# Validate the model on the validation set
print("\nValidating the model on validation data...")
y_val_pred = model.predict(X_val_split)
print("Validation Accuracy:", accuracy_score(y_val_split, y_val_pred))
print("\nValidation Classification Report:")
print(classification_report(y_val_split, y_val_pred))

# Evaluate the model on the test set
print("\nEvaluating the model on test data...")
y_test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

# Save the trained model (optional)
import joblib
joblib.dump(model, 'activity_model.pkl')
print("\nModel saved as 'activity_model.pkl'")
