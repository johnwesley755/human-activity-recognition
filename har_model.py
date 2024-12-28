import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Paths to dataset
DATASET_PATH = "UCI HAR Dataset/"

def load_data(file_path):
    """Load data from a given file path."""
    return pd.read_csv(file_path, delim_whitespace=True, header=None)

# Load training and testing datasets
X_train = load_data(DATASET_PATH + "train/X_train.txt")
y_train = load_data(DATASET_PATH + "train/y_train.txt")
X_test = load_data(DATASET_PATH + "test/X_test.txt")
y_test = load_data(DATASET_PATH + "test/y_test.txt")

# Encode target labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train.values.ravel())
y_test = encoder.transform(y_test.values.ravel())

# One-hot encode target labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Generate a classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
print(classification_report(y_test_classes, y_pred_classes, target_names=encoder.classes_.astype(str)))

# Plot training and validation accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
