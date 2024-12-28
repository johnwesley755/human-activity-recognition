# Human Activity Recognition (HAR)

This project involves the use of machine learning techniques to recognize human activities based on sensor data from smartphones. It uses the UCI Human Activity Recognition (HAR) dataset, which contains data from accelerometers and gyroscopes to classify different human activities such as walking, sitting, and standing.

## Table of Contents
- [About](#about)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)


## About
Human Activity Recognition (HAR) is a classification problem where the objective is to recognize various human activities from sensor data (e.g., accelerometer, gyroscope). The dataset contains labeled time-series data representing different activities, which can be used to train a machine learning model to predict activities based on new data.

This project implements a solution using the UCI HAR dataset to classify human activities using machine learning algorithms.

## Dataset
The dataset used in this project is the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).

The dataset consists of 561 features recorded from the accelerometer and gyroscope of smartphones worn by 30 participants during activities like walking, sitting, and standing.

Key files in the dataset:
- `X_train.txt`: Training set of feature data (562 columns).
- `Y_train.txt`: Training labels indicating the activity class.
- `X_test.txt`: Test set of feature data.
- `Y_test.txt`: Test labels.
- `features.txt`: List of feature names.

## Installation

### Prerequisites:
- **Python 3.x** (or newer)
- **pip** (Python package manager)

### Steps to Install:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/human-activity-recognition.git
   ```
   
2. Navigate to the project folder:
   ```bash
   cd human-activity-recognition
   ```

3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:
   - **On Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - **On macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Once the dependencies are installed, you can use the scripts to load the dataset, preprocess the data, and train a machine learning model.

### Example for running the script to train the model:
```bash
python train_model.py
```

This will train the model using the training data and save the model for future predictions.

### Example for running predictions:
```bash
python predict_activity.py --input new_data.txt
```

This will use the trained model to predict the human activity based on new input data.

## Model Training

The model is trained using a machine learning algorithm (e.g., Random Forest, SVM). You can change the algorithm and parameters in the `train_model.py` script.

### Steps for training:
1. Load the training dataset (`X_train.txt` and `Y_train.txt`).
2. Preprocess the data (normalization, feature extraction).
3. Train a classifier model.
4. Evaluate the model on the test data (`X_test.txt` and `Y_test.txt`).
5. Save the trained model for inference.

## Contributing
We welcome contributions to improve the project. If you'd like to contribute, follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add new feature'`)
4. Push the changes to your branch (`git push origin feature-branch`)
5. Open a pull request

---
