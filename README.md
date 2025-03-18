# SPY k-NN Classifier

This repository implements a k-Nearest Neighbors (k-NN) classifier in Python to predict positive entry days for the SPY stock index. The classifier uses MinMax normalization and hyperparameter tuning via RandomizedSearchCV.

## Features

- **Data Normalization:**  
  Uses MinMaxScaler to normalize the feature data.
  
- **Hyperparameter Tuning:**  
  Utilizes RandomizedSearchCV to explore various parameters for the k-NN classifier.
  
- **Model Evaluation:**  
  Generates a classification report and confusion matrix to assess the model's performance.

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - scipy
  - matplotlib
  - seaborn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/runciter2078/Classification_kNN.git
   ```

2. (Optional) Rename the repository folder to `SPY_kNN_Classifier` for clarity.

3. Navigate to the project directory:

   ```bash
   cd Classification_kNN
   ```

## Usage

1. Place your CSV file (e.g., `SPYV3.csv`) in the project directory.

2. Run the script:

   ```bash
   python knn_spyv3.py
   ```

The script will:
- Load and normalize the dataset.
- Split the data into training and testing sets.
- Perform hyperparameter tuning for the k-NN classifier.
- Train the final model using the chosen parameters.
- Evaluate the model and display the classification report and confusion matrix.

## Notes

- **Hyperparameter Tuning:**  
  The hyperparameter search uses 168 iterations by default. Adjust `n_iter_search` as needed based on your dataset size and available computational resources.

## License

This project is distributed under the [MIT License](LICENSE).
