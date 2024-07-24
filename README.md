# Custom Model Trainer

## Overview

The Custom Model Trainer is a Python tool designed to simplify the process of training and saving machine learning models. Users can specify the dataset and input/output columns, and the system handles the training and model saving automatically. Supported models include RandomForestRegressor, LinearRegression, and SVR.

## Features

- **Dynamic Model Training**: Train different types of models (RandomForestRegressor, LinearRegression, SVR) based on user input.
- **Custom Input/Output Columns**: Specify multiple input columns and a single output column for training.
- **Model Saving**: Automatically saves the trained model to a file.
- **Evaluation**: Provides R^2 score for evaluating model performance.

## Installation

To use the Custom Model Trainer, ensure you have Python and the necessary libraries installed. You can install the required libraries using `pip`:

```bash
pip install pandas scikit-learn matplotlib
```

## Usage

1. **Initialize the Trainer**

   Create an instance of the `CustomTraining` class with the path to your CSV file, a list of input column names, and the output column name.

   ```python
   from utils.custom_training import CustomTraining

   csv_file_path = 'path/to/your/csvfile.csv'
   input_column_names = ['input1', 'input2', 'input3']  # List of input columns
   output_column_name = 'output'  # Output column name

   trainer = CustomTraining(csv_file_path, input_column_names, output_column_name)
   ```

2. **Train Models**

   You can train different models by calling the respective methods:

   - **Random Forest Regressor**
     ```python
     rf_score = trainer.train_random_forest_regressor()
     print(f"Random Forest Regressor R^2 Score: {rf_score}")
     ```

   - **Linear Regression**
     ```python
     lr_score = trainer.train_linear_regression()
     print(f"Linear Regression R^2 Score: {lr_score}")
     ```

   - **Support Vector Regressor (SVR)**
     ```python
     svr_score = trainer.train_svr()
     print(f"SVR R^2 Score: {svr_score}")
     ```

3. **Model Prediction**

   Load the saved model and make predictions:

   ```python
   predictions = trainer.model_prediction([[value1, value2, value3]])
   print(f"Predictions: {predictions}")
   ```

## Accuracy Comparison

The performance of three regression models—Linear Regression, Random Forest Regressor, and Support Vector Regressor (SVR)—was evaluated using the R^2 score to determine their predictive accuracy. The R^2 score, representing the proportion of variance in the target variable explained by the model, was calculated for each algorithm after training on identical data subsets. The Linear Regression model provided a baseline performance, capturing linear relationships with an R^2 score of ```0.4205936575054805```. In contrast, the Random Forest Regressor, which utilizes ensemble learning to model complex, non-linear interactions, achieved an R^2 score of ```0.9229556734847897```, reflecting its superior capacity to fit the data. The SVR, leveraging support vector machines, delivered an R^2 score of ```0.39809432477257944```, demonstrating its effectiveness in high-dimensional spaces with a non-linear decision boundary. This comparative analysis highlights the relative strengths of each model, with Random Forest Regressor exhibiting the highest accuracy, thereby offering the most robust predictive performance in this context.

![accuracy_comparison](https://github.com/user-attachments/assets/482f94fb-46ab-429b-9921-6c766df7aed9)

## Contributing

Feel free to fork the repository and submit pull requests. Issues and feature requests are also welcome.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
