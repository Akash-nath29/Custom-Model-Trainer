import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt

class CustomTraining:
    def __init__(self, csv_file_path: str, input_column_name: list, output_column_name: str):
        """
        Initialize the CustomTraining class with the path to the CSV file, input column names, and output column name.

        Args:
            csv_file_path (str): Path to the CSV file.
            input_column_name (list): List of input column names.
            output_column_name (str): Output column name.
        """
        self.csv_file_path = csv_file_path
        self.input_column_name = input_column_name
        self.output_column_name = output_column_name
        
    def data_preparation(self):
        """
        Prepare the data by loading the CSV file, dropping missing values, splitting into train and test sets, 
        and standardizing the features.

        Returns:
            tuple: Standardized training and testing data (X_train, y_train, X_test, y_test).
        """
        try:
            df = pd.read_csv(self.csv_file_path)
        except FileNotFoundError:
            raise Exception(f"File {self.csv_file_path} not found.")

        if not all(col in df.columns for col in self.input_column_name + [self.output_column_name]):
            raise Exception("One or more specified columns do not exist in the CSV file.")
        data = df.dropna()
        
        X = data[self.input_column_name]
        y = data[self.output_column_name]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, y_train, X_test, y_test
    
    def train_random_forest_regressor(self):
        """
        Train a Random Forest Regressor model on the prepared data and save the model to a file.

        Returns:
            float: R^2 score of the trained model on the test data.
        """
        X_train, y_train, X_test, y_test = self.data_preparation()
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        y_pred = model.predict(X_test)
        
        return r2_score(y_test, y_pred)
    
    def train_linear_regression(self):
        """
        Train a Linear Regression model on the prepared data and save the model to a file.

        Returns:
            float: R^2 score of the trained model on the test data.
        """
        X_train, y_train, X_test, y_test = self.data_preparation()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save the model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        
        return r2_score(y_test, y_pred)
    
    def train_svr(self):
        """
        Train a Support Vector Regressor (SVR) model on the prepared data and save the model to a file.

        Returns:
            float: R^2 score of the trained model on the test data.
        """
        X_train, y_train, X_test, y_test = self.data_preparation()
        
        model = LinearSVR()
        model.fit(X_train, y_train)
        
        # Save the model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        
        return r2_score(y_test, y_pred)
    
    def model_prediction(self, value:list):
        """
        Load the trained model from a file and make predictions on the given input data.

        Args:
            value (list): Input data for making predictions.

        Returns:
            numpy.ndarray: Predictions made by the loaded model.
        """
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        X = pd.DataFrame(value)
        return model.predict(X)
    
    def compare_model_accuracies(self):
        """
        Compare the R^2 scores of Linear Regression, Random Forest Regressor, and SVR models 
        and plot the comparison in a bar chart.

        Returns:
            None
        """
        lr_score = self.train_linear_regression()
        rf_score = self.train_random_forest_regressor()
        svr_score = self.train_svr()
        
        models = ['Linear Regression', 'Random Forest Regressor', 'SVR']
        scores = [lr_score, rf_score, svr_score]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, scores, color=['blue', 'green', 'red'])
        plt.xlabel('Models')
        plt.ylabel('R^2 Score')
        plt.title('Model Comparison')
        plt.ylim(0, 1)
        plt.show()