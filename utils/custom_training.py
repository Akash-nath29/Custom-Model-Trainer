import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

class CustomTraining:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        
    def data_preparation(self):
        df = pd.read_csv(self.csv_file_path)
        data = df.dropna()
        X = data[['SAT']]
        y = data['GPA']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, y_train, X_test, y_test
    
    def model_training(self):
        X_train, y_train, X_test, y_test = self.data_preparation()
        # Your model training code here
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        return model.score(X_test, y_test)
    
    def model_prediction(self, value:list):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        X = pd.DataFrame(value)
        return model.predict(X)