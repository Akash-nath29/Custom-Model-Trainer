import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class CustomTraining:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        
    def data_preparation(self):
        df = pd.read_csv(self.csv_file_path)
        data = df.dropna()
        X = data[['SAT']]
        y = data['GPA']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, y_train, X_test, y_test
    
    def train_random_forest_regressor(self):
        X_train, y_train, X_test, y_test = self.data_preparation()
        
        # Use RandomForestRegressor instead of LinearRegression
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save the model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        
        return r2_score(y_test, y_pred)
    
    def train_linear_regression(self):
        X_train, y_train, X_test, y_test = self.data_preparation()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save the model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        
        return r2_score(y_test, y_pred)
    
    def model_prediction(self, value:list):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        X = pd.DataFrame(value)
        return model.predict(X)