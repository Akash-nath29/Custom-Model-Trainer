from utils.custom_training import CustomTraining
import pandas as pd

# Create an instance of CustomTraining
custom_training = CustomTraining('replicated_data.csv')
score = custom_training.train_random_forest_regressor()

print("Score: ", str(score))

gpa = custom_training.model_prediction([[1700]])
print("GPA: ", str(gpa[0]))

custom_training = CustomTraining('replicated_data.csv')
score = custom_training.train_linear_regression()

print("Score: ", str(score))

gpa = custom_training.model_prediction([[1700]])
print("GPA: ", str(gpa[0]))