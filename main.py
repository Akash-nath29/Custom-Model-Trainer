from utils.custom_training import CustomTraining
import pandas as pd

# Create an instance of CustomTraining
custom_training = CustomTraining('data.csv')
score = custom_training.model_training()

print("Score: ", str(score))

gpa = custom_training.model_prediction([[1700]])
print("GPA: ", str(gpa[0]))