from utils.custom_training import CustomTraining

# RandomForestRegressor
custom_training = CustomTraining('continuous_dataset.csv', ['SAT'], 'GPA')
score = custom_training.train_random_forest_regressor()

print("Score: ", str(score))

gpa = custom_training.model_prediction([[1700]])
print("GPA: ", str(gpa[0]))

custom_training.compare_model_accuracies()