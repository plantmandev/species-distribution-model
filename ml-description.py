import pandas as pd

# Load the training data
training_data = pd.read_csv('vanessa-atalanta_training_data.csv')

print("Training data summary:")
print(training_data.describe())

print("\nPresences vs Background:")
print(training_data['presence'].value_counts())

print("\nLand cover values:")
print(training_data['landcover'].value_counts())

print("\nClimate ranges by presence:")
print(training_data.groupby('presence')[['tmax', 'tmin', 'ppt']].describe())