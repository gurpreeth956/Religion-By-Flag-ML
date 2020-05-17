# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Import data from csv file
data = pd.read_csv("flags_data.csv")

# Pick the important attributes
x = data[['landmass', 'language', 'crescent', 'crosses', 'sunstars']].values
y = data['religion'].values
countries = data['name'].values

# Split into testing and training set
x_train, x_test, y_train, y_test, countries_train, countries_test = train_test_split(x, y, countries, test_size=.3)

# Create decision tree classifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(x_train, y_train)
y_prediction = classifier.predict(x_test)
features_arr = ['landmass', 'language', 'crosses', 'sunstars', 'crescent']
classes_arr = ['Christian', 'Muslim', 'Other']

# Compute accuracy of the decision tree
accuracy = accuracy_score(y_test, y_prediction)
print('Accuracy of decision tree: ', accuracy, '\n')
fig, ax = plt.subplots(figsize=(30, 30))

# Get accuracy of total prediction
y_total = len(y_test)
y_correct = 0
for i in range(0, y_total):
    if y_test[i] == y_prediction[i]:
        y_correct += 1

# Print test results with predictions
np_countries_test = np.array(countries_test)
np_y_test = np.array(y_test)
np_y_prediction = np.array(y_prediction)
results_array = np.column_stack((np_countries_test, np_y_test))
results_array = np.column_stack((results_array, np_y_prediction))
print('Below is the test set of the country, actual religion, predicted religion '
      '(1 for Christian, 2 for Islam, 3 for Other): ')
print(results_array, '\n')
print('The Decision Tree predicted', y_correct, 'out of', y_total, 'correctly, meaning the accuracy is:', accuracy)

# Plot decision tree graph (change estimator index for which tree you want to show)
tree.plot_tree(classifier.estimators_[0], feature_names=features_arr, class_names=classes_arr, filled=True,
               fontsize=8, ax=ax)

# Save tree as png to view decision tree (zoom in)
plt.savefig('religion_decision_tree', dpi=100)
plt.tight_layout()
#plt.show()
