# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# Import data from csv file
data = pd.read_csv('flags_data.csv', delimiter=',')

# Pick the important attributes
x = data.iloc[:, [1, 5]].values
y = data.iloc[:, 29].values
countries = data.iloc[:, 0].values

# Split into testing and training set
x_train, x_test, y_train, y_test, countries_train, countries_test = train_test_split(x, y, countries, test_size=.3)

# Normalize data before making predictions
# normalize_scale = StandardScaler()
# normalize_scale.fit(x_train)
# x_train = normalize_scale.transform(x_train)
# x_test = normalize_scale.transform(x_test)

# Create x and y ranges for the plot
x_min = x_train[:, 0].min() - 1
x_max = x_train[:, 0].max() + 1
y_min = x_train[:, 1].min() - 1
y_max = x_train[:, 1].max() + 1
x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01))

# Use KNN classifier to predict religion
kNN_class = KNeighborsClassifier(n_neighbors=5, weights='distance')
kNN_class.fit(x_train, y_train)
z = kNN_class.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
z_grid = z.reshape(x_grid.shape)

# Plot the results of the training set
plt.figure(1)
light_Colors = ListedColormap(['#F39D9D', '#93F088', '#67AAF1'])
dark_Colors = ListedColormap(['#F32C2C', '#219912', '#0E67C5'])
plt.pcolormesh(x_grid, y_grid, z_grid, cmap=light_Colors)
plt.scatter(x_train[:, 0], x_train[:, 1], cmap=dark_Colors, c=y_train)
plt.xlim(x_grid.min(), x_grid.max())
plt.ylim(y_grid.min(), y_grid.max())
plt.title('Religion 5-NN Classifier')
plt.xlabel('landmass')
plt.ylabel('language')
red_patch = patches.Patch(color='red', label='Christian')
green_patch = patches.Patch(color='green', label='Islam')
blue_patch = patches.Patch(color='blue', label='Other')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=[red_patch, green_patch, blue_patch])
plt.tight_layout()

# Get prediction results and report
y_prediction = kNN_class.predict(x_test)
print(classification_report(y_test, y_prediction))

# Calculate error for KNN from 1 to 25
errors = []
for i in range(1, 25):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance')
    knn.fit(x_train, y_train)
    y_prediction_i = knn.predict(x_test)
    errors.append(np.mean(y_prediction_i != y_test))

# Plot KNN Classifier error based on K
plt.figure(2)
plt.plot(range(1, 25), errors, color='red', marker='o', markersize=8)
plt.title('K-Value Error Rate')
plt.xlabel('K-Value')
plt.ylabel('Error')
plt.tight_layout()
plt.show()

# Print test results with predictions
np_countries_test = np.array(countries_test)
np_y_test = np.array(y_test)
np_y_prediction = np.array(y_prediction)
results_array = np.column_stack((np_countries_test, np_y_test))
results_array = np.column_stack((results_array, np_y_prediction))
print('Below is the test set [country, actual religion, predicted religion '
      '(1 for Christian, 2 for Islam, 3 for Other: ')
print(results_array)
