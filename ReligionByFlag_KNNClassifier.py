# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler


# Import data from csv file
data = pd.read_csv('flags_data.csv')

# Pick the important attributes
x = data[['landmass', 'language']].values
y = data['religion'].values
countries = data['name'].values

# Split into testing and training set
x_train, x_test, y_train, y_test, countries_train, countries_test = train_test_split(x, y, countries, test_size=.3)

# Normalize data if you need to
# normalize_scale = StandardScaler().fit(x_train)
# x_train = normalize_scale.transform(x_train)
# x_test = normalize_scale.transform(x_test)

# Create x and y ranges for the plot
x_min = x_train[:, 0].min() - 1
x_max = x_train[:, 0].max() + 1
y_min = x_train[:, 1].min() - 1
y_max = x_train[:, 1].max() + 1
x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01))

# Use KNN classifier to predict religion
knn_class = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_class.fit(x_train, y_train)

# Plot the results of the training set
plt.figure(1)
light_Colors = ListedColormap(['#F39D9D', '#93F088', '#67AAF1'])
dark_Colors = ListedColormap(['#F32C2C', '#219912', '#0E67C5'])
z = knn_class.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
z_grid = z.reshape(x_grid.shape)
plt.pcolormesh(x_grid, y_grid, z_grid, cmap=light_Colors)
plt.scatter(x_train[:, 0], x_train[:, 1], cmap=dark_Colors, c=y_train)
plt.xlim(x_grid.min(), x_grid.max())
plt.ylim(y_grid.min(), y_grid.max())
plt.title('Religion 5-NN Classifier (Landmass and Language)')
plt.xlabel('landmass')
plt.ylabel('language')
red_patch = patches.Patch(color='red', label='Christian')
green_patch = patches.Patch(color='green', label='Islam')
blue_patch = patches.Patch(color='blue', label='Other')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=[red_patch, green_patch, blue_patch])
plt.tight_layout()

# Get prediction results and report
y_prediction = knn_class.predict(x_test)
print(classification_report(y_test, y_prediction))

# Get accuracy of total prediction
y_total = len(y_test)
y_correct = 0
for i in range(0, y_total):
    if y_test[i] == y_prediction[i]:
        y_correct += 1
y_accuracy = y_correct/y_total

# Print test results with predictions
np_countries_test = np.array(countries_test)
np_y_test = np.array(y_test)
np_y_prediction = np.array(y_prediction)
results_array = np.column_stack((np_countries_test, np_y_test))
results_array = np.column_stack((results_array, np_y_prediction))
print('Below is the test set [country, actual religion, predicted religion '
      '(1 for Christian, 2 for Islam, 3 for Other: ')
print(results_array, '\n')
print('The KNN Classifier predicted', y_correct, 'out of', y_total, 'correctly, meaning the accuracy is:', y_accuracy)
print('')

# Calculate error for KNN from 1 to 25
errors = []
for i in range(1, 25):
    knn_class_i = KNeighborsClassifier(n_neighbors=i, weights='distance')
    knn_class_i.fit(x_train, y_train)
    y_prediction_i = knn_class_i.predict(x_test)
    errors.append(np.mean(y_prediction_i != y_test))

# Plot KNN Classifier error based on K
plt.figure(2)
plt.plot(range(1, 25), errors, color='red', marker='o', markersize=8)
plt.title('K-Value Error Rate')
plt.xlabel('K-Value')
plt.ylabel('Error')
plt.tight_layout()

# Do a 5-fold cross validation
knn_cross_valid = KNeighborsClassifier(n_neighbors=5, weights='distance')
cross_val_score = cross_val_score(knn_cross_valid, x, y, cv=5)
cross_val_mean = np.mean(cross_val_score)
print('5-fold Cross Validation Accuracy:', cross_val_score)
print('Cross Validation Mean:{}', cross_val_mean)

# Show plots after everything
plt.show()
