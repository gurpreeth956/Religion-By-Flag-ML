# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches


# Import data from csv file
data = pd.read_csv('flags_data.csv', delimiter=',')

# Pick the important attributes
x = data.iloc[:, [3, 4]].values
y = data.iloc[:, 29].values

# Split into testing and training set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

# Normalize data before making predictions
normalize_scale = StandardScaler()
normalize_scale.fit(x_train)
x_train = normalize_scale.transform(x_train)
x_test = normalize_scale.transform(x_test)

# Create x and y ranges for the plot
x_min = x_train[:, 0].min() - 1
x_max = x_train[:, 0].max() + 1
y_min = x_train[:, 1].min() - 1
y_max = x_train[:, 1].max() + 1
x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01))

# Use KNN classifier to predict religion
neighbors = 5
kNN_class = KNeighborsClassifier(neighbors, weights='distance')
kNN_class.fit(x_train, y_train)
z = kNN_class.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
z_grid = z.reshape(x_grid.shape)

# Plot the results of the training set
plt.figure()
light_Colors = ListedColormap(['#F39D9D', '#93F088', '#67AAF1'])
dark_Colors = ListedColormap(['#F32C2C', '#219912', '#0E67C5'])
plt.pcolormesh(x_grid, y_grid, z_grid, cmap=light_Colors)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=dark_Colors)
plt.xlim(x_grid.min(), x_grid.max())
plt.ylim(y_grid.min(), y_grid.max())
plt.title('Religion 5-NN Classifier')
plt.xlabel('area')
plt.ylabel('population')
red_patch = patches.Patch(color='red', label='Christian')
green_patch = patches.Patch(color='green', label='Islam')
blue_patch = patches.Patch(color='blue', label='Other')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=[red_patch, green_patch, blue_patch])
plt.tight_layout()
plt.show()

