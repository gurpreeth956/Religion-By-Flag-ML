import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("flags_data.csv")

# Drop more data later, tree is too large
#'landmass''language','crescent','crosses', 'sunstars'
x = data.drop(['mainhue', 'green', 'religion', 'topleft', 'botright', 'name','zone','area','population','bars','stripes','colors','red','blue','gold','white','black','orange','circles','saltires','quarters','triangle','icon','animate','text'], axis=1)
y = data['religion']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
feature_arr = ['landmass', 'language', 'crosses', 'sunstars', 'crescent']
class_arr = ['Christian', 'Muslim', 'Other']

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)
fig, ax = plt.subplots(figsize=(10, 10))

# Tree is too large to display properly, change max depth, fontsize and DPI to adjust info/size
tree.plot_tree(classifier, feature_names=feature_arr, class_names=class_arr, ax=ax, filled=True, fontsize=5, max_depth=4) #max_depth=4
# Save tree as png if you want to zoom in
plt.savefig('flag_tree', dpi=100)
plt.tight_layout()
plt.show()
