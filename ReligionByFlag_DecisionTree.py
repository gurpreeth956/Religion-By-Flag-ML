import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

data = pd.read_csv("flags_data.csv")

#Drop more data later, tree is too large
X = data.drop(['religion', 'topleft', 'botright', 'name', 'mainhue'], axis=1)
y = data['religion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

feature_arr = ['landmass','zone','area','population','language','bars','stripes','colors','red','green','blue','gold','white','black','orange','circles','crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text']
class_arr = ['Catholic', 'Other Christian', 'Muslim', 'Buddhist', 'Hindu', 'Ethnic', 'Marxist', 'Others']

fig, ax = plt.subplots(figsize=(20,20))

#Tree is too large to display properly, change max depth, fontsize and DPI to adjust info/size
tree.plot_tree(classifier, feature_names= feature_arr, class_names= class_arr, ax=ax, filled=True, fontsize = 5) #max_depth=4
#save tree as png if you want to zoom in
plt.savefig('flag_tree', dpi=100)
plt.show()
