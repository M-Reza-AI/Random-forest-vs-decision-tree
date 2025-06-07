from Helper.AI_Helper import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

Iris_dataset = pd.read_csv('iris.csv')

y = Iris_dataset['species']
x = Iris_dataset.drop('species', axis=1)

x_train , x_val , train_y , val_y = train_test_split(x, y, test_size=0.2, random_state=0)

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(x_train, train_y)

forest = RandomForestClassifier(n_estimators=5)
forest.fit(x_train, train_y)

tree_y_pred = tree.predict(x_val)
forest_y_pred = forest.predict(x_val)

tree_accuracy = accuracy_score(val_y, tree_y_pred)
forest_accuracy = accuracy_score(val_y, forest_y_pred)

print("The tree accuracy is: " ,tree_accuracy)
print("The forest accuracy is: " ,forest_accuracy)