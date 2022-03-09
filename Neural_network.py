from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
target = iris.target
data = iris.data

data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=0, train_size=0.7)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(data_train, target_train)

result = clf.predict(data_test)
print(result)
print(accuracy_score(result, target_test))