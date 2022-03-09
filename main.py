from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

clf = GaussianNB()

iris = datasets.load_iris()

print(dir(iris))
print(iris.feature_names)

target = iris.target

data = iris.data

clf.fit(data, target)

result = clf.predict(data)
print(accuracy_score(result, target))
print(confusion_matrix(target, result))
