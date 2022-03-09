from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
target = iris.target
data = iris.data

data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=0, train_size=0.7)

reg = linear_model.LinearRegression()
reg.fit(data_train, target_train)

print(reg.coef_)

result = reg.predict(data_test)


