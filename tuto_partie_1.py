from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

housing = datasets.fetch_california_housing()

x = housing.data
y = housing.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 432)  # test_size = 0.2, on garde 20% du dataset pour le test 

model = LinearRegression()

model.fit(x_train, y_train)   # training 

y_pred = model.predict(x_test)  #testing 
r2 = r2_score(y_test, y_pred)


# print(housing.feature_names)
# print(x[0])
# print(y[0])
print(r2)















