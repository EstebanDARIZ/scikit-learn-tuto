from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

housing = datasets.fetch_california_housing()

x = housing.data
y = housing.target

# print(x.shape)
poly = PolynomialFeatures()
x = poly.fit_transform(x)
# print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 432)  # test_size = 0.2, on garde 20% du dataset pour le test 


r2_max = 0
lr_best = 0
iter_best = 0


# A loop to find the best combo of hyperparameter (res lr 0.1 , iter_max = 250)
for lr in [0.1, 0.05, 0.001]:   #find a good learning rate 
    for i in [100, 150, 200, 250, 300, 350]:    #nomber of trees generated in the model
        model = HistGradientBoostingRegressor(max_iter = i,
                                              learning_rate=lr)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        if r2 > r2_max:
            r2_max = r2
            lr_best = lr
            iter_best = i

print(iter_best, lr_best, r2_max)  # res 250, 0.1, 0.8483 
















