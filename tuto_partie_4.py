from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import joblib

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


model = HistGradientBoostingRegressor(max_iter = 250,
                                    learning_rate=0.1)
model.fit(x_train, y_train)

joblib.dump(model, "HGBR_model.joblib")

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)


















