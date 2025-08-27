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

LR = LinearRegression()
RFR = RandomForestRegressor(n_jobs = -1)
HGBR = HistGradientBoostingRegressor()

for i in [LR, RFR, HGBR]:
    i.fit(x_train, y_train)
    y_pred = i.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(i)
    print(r2)
















