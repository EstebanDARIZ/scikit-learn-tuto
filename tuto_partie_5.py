from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.preprocessing import PolynomialFeatures
import joblib

housing = datasets.fetch_california_housing()

x = housing.data
y = housing.target

poly = PolynomialFeatures()
x = poly.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 432)  # test_size = 0.2, on garde 20% du dataset pour le test 

local_model = joblib.load("HGBR_model.joblib")  #load the model on the machine
y_pred = local_model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(r2)





