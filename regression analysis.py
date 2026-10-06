import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("Dummy Data HSS.csv")

print(data.head())
print(data.dtypes)
print(data.isnull().sum())
print("Duplicate rows:", data.duplicated().sum())

data = data.dropna()
print("Rows after cleaning:", len(data))

# I used a scatter plot because TV and Sales are numeric variables.
# The plot shows a strong positive relationship between TV and Sales
# I removed 26 incomplete rows, and 4,546 complete rows remained
plt.scatter(data["TV"], data["Sales"])
plt.xlabel("TV")
plt.ylabel("Sales")
plt.show()

plt.scatter(data["Radio"], data["Sales"])
plt.xlabel("Radio")
plt.ylabel("Sales")
plt.show()

plt.scatter(data["Social Media"], data["Sales"])
plt.xlabel("Social Media")
plt.ylabel("Sales")
plt.show()

X = data[["TV", "Radio", "Social Media"]]
y = data["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predicted_sales = model.predict(X_test)

r_squared = r2_score(y_test, predicted_sales)
rmse = mean_squared_error(y_test, predicted_sales) ** 0.5

print("R-squared:", r_squared)
print("RMSE:", rmse)

plt.scatter(y_test, predicted_sales)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.show()

errors = y_test - predicted_sales

# This plot checks whether the prediction errors show a clear pattern

plt.scatter(predicted_sales, errors)
plt.axhline(0, color="black")
plt.xlabel("Predicted Sales")
plt.ylabel("Error")
plt.show()
