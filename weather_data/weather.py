import pandas as pd
df = pd.read_csv("seattle-weather.csv")
df.describe()
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("seattle-weather.csv")
df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
x = df[["day", "month", "year", "wind"]]
y = df["weather"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2)
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", accuracy)
input_date = input("Enter date (YYYY-MM-DD): ")
input_wind = float(input("Enter wind speed: "))
date_obj = pd.to_datetime(input_date)
day = date_obj.day
month = date_obj.month
year = date_obj.year
input_data = pd.DataFrame([[day, month, year, input_wind]], columns=["day", "month", "year", "wind"])
pred_encoded = model.predict(input_data)
pred_weather = le.inverse_transform(pred_encoded)

print("Predicted Weather:", pred_weather[0])
import joblib
joblib.dump(model, "weather_model.pkl")

