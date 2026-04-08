import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

plt.figure()
plt.plot(df['Close'])
plt.title("Prix de clôture Apple")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.show()

df['Target'] = df['Close'].shift(-1)
df = df.dropna()

X = df[['Close']]
y = df['Target']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Exemple prédiction :", predictions[:5])

df['MA_20'] = df['Close'].rolling(window=20).mean()
