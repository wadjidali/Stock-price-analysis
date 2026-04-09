import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


print("Téléchargement des données...")
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

print(df.head())


df['MA_20'] = df['Close'].rolling(window=20).mean()
df['MA_50'] = df['Close'].rolling(window=50).mean()

df['Return'] = df['Close'].pct_change()

df['Target'] = df['Close'].shift(-1)

df = df.dropna()


plt.figure(figsize=(10,5))
plt.plot(df['Close'], label='Prix')
plt.plot(df['MA_20'], label='MA 20')
plt.plot(df['MA_50'], label='MA 50')
plt.legend()
plt.title("Analyse technique - Apple")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.show()


X = df[['Close', 'MA_20', 'MA_50', 'Return']]

y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


print("Entraînement du modèle...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("\nRésultats :")
print("Mean Absolute Error :", mae)


plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Valeurs réelles')
plt.plot(predictions, label='Prédictions')
plt.legend()
plt.title("Comparaison Réel vs Prédiction")
plt.xlabel("Temps")
plt.ylabel("Prix")
plt.show()
