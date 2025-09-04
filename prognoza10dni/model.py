import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error



df = pd.read_csv('DaneBezSierpnia.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df = df.sort_values('timestamp')

daneTestowe = ['temperatura', 'radiacja', 'Vwiatru', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
target = ['EnergiaPozyskana']

df = df.dropna(subset=daneTestowe + target)

X = df[daneTestowe]
y = df[target].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_Predykcja = model.predict(X_test)

mse = mean_squared_error(y_test, y_Predykcja)
print(f"Mean Squared Error: {mse:.2f}")
mae = mean_absolute_error(y_test, y_Predykcja)
print(f"Mean absolute Error: {mae:.2f}")



latitude = 54.29808
longitude = 17.20368

response = requests.get("https://api.open-meteo.com/v1/forecast", params={
    "latitude": latitude,
    "longitude": longitude,
    "hourly": "temperature_2m,windspeed_10m,shortwave_radiation",
    "timezone": "UTC",
    "forecast_days": 10
})

data = response.json()

prognozaPogody = pd.DataFrame({
    'timestamp': pd.to_datetime(data['hourly']['time']).tz_localize('UTC'),
    'temperatura': data['hourly']['temperature_2m'],
    'Vwiatru': data['hourly']['windspeed_10m'],
    'radiacja': data['hourly']['shortwave_radiation']
})


prognozaPogody['Vwiatru'] = prognozaPogody['Vwiatru'] * (5/18)


prognozaPogody = prognozaPogody.set_index('timestamp')


idx = pd.date_range(start=prognozaPogody.index.min(), end=prognozaPogody.index.max(), freq='15min', tz='UTC')
prognozaPogody_15min = prognozaPogody.reindex(idx).interpolate(method='time').reset_index().rename(columns={'index': 'timestamp'})


prognozaPogody_15min['hour'] = prognozaPogody_15min['timestamp'].dt.hour
prognozaPogody_15min['month'] = prognozaPogody_15min['timestamp'].dt.month
prognozaPogody_15min['hour_sin'] = np.sin(2 * np.pi * prognozaPogody_15min['hour'] / 24)
prognozaPogody_15min['hour_cos'] = np.cos(2 * np.pi * prognozaPogody_15min['hour'] / 24)
prognozaPogody_15min['month_sin'] = np.sin(2 * np.pi * prognozaPogody_15min['month'] / 12)
prognozaPogody_15min['month_cos'] = np.cos(2 * np.pi * prognozaPogody_15min['month'] / 12)

prognozaPogody_15min_10d = prognozaPogody_15min.head(960).copy()


X_pogoda_15min = prognozaPogody_15min_10d[daneTestowe]
przewidywanaEnergia_15min = model.predict(X_pogoda_15min)
prognozaPogody_15min_10d['EnergiaWytworzona'] = przewidywanaEnergia_15min


prognozaPogody_15min_10d.set_index('timestamp')['EnergiaWytworzona'].plot(figsize=(12, 4), title='Prognozowana produkcja energii co 15 minut (10 dni)')
plt.ylabel('Energia (kWh)')
plt.grid(True)
plt.show()


prognozaPogody_15min_10d['date'] = prognozaPogody_15min_10d['timestamp'].dt.date
suma_dzienna_15min = prognozaPogody_15min_10d.groupby('date')['EnergiaWytworzona'].sum().reset_index()

print("Suma energii wytworzonej na każdy dzień (10 dni):")
print(suma_dzienna_15min)

#slupki
plt.figure(figsize=(10, 5))
plt.bar(suma_dzienna_15min['date'].astype(str), suma_dzienna_15min['EnergiaWytworzona'])
plt.xticks(rotation=45)
plt.title("Prognozowana suma energii dziennie (10 dni)")
plt.ylabel("Energia (kWh)")
plt.grid(axis='y')
plt.show()


suma_dzienna_15min.to_csv('PrognozaEnergii_SumaDzienna_10dni.csv', index=False)

prognozaPogody_15min_10d_ToSave = prognozaPogody_15min_10d.drop(columns=['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'hour', 'month', 'date'])
prognozaPogody_15min_10d_ToSave.to_csv('PrognozaEnergii_15min_10dni.csv', index=False)
