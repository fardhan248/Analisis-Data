import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import matlib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Baca data
indir = "Minggu (12)/"
data = pd.read_excel(indir+"Data TMA RAW.xlsx", sheet_name="Nanjung")

# Pre-processing data
## Baca data csv
data1 = data.set_index("Data", drop=True) # Mengubah index menjadi tanggal
data2 = data1.mean(axis=1) # Merata-ratakan semua kolom
tma = data2["2018-02-01":"2018-03-31"] # Slice data
tma = tma.sort_index() # Mengurutkan data berdasarkan indeks tanggal 
if tma.isna().any() == True: # Mengecek data nan
    tma = tma.dropna() # Drop data nan

## Plot data time series
t2 = tma.index.strftime("%m-%d")
fig, ax = plt.subplots(figsize=(12,10))
ax.plot(t2, tma, color="blue")
ax.plot(t2, np.squeeze((matlib.repmat(tma.mean(), 1, tma.index.size))))
ax.set_xlabel("Waktu (Bulan-tanggal)", fontsize=15)
ax.set_ylabel("TMA (m)", fontsize=15)
ax.set_title("Deret Waktu TMA Nanjung (2018-02-01 s.d. 2018-03-31)", fontsize=20, weight="bold")
ax.set_xticks(t2[::5], t2[::5], rotation=30, fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.tight_layout()
fig.savefig(indir+"hasil/ts.png")

## Bagi data menjadi train dan test data
train, test = tma[:"2018-03-24"], tma["2018-03-25":]

# Plot ACF dan PACF
## Plot ACF
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14,10))
plot_acf(train, ax=ax[0], lags=12, auto_ylims=True)
ax[0].set_xlabel("Lag", fontsize=15)
ax[0].set_ylabel("Auto-korelasi", fontsize=15)
ax[0].set_title("Auto-korelasi", fontsize=20, weight="bold")
ax[0].tick_params(labelsize=15)
## Plot PACF
plot_pacf(train, ax=ax[1], lags=12, auto_ylims=True)
ax[1].set_xlabel("Lag", fontsize=15)
ax[1].set_ylabel("Parsial Auto-korelasi", fontsize=15)
ax[1].set_title("Parsial Auto-korelasi", fontsize=20, weight="bold")
ax[1].tick_params(labelsize=15)
plt.tight_layout()
fig.savefig(indir+"hasil/acf_pacf.png", dpi=300)
# Berdasarkan hasil plot ACF, data sudah stasioner

# fit model ARIMA
mod1 = ARIMA(train.values, order=(1,0,0))
param1 = mod1.fit() # Parameter fungsi model

mod2 = ARIMA(train.values, order=(0,0,1))
param2 = mod2.fit() # Parameter fungsi model

mod3 = ARIMA(train.values, order=(0,0,2))
param3 = mod3.fit() # Parameter fungsi model

mod4 = ARIMA(train.values, order=(0,0,3))
param4 = mod4.fit() # Parameter fungsi model

mod5 = ARIMA(train.values, order=(1,0,1))
param5 = mod5.fit() # Parameter fungsi model

mod6 = ARIMA(train.values, order=(1,0,2))
param6 = mod6.fit() # Parameter fungsi model

mod7 = ARIMA(train.values, order=(1,0,3))
param7 = mod7.fit() # Parameter fungsi model

# Prediksi model dengan fungsi .predict
prediksi1 = param1.predict(start=len(train)+1, end=len(train)+len(test))
prediksi2 = param2.predict(start=len(train)+1, end=len(train)+len(test))
prediksi3 = param3.predict(start=len(train)+1, end=len(train)+len(test))
prediksi4 = param4.predict(start=len(train)+1, end=len(train)+len(test))
prediksi5 = param5.predict(start=len(train)+1, end=len(train)+len(test))
prediksi6 = param6.predict(start=len(train)+1, end=len(train)+len(test))
prediksi7 = param7.predict(start=len(train)+1, end=len(train)+len(test))

# Prediksi model dengan fungsi .forecast (rolling)
def rolling(train_data, test_data, order):
    train1 = train_data.copy().tolist()
    prediksi11 = []
    for i in range(len(test_data)):
        model11 = ARIMA(train1, order=order)
        param11 = model11.fit()
        fcst = param11.forecast()
        prediksi11.append(fcst[0])
        train1.append(test_data.values[i])
    return prediksi11

rol1 = rolling(train, test, (1,0,0))
rol2 = rolling(train, test, (0,0,1))
rol3 = rolling(train, test, (0,0,2))
rol4 = rolling(train, test, (0,0,3))
rol5 = rolling(train, test, (1,0,1))
rol6 = rolling(train, test, (1,0,2))
rol7 = rolling(train, test, (1,0,3))

# Plot prediksi dengan fungsi .predict
t2 = test.index.strftime("%m-%d")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(28,10), tight_layout=True)
ax[0].plot(t2, test, color="black", label="Observasi") # Observasi
ax[0].plot(t2, prediksi1, color="blue", label="Model 1") # Model 1, ARIMA(1,0,0)
ax[0].plot(t2, prediksi2, color="violet", label="Model 2") # Model 2, ARIMA(0,0,1)
ax[0].plot(t2, prediksi3, color="purple", label="Model 3") # Model 3, ARIMA(0,0,2)
ax[0].plot(t2, prediksi4, color="green", label="Model 4") # Model 4, ARIMA(0,0,3)
ax[0].plot(t2, prediksi5, color="cyan", label="Model 5") # Model 5, ARIMA(1,0,1)
ax[0].plot(t2, prediksi6, color="magenta", label="Model 6") # Model 6, ARIMA(1,0,2)
ax[0].plot(t2, prediksi7, color="orange", label="Model 7") # Model 7, ARIMA(1,0,3)
ax[0].set_xlabel("Waktu (Bulan-tanggal)", fontsize=18)
ax[0].set_ylabel("TMA (m)", fontsize=18)
ax[0].set_title("Hasil Prediksi Model\nTMA Nanjung (2018-02-01 s.d. 2018-03-31)", weight="bold", fontsize=20)
ax[0].tick_params(labelsize=18)
ax[0].grid()
ax[0].legend(fontsize=18)

# Plot prediksi dengan fungsi .forecast (rolling)
ax[1].plot(t2, test, color="black", label="Observasi") # Observasi
ax[1].plot(t2, rol1, color="blue", label="Model 1") # Model 1, ARIMA(1,0,0)
ax[1].plot(t2, rol2, color="violet", label="Model 2") # Model 2, ARIMA(0,0,1)
ax[1].plot(t2, rol3, color="purple", label="Model 3") # Model 3, ARIMA(0,0,2)
ax[1].plot(t2, rol4, color="green", label="Model 4") # Model 4, ARIMA(0,0,3)
ax[1].plot(t2, rol5, color="cyan", label="Model 5") # Model 5, ARIMA(1,0,1)
ax[1].plot(t2, rol6, color="magenta", label="Model 6") # Model 6, ARIMA(1,0,2)
ax[1].plot(t2, rol7, color="orange", label="Model 7") # Model 7, ARIMA(1,0,3)
ax[1].set_xlabel("Waktu (Bulan-tanggal)", fontsize=18)
ax[1].set_ylabel("TMA (m)", fontsize=18)
ax[1].set_title("Hasil Prediksi Rolling Model\nTMA Nanjung (2018-02-01 s.d. 2018-03-31)", weight="bold", fontsize=20)
ax[1].tick_params(labelsize=18)
ax[1].grid()
ax[1].legend(fontsize=18)
fig.savefig(indir+"hasil/hasil.png", dpi=300, bbox_inches="tight")

# Evaluasi model
rmse = {"Model 1": np.sqrt(mean_squared_error(test, prediksi1)), "Model 2": np.sqrt(mean_squared_error(test, prediksi2)),
        "Model 3": np.sqrt(mean_squared_error(test, prediksi3)), "Model 4": np.sqrt(mean_squared_error(test, prediksi4)),
        "Model 5": np.sqrt(mean_squared_error(test, prediksi5)), "Model 6": np.sqrt(mean_squared_error(test, prediksi6)),
        "Model 7": np.sqrt(mean_squared_error(test, prediksi7))} # Dict RMSE model
rmse_rol = {"Model 1": np.sqrt(mean_squared_error(test, rol1)), "Model 2": np.sqrt(mean_squared_error(test, rol2)),
            "Model 3": np.sqrt(mean_squared_error(test, rol3)), "Model 4": np.sqrt(mean_squared_error(test, rol4)),
            "Model 5": np.sqrt(mean_squared_error(test, rol5)), "Model 6": np.sqrt(mean_squared_error(test, rol6)),
            "Model 7": np.sqrt(mean_squared_error(test, rol7))} # Dict RMSE model
## Plot RMSE
fig, ax = plt.subplots(figsize=(10,8))
ax.bar(np.arange(len(rmse))-0.2, rmse.values(), width=0.4, color="blue", label="RMSE Biasa")
ax.bar(np.arange(len(rmse))+0.2, rmse_rol.values(), width=0.4, color="cyan", label="RMSE Rolling")
ax.set_ylabel("RMSE (m)", fontsize=15)
ax.set_title("RMSE Setiap Model", fontsize=20, weight="bold")
ax.set_yticks(np.arange(0, 0.7, 0.05))
ax.set_xticks(np.arange(len(rmse)), rmse.keys())
ax.tick_params(labelsize=15)
ax.grid(axis="y")
ax.legend()
fig.savefig(indir+"hasil/rmse.png", dpi=300)
