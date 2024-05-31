import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import wvelet as wve

# Baca data
data = xr.open_dataarray("D:/[2]DATA ANDAT/chirps-v2.0.1981.2015.days_p10.nc")

# Potong data sesuai waktu kajian (minimal 20 tahun) (1986-01-01, 2010-12-31)
data1 = data.sel(time=slice("1986-01-01", "2010-12-31"))

# Potong data menjadi satu titik
ykt1 = data1.sel(lat=-7.8, lon=110.39111111111, method="nearest") # Kota Yogyakarta
jyp1 = data1.sel(lat=-2.592919, lon=140.682325, method="nearest") # Kota Jayapura

# Merata-ratakan dan me-resample data menjadi per bulan
ykt = ykt1.resample(time="1MS").mean() # Resample data menjadi setiap awal bulan
jyp = jyp1.resample(time="1MS").mean() # Resample data menjadi setiap awal bulan

# Normalisasi data
## Kota Yogyakarta
var_ykt = np.std(ykt.values, ddof=1)**2
norm_ykt = (ykt - np.mean(ykt))
## Kota Jayapura
var_jyp = np.std(jyp.values, ddof=1)**2
norm_jyp = (jyp - np.mean(jyp))

# Array waktu
t1 = pd.to_datetime(jyp.time.values)
t2 = t1.strftime("%Y-%m")

# Plot data time series per bulan
## Kota Yogyakarta
fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(211)
ax.plot(t2, norm_ykt, color="r")
ax.set_xlabel("Waktu (Tahun-Bulan)", fontsize=10)
ax.set_ylabel("Curah Hujan\nNormal (mm)", fontsize=10)
ax.set_title("Curah Hujan Normal Per Bulan\nKota Yogyakarta", fontsize=15, weight="bold", pad=8)
ax.set_xticks(t2[19::20], t2[19::20], rotation=30, fontsize=10)
plt. yticks(fontsize=10)
## Kota Jayapura
ax1 = fig.add_subplot(212)
ax1.plot(t2, norm_jyp, color="b")
ax1.set_xlabel("Waktu (Tahun-Bulan)", fontsize=10)
ax1.set_ylabel("Curah Hujan\nNormal (mm)", fontsize=10)
ax1.set_title("Curah Hujan Normal Per Bulan\nKota Jayapura", fontsize=15, weight="bold", pad=8)
ax1.set_xticks(t2[19::20], t2[19::20], rotation=30, fontsize=10)
plt. yticks(fontsize=10)
plt.tight_layout(pad=2)
fig.savefig("Minggu (11)/hasil/ts.png", dpi=300)

# Wavelet
## DOG
### Kota Yogyakarta
wve.wvelet(norm_ykt, var_ykt, t2, "DOG", "Yogyakarta")
### Kota Jayapura
wve.wvelet(norm_jyp, var_jyp, t2, "DOG", "Jayapura")

## PAUL
### Kota Yogyakarta
wve.wvelet(norm_ykt, var_ykt, t2, "PAUL", "Yogyakarta")
### Kota Jayapura
wve.wvelet(norm_jyp, var_jyp, t2, "PAUL", "Jayapura")
