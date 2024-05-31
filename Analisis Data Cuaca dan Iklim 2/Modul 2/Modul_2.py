import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq
from scipy.fftpack import fft
import xarray as xr
import pandas as pd
import fourier as fr
import sys
wdir = "C:/Users/acidc/OneDrive - Institut Teknologi Bandung/ITB/Semester 6 (2023 - 2024)/KULIAH/Analisis Data Cuaca dan Iklim II/Minggu (7)/"
sys.path.insert(0, wdir)

# Baca data
data = xr.open_dataarray("D:/[2]DATA ANDAT/MSWEP_MON_INA_197902-202011.nc")
data = data.sel(lat=-6.95, lon=107.6, method="nearest").sel(time=slice("1996-01", "2015-12"))

# Standarisasi data
data_mean = data.mean(skipna=True)
data_std = np.std(data)
data = (data - data_mean)/data_std

# Visualisasi data time series
fig = plt.figure(figsize=(18,9))
ax = fig.add_subplot(111)

bulan = pd.to_datetime(data.time.values)
bulan6 = bulan.strftime("%Y-%m")

ax.plot(bulan, data, color="blue", label="Data standar")
ax.set_xlabel("Tahun-Bulan", fontsize=15)
ax.set_ylabel("CH Standar (mm/bulan)", fontsize=15)
ax.set_title("Time Series Data Standar Tahun 1996 - 2015\n(-6.95, 107.6 / Bandung)", fontsize=20, weight="bold")
ax.set_xticks(bulan[::6], bulan6[::6], rotation=45, fontsize=10)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.grid()
fig.savefig(wdir + "hasil/ts.png", dpi=300)

# FFT Time Series
n = len(bulan) # Panjang data = 240
my = np.arange(1, n+1, 1) # Month of year
## Jarak antar data
s = my[1] - my[0] # 1 bulan

## FFT
F = fft(data.values)

## Periode
T = n/np.arange(1, n/2) # T (bulan) = n/k 

## Frekuensi sampel
f = fftfreq(n, s) # per bulan

F = abs(F[np.where(f > 0)])
f = abs(f[np.where(f > 0)])

## Plot FFT
fig = plt.figure(figsize=(15,13))
ax = fig.add_subplot(211)

ax.plot(f, F, color="blue", label="FFT")
ax.set_ylabel("Amplitudo", fontsize=15)
ax.set_title("Periodogram FFT", fontsize=20, weight="bold")
ax.set_xticks(f[::10])
ax.set_xlabel("Frekuensi ($Bulan^{-1}$)", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.grid()

ax1 = fig.add_subplot(212)

ax1.semilogx(T, F, color="blue", label="FFT")
ax1.set_ylabel("Amplitudo", fontsize=15)
ax1.set_title("Periodogram FFT", fontsize=20, weight="bold")
ax1.set_xticks(np.arange(12, 12*(10+1), 12), np.arange(1, 10+1, 1))
ax1.set_xlabel("Periode (Tahun)", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.grid()
fig.savefig(wdir + "hasil/periodogram.png", dpi=300)

# Filtering Data
## Lowpass
F_low = fr.filter(1, 1/(12*5), 6, data, "low", f)
fr.filterplot(T, F, F_low, 1/(12*5), "Lowpass", True, wdir + "hasil/periodogram_low.png")

## Highpass
F_high = fr.filter(1, 1/(12*4), 6, data, "high", f)
fr.filterplot(T, F, F_high, 1/(12*4), "Highpass", True, wdir + "hasil/periodogram_high.png")

## Bandpass
cutoff = np.array([1/(12*7), 1/(12*3)])
F_band = fr.filter(1, cutoff, 6, data, "bandpass", f)
fr.filterplot(T, F, F_band, cutoff, "Bandpass", True, wdir + "hasil/periodogram_band.png")
