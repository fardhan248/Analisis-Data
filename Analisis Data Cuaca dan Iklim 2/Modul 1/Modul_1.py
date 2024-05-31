import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt
import harmonik as hr

wdir = "C:/Users/acidc/OneDrive - Institut Teknologi Bandung/ITB/Semester 6 (2023 - 2024)/KULIAH/Analisis Data Cuaca dan Iklim II/Minggu (5)/"
data = xr.open_dataarray(wdir + "gsmap.1hr.citarum.nc")

tspwk = data.groupby("time.month").mean().sel(lat=-6.542, lon=107.44, method="nearest")
tsdpk = data.groupby("time.month").mean().sel(lat=-6.403, lon=106.794, method="nearest")

bulan = tspwk.month.values
n = len(bulan)
pwkbar = tspwk.mean().values #Rata-Rata
dpkbar = tsdpk.mean().values # Rata-Rata

# Plot Time Series
plt.figure(figsize=(12, 9))
plt.plot(tspwk, color="green", label="Purwakarta")
plt.plot(tsdpk, color="orange", label="Depok")
plt.title("Time Series Curah Hujan Per Bulan", fontsize=20, weight="bold")
plt.legend(fontsize=15)
plt.grid()
plt.xticks(np.arange(0, 12, 1), bulan, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Bulan", fontsize=15)
plt.ylabel("Curah Hujan (mm/jam)", fontsize=15)
plt.savefig(wdir+"hasil/ts.png", dpi=300)

# Hitung Harmonik
## Purwakarta
hrpwk = hr.hitung_harmonik(tspwk-pwkbar)

## Depok
hrdpk = hr.hitung_harmonik(tsdpk-dpkbar)

# Plot Periodogram
## Purwakarta
sypwk = np.var(tspwk.values - pwkbar, ddof=1) # Variansi
Rpwk = (n*hrpwk.C**2) / (2*(n-1)*sypwk)

## Depok
sydpk = np.var(tsdpk.values - dpkbar, ddof=1) # Variansi
Rdpk = (n*hrdpk.C**2) / (2*(n-1)*sydpk)

plt.figure(figsize=(12, 9))
plt.bar(np.arange(int(n/2))-0.2, Rpwk, color="green", width=0.4, label="Purwakarta")
plt.bar(np.arange(int(n/2))+0.2, Rdpk, color="orange", width=0.4, label="Depok")
plt.title("Periodogram CH Per Bulan di Purwakarta \ndan Depok", fontsize=20, weight="bold")
plt.xlabel("k (Bilangan Gelombang)", fontsize=15)
plt.ylabel(r"$R^2$ (Normalized Amplitude)", fontsize=15)
plt.xticks(ticks=np.arange(int(n/2)), labels=np.arange(1, int(n/2)+1), fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.grid()
plt.savefig(wdir+"hasil/periodogram.png", dpi=300)

# Plot Harmonik Purwakarta
plt.figure(figsize=(12, 9))
plt.plot(tspwk, "*-", label="Time Series", color="black")
plt.plot(pwkbar+hrpwk.ykt[0], color="blue", label="k = 1")
plt.plot(pwkbar+hrpwk.ykt[1], color="purple", label="k = 2")
plt.plot(pwkbar+hrpwk.ykt[2], color="orange", label="k = 3")
plt.plot(pwkbar+hrpwk.ykt[3], color="red", label="k = 4")
plt.title("Harmonik CH Purwakarta Per Bulan", fontsize=20, weight="bold")
plt.xticks(np.arange(0, 12, 1), bulan, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Bulan", fontsize=15)
plt.ylabel("Curah Hujan (mm/jam)", fontsize=15)
plt.legend(fontsize=15)
plt.grid()
plt.savefig(wdir + "hasil/hrpwk.png", dpi=300)

# Plot Harmonik Depok
plt.figure(figsize=(12, 9))
plt.plot(tsdpk, "*-", label="Time Series", color="black")
plt.plot(dpkbar+hrdpk.ykt[0], color="blue", label="k = 1")
plt.plot(dpkbar+hrdpk.ykt[1], color="purple", label="k = 2")
plt.plot(dpkbar+hrdpk.ykt[2], color="orange", label="k = 3")
plt.title("Harmonik CH Depok Per Bulan", fontsize=20, weight="bold")
plt.xticks(np.arange(0, 12, 1), bulan, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Bulan", fontsize=15)
plt.ylabel("Curah Hujan (mm/jam)", fontsize=15)
plt.legend(fontsize=15)
plt.grid()
plt.savefig(wdir + "hasil/hrdpk.png", dpi=300)

# Plot Spasial
## Harmonik seluruh titik
monthly = data.groupby("time.month").mean()
lat = monthly.lat.values
lon = monthly.lon.values
hrmon = np.array(np.zeros((len(lat), len(lon))), dtype="object")
R2 = np.zeros((int(n/2), len(lat), len(lon)))
for i in range(len(lat)):
    for j in range(len(lon)):
        n = len(monthly[:, i, j].month)
        ybar = monthly[:, i, j].mean()
        hrmon[i, j] = hr.hitung_harmonik(monthly[:, i, j] - ybar)
        vars = np.var(monthly[:, i, j] - ybar, ddof=1).values
        R2[:, i, j] = (n/2 *hrmon[i,j].C**2) / ((n-1)*vars)

## Plot 2D
### k = 1
hr.plot2d(R2[0,:,:], lat, lon, "Normalized Amplitude Harmonic k = 1", "Amplitudo", wdir+"hasil/spasial_k1.png", np.linspace(0.5, 1, 20))        
        
### k = 2
hr.plot2d(R2[1,:,:], lat, lon, "Normalized Amplitude Harmonic k = 2", "Amplitudo", wdir+"hasil/spasial_k2.png", np.linspace(0, 0.5, 20))

### k = 3
hr.plot2d(R2[2,:,:], lat, lon, "Normalized Amplitude Harmonic k = 3", "Amplitudo", wdir+"hasil/spasial_k3.png", np.linspace(0, 0.3, 20))
