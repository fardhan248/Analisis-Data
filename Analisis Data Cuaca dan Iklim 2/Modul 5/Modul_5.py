import xarray as xr
import pandas as pd
import numpy as np
import sklearn.decomposition as skldec
import matplotlib.pyplot as plt
import cartopy as crt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import seaborn as sns
sns.set_theme(context="talk", font="georgia", palette="muted")

indir_save = "Minggu (14)/tugas/hasil/"
# Baca data
indir_data = "D:/[2]DATA ANDAT/"
temp = xr.open_dataset(indir_data+"sst.mon.mean.nc")
temp = temp.sst
waktu0 = temp.time.values
waktu_str = pd.to_datetime(waktu0).strftime("%Y-%m")

# DETREND, STANDARISASI
# Cek trend
temp_mean = temp.where(temp > 0).mean(dim=["lon", "lat"])
param_trend = np.polyfit(np.arange(0,len(temp_mean)), temp_mean.values, deg=1)
temp_cek = np.poly1d(param_trend)

## Plot trend
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(waktu0, temp_mean, color="blue")
ax.plot(waktu0, temp_cek(np.arange(0,len(waktu0))), color="orange", label=r"$y={:.2e}x+{:.3f}$".format(param_trend[0], param_trend[1]))
ax.set_title("SST Mean", fontsize=25, weight="bold")
ax.set_xlabel("Waktu", fontsize=20)
ax.set_ylabel(r"SST ($^\circ C$)", fontsize=20)
ax.grid(True)
ax.legend(fontsize=15)
ax.tick_params(labelsize=20)
fig.savefig(indir_save+"tren.png", dpi=300, bbox_inches="tight")

# Potong wilayah dan waktu
temp_hindia = temp.where(temp>0).sel(lat=slice(-25,30), lon=slice(30,120)) 
temp_hindia = temp_hindia.sel(time=slice("1991-01", "2020-12"))
waktu = temp_hindia.time.values
waktu_str = pd.to_datetime(waktu).strftime("%Y-%m")

# Detrend
param = temp_hindia.polyfit(dim="time", deg=1, skipna=True).polyfit_coefficients
fitting = xr.polyval(temp_hindia["time"], param)
temp_hindia = temp_hindia - fitting

## Plot detrend
temp_mean = temp_hindia.mean(dim=["lon", "lat"])
param_trend = np.polyfit(np.arange(0,len(temp_mean)), temp_mean.values, deg=1)
temp_cek = np.poly1d(param_trend)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(waktu, temp_mean, color="blue")
ax.plot(waktu, temp_cek(np.arange(0,len(waktu))), color="orange", label=r"$y={:.2e}x+{:.3f}$".format(param_trend[0], param_trend[1]))
ax.set_title("SST Mean Samudra Hindia (Detrend)", fontsize=25, weight="bold")
ax.set_xlabel("Waktu", fontsize=20)
ax.set_ylabel(r"SST ($^\circ C$)", fontsize=20)
ax.grid(True)
ax.legend(fontsize=15)
ax.tick_params(labelsize=20)
fig.savefig(indir_save+"detren.png", dpi=300, bbox_inches="tight")

# Standarisasi SST
temp_mean = temp_hindia.groupby("time.month").mean()
temp_std = temp_hindia.groupby("time.month").std()
temp_anom = temp_hindia.copy()
for i in range(12):
    temp_anom = temp_anom.where(temp_hindia["time.month"] != i+1, (temp_hindia - temp_mean.isel(month=i)) / temp_std.isel(month=i))

# Konversi data menjadi dua dimensi
lon = temp_anom.lon
lat = temp_anom.lat

temp_reshape = np.reshape(temp_anom.values, newshape=(len(waktu), len(lon)*len(lat)))  # Reshape data
kolom = pd.MultiIndex.from_product([temp_anom.lat.values, temp_anom.lon.values], names=["latitude", "longitude"]) # Membuat multiindex
temp_reshape = pd.DataFrame(temp_reshape, columns=kolom, index=waktu)  # Membuat dataframe
temp_nan = temp_reshape.isna().any()  # Menyimpan nilai nan
temp_clean = temp_reshape.loc[:, ~temp_nan].T  # Transpose

# PCA
matriks = temp_clean.copy() # Copy data yang sudah dihilangkan nilai nan-nya
## Hitung PCA
pca = skldec.PCA()
pca.fit(matriks)
evec = pca.components_  # Vektor Eigen
eval = pca.explained_variance_  # Nilai Eigen
persen = pca.explained_variance_ratio_ * 100  # Nilai Eigen dalam bentuk persen rasio
proyeksi = pca.transform(matriks)  # Proyeksi data ke basis vektor eigen. Nilai PC
kolom = ["PC-" + str(i) for i in range(1, len(matriks.columns)+1)]
proyeksi = pd.DataFrame(proyeksi, index=matriks.index, columns=kolom)

## Plot PCA
fig, ax = plt.subplots(figsize=(11,7))
ax.plot(kolom[:12], eval[:12], color="orange", marker="o")
ax.set_title("Scree Plot (PC1 - PC12)", fontsize=25, weight="bold")
ax.set_ylabel("Eigen Values", fontsize=15)
ax.tick_params(labelsize=15)
ax.grid(True)
fig.savefig(indir_save+"scree.png", dpi=300, bbox_inches="tight")
plt.clf()

pair = sns.pairplot(proyeksi.iloc[:,:4], plot_kws={"s":5}, diag_kind="kde")
plt.suptitle("Pairplot PC1 - PC4", y = 1.02, fontsize=25, weight="bold")
pair.savefig(indir_save+"pairplot.png", dpi=300, bbox_inches="tight")

## Ubah hasil proyeksi menjadi data tiga dimensi
proyeksi2 = temp_reshape.copy().values
proyeksi2[:, ~temp_nan] = proyeksi.T.values
proyeksi2 = np.reshape(proyeksi2, newshape=(len(proyeksi.columns), len(lat), len(lon)))
temp_pc = xr.Dataset(data_vars={"proyeksi": (["PC", "lat", "lon"], proyeksi2), "vektoreigen": (["waktu", "PC"], evec), "nilaieigen": (["PC"], eval), "eigen_persen": (["PC"], persen)},
                     coords={"PC": np.arange(1, len(proyeksi.columns)+1), "lat": lat.values, "lon": lon.values, "waktu": waktu})  # Dataset hasil PCA

indeks1 = pd.read_excel('D:/[2]DATA ANDAT/Indices.xlsx',index_col='Time',na_values=[-999.9, -9999])
indeks = indeks1.loc[(indeks1.index >= "1991-01-01") & (indeks1.index <="2020-12-01")]

## Plot PCA
for i in range(1,6+1):
    mode = temp_pc.proyeksi.sel(PC = i)
    temporal = temp_pc.vektoreigen.sel(PC = i)
    varians = temp_pc.eigen_persen.sel(PC = i)

    # Spasial
    fig = plt.figure(figsize=(15, 14))
    ax1 = plt.subplot2grid((4,1), (0,0), rowspan=3, fig=fig, projection=ccrs.Mercator())
    peta = mode.plot.contourf(ax=ax1, cmap = "RdYlBu_r", levels=25, transform=ccrs.PlateCarree(), add_colorbar=False)
    cbar = plt.colorbar(peta, shrink = 0.7, pad = 0.03)
    cbar.set_label("Magnitudo", fontsize=30)
    cbar.ax.tick_params(labelsize=30)
    ax1.set_title("Pola Spasial PC-" + str(i), fontsize=33)
    ax1.coastlines(lw=2, color="k", zorder=2)
    ax1.add_feature(cfeature.LAND, facecolor="white", zorder=0)
    gl = ax1.gridlines(draw_labels=True, linewidth=1, color="grey", alpha=0.5)
    gl.top_labels=False
    gl.right_labels=False
    gl.left_labels=True
    gl.bottom_labels=True
    gl.xlabel_style = {"size":30}
    gl.ylabel_style = {"size":30}

    # Temporal
    ax2 = plt.subplot2grid((4,1), (3,0), rowspan=1, fig=fig)
    ax3 = ax2.twinx()
    ax3.plot(indeks.index, indeks["DMI"].values,ls="--", label="DMI")
    ax2.plot(waktu, temporal, color="black", label="PC-"+str(i))
    ax2.set_title("Pola Temporal PC-" + str(i), fontsize=33)
    ax2.set_xlabel("Waktu", fontsize=30)
    ax2.set_ylabel("Magnitudo", fontsize=30)
    ax3.set_ylabel("Indeks", fontsize=30)
    ax2.tick_params(labelsize=30)
    ax2.grid(True)
    ax3.figure.legend(loc="lower right", bbox_to_anchor=(0.9,0.09))
    
    plt.suptitle("Principal Component " + str(i) + " (Variansi: {:.2f}%)".format(varians.values), fontsize=35, weight="bold", y=0.93)
    fig.savefig(indir_save+"pc_"+ str(i) +".png", dpi=300, bbox_inches="tight")


# NON-DETREND, STANDARISASI
# Potong wilayah dan waktu
temp_hindia = temp.sel(lat=slice(-25,30), lon=slice(30,120)) 
temp_hindia = temp_hindia.sel(time=slice("1991-01", "2020-12"))
waktu = temp_hindia.time.values
waktu_str = pd.to_datetime(waktu).strftime("%Y-%m")

# Cek trend
temp_mean = temp_hindia.where(temp_hindia > 0).mean(dim=["lon", "lat"])
param_trend = np.polyfit(np.arange(0,len(temp_mean)), temp_mean.values, deg=1)
temp_cek = np.poly1d(param_trend)

## Plot trend
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(waktu, temp_mean, color="blue")
ax.plot(waktu, temp_cek(np.arange(0,len(waktu))), color="orange", label=r"$y={:.2e}x+{:.3f}$".format(param_trend[0], param_trend[1]))
ax.set_title("SST Mean Samudra Hindia (Non-Detrend)", fontsize=25, weight="bold")
ax.set_xlabel("Waktu", fontsize=20)
ax.set_ylabel(r"SST ($^\circ C$)", fontsize=20)
ax.grid(True)
ax.legend(fontsize=15)
ax.tick_params(labelsize=20)
fig.savefig(indir_save+"tren2.png", dpi=300, bbox_inches="tight")

# Anomali SST
temp_mean = temp_hindia.groupby("time.month").mean()
temp_std = temp_hindia.groupby("time.month").std()
temp_anom = temp_hindia.copy()
for i in range(12):
    temp_anom = temp_anom.where(temp_hindia["time.month"] != i+1, (temp_hindia - temp_mean.isel(month=i)) / temp_std.isel(month=i))

# Konversi data menjadi dua dimensi
lon = temp_anom.lon
lat = temp_anom.lat

temp_reshape = np.reshape(temp_anom.values, newshape=(len(waktu), len(lon)*len(lat)))  # Reshape data
kolom = pd.MultiIndex.from_product([temp_anom.lat.values, temp_anom.lon.values], names=["latitude", "longitude"]) # Membuat multiindex
temp_reshape = pd.DataFrame(temp_reshape, columns=kolom, index=waktu)  # Membuat dataframe
temp_nan = temp_reshape.isna().any()  # Menyimpan nilai nan
temp_clean = temp_reshape.loc[:, ~temp_nan].T  # Transpose

# PCA
matriks = temp_clean.copy() # Copy data yang sudah dihilangkan nilai nan-nya
## Hitung PCA
pca = skldec.PCA()
pca.fit(matriks)
evec = pca.components_  # Vektor Eigen
eval = pca.explained_variance_  # Nilai Eigen
persen = pca.explained_variance_ratio_ * 100  # Nilai Eigen dalam bentuk persen rasio
proyeksi = pca.transform(matriks)  # Proyeksi data ke basis vektor eigen. Nilai PC
kolom = ["PC-" + str(i) for i in range(1, len(matriks.columns)+1)]
proyeksi = pd.DataFrame(proyeksi, index=matriks.index, columns=kolom)

## Plot PCA
fig, ax = plt.subplots(figsize=(11,7))
ax.plot(kolom[:12], eval[:12], color="orange", marker="o")
ax.set_title("Scree Plot (PC1 - PC12) Non-Detrend", fontsize=25, weight="bold")
ax.set_ylabel("Eigen Values", fontsize=15)
ax.tick_params(labelsize=15)
ax.grid(True)
fig.savefig(indir_save+"scree2.png", dpi=300, bbox_inches="tight")
plt.clf()

pair = sns.pairplot(proyeksi.iloc[:,:4], plot_kws={"s":5}, diag_kind="kde")
plt.suptitle("Pairplot PC1 - PC4 Non-Detrend", y = 1.02, fontsize=25, weight="bold")
pair.savefig(indir_save+"pairplot2.png", dpi=300, bbox_inches="tight")

## Ubah hasil proyeksi menjadi data tiga dimensi
proyeksi2 = temp_reshape.copy().values
proyeksi2[:, ~temp_nan] = proyeksi.T.values
proyeksi2 = np.reshape(proyeksi2, newshape=(len(proyeksi.columns), len(lat), len(lon)))
temp_pc = xr.Dataset(data_vars={"proyeksi": (["PC", "lat", "lon"], proyeksi2), "vektoreigen": (["waktu", "PC"], evec), "nilaieigen": (["PC"], eval), "eigen_persen": (["PC"], persen)},
                     coords={"PC": np.arange(1, len(proyeksi.columns)+1), "lat": lat.values, "lon": lon.values, "waktu": waktu})  # Dataset hasil PCA

## Plot PCA
for i in range(1,6+1):
    mode = temp_pc.proyeksi.sel(PC = i)
    temporal = temp_pc.vektoreigen.sel(PC = i)
    varians = temp_pc.eigen_persen.sel(PC = i)

    # Spasial
    fig = plt.figure(figsize=(15, 14))
    ax1 = plt.subplot2grid((4,1), (0,0), rowspan=3, fig=fig, projection=ccrs.Mercator())
    peta = mode.plot.contourf(ax=ax1, cmap = "RdYlBu_r", levels=25, transform=ccrs.PlateCarree(), add_colorbar=False)
    cbar = plt.colorbar(peta, shrink = 0.7, pad = 0.03)
    cbar.set_label("Magnitudo", fontsize=30)
    cbar.ax.tick_params(labelsize=30)
    ax1.set_title("Pola Spasial PC-" + str(i), fontsize=33)
    ax1.coastlines(lw=2, color="k", zorder=2)
    ax1.add_feature(cfeature.LAND, facecolor="white", zorder=0)
    gl = ax1.gridlines(draw_labels=True, linewidth=1, color="grey", alpha=0.5)
    gl.top_labels=False
    gl.right_labels=False
    gl.left_labels=True
    gl.bottom_labels=True
    gl.xlabel_style = {"size":30}
    gl.ylabel_style = {"size":30}

    # Temporal
    ax2 = plt.subplot2grid((4,1), (3,0), rowspan=1, fig=fig)
    ax3 = ax2.twinx()
    ax3.plot(indeks.index, indeks["DMI"].values,ls="--", label="DMI")
    ax2.plot(waktu, temporal, color="black", label="PC-"+str(i))
    ax2.set_title("Pola Temporal PC-" + str(i), fontsize=33)
    ax2.set_xlabel("Waktu", fontsize=30)
    ax2.set_ylabel("Magnitudo", fontsize=30)
    ax3.set_ylabel("Indeks", fontsize=30)
    ax2.tick_params(labelsize=30)
    ax2.grid(True)
    ax3.figure.legend(loc="lower right", bbox_to_anchor=(0.9,0.09))
    
    plt.suptitle("Principal Component " + str(i) + " Non-Detrend (Variansi: {:.2f}%)".format(varians.values), fontsize=35, weight="bold", y=0.93)
    fig.savefig(indir_save+"pc__"+ str(i) +".png", dpi=300, bbox_inches="tight")
