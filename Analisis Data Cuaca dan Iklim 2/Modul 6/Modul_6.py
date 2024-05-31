# Perhitungan
import numpy as np
import sklearn.preprocessing as skpp
import cluster_func

# Olah data
import xarray as xr
import pandas as pd

# Visualisasi
import seaborn as sns; sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

indir = "D:/[2]DATA ANDAT/"
dir_save = "Minggu (15)/hasil/"
hujan0 = xr.open_dataset(indir+"prate.sfc.mon.mean.nc")
hujan1 = hujan0.assign_coords(lon=(((hujan0.lon + 180) % 360) - 180)).sortby("lon")

# Pre-processing data
# Slice Indonesia
hujan = hujan1.sel(lat=slice(10, -13), lon=slice(90,150))
precip0 = hujan.prate * 86400  # Konversi satuan menjadi mm/hari

# Rata-rata per bulan
precip0 = precip0.sel(time=slice("1991-01-01", "2020-12-01"))
precip1 = precip0.groupby("time.month").mean()

# Stack lat lon menjadi multiindex
precip2 = precip1.stack(latlon=("lat", "lon"))

# Konversi data ke Pandas
precip = precip2.to_pandas().T

# Standarisasi data
param = skpp.PowerTransformer().fit(precip)
scaled = param.transform(precip)
precip_scaled = pd.DataFrame(scaled, index = precip.index, columns = precip.columns)

# Clustering
## Hierarchical Clustering
### Variasi 1
hcc = cluster_func.hierarchical(precip_scaled, 5, "euclidean", "ward")

### Variasi 2
hcc2 = cluster_func.hierarchical(precip_scaled, 5, "manhattan", "average")

## K-Means
### Variasi 1
kcc = cluster_func.kmeans(precip_scaled, 5, "k-means++")

### Variasi 2
centroid = cluster_func.get_centroid(precip_scaled, 5, "euclidean", "ward")
kcc2 = cluster_func.kmeans(precip_scaled, 5, centroid)

# Xarray hasil clustering
label1 = pd.DataFrame([hcc.labels_, hcc2.labels_, kcc.labels_, kcc2.labels_], index=["Agglomerative_1", "Agglomerative_2", "KMeans_1", "KMeans_2"], columns=precip_scaled.index).T
label = label1.to_xarray()

# Evaluasi
dbi, inertia, distances = cluster_func.evaluation(precip_scaled, hcc, hcc2, 18)  # DBI indeks, inersia KMeans, jarak agglomerative
    
# Plot hasil evaluasi
## DBI
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(dbi, "o-", label=dbi.columns)

ax.set_title("Nilai DBI Setiap Banyaknya Cluster", fontsize=30, weight="bold")
ax.set_xlabel("Banyak Cluster", fontsize = 20)
ax.set_ylabel("DBI", fontsize = 20)
ax.legend(fontsize=20)
ax.tick_params(labelsize=20)
fig.savefig(dir_save+"dbi.png", dpi=300, bbox_inches="tight")
plt.close()

## Inertia dan distance
color1 = ["red", "blue"]
color2 = ["green", "orange"]
fig, ax = plt.subplots(figsize=(12,8))
for column, color in zip(inertia.columns, color1):
    ax.plot(inertia[column], "o-", label=column, color=color)
ax2 = ax.twinx()
for column, color in zip(distances.columns, color2):
    ax2.plot(distances[1:18][column], "o-", label=column, color=color)

ax.set_title("Nilai Inersia KMeans dan Jarak Agglomerative\nSetiap Banyaknya Cluster", fontsize=30, weight="bold")
ax.set_xlabel("Banyak Cluster", fontsize = 20)
ax.set_ylabel("Inersia KMeans", fontsize = 20)
ax.legend(fontsize=20)
ax.tick_params(labelsize=20)

ax2.set_ylabel("Jarak Agglomerative", fontsize=20)
ax2.tick_params(labelsize=20)
ax2.legend(fontsize=20, loc=7)
fig.savefig(dir_save+"elbow.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot spasial dan temporal hasil clustering
precip_cluster = pd.concat([precip_scaled, label1], axis=1)
variasi = ["Agglomerative_1", "Agglomerative_2", "KMeans_1", "KMeans_2"]
judul = ["Agglomerative Variasi 1", "Agglomerative Variasi 2", "KMeans Variasi 1", "KMeans Variasi 2"]
for i in range(len(variasi)):
    # Plot spasial
    prj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = plt.subplot(projection=prj)
    ax.coastlines()

    levels = np.arange(-0.5, 5.5, 1)
    plot = label[variasi[i]].plot(ax=ax, levels=levels, cmap="Set1", cbar_kwargs={'ticks': range(5), "shrink":0.45, "aspect":18})

    plot.colorbar.ax.tick_params(labelsize=20)
    plot.colorbar.set_label(label='Cluster', size=20)

    ax.set_title(judul[i], fontsize=30, weight="bold")

    gl = ax.gridlines(draw_labels=True, linewidth=1, color="grey", alpha=0.5)
    gl.top_labels=False
    gl.right_labels=False
    gl.left_labels=True
    gl.bottom_labels=True
    gl.xlabel_style = {"size":18}
    gl.ylabel_style = {"size":18}

    fig.savefig(dir_save+variasi[i]+".png", bbox_inches="tight", dpi=300)
    plt.close()

    # Plot temporal
    cluster = np.sort(precip_cluster[variasi[i]].unique())
    for j in range(len(cluster)):
        data_plot = precip_cluster[np.arange(1,12+1,1)].loc[precip_cluster[variasi[i]] == j].mean(axis=0)
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(data_plot, "o-", color="orange", linewidth=4)

        ax.set_title(judul[i]+" Cluster "+str(j), fontsize=30, weight="bold")
        ax.set_ylabel("Curah Hujan Standarisasi", fontsize=20)
        ax.set_xlabel("Bulan", fontsize=20)
        ax.tick_params(labelsize=20)
        fig.savefig(dir_save+variasi[i]+"_c"+str(j)+".png", bbox_inches="tight", dpi=300)
        plt.close()
