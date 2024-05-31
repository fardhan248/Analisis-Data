# Tugas 1: Menghitung Numerical Summary Tabel A1
# Numerical Summary:
# - Location: Mean, median, trimean, trimmed mean
# - Spread: Standar deviasi, IQR, MAD
# - Symmetry: Skewness, Yule-Kendall

# Import module
import numpy as np
import pandas as pd
from scipy import stats

indir = "../DATA/" #input directory

#=================================================================

# Baca data tabel A1
data = pd.read_csv(indir + "tableA1.txt", delimiter = "\t", header = None, names = ["date", "precip1", "tmax1", "tmin1", "precip2", "tmax2", "tmin2"])
data = data.drop(columns = "date") # drop kolom date

#=================================================================

# Location
# Mean, median, trimean, trimmed mean

# Mean
data_mean = data.mean()
data_mean = pd.DataFrame(data_mean, columns = ["Mean"]) # Men-dataframe-kan suatu data

#=================================================================

# Median
data_median = data.median()
data_median = pd.DataFrame(data_median, columns = ["Median"])
# atau
#data_q2 = data.quantile(q = 0.5)

#=================================================================

# Trimean
data_q1 = data.quantile(q = 0.25) # quartil 1
data_q2 = data.quantile(q = 0.5)  # quartil 2 / median
data_q3 = data.quantile(q = 0.75) # quartil 3
data_trmean = (data_q1 + 2*data_q2 + data_q3)/4
data_trmean = pd.DataFrame(data_trmean, columns = ["Trimean"])

#=================================================================

# Trimmed mean
data_trim = stats.trim_mean(data, 0.1)
data_trim = pd.DataFrame(data_trim, columns = ["Trimmed Mean"])
data_trim.index = ["precip1", "tmax1", "tmin1", "precip2", "tmax2", "tmin2"] # Menamakan indeks berdasarkan nama kolom

#=================================================================

# Spread
# Standar deviasi, IQR, MAD

# Standar deviasi
data_std = data.std()
data_std = pd.DataFrame(data_std, columns = ["Standar Deviasi"])

#=================================================================

# Interquartile Range (IQR)
data_iqr = data_q3 - data_q1
data_iqr_df = pd.DataFrame(data_iqr, columns = ["IQR"])

#=================================================================

# Median Absolute Deviation (MAD)
data_mad = abs(data - data_q2).median()
data_mad = pd.DataFrame(data_mad, columns = ["MAD"])

#=================================================================

# Symmetry
# Skewness, Yule-Kendall

# Skewness
data_skew = stats.skew(data)
data_skew = pd.DataFrame(data_skew, columns = ["Skewness"])
data_skew.index = ["precip1", "tmax1", "tmin1", "precip2", "tmax2", "tmin2"]

#=================================================================

# Yule-Kendall
data_yk = (data_q1 - 2*data_q2 + data_q3)/data_iqr
data_yk = pd.DataFrame(data_yk, columns = ["Yule-Kendall"])

#=================================================================
#=================================================================

# Tugas 2: Mempetakan Numerical Summary Data Spasio-Temporal Setiap Musim
# Numerical summary: Trimean (location), Skewness (symmetry)

# Import module
import matplotlib.pyplot as plt
import xarray as xr
import cartopy as crt

#=================================================================

# Baca data
datapr = xr.open_dataarray(indir + "MSWEP_MON_INA_197902-202011.nc")
datapr = xr.DataArray(datapr)
lon = datapr["lon"]
lat = datapr["lat"]
musim = ["DJF", "MAM", "JJA", "SON"]

#=================================================================

# Location: Trimean

# Trimean (+- 3 menit)
data_q1 = datapr.groupby("time.season").quantile(q = 0.25)
data_q2 = datapr.groupby("time.season").quantile(q = 0.5)
data_q3 = datapr.groupby("time.season").quantile(q = 0.75)
data_tri = (data_q1 + 2*data_q2 + data_q3)/4

#=================================================================

# Untuk mengatur tickmarks gambar
import matplotlib.ticker as mticker

# Menghilangkan pesan warning yang tidak esensial
import warnings
warnings.filterwarnings("ignore")

#=================================================================

# Looping peta
for k in range(4):
    # Proyeksi peta
    proj = crt.crs.PlateCarree()

    # figure dan axis
    fig = plt.figure(figsize = (16, 8))
    ax = plt.axes(projection = proj)

    # Plot data probablity SOI
    p = data_tri.sel(season=musim[k]).plot(ax=ax, # Sumbu axis yang digunakan 
                     transform = proj, # Transformasi data ke dalam proyeksi peta
                     levels = np.arange(0.0, 1100, 90), # Colorbar kontur
                     cmap = "YlGnBu", # Colormap
                     cbar_kwargs = {"orientation" : "horizontal", "shrink" : 0.6}, # Pengaturan colorbar
                     )

    # Fitur tambahan lainnya
    ax.set_title("Trimean Data Presipitasi di " + musim[k], fontsize = 20, fontweight = "bold") # Judul Peta
    ax.coastlines(resolution = "10m", linewidth = 1.3, color = "black") # Garis pantai

    # Set area gambar
    ax.set_extent([93, 145, 13.5, -13.5]) # Peta Indonesia

    # Set gridline dan label peta
    gl = p.axes.gridlines(draw_labels = True, linestyle = ":", linewidth = 2, color = "gray")
    gl.xlocator = mticker.FixedLocator([100, 110, 120, 130, 140])
    gl.ylocator = mticker.FixedLocator([-15, -10, -5, 0, 5, 10, 15])
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True
    gl.xlabel_style = {"size": 20, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"size": 20, "color": "black", "weight": "bold"}

    # Pengaturan colorbar
    p.colorbar.ax.tick_params(labelsize = 20)
    p.colorbar.set_label(label = "Trimean", size = 20, weight = "bold")

    # Save gambar
    fig.savefig("trimean/" + musim[k] + ".png", dpi = 300)
    plt.clf() # Reset gambar    

#=================================================================

# Symmetry: Skewness

# Skewness
skew_data = np.zeros((4, 300, 600)) # Matriks kosong, untuk menyimpan nilai skewness per musim
for k in range(4):
    skew_data[k, :, :] = stats.skew(datapr.groupby("time.season")[musim[k]])
data_skew = xr.DataArray(skew_data,
                         dims = ("season", "lat", "lon"), 
                         coords = {"season":musim, "lat":lat, "lon":lon}) # Memberi nama koordinat pada data skewness
                       
#=================================================================

# Untuk mengatur tickmarks gambar
import matplotlib.ticker as mticker

# Menghilangkan pesan warning yang tidak esensial
import warnings
warnings.filterwarnings("ignore")

#=================================================================

# Looping peta
for k in range(4):
    # Proyeksi peta
    proj = crt.crs.PlateCarree()

    # figure dan axis
    fig = plt.figure(figsize = (16, 8))
    ax = plt.axes(projection = proj)

    # Plot data probablity SOI
    p = data_skew.sel(season=musim[k]).plot(ax=ax, # Sumbu axis yang digunakan 
                      transform = proj, # Transformasi data ke dalam proyeksi peta
                      levels = np.arange(-2, 11, 1.5), # Colorbar kontur
                      cmap = "YlGnBu", # Colormap
                      cbar_kwargs = {"orientation" : "horizontal", "shrink" : 0.6}, # Pengaturan colorbar
                      )

    # Fitur tambahan lainnya
    ax.set_title("Skewness Data Presipitasi di " + musim[k], fontsize = 20, fontweight = "bold") # Judul Peta
    ax.coastlines(resolution = "10m", linewidth = 1.3, color = "black") # Garis pantai

    # Set area gambar
    ax.set_extent([93, 145, 13.5, -13.5]) # Peta Indonesia

    # Set gridline dan label peta
    gl = p.axes.gridlines(draw_labels = True, linestyle = ":", linewidth = 2, color = "gray")
    gl.xlocator = mticker.FixedLocator([100, 110, 120, 130, 140])
    gl.ylocator = mticker.FixedLocator([-15, -10, -5, 0, 5, 10, 15])
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True
    gl.xlabel_style = {"size": 20, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"size": 20, "color": "black", "weight": "bold"}

    # Pengaturan colorbar
    p.colorbar.ax.tick_params(labelsize = 20)
    p.colorbar.set_label(label = "Skewness", size = 20, weight = "bold")

    # Save gambar
    fig.savefig("skewness/" + musim[k] + ".png", dpi = 300)
    plt.clf() # Reset gambar
    
