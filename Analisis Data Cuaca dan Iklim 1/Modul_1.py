# Data yang digunakan:
# 1. Curah hujan : GPCC_INA.dat
# 2. Indeks iklim: CLIM_INDEX.csv
# dengan panjang data sebanyak 55 tahun (1950 - 2005) per bulan

#=================================================================

# Import module
import numpy as np
import pandas as pd
import xarray as xr
import cartopy as crt
import matplotlib.pyplot as plt

#=================================================================

# Data presipitasi
dat_ch = np.fromfile('GPCC_INA.dat', dtype = np.float32)

# Reshape data
dat_ch = np.reshape(dat_ch,(768, 41, 101))

# Panjang dimensi, menentukan koordinat lon, lat, dan time sesuai file GPCC_INA.ctl
lon = np.linspace(start = 94.75, stop = 94.75 + 0.5*100, num = 101)
lat = np.linspace(start = -10.25, stop = -10.25 + 0.5*40, num = 41)
time = pd.date_range(start = '1950', periods = 768, freq = 'MS')

# Data array
data = xr.DataArray(dat_ch, dims = ("time", "lat", "lon"), coords = {"time":time, "lat":lat, "lon":lon})
data = data.where(data >= 0)
data = data.loc["1950-01-01" : "2005-12-01"]

#=================================================================

# Data index
data_idx = pd.read_csv("CLIM_INDEX.csv", delimiter = ";")
data_idx.index = pd.date_range(start = "19500101", end = "20091201", freq = "MS")
data_idx = data_idx.drop(columns = ["YEAR", "MONTH"])
data_idx.index.name = "Time"

#=================================================================

# Index SOI 1950 - 2005
soi = data_idx["SOI"].loc["1950-01-01" : "2005-12-01"]

# Index DMI (IOD) 1950 - 2005
iod = data_idx["DMI"].loc["1950-01-01" : "2005-12-02"]

#=================================================================

# Ambang batas curah hujan per musim
# Curah hujan klimatologi
climPr = data.groupby("time.month").mean()

# Menentukan ambang batas di atas dan di bawah normal
AN = climPr + 0.15*climPr
BN = climPr - 0.15*climPr

# Anomali curah hujan berdasarkan ambang batas
anomAN = data.groupby("time.month") - AN
anomBN = data.groupby("time.month") - BN
#-----------------------------------------------------
# Ambang batas curah hujan di seluruh periode waktu
# Curah hujan klimatologi
climPr_all = data.mean()

# Menentukan ambang batas di atas dan di bawah normal
AN_all = climPr_all + 0.15*climPr_all
BN_all = climPr_all - 0.15*climPr_all

# Anomali curah hujan berdasarkan ambang batas
anomAN_all = data - AN_all
anomBN_all = data - BN_all

#=================================================================

# Memisahkan data anomali per musim (CH AN)
anomAN_msm = anomAN.groupby("time.season")
anomAN_msm_son = anomAN_msm["SON"]
anomAN_msm_djf = anomAN_msm["DJF"]
anomAN_msm_mam = anomAN_msm["MAM"]
anomAN_msm_jja = anomAN_msm["JJA"]

# Ubah data menjadi logical dengan: 1 artinya CH di atas ambang, 0 artinya CH di bawah ambang
anomAN_msm_son = (anomAN_msm_son>0) * 1
anomAN_msm_djf = (anomAN_msm_djf>0) * 1
anomAN_msm_mam = (anomAN_msm_mam>0) * 1
anomAN_msm_jja = (anomAN_msm_jja>0) * 1
anomAN_all = (anomAN_all>0) * 1 # Seluruh periode waktu
#---------------------------------------------
# Memisahkan data anomali per musim (CH BN)
anomBN_msm = anomBN.groupby("time.season")
anomBN_msm_son = anomBN_msm["SON"]
anomBN_msm_djf = anomBN_msm["DJF"]
anomBN_msm_mam = anomBN_msm["MAM"]
anomBN_msm_jja = anomBN_msm["JJA"]

# Ubah data menjadi logical dengan: 1 artinya CH di bawah ambang, 0 artinya CH di atas ambang
anomBN_msm_son = (anomBN_msm_son<0) * 1
anomBN_msm_djf = (anomBN_msm_djf<0) * 1
anomBN_msm_mam = (anomBN_msm_mam<0) * 1
anomBN_msm_jja = (anomBN_msm_jja<0) * 1
anomBN_all = (anomBN_all<0) * 1 # Seluruh periode waktu

#=================================================================

# SOI
# Memisahkan data indeks SOI per musim
soi_msm = soi.to_xarray().groupby("Time.season")
soi_msm_son = soi_msm["SON"] # data indeks 1 musim
soi_msm_djf = soi_msm["DJF"]
soi_msm_mam = soi_msm["MAM"]
soi_msm_jja = soi_msm["JJA"]
soi_all = soi.to_xarray() # Seluruh periode waktu

# El-Nino
# Threshold indeks El-Nino
thold_el = -1.0

# Ubah data menjadi logical dengan: 1 artinya kejadian El-Nino, 0 artinya bukan kejadian El-Nino
el_msm_son = (soi_msm_son <= thold_el)*1
el_msm_djf = (soi_msm_djf <= thold_el)*1
el_msm_mam = (soi_msm_mam <= thold_el)*1
el_msm_jja = (soi_msm_jja <= thold_el)*1
el_all = (soi_all <= thold_el)*1 # Seluruh periode waktu
#---------------------------------------------------
# La-Nina
# Threshold indeks La-Nina
thold_la = 1.0

# Ubah data menjadi logical dengan: 1 artinya kejadian La-Nina, 0 artinya bukan kejadian La-Nina
la_msm_son = (soi_msm_son >= thold_la)*1
la_msm_djf = (soi_msm_djf >= thold_la)*1
la_msm_mam = (soi_msm_mam >= thold_la)*1
la_msm_jja = (soi_msm_jja >= thold_la)*1
la_all = (soi_all >= thold_la)*1 # Seluruh periode waktu

#==============================================================================================

# IOD
# Memisahkan data indeks IOD per musim
iod_msm = iod.to_xarray().groupby("Time.season")
iod_msm_son = iod_msm["SON"] # data indeks 1 musim
iod_msm_djf = iod_msm["DJF"]
iod_msm_mam = iod_msm["MAM"]
iod_msm_jja = iod_msm["JJA"]
iod_all = iod.to_xarray() # Seluruh periode waktu

# IOD+ (dp)
# Threshold indeks IOD+
thold_dp = 0.4

# Ubah data menjadi logical dengan: 1 artinya IOD+, 0 artinya bukan IOD+
dp_msm_son = (iod_msm_son >= thold_dp)*1 
dp_msm_djf = (iod_msm_djf >= thold_dp)*1
dp_msm_mam = (iod_msm_mam >= thold_dp)*1
dp_msm_jja = (iod_msm_jja >= thold_dp)*1
dp_all = (iod_all >= thold_dp)*1 # Seluruh periode waktu
#---------------------------------------------------
#IOD- (dm)
# Threshold indeks IOD-
thold_dm = -0.4

# Ubah data menjadi logical dengan: 1 artinya IOD-, 0 artinya bukan IOD-
dm_msm_son = (iod_msm_son <= thold_dm)*1 
dm_msm_djf = (iod_msm_djf <= thold_dm)*1
dm_msm_mam = (iod_msm_mam <= thold_dm)*1
dm_msm_jja = (iod_msm_jja <= thold_dm)*1
dm_all = (iod_all <= thold_dm)*1 # Seluruh periode waktu

#=================================================================

# Looping menghitung probabilitas kondisional SOI dan IOD
prob_el_son = data[0, ...].copy() # SOI: El-Nino
prob_el_djf = data[0, ...].copy()
prob_el_mam = data[0, ...].copy()
prob_el_jja = data[0, ...].copy()
prob_el_all = data[0, ...].copy()
prob_la_son = data[0, ...].copy() # SOI: La-Nina
prob_la_djf = data[0, ...].copy()
prob_la_mam = data[0, ...].copy()
prob_la_jja = data[0, ...].copy()
prob_la_all = data[0, ...].copy()
prob_dp_son = data[0, ...].copy() # IOD+
prob_dp_djf = data[0, ...].copy()
prob_dp_mam = data[0, ...].copy()
prob_dp_jja = data[0, ...].copy()
prob_dp_all = data[0, ...].copy()
prob_dm_son = data[0, ...].copy() # IOD-
prob_dm_djf = data[0, ...].copy()
prob_dm_mam = data[0, ...].copy()
prob_dm_jja = data[0, ...].copy()
prob_dm_all = data[0, ...].copy()
nt, ny, nx = data.shape
for j in range(ny):
    for i in range(nx):
        if ~np.isnan(data[0, j, i]):
            tsAN_son = anomAN_msm_son[:, j, i]
            tsAN_djf = anomAN_msm_djf[:, j, i]
            tsAN_mam = anomAN_msm_mam[:, j, i]
            tsAN_jja = anomAN_msm_jja[:, j, i]
            tsAN_all = anomAN_all[:, j, i] # Seluruh periode waktu
            tsBN_son = anomBN_msm_son[:, j, i]
            tsBN_djf = anomBN_msm_djf[:, j, i]
            tsBN_mam = anomBN_msm_mam[:, j, i]
            tsBN_jja = anomBN_msm_jja[:, j, i]
            tsBN_all = anomBN_all[:, j, i] # Seluruh periode waktu
            # Probibilitas kondisional
            # SOI: El-Nino
            prob_el_son[j, i] = np.sum((tsBN_son.values & el_msm_son.values))/np.sum(el_msm_son.values)
            prob_el_djf[j, i] = np.sum((tsBN_djf.values & el_msm_djf.values))/np.sum(el_msm_djf.values)
            prob_el_mam[j, i] = np.sum((tsBN_mam.values & el_msm_mam.values))/np.sum(el_msm_mam.values)
            prob_el_jja[j, i] = np.sum((tsBN_jja.values & el_msm_jja.values))/np.sum(el_msm_jja.values)
            prob_el_all[j, i] = np.sum((tsBN_all.values & el_all.values))/np.sum(el_all.values)
            # SOI: La-Nina
            prob_la_son[j, i] = np.sum((tsAN_son.values & la_msm_son.values))/np.sum(la_msm_son.values)
            prob_la_djf[j, i] = np.sum((tsAN_djf.values & la_msm_djf.values))/np.sum(la_msm_djf.values)
            prob_la_mam[j, i] = np.sum((tsAN_mam.values & la_msm_mam.values))/np.sum(la_msm_mam.values)
            prob_la_jja[j, i] = np.sum((tsAN_jja.values & la_msm_jja.values))/np.sum(la_msm_jja.values)
            prob_la_all[j, i] = np.sum((tsAN_all.values & la_all.values))/np.sum(la_all.values)
            # IOD: IOD+
            prob_dp_son[j, i] = np.sum((tsBN_son.values & dp_msm_son.values))/np.sum(dp_msm_son.values)
            prob_dp_djf[j, i] = np.sum((tsBN_djf.values & dp_msm_djf.values))/np.sum(dp_msm_djf.values)
            prob_dp_mam[j, i] = np.sum((tsBN_mam.values & dp_msm_mam.values))/np.sum(dp_msm_mam.values)
            prob_dp_jja[j, i] = np.sum((tsBN_jja.values & dp_msm_jja.values))/np.sum(dp_msm_jja.values)
            prob_dp_all[j, i] = np.sum((tsBN_all.values & dp_all.values))/np.sum(dp_all.values)
            # IOD: IOD-
            prob_dm_son[j, i] = np.sum((tsAN_son.values & dm_msm_son.values))/np.sum(dm_msm_son.values)
            prob_dm_djf[j, i] = np.sum((tsAN_djf.values & dm_msm_djf.values))/np.sum(dm_msm_djf.values)
            prob_dm_mam[j, i] = np.sum((tsAN_mam.values & dm_msm_mam.values))/np.sum(dm_msm_mam.values)
            prob_dm_jja[j, i] = np.sum((tsAN_jja.values & dm_msm_jja.values))/np.sum(dm_msm_jja.values)
            prob_dm_all[j, i] = np.sum((tsAN_all.values & dm_all.values))/np.sum(dm_all.values)
            
#=================================================================            
            
# Untuk mengatur tickmarks gambar
import matplotlib.ticker as mticker

# Menghilangkan pesan warning yang tidak esensial
import warnings
warnings.filterwarnings("ignore")            

#=================================================================

# Variabel/text yang dapat diubah
# - Baris ke-258: variabel prob_##_###
# - Baris ke-266: text judul
# - Baris ke-290: save gambar

# Proyeksi peta
proj = crt.crs.PlateCarree()

# figure dan axis
fig = plt.figure(figsize = (16, 8))
ax = plt.axes(projection = proj)

# Plot data probablity SOI
p = prob_la_son.plot(ax=ax, # Sumbu axis yang digunakan 
                     transform = proj, # Transformasi data ke dalam proyeksi peta
                     levels = np.arange(0.0, 1.0, 0.1), # Level ketinggian kontur
                     cmap = "jet", # Colormap
                     cbar_kwargs = {"orientation" : "horizontal", "shrink" : 0.9}, # Pengaturan colorbar
                     )

# Fitur tambahan lainnya
ax.set_title("Probabilitas CH AN ketika La-Nina di SON", fontsize = 20, fontweight = "bold") # Judul Peta
ax.coastlines(resolution = "10m", linewidth = 2, edgecolor = "black") # Garis pantai
ocean = crt.feature.NaturalEarthFeature("physical", "ocean", scale = "10m", edgecolor = "none", facecolor = crt.feature.COLORS["water"]) # Laut
ax.add_feature(ocean, linewidth = 0.2)

# Set area gambar
ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]]) # Peta Indonesia

# Set gridline dan label peta
gl = p.axes.gridlines(draw_labels = True, linestyle = ":", linewidth = 2, color = "gray")
gl.xlocator = mticker.FixedLocator([100, 110, 120, 130, 140])
gl.ylocator = mticker.FixedLocator([-8, -4, 0, 4, 8])
gl.xlabels_top = False
gl.xlabels_bottom = True
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {"size": 20, "color": "black", "weight": "bold"}
gl.ylabel_style = {"size": 20, "color": "black", "weight": "bold"}

# Pengaturan colorbar
p.colorbar.ax.tick_params(labelsize = 20)
p.colorbar.set_label(label = "Probabilitas", size = 20, weight = "bold")

# Save gambar
fig.savefig("La-Nina/SON.png", dpi = 300)
