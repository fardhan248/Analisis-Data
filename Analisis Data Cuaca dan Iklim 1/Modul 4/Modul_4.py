import warnings
warnings.simplefilter('ignore') #ignores simple warning
# Untuk mengatur tickmarks gambar
import matplotlib.ticker as mticker

import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime

from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#======================================================

def timelagGFS(file_list,varname,domain,valid_time_start,valid_time_end):
  '''
  Fungsi sederhana untuk membuat time-lag ensemble dari data GFS
  '''
  #member-1
  m = TDSCatalog(file_list[0])
  print(m)
  ds = list(m.datasets.values())[0]
  ncss = ds.subset()
  query = ncss.query()
  query.lonlat_box(north=domain['north'], south=domain['south'], east=domain['east'], west=domain['west'])
  query.time_range(start=valid_time_start,end=valid_time_end)
  query.accept('netcdf4')
  query.variables(varname)
  #get data
  data = xr.open_dataset(NetCDF4DataStore(ncss.get_data(query)))
  data = data.rename({data[varname].dims[0]: 'time'})

  #member lainnya
  for mem in file_list[1:]:
    m = TDSCatalog(mem)
    print(m)
    ds = list(m.datasets.values())[0]
    ncss = ds.subset()

    query = ncss.query()
    query.lonlat_box(north=domain['north'], south=domain['south'], east=domain['east'], west=domain['west'])
    query.time_range(start=valid_time_start,end=valid_time_end)
    query.accept('netcdf4')
    query.variables(varname)

    #get data
    dat = xr.open_dataset(NetCDF4DataStore(ncss.get_data(query)))
    dat = dat.rename({dat[varname].dims[0]: 'time'})

    #concatenate
    data = xr.concat([data,dat],"member")

  return data

#====================

# Menentukan tanggal dan waktu data
tgl = [14, 15, 16, 17, 18, 19]
jam = ["00", "06", "12", "18"]
tlag_list = []

for tanggal in tgl:
    for waktu in jam:
        link = "https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_202310" +  str(tanggal) + "_" + waktu + "00.grib2/catalog.xml"
        tlag_list.append(link)

#===========================
# PRESIPITASI

#parameter subset
varname='Precipitation_rate_surface'
llbox={'north':15,
       'south':-15,
       'west':90,
       'east':150}
valid_s=datetime(2023,10,20,0,0)
valid_e=datetime(2023,10,22,21,0)

tlag_ens=timelagGFS(tlag_list,varname,llbox,valid_s,valid_e)

tlag_ens.to_netcdf('data_precip_14_to_19.nc') # Konversi data ke file .nc

# Baca data
precip = xr.open_dataset("data_precip_14_to_19.nc")
precip = precip.resample(time = "6H").sum()
precip_res = precip*3600*6 # Konversi satuan dari mm/s ke mm/6 jam

#=============================
## Probabilitas

t = precip_res["Precipitation_rate_surface"]["time"].values
thold = 0.5  # threshold

# Set up subplots
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 16), subplot_kw={'projection': ccrs.PlateCarree()})
fig.subplots_adjust(hspace=0.01)

for i, ax in enumerate(axes.flatten()):
    dat = precip_res["Precipitation_rate_surface"].sel(time=t[i])
    prob = np.sum(dat > thold, axis=0) / 24

    p = prob.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Blues',
                  cbar_kwargs={'shrink': 0.8, "label": "Probabilitas", "orientation":"horizontal"}, vmin=0.0, vmax=1.0)

    ax.add_feature(cfeature.BORDERS, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    ax.set_extent([95, 150, -10, 10])
    ax.set_title(str(t[i]), fontsize=9)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linestyle=":", linewidth=2, color="gray")
    gl.xlocator = mticker.FixedLocator([90, 100, 110, 120, 130, 140, 150])
    gl.ylocator = mticker.FixedLocator([-10, -5, 0, 5, 10])
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True
    gl.xlabel_style = {"size": 8, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"size": 8, "color": "black", "weight": "bold"}

plt.suptitle("Probabilitas Presipitasi di Atas 0.5 mm/(6 jam)", fontsize=15, fontweight="bold", y = 0.85)

# Save the figure
fig.savefig("Probabilitas/Presipitasi/all.png", dpi=300)
plt.show()

#=================
## Meteogram

# Me-resample data di satu titik dengan merata-ratakan member
precip_ts = pd.Series(precip_res["Precipitation_rate_surface"].sel(latitude=-6.891443681017748, longitude=107.61103629128853, method="nearest").mean(dim="member").values, 
                      index = precip_res["Precipitation_rate_surface"]["time"].values)

# Resample data menjadi rata-rata per hari
precip_mean = precip_res["Precipitation_rate_surface"].groupby("time.day").mean().sel(latitude=-6.891443681017748, longitude=107.61103629128853, method="nearest").mean(dim="member")

fig, ax = plt.subplots(figsize = (11, 5), dpi = 300)
sns.boxplot(x = precip_ts.index.date, y = precip_ts, ax = ax, width = 0.5, color = "cyan") # Boxplot
sns.lineplot(precip_mean, color = "blue") # Lineplot
ax.set_xlabel("Tanggal", fontsize=15)
ax.set_ylabel("Presipitasi (mm/6 jam)", fontsize=15)
ax.set_title("Meteogram Presipitasi Tanggal 20, 21, 22 Oktober 2023", fontsize=18)
plt.tick_params(axis='both', labelsize=12) # Ukuran font nilai x dan y
sns.set_style("whitegrid") # Menambahkan gridlines
plt.show()

# Save gambar
fig.savefig("Meteogram/presipitasi.png", dpi = 300)

#======================
# TEKANAN

#parameter subset
varname2='Pressure_surface'
llbox2={'north':15,
       'south':-15,
       'west':90,
       'east':150}
valid_s2=datetime(2023,10,20,0,0)
valid_e2=datetime(2023,10,22,21,0)

tlag_ens2=timelagGFS(tlag_list,varname2,llbox2,valid_s2,valid_e2)

tlag_ens2.to_netcdf('data_pressure_14_to_19.nc') # Konversi data ke file .nc

# Baca data
tekan = xr.open_dataset("data_pressure_14_to_19.nc")
tekan = tekan.resample(time = "6H").mean()
tekan_res = tekan*0.01 # Konversi satuan dari Pascal ke mbar

#=======================
## Probabilitas

t2 = tekan_res["Pressure_surface"]["time"].values
thold2 = 950  # threshold

# Set up subplots
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 16), subplot_kw={'projection': ccrs.PlateCarree()})
fig.subplots_adjust(hspace=0.01)

for i, ax in enumerate(axes.flatten()):
    dat2 = tekan_res["Pressure_surface"].sel(time=t2[i])
    prob2 = np.sum(dat2 < thold2, axis=0) / 24

    p2 = prob2.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Wistia',
                  cbar_kwargs={'shrink': 0.8, "label": "Probabilitas", "orientation":"horizontal"}, vmin=0.0, vmax=1.0)

    ax.add_feature(cfeature.BORDERS, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    ax.set_extent([95, 150, -10, 10])
    ax.set_title(str(t2[i]), fontsize=9)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linestyle=":", linewidth=2, color="gray")
    gl.xlocator = mticker.FixedLocator([90, 100, 110, 120, 130, 140, 150])
    gl.ylocator = mticker.FixedLocator([-10, -5, 0, 5, 10])
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True
    gl.xlabel_style = {"size": 8, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"size": 8, "color": "black", "weight": "bold"}

plt.suptitle("Probabilitas Tekanan di Bawah 950 mbar", fontsize=15, fontweight="bold", y = 0.85)

# Save the figure
fig.savefig("Probabilitas/Tekanan/all.png", dpi=300)
plt.show()

#=================
## Meteogram

# Me-resample data di satu titik dengan merata-ratakan member
tekan_ts = pd.Series(tekan_res["Pressure_surface"].sel(latitude=-6.891443681017748, longitude=107.61103629128853, method="nearest").mean(dim="member").values, 
                      index = tekan_res["Pressure_surface"]["time"].values)

# Resample data menjadi rata-rata per hari
tekan_mean = tekan_res["Pressure_surface"].groupby("time.day").mean().sel(latitude=-6.891443681017748, longitude=107.61103629128853, method="nearest").mean(dim="member")


fig, ax = plt.subplots(figsize = (11, 5), dpi = 300)
sns.boxplot(x = tekan_ts.index.date, y = tekan_ts, ax = ax, width = 0.5, color = "green") # Boxplot
sns.lineplot(tekan_mean, color = "blue") # Lineplot
ax.set_xlabel("Tanggal", fontsize=15)
ax.set_ylabel("Tekanan (mbar)", fontsize=15)
ax.set_title("Meteogram Tekanan Tanggal 20, 21, 22 Oktober 2023", fontsize=18)
plt.tick_params(axis='both', labelsize=12) # Ukuran font nilai x dan y
sns.set_style("whitegrid") # Menambahkan gridlines
plt.show()

# Save gambar
fig.savefig("Meteogram/tekanan.png", dpi = 300)

#================================
# TEMPERATUR

#parameter subset
varname3='Temperature_surface'
llbox3={'north':15,
       'south':-15,
       'west':90,
       'east':150}
valid_s3=datetime(2023,10,20,0,0)
valid_e3=datetime(2023,10,22,21,0)

tlag_ens3=timelagGFS(tlag_list,varname3,llbox3,valid_s3,valid_e3)

tlag_ens3.to_netcdf('data_temp_14_to_19.nc') # Konversi data ke file .nc

# Baca data
temp = xr.open_dataset("data_temp_14_to_19.nc")
temp = temp.resample(time="6H").mean()
temp_res = temp - 273.15 # Konversi satuan dari Kelvin ke Celcius

#============================
## Probabilitas 

t3 = temp_res["Temperature_surface"]["time"].values
thold3 = 30  # threshold

# Set up subplots
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 16), subplot_kw={'projection': ccrs.PlateCarree()})
fig.subplots_adjust(hspace=0.01)

for i, ax in enumerate(axes.flatten()):
    dat3 = temp_res["Temperature_surface"].sel(time=t3[i])
    prob3 = np.sum(dat3 > thold3, axis=0) / 24

    p3 = prob3.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Reds',
                  cbar_kwargs={'shrink': 0.8, "label": "Probabilitas", "orientation":"horizontal"}, vmin=0.0, vmax=1.0)

    ax.add_feature(cfeature.BORDERS, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    ax.set_extent([95, 150, -10, 10])
    ax.set_title(str(t3[i]), fontsize=9)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linestyle=":", linewidth=2, color="gray")
    gl.xlocator = mticker.FixedLocator([90, 100, 110, 120, 130, 140, 150])
    gl.ylocator = mticker.FixedLocator([-10, -5, 0, 5, 10])
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True
    gl.xlabel_style = {"size": 8, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"size": 8, "color": "black", "weight": "bold"}

plt.suptitle("Probabilitas Temperatur di Atas $30 \degree C$", fontsize=15, fontweight="bold", y = 0.85)

# Save the figure
fig.savefig("Probabilitas/Temperatur/all.png", dpi=300)
plt.show()

#=====================
## Meteogram

# Me-resample data di satu titik dengan merata-ratakan member
temp_ts = pd.Series(temp_res["Temperature_surface"].sel(latitude=-6.891443681017748, longitude=107.61103629128853, method="nearest").mean(dim="member").values, 
                      index = temp_res["Temperature_surface"]["time"].values)

# Resample data menjadi rata-rata per hari
temp_mean = temp_res["Temperature_surface"].groupby("time.day").mean().sel(latitude=-6.891443681017748, longitude=107.61103629128853, method="nearest").mean(dim="member")

fig, ax = plt.subplots(figsize = (11, 5), dpi = 300)
sns.boxplot(x = temp_ts.index.date, y = temp_ts, ax = ax, width = 0.5, color = "red") # Boxplot
sns.lineplot(temp_mean, color = "black") # Lineplot
ax.set_xlabel("Tanggal", fontsize=15)
ax.set_ylabel("Temperatur ($\degree C$)", fontsize=15)
ax.set_title("Meteogram Temperatur Tanggal 20, 21, 22 Oktober 2023", fontsize=18)
plt.tick_params(axis='both', labelsize=12) # Ukuran font nilai x dan y
plt.show()
sns.set_style("whitegrid") # Menambahkan gridlines

# Save gambar
fig.savefig("Meteogram/temperatur.png", dpi = 300)
