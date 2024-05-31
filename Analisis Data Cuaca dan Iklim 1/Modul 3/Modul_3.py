import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.special import comb 
import scipy.stats as st
import cartopy as crt

#=======================================================================================
# TUGAS 1

# Nama-nama file data
files = ["*" for i in range(18+1)]
for i in range(18+1):
    files[i] = "data_temp/tmax.2m.gauss." + str(i+2005) + ".nc"

#=======================================================================================

# Data klimatologi
data_temp = xr.open_mfdataset(files, combine='by_coords')
data_temp = xr.DataArray(data_temp["tmax"]) - 273.15

# Data klimatologi Indonesia
datatemp_indo = data_temp.sel(lat=slice(16, -16), lon=slice(90, 145), level=2)

# Data klimatologi Indonesia musim JJA
datatemp_jja = datatemp_indo.groupby("time.season")["JJA"]

# Data klimatologi bulan Juni di Indonesia
juntemp = datatemp_indo.groupby("time.month")[6]

# Data klimatologi bulan Juli di Indonesia
jultemp = datatemp_indo.groupby("time.month")[7]

# Data tanggal 20 Juni 2023 di Indonesia
juntemp_23 = juntemp.sel(time="2023-06-20")

# Data tanggal 20 Juli 2023 di Indonesia
jultemp_23 = jultemp.sel(time="2023-07-20")

# Data JJA 2023 di Indonesia
temp_jja_23 = datatemp_jja.sel(time=(datatemp_jja["time.year"] == 2023)).mean(dim="time")

lat = datatemp_indo["lat"]  # Nilai lintang
lon = datatemp_indo["lon"]  # Nilai bujur

#=======================================================================================

# Dist Gamma 20 Juni 2023
prob = xr.DataArray(np.zeros((16, 30)), dims=("lat","lon"), coords={"lat":lat,"lon":lon})
for j in range(16):
    for i in range(30):
        param = st.gamma.fit(juntemp.sel(lat=lat[j], lon=lon[i]))   # Parameter di setiap titik
        prob[j, i] = st.gamma.cdf(juntemp_23[j, i], *param)         # Probabilitas (persentil) di tiap titik
            
prob.to_netcdf('persentil_20jun2023.nc') # Menyimpan data ke file .nc

# Dist Gamma 20 Juli 2023
prob2 = xr.DataArray(np.zeros((16, 30)), dims=("lat","lon"), coords={"lat":lat,"lon":lon})
for j in range(16):
    for i in range(30):
        param = st.gamma.fit(jultemp.sel(lat=lat[j], lon=lon[i]))   # Parameter di setiap titik
        prob2[j, i] = st.gamma.cdf(jultemp_23[j, i], *param)        # Probabilitas (persentil) di tiap titik
            
prob2.to_netcdf('persentil_20jul2023.nc')

# Dist Gamma JJA 2023
prob3 = xr.DataArray(np.zeros((16, 30)), dims=("lat","lon"), coords={"lat":lat,"lon":lon})
for j in range(16):
    for i in range(30):
        param = st.gamma.fit(datatemp_jja.sel(lat=lat[j], lon=lon[i]))   # Parameter di setiap titik
        prob3[j, i] = st.gamma.cdf(temp_jja_23[j, i], *param)            # Probabilitas (persentil) di tiap titik
            
prob3.to_netcdf('persentil_jja20231.nc')

# Dist Gamma Max Juni 2023 
prob4 = xr.DataArray(np.zeros((16, 30)), dims=("lat","lon"), coords={"lat":lat,"lon":lon})
for j in range(16):
    for i in range(30):
        param = st.gamma.fit(juntemp.sel(lat=lat[j], lon=lon[i]))           # Parameter di setiap titik
        prob4[j, i] = st.gamma.cdf(juntemp_23.values.max(), *param)         # Probabilitas (persentil) di tiap titik
            
prob4.to_netcdf('persentil_junmax2023.nc')

# Dist Gamma Max Juli 2023
prob5 = xr.DataArray(np.zeros((16, 30)), dims=("lat","lon"), coords={"lat":lat,"lon":lon})
for j in range(16):
    for i in range(30):
        param = st.gamma.fit(jultemp.sel(lat=lat[j], lon=lon[i]))           # Parameter di setiap titik
        prob5[j, i] = st.gamma.cdf(jultemp_23.values.max(), *param)         # Probabilitas (persentil) di tiap titik
            
prob5.to_netcdf('persentil_julmax2023.nc')

# Dist Gamma Max JJA 2023
prob6 = xr.DataArray(np.zeros((16, 30)), dims=("lat","lon"), coords={"lat":lat,"lon":lon})
for j in range(16):
    for i in range(30):
        param = st.gamma.fit(datatemp_jja.sel(lat=lat[j], lon=lon[i]))       # Parameter di setiap titik
        prob6[j, i] = st.gamma.cdf(temp_jja_23.values.max(), *param)         # Probabilitas (persentil) di tiap titik
            
prob6.to_netcdf('persentil_jjamax2023.nc')

#=======================================================================================

# Untuk mengatur tickmarks gambar
import matplotlib.ticker as mticker

# Nama file dan judul gambar
dat = ["20jun2023", "20jul2023", "jja2023", "junmax2023", "julmax2023", "jjamax20231"]
tgl = ["2023-06-20", "2023-07-20", "JJA 2023", "T = "+str(juntemp_23.values.max())+" oC Juni 2023", "T = "+str(jultemp_23.values.max())+" oC Juli 2023", "T = "+str(temp_jja_23.values.max())+" oC JJA 2023"]

# Looping membuat peta
for i in range(6):
    # Proyeksi peta
    proj = crt.crs.PlateCarree()

    # figure dan axis
    fig = plt.figure(figsize = (16, 8))
    ax = plt.axes(projection = proj)

    # Plot data probablity SOI
    p = xr.open_dataarray("persentil_" + dat[i] + ".nc").plot(ax=ax, # Sumbu axis yang digunakan 
               transform = proj, # Transformasi data ke dalam proyeksi peta
               levels = np.arange(0, 1, 0.05), # Colorbar kontur
               cmap = "turbo", # Colormap
               cbar_kwargs = {"orientation" : "horizontal", "shrink" : 0.6}, # Pengaturan colorbar
               )

    # Fitur tambahan lainnya
    ax.set_title("Persentil Tmax " + tgl[i] , fontsize = 20, fontweight = "bold") # Judul Peta
    ax.coastlines(resolution = "10m", linewidth = 1.3, color = "black") # Garis pantai
    #ocean = crt.feature.NaturalEarthFeature("physical", "ocean", scale = "10m", edgecolor = "none", facecolor = crt.feature.COLORS["water"]) # Laut
    #ax.add_feature(ocean, linewidth = 0.2)

    # Set area gambar
    ax.set_extent([93, 138.75, 13.5, -13.5]) # Peta Indonesia

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
    p.colorbar.set_label(label = "Persentil", size = 20, weight = "bold")
    
    # Save gambar
    fig.savefig("tugas1_temp/" + dat[i] + ".png", dpi = 300)
    plt.clf() # Reset gambar

#=======================================================================================
# TUGAS 2

from scipy.stats import gumbel_r

# Nama-nama file data
files2 = ["*" for i in range(25+1)]
for i in range(25+1):
    files2[i] = "data_precp/prate.sfc.gauss." + str(i+1998) + ".nc"

# Data klimatologi
data_pr = xr.open_mfdataset(files2, combine='by_coords')
data_pr = data_pr["prate"].groupby("time.year").max()

# Konversi satuan dari kg/m^2/s ke mm/hari
data_pr_day = data_pr*24*3600

# Data klimatologi Indonesia
datapr_indo = data_pr_day.sel(lat=slice(16, -16), lon=slice(90, 145))

lat2 = datapr_indo["lat"]
lon2 = datapr_indo["lon"]

#=======================================================================================

# Dist Gumbel Right, Periode ulang 10 tahun, annual maxima
periode_ulang = xr.DataArray(np.zeros((16, 30)), dims=("lat","lon"), coords={"lat":lat2,"lon":lon2})

w = 1       # sampling frekuensi. Karena menggunakan metode annual maxima, digunakan w = 1
R = 10      # Periode Ulang
Fx = 1 - (1/(R*w))

for j in range(16):
    for i in range(30):
        param = gumbel_r.fit(datapr_indo.sel(lat=lat2[j], lon=lon2[i]))      # Parameter gumbel_r di setiap titik
        Rx = gumbel_r.ppf(Fx, *param)
        periode_ulang[j, i] = Rx                                             # Curah hujan periode ulang di setiap titik

periode_ulang.to_netcdf('tugas2_pr/periode_ulang_gum_10thn.nc')

# Dist Gumbel Right, Periode ulang 25 tahun, annual maxima
periode_ulang2 = xr.DataArray(np.zeros((16, 30)), dims=("lat","lon"), coords={"lat":lat2,"lon":lon2})

w = 1       # sampling frekuensi. Karena menggunakan metode annual maxima, digunakan w = 1
R = 25      # Periode Ulang
Fx = 1 - (1/(R*w))

for j in range(16):
    for i in range(30):
        param = gumbel_r.fit(datapr_indo.sel(lat=lat2[j], lon=lon2[i]))       # Parameter gumbel_r di setiap titik
        Rx = gumbel_r.ppf(Fx, *param)
        periode_ulang2[j, i] = Rx                                             # Curah hujan periode ulang di setiap titik

periode_ulang2.to_netcdf('tugas2_pr/periode_ulang_gum_25thn.nc')

# Dist Gumbel Right, Periode ulang 50 tahun, annual maxima
periode_ulang3 = xr.DataArray(np.zeros((16, 30)), dims=("lat","lon"), coords={"lat":lat2,"lon":lon2})

w = 1       # sampling frekuensi. Karena menggunakan metode annual maxima, digunakan w = 1
R = 50      # Periode Ulang
Fx = 1 - (1/(R*w))

for j in range(16):
    for i in range(30):
        param = gumbel_r.fit(datapr_indo.sel(lat=lat2[j], lon=lon2[i]))       # Parameter gumbel_r di setiap titik
        Rx = gumbel_r.ppf(Fx, *param)
        periode_ulang3[j, i] = Rx                                             # Curah hujan periode ulang di setiap titik

periode_ulang3.to_netcdf('tugas2_pr/periode_ulang_gum_50thn.nc')

# Dist Gumbel Right, Periode ulang 100 tahun, annual maxima
periode_ulang4 = xr.DataArray(np.zeros((16, 30)), dims=("lat","lon"), coords={"lat":lat2,"lon":lon2})

w = 1       # sampling frekuensi. Karena menggunakan metode annual maxima, digunakan w = 1
R = 100      # Periode Ulang
Fx = 1 - (1/(R*w))

for j in range(16):
    for i in range(30):
        param = gumbel_r.fit(datapr_indo.sel(lat=lat2[j], lon=lon2[i]))       # Parameter gumbel_r di setiap titik
        Rx = gumbel_r.ppf(Fx, *param)
        periode_ulang4[j, i] = Rx                                             # Curah hujan periode ulang di setiap titik

periode_ulang4.to_netcdf('tugas2_pr/periode_ulang_gum_100thn.nc')

#=======================================================================================

# Untuk mengatur tickmarks gambar
import matplotlib.ticker as mticker

# Nama file dan judul gambar
period = ["10", "25", "50", "100"] 

# Looping membuat peta
for i in range(4):
    # Proyeksi peta
    proj = crt.crs.PlateCarree()

    # figure dan axis
    fig = plt.figure(figsize = (16, 8))
    ax = plt.axes(projection = proj)

    # Plot data probablity SOI
    p = xr.open_dataarray("tugas2_pr/periode_ulang_gum_" + period[i] + "thn.nc").plot(ax=ax, # Sumbu axis yang digunakan 
               transform = proj, # Transformasi data ke dalam proyeksi peta
               levels = np.arange(80, 160, 10), # Colorbar kontur
               cmap = "RdYlBu", # Colormap
               cbar_kwargs = {"orientation" : "horizontal", "shrink" : 0.6}, # Pengaturan colorbar
               )

    # Fitur tambahan lainnya
    ax.set_title("Periode Ulang Curah Hujan " + period[i] + " Tahun", fontsize = 20, fontweight = "bold") # Judul Peta
    ax.coastlines(resolution = "10m", linewidth = 1.3, color = "black") # Garis pantai
    ocean = crt.feature.NaturalEarthFeature("physical", "ocean", scale = "10m", edgecolor = "none", facecolor = "white") # Laut
    ax.add_feature(ocean, linewidth = 0.2)

    # Set area gambar
    ax.set_extent([93, 138.75, 13.5, -13.5]) # Peta Indonesia

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
    p.colorbar.set_label(label = "Curah Hujan (mm/hari)", size = 20, weight = "bold")
    
    # Save gambar
    fig.savefig("tugas2_pr/" + period[i] + ".png", dpi = 300)
    plt.clf() # Reset gambar
    
