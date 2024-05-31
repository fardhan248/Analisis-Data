# Import modul
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import xskillscore as xs
import cartopy as crt

# Untuk mengatur tickmarks gambar
import matplotlib.ticker as mticker

# Menghilangkan pesan warning yang tidak esensial
import warnings
warnings.filterwarnings("ignore")

#=============================================
# Data yang digunakan adalah data model 1.CMCC-CM2-SR5 dan 2.MIROC6 L1

# Baca data
# MODEL 1
datdir = "../Data/model/"
modln = "CMCC-CM2-SR5"
dmodl0 = xr.open_dataarray(datdir + modln + "/pr_Amon_" + modln + "_dcppA-hindcast-ens_ina_196201-201912_L1.nc") # mm/s

# MODEL 2
modln2 = "MIROC6"
dmodl02 = xr.open_dataarray(datdir + modln2 + "/pr_Amon_" + modln2 + "_dcppA-hindcast-ens_ina_196201-202012_L1.nc") # mm/s

# Baca data observasi
dobs0 = xr.open_dataarray("../Data/MSWEP_MON_INA_197902-202011.nc") # Data akumulasi bulanan dalam mm/bulan

#====================================================
# Agregat data

# MODEL 1
# Konversi ke akumulasi bulanan
ndays = dmodl0.time.dt.days_in_month
dmodl1 = dmodl0*24*3600*ndays

# Konversi ke akumulasi tahunan
dmodl = dmodl1.resample(time = "YS").sum() # mm/tahun

# Ubah datetime format, memastikan format waktunya
if np.issubdtype(dmodl["time"].dtype, np.datetime64):
    print("not cftime")
else:
    print("convert cftime") # cf: Climate forecast
    dmodl["time"] = dmodl.indexes["time"].to_datetimeindex()

# MODEL 2
# Konversi ke akumulasi bulanan
ndays2 = dmodl02.time.dt.days_in_month
dmodl12 = dmodl02*24*3600*ndays2

# Konversi ke akumulasi tahunan
dmodl2 = dmodl12.resample(time = "YS").sum() # mm/tahun

# Ubah datetime format, memastikan format waktunya
if np.issubdtype(dmodl2["time"].dtype, np.datetime64):
    print("not cftime")
else:
    print("convert cftime") # cf: Climate forecast
    dmodl2["time"] = dmodl2.indexes["time"].to_datetimeindex()

# Resample data observasi
dobs = dobs0.resample(time = "YS").sum() # mm/tahun

#========================================================
# Interpolasi data observasi dan olah data anomali

# MODEL 1
ts_modl = dmodl.sel(lat = slice(-12, -4), lon = slice(100, 125)).sel(time = slice("1980", "2019"))
ts_obs = dobs.sel(lat = slice(-2, -14), lon = slice(98, 127)).sel(time = slice("1980", "2019"))

# Perhitungan anomali
klim_obs = ts_obs.mean(dim = ["time"])
klim_modl = ts_modl.mean(dim = ["time"])

anom_obs = ts_obs - klim_obs
anom_modl = ts_modl - klim_modl
ens_anom = anom_modl.mean(dim = "realization")

# Interpolasi koordinat latitude dan longitude observasi menyesuaikan model
anom_obs_interp = anom_obs.interp(lon = anom_modl.lon, lat = anom_modl.lat, method = "cubic")

nx = len(ens_anom.lon) # jumlah grid kolom longitude
ny = len(ens_anom.lat) # jumlah grid kolom latitude
lon_x = ens_anom.lon
lat_y = ens_anom.lat

# MODEL 2
ts_modl2 = dmodl2.sel(lat = slice(-12, -4), lon = slice(100, 125)).sel(time = slice("1980", "2019"))
ts_obs2 = dobs.sel(lat = slice(-2, -14), lon = slice(98, 127)).sel(time = slice("1980", "2019"))

# Perhitungan anomali
klim_obs2 = ts_obs2.mean(dim = ["time"])
klim_modl2 = ts_modl2.mean(dim = ["time"])

anom_obs2 = ts_obs2 - klim_obs2
anom_modl2 = ts_modl2 - klim_modl2
ens_anom2 = anom_modl2.mean(dim = "realization")

# Interpolasi koordinat latitude dan longitude observasi menyesuaikan model
anom_obs_interp2 = anom_obs2.interp(lon = anom_modl2.lon, lat = anom_modl2.lat, method = "cubic")

nx2 = len(ens_anom2.lon) # jumlah grid kolom longitude
ny2 = len(ens_anom2.lat) # jumlah grid kolom latitude
lon_x2 = ens_anom2.lon
lat_y2 = ens_anom2.lat

#==============================================
# Peta deterministik

# Kontinu: RMSE, Corr (Pearson)
# MODEL 1
cor = xr.DataArray(np.zeros((ny, nx)), dims = ("lat", "lon"), coords = {"lat" : lat_y, "lon" : lon_x})
rmse = xr.DataArray(np.zeros((ny, nx)), dims = ("lat", "lon"), coords = {"lat" : lat_y, "lon" : lon_x})
for i in range(nx):
    for j in range(ny):
        # Korelasi
        korr = xr.corr(ens_anom.sel(lon = lon_x[i], lat = lat_y[j]), anom_obs_interp.sel(lon = lon_x[i], lat = lat_y[j]))
        cor[j, i] = korr
        
        # RMSE
        rmse[j, i] = xs.rmse(ens_anom.sel(lon = lon_x[i], lat = lat_y[j]), anom_obs_interp.sel(lon = lon_x[i], lat = lat_y[j]))

cor.to_netcdf("korelasi1.nc")
rmse.to_netcdf("rmse1.nc")

# MODEL 2
cor2 = xr.DataArray(np.zeros((ny2, nx2)), dims = ("lat", "lon"), coords = {"lat" : lat_y2, "lon" : lon_x2})
rmse2 = xr.DataArray(np.zeros((ny2, nx2)), dims = ("lat", "lon"), coords = {"lat" : lat_y2, "lon" : lon_x2})
for i in range(nx2):
    for j in range(ny2):
        # Korelasi
        korr = xr.corr(ens_anom2.sel(lon = lon_x2[i], lat = lat_y2[j]), anom_obs_interp2.sel(lon = lon_x2[i], lat = lat_y2[j]))
        cor2[j, i] = korr
        
        # RMSE
        rmse2[j, i] = xs.rmse(ens_anom2.sel(lon = lon_x2[i], lat = lat_y2[j]), anom_obs_interp2.sel(lon = lon_x2[i], lat = lat_y2[j]))

cor2.to_netcdf("korelasi2.nc")
rmse2.to_netcdf("rmse2.nc")

# Plot
# Nama file dan judul gambar
model = ["korelasi1", "korelasi2"]
jdl = ["Korelasi Model 1", "Korelasi Model 2"]

# Looping membuat peta
for i in range(2):
    # Proyeksi peta
    proj = crt.crs.PlateCarree()

    # figure dan axis
    fig = plt.figure(figsize = (16, 8))
    ax = plt.axes(projection = proj)

    # Plot data probablity SOI
    p = xr.open_dataarray(model[i] + ".nc").plot(ax=ax, # Sumbu axis yang digunakan 
               transform = proj, # Transformasi data ke dalam proyeksi peta
               levels = np.arange(-1, 1, 0.1), # Colorbar kontur
               cmap = "PiYG", # Colormap
               cbar_kwargs = {"orientation" : "horizontal", "shrink" : 0.6}, # Pengaturan colorbar
               )

    # Fitur tambahan lainnya
    ax.set_title(jdl[i], fontsize = 20, fontweight = "bold") # Judul Peta
    ax.coastlines(resolution = "10m", linewidth = 1.3, color = "black") # Garis pantai
    
    # Set area gambar
    ax.set_extent([104, 125, -12, -4]) # Peta Indonesia

    # Set gridline dan label peta
    gl = p.axes.gridlines(draw_labels = True, linestyle = ":", linewidth = 2, color = "gray")
    gl.xlocator = mticker.FixedLocator([100, 105, 110, 115, 120, 125])
    gl.ylocator = mticker.FixedLocator([-15, -13, -11, -9, -7, -5, -3])
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True
    gl.xlabel_style = {"size": 20, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"size": 20, "color": "black", "weight": "bold"}

    # Pengaturan colorbar
    p.colorbar.ax.tick_params(labelsize = 20)
    p.colorbar.set_label(label = jdl[i], size = 20, weight = "bold")
    
    # Save gambar
    fig.savefig(model[i] + ".png", dpi = 300)
    plt.clf() # Reset gambar

# Nama file dan judul gambar
model = ["rmse1", "rmse2"]
jdl = ["RMSE Model 1", "RMSE Model 2"]

# Looping membuat peta
for i in range(2):
    # Proyeksi peta
    proj = crt.crs.PlateCarree()

    # figure dan axis
    fig = plt.figure(figsize = (16, 8))
    ax = plt.axes(projection = proj)

    # Plot data probablity SOI
    p = xr.open_dataarray(model[i] + ".nc").plot(ax=ax, # Sumbu axis yang digunakan 
               transform = proj, # Transformasi data ke dalam proyeksi peta
               levels = np.arange(100, 1000, 100), # Colorbar kontur
               cmap = "YlGnBu", # Colormap
               cbar_kwargs = {"orientation" : "horizontal", "shrink" : 0.6}, # Pengaturan colorbar
               )

    # Fitur tambahan lainnya
    ax.set_title(jdl[i], fontsize = 20, fontweight = "bold") # Judul Peta
    ax.coastlines(resolution = "10m", linewidth = 1.3, color = "black") # Garis pantai
    
    # Set area gambar
    ax.set_extent([104, 125, -12, -4]) # Peta Indonesia

    # Set gridline dan label peta
    gl = p.axes.gridlines(draw_labels = True, linestyle = ":", linewidth = 2, color = "gray")
    gl.xlocator = mticker.FixedLocator([100, 105, 110, 115, 120, 125])
    gl.ylocator = mticker.FixedLocator([-15, -13, -11, -9, -7, -5, -3])
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True
    gl.xlabel_style = {"size": 20, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"size": 20, "color": "black", "weight": "bold"}

    # Pengaturan colorbar
    p.colorbar.ax.tick_params(labelsize = 20)
    p.colorbar.set_label(label = jdl[i], size = 20, weight = "bold")
    
    # Save gambar
    fig.savefig(model[i] + ".png", dpi = 300)
    plt.clf() # Reset gambar

#================================================
# Peta deterministik

# Kontinu: Contingency (PC, TS)
# MODEL 1
pc = xr.DataArray(np.zeros((ny, nx)), dims = ("lat", "lon"), coords = {"lat" : lat_y, "lon" : lon_x})
ts = xr.DataArray(np.zeros((ny, nx)), dims = ("lat", "lon"), coords = {"lat" : lat_y, "lon" : lon_x})
for i in range(nx):
    for j in range(ny):
        # Kategori: True/1 jika anomali positif (anom > 0) dan False/0 jika anomali negatif (anom < 0)
        obs = (anom_obs_interp.sel(lon = lon_x[i], lat = lat_y[j]) > 0).values*1 # Observasi: True/1 dan False/0
        fcst = (ens_anom.sel(lon = lon_x[i], lat = lat_y[j]) > 0).values*1 # Ens. mean: True/1 dan False/0

        # Simpan obs dan fcst dalam dataframe
        df = pd.DataFrame({"OBS": pd.Series(obs), "FCST": pd.Series(fcst)})

        # Contingency table
        ct = pd.crosstab(df["FCST"], df["OBS"])

        a = ct[1][1] # Hit: obs = 1 dan fcst = 1
        b = ct[0][1] # False alarm: obs = 0 dan fcst = 1
        c = ct[1][0] # Miss: obs = 1 dan fcst = 0
        d = ct[0][0] # Correct negative: obs = 0 dan fcst = 0
        n = a + b + c + d # Total sample
        
        pc[j, i] = (a + d)/n # Percent of correct
        ts[j, i] = a/(a + b + c) # Threat score

pc.to_netcdf("pc1.nc")
ts.to_netcdf("ts1.nc")

# MODEL 2
pc2 = xr.DataArray(np.zeros((ny2, nx2)), dims = ("lat", "lon"), coords = {"lat" : lat_y2, "lon" : lon_x2})
ts2 = xr.DataArray(np.zeros((ny2, nx2)), dims = ("lat", "lon"), coords = {"lat" : lat_y2, "lon" : lon_x2})
for i in range(nx2):
    for j in range(ny2):
        # Kategori: True/1 jika anomali positif (anom > 0) dan False/0 jika anomali negatif (anom < 0)
        obs = (anom_obs_interp2.sel(lon = lon_x2[i], lat = lat_y2[j]) > 0).values*1 # Observasi: True/1 dan False/0
        fcst = (ens_anom2.sel(lon = lon_x2[i], lat = lat_y2[j]) > 0).values*1 # Ens. mean: True/1 dan False/0

        # Simpan obs dan fcst dalam dataframe
        df = pd.DataFrame({"OBS": pd.Series(obs), "FCST": pd.Series(fcst)})

        # Contingency table
        ct = pd.crosstab(df["FCST"], df["OBS"])

        a = ct[1][1] # Hit: obs = 1 dan fcst = 1
        b = ct[0][1] # False alarm: obs = 0 dan fcst = 1
        c = ct[1][0] # Miss: obs = 1 dan fcst = 0
        d = ct[0][0] # Correct negative: obs = 0 dan fcst = 0
        n = a + b + c + d # Total sample
        
        pc2[j, i] = (a + d)/n # Percent of correct
        ts2[j, i] = a/(a + b + c) # Threat score

pc2.to_netcdf("pc2.nc")
ts2.to_netcdf("ts2.nc")

# Plot
# Nama file dan judul gambar
model = ["pc1", "pc2", "ts1", "ts2"]
jdl = ["Percent of Correct Model 1", "Percent of Correct Model 2", "Threat Score Model 1", "Threat Score Model 2"]

# Looping membuat peta
for i in range(4):
    # Proyeksi peta
    proj = crt.crs.PlateCarree()

    # figure dan axis
    fig = plt.figure(figsize = (16, 8))
    ax = plt.axes(projection = proj)

    # Plot data probablity SOI
    p = xr.open_dataarray(model[i] + ".nc").plot(ax=ax, # Sumbu axis yang digunakan 
               transform = proj, # Transformasi data ke dalam proyeksi peta
               levels = np.arange(0, 1, 0.02), # Colorbar kontur
               cmap = "YlGnBu", # Colormap
               cbar_kwargs = {"orientation" : "horizontal", "shrink" : 0.6}, # Pengaturan colorbar
               )

    # Fitur tambahan lainnya
    ax.set_title(jdl[i], fontsize = 20, fontweight = "bold") # Judul Peta
    ax.coastlines(resolution = "10m", linewidth = 1.3, color = "black") # Garis pantai
    
    # Set area gambar
    ax.set_extent([104, 125, -12, -4]) # Peta Indonesia

    # Set gridline dan label peta
    gl = p.axes.gridlines(draw_labels = True, linestyle = ":", linewidth = 2, color = "gray")
    gl.xlocator = mticker.FixedLocator([100, 105, 110, 115, 120, 125])
    gl.ylocator = mticker.FixedLocator([-15, -13, -11, -9, -7, -5, -3])
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True
    gl.xlabel_style = {"size": 20, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"size": 20, "color": "black", "weight": "bold"}

    # Pengaturan colorbar
    p.colorbar.ax.tick_params(labelsize = 20)
    p.colorbar.set_label(label = jdl[i], size = 20, weight = "bold")
    
    # Save gambar
    fig.savefig(model[i] + ".png", dpi = 300)
    plt.clf() # Reset gambar

#===========================================================
# Peta Probabilistik

# Brier Score (BS)
# MODEL 1
bs = xr.DataArray(np.zeros((ny, nx)), dims = ("lat", "lon"), coords = {"lat" : lat_y, "lon" : lon_x})
for i in range(nx):
    for j in range(ny):
        # Tentukan threshold
        thold = 0.0

        # Probability forecast (forecast > thold)
        fcst = (anom_modl.sel(lon = lon_x[i], lat = lat_y[j]) > 0).sum(dim = "realization")/10
        obs = (anom_obs_interp.sel(lon = lon_x[i], lat = lat_y[j]) > 0)*1 
        
        # Brier Score
        bs[j, i] = xs.brier_score(obs, fcst).values

bs.to_netcdf("bs1.nc")

# MODEL 2
bs2 = xr.DataArray(np.zeros((ny2, nx2)), dims = ("lat", "lon"), coords = {"lat" : lat_y2, "lon" : lon_x2})
for i in range(nx2):
    for j in range(ny2):
        # Tentukan threshold
        thold = 0.0

        # Probability forecast (forecast > thold)
        fcst = (anom_modl2.sel(lon = lon_x2[i], lat = lat_y2[j]) > 0).sum(dim = "realization")/10
        obs = (anom_obs_interp2.sel(lon = lon_x2[i], lat = lat_y2[j]) > 0)*1 
        
        # Brier Score
        bs2[j, i] = xs.brier_score(obs, fcst).values

bs2.to_netcdf("bs2.nc")

# Plot
# Nama file dan judul gambar
model = ["bs1", "bs2"]
jdl = ["Brier Score Model 1", "Brier Score Model 2"]

# Looping membuat peta
for i in range(2):
    # Proyeksi peta
    proj = crt.crs.PlateCarree()

    # figure dan axis
    fig = plt.figure(figsize = (16, 8))
    ax = plt.axes(projection = proj)

    # Plot data probablity SOI
    p = xr.open_dataarray(model[i] + ".nc").plot(ax=ax, # Sumbu axis yang digunakan 
               transform = proj, # Transformasi data ke dalam proyeksi peta
               levels = np.arange(0, 1, 0.02), # Colorbar kontur
               cmap = "YlGnBu", # Colormap
               cbar_kwargs = {"orientation" : "horizontal", "shrink" : 0.6}, # Pengaturan colorbar
               )

    # Fitur tambahan lainnya
    ax.set_title(jdl[i], fontsize = 20, fontweight = "bold") # Judul Peta
    ax.coastlines(resolution = "10m", linewidth = 1.3, color = "black") # Garis pantai
    
    # Set area gambar
    ax.set_extent([104, 125, -12, -4]) # Peta Indonesia

    # Set gridline dan label peta
    gl = p.axes.gridlines(draw_labels = True, linestyle = ":", linewidth = 2, color = "gray")
    gl.xlocator = mticker.FixedLocator([100, 105, 110, 115, 120, 125])
    gl.ylocator = mticker.FixedLocator([-15, -13, -11, -9, -7, -5, -3])
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True
    gl.xlabel_style = {"size": 20, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"size": 20, "color": "black", "weight": "bold"}

    # Pengaturan colorbar
    p.colorbar.ax.tick_params(labelsize = 20)
    p.colorbar.set_label(label = jdl[i], size = 20, weight = "bold")
    
    # Save gambar
    fig.savefig(model[i] + ".png", dpi = 300)
    plt.clf() # Reset gambar

#=====================================================
# Peta Probabilistik

# Brier Skill Score (BSS)
# MODEL 1
bss = xr.DataArray(np.zeros((ny, nx)), dims = ("lat", "lon"), coords = {"lat" : lat_y, "lon" : lon_x})
for i in range(nx):
    for j in range(ny):
        # Tentukan threshold
        thold = 0.0

        # Probability forecast (forecast > thold)
        fcst = (anom_modl.sel(lon = lon_x[i], lat = lat_y[j]) > 0).sum(dim = "realization")/10
        obs = (anom_obs_interp.sel(lon = lon_x[i], lat = lat_y[j]) > 0)*1 
        clim = np.sum(obs)/obs.size
        fcst_ref = np.ones(len(fcst.values))*clim.values

        # Dataframe
        df_ref = pd.DataFrame({"OBS": pd.Series(obs), "FCST": pd.Series(fcst_ref)})

        # BS Referensi
        bs_ref = xs.brier_score(df_ref["OBS"].values, df_ref["FCST"].values)

        # Brier Score
        bs = xs.brier_score(obs, fcst).values

        # BSS: Peningkatan akurasi relatif terhadap suatu referensi (%)
        bss[j, i] = (1 - (bs/bs_ref))*100
 
bss.to_netcdf("bss1.nc")

# MODEL 2
bss2 = xr.DataArray(np.zeros((ny2, nx2)), dims = ("lat", "lon"), coords = {"lat" : lat_y2, "lon" : lon_x2})
for i in range(nx2):
    for j in range(ny2):
        # Tentukan threshold
        thold = 0.0

        # Probability forecast (forecast > thold)
        fcst = (anom_modl2.sel(lon = lon_x2[i], lat = lat_y2[j]) > 0).sum(dim = "realization")/10
        obs = (anom_obs_interp2.sel(lon = lon_x2[i], lat = lat_y2[j]) > 0)*1 
        clim = np.sum(obs)/obs.size
        fcst_ref = np.ones(len(fcst.values))*clim.values

        # Dataframe
        df_ref = pd.DataFrame({"OBS": pd.Series(obs), "FCST": pd.Series(fcst_ref)})

        # BS Referensi
        bs_ref = xs.brier_score(df_ref["OBS"].values, df_ref["FCST"].values)

        # Brier Score
        bs = xs.brier_score(obs, fcst).values

        # BSS: Peningkatan akurasi relatif terhadap suatu referensi (%)
        bss2[j, i] = (1 - (bs/bs_ref))*100
 
bss2.to_netcdf("bss2.nc")

# Plot
# Nama file dan judul gambar
model = ["bss1", "bss2"]
jdl = ["Brier Skill Score Model 1", "Brier Skill Score Model 2"]

# Looping membuat peta
for i in range(2):
    # Proyeksi peta
    proj = crt.crs.PlateCarree()

    # figure dan axis
    fig = plt.figure(figsize = (16, 8))
    ax = plt.axes(projection = proj)

    # Plot data probablity SOI
    p = xr.open_dataarray(model[i] + ".nc").plot(ax=ax, # Sumbu axis yang digunakan 
               transform = proj, # Transformasi data ke dalam proyeksi peta
               levels = np.arange(-80, 30, 5), # Colorbar kontur
               cmap = "YlGnBu", # Colormap
               cbar_kwargs = {"orientation" : "horizontal", "shrink" : 0.6}, # Pengaturan colorbar
               )

    # Fitur tambahan lainnya
    ax.set_title(jdl[i], fontsize = 20, fontweight = "bold") # Judul Peta
    ax.coastlines(resolution = "10m", linewidth = 1.3, color = "black") # Garis pantai
    
    # Set area gambar
    ax.set_extent([104, 125, -12, -4]) # Peta Indonesia

    # Set gridline dan label peta
    gl = p.axes.gridlines(draw_labels = True, linestyle = ":", linewidth = 2, color = "gray")
    gl.xlocator = mticker.FixedLocator([100, 105, 110, 115, 120, 125])
    gl.ylocator = mticker.FixedLocator([-15, -13, -11, -9, -7, -5, -3])
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True
    gl.xlabel_style = {"size": 20, "color": "black", "weight": "bold"}
    gl.ylabel_style = {"size": 20, "color": "black", "weight": "bold"}

    # Pengaturan colorbar
    p.colorbar.ax.tick_params(labelsize = 20)
    p.colorbar.set_label(label = jdl[i], size = 20, weight = "bold")
    
    # Save gambar
    fig.savefig(model[i] + ".png", dpi = 300)
    plt.clf() # Reset gambar
    
