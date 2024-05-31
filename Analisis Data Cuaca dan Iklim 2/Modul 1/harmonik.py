class harmonik:
    def __init__(self, Ak, Bk, Ck, phik, y):
        self.A = Ak
        self.B = Bk
        self.C = Ck
        self.phi = phik
        self.ykt = y

def hitung_harmonik(data):
    """
    Perhitungan harmonik
    data : time series 1D
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression as lr 
    n = len(data)
    waktu = np.arange(1, n+1)
    y = np.zeros((int(n/2), n))*np.nan
    phik = np.zeros(6)*np.nan
    x = np.empty((0, n))

    for k in range(int(n/2)):
        x1 = np.cos(2 * np.pi * (k+1) * waktu / n)
        x2 = np.sin(2 * np.pi * (k+1) * waktu / n)
        x = np.vstack((x, x1, x2))

    x = x.transpose()
    lin = lr().fit(x, data)
    koef = lin.coef_
    Ak = koef[0:n:2]
    Bk = koef[1:n:2]
    Ck = np.sqrt(Ak**2 + Bk**2)

    for k in range(int(n/2)):
        if Ak[k] > 0:
            phik[k] = np.arctan(Bk[k]/Ak[k])
        elif Ak[k] < 0:
            phik[k] = np.arctan(Bk[k]/Ak[k]) + np.pi
        else: # Ak[k] = 0
            phik[k] = np.pi/2

        y[k, :] = Ck[k] * np.cos((2 * np.pi * waktu * (k+1) / n) - phik[k])
    return harmonik(Ak, Bk, Ck, phik, y)

def plot2d(data, lat, lon, judul, clabel, direc, level):
    """"
    Plot kontur, data 2D
    data   : Data 2D
    lat    : Latitude (1D)
    lon    : Longitude (1D)
    judul  : Judul plot (string)
    clabel : Judul colorbar (string)
    direc  : Direktori save image (string)
    level  : Nilai colorbar (array)
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt

    prj = cimgt.GoogleTiles()
    prj2 = ccrs.PlateCarree()

    plt.figure(figsize=(11, 8.5))
    ax = plt.axes(projection=prj2)
    plot = ax.contourf(lon, lat, data, levels=level, trasform=prj2, cmap="gnuplot", alpha=0.7)

    cbar = plt.colorbar(plot, orientation="vertical")
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label=clabel, size=15)
    ax.add_image(prj, 10)
    ax.set_title(judul, fontsize=20, weight="bold")
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]])
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")

    gl = ax.gridlines(draw_labels=True, linewidth=0.9, color="gray", alpha=0.8, linestyle="--")
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style = {"size":15, "label":"Longitude"}
    gl.ylabel_style = {"size":15}
    plt.tight_layout()
    plt.savefig(direc, dpi=300)
