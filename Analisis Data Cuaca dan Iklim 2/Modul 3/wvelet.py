def wvelet(data, var, t2, mom, kota):
    """
    Fungsi hitung dan plot wavelet
    data    : Data time series 1D (array float)
    var     : Variansi data (float)
    t2      : Data waktu (array)
    mom     : Jenis motherwavelet (str)
    kota    : Kota kajian untuk judul (str)
    """
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    import numpy as np
    import waveletFunctions as wf
    import pandas as pd

    # Autokorelasi lag-1
    a = pd.Series(data)
    b = a.shift(1)
    df = pd.concat([a, b], axis=1)
    corr = df.corr()

    # Wavelet
    # a. Pengaturan parameter wavelet
    n = len(data)
    dt = 1/12
    pad = 0                                         # Set nilai 0 untuk time series
    dj = 0.25                                       # Kenaikan scale
    s0 = 2 * dt                                     # Default, scale dimulai 2 bulanan. Scale terkecil wavelet
    j1 = 7 / dj                                     # Default, jumlah scale
    mother = mom                                    # Jenis wavelet

    # b. Transformasi wavelet
    wave, period, scale, coi = wf.wavelet(data, dt=dt, pad=pad, dj=dj, s0=s0, J1=j1, mother=mother)  # Fungsi perhitungan wavelet
    power = (np.abs(wave))**2  # Power wavelet spectrum
    global_wave_spec = power.sum(axis=1) / n  # Untuk global wavelet spectrum (time-average)

    # c. Pengaturan selang kepercayaan
    lag1 = corr[0][1]                  # Lag-1 autocorrelation untuk level signifikansi
    signif = wf.wave_signif(([var]), dt=dt, scale=scale, sigtest=0, lag1=lag1, mother=mother)  # Fungsi untuk signifikansi (sigtest=0, chi-square test)
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # Membuat array ukuran (J+1)*N
    sig95 = power / sig95 # Power dapat disebut signifikan, apabila rasio power dengan signif > 1

    # d. Spektrum wavelet global dan selang kepercayaan
    dof = n - scale # Degree of freedom
    global_signif = wf.wave_signif(var, dt=dt, scale=scale, sigtest=1, lag1=lag1, dof=dof, mother=mother) # Fungsi sigifikansi untuk global wavelet spectrum (sigtest=1)

    # e. Scale-average pada rentang periode 2 - 8 tahun
    avg = np.logical_and(scale >= 0.5, scale < 8)  # Logika True untuk rentang skala 2 - 8 tahun 
    if (mother == "DOG"):  # Conditional nilai Cdelta sesuai dengan motherwavelet-nya
        Cdelta = 1.966
    elif (mother == "PAUL"):    
        Cdelta = 1.132   
    elif (mother == "MORLET"):
        Cdelta = 0.776
    scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # Membuat array ukuran (J+1)*N
    scale_avg = power /scale_avg  # Sesuai dengan persamaan 24 di referensi
    scale_avg = dj * dt / Cdelta * sum(scale_avg[avg, :])  # Sesuai dengan persamaan 24 di referensi, untuk scale-average
    scale_signif = wf.wave_signif(var, dt=dt, scale=scale, sigtest=2, lag1=lag1, dof=([0.5,7.9]), mother=mother)  # Fungsi signifikansi untuk scale-average

    # Plot wavelet
    ## a. Time series
    fig = plt.figure(figsize=(9,10))
    gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0, hspace=0)

    plt.subplot(gs[0, 0:3])
    plt.plot(t2, data, color="blue")
    plt.xticks(t2[19::20], t2[19::20], rotation=30)
    plt.xlabel("Waktu (Tahun-Bulan)")
    plt.ylabel("Curah Hujan Normal (mm)")
    plt.title("a) Curah Hujan Normal Per Bulan Kota " + kota)

    ## b. Plot kontur wavelet power spectrum
    plt2 = plt.subplot(gs[1, 0:3])
    levels = np.linspace(0, power.max(), 11)
    CS = plt.contourf(t2, period, power, len(levels))
    im = plt.contourf(CS, levels=levels, cmap="Blues")
    plt.xlabel("Waktu (Tahun-Bulan)")
    plt.ylabel("Periode (Tahun)")
    plt.title("b) Wavelet Power Spectrum (mm$^2$)")
    # Kontur signifikansi
    plt.contour(t2, period, sig95, [-99, 1], colors="k")
    plt.xticks(t2[19::20], t2[19::20], rotation=30)
    # Area Cone-of-Influence (COI)
    ts = t2
    coi_area = np.concatenate([[4*np.max(scale)], coi, [4*np.max(scale)], [4*np.max(scale)]])
    ts_area = np.concatenate([[ts[0]], ts, [ts[-1]], [ts[0]]])
    L = plt.plot(ts_area, (coi_area), "grey", linewidth=1)
    F = plt.fill(ts_area, (coi_area), "grey", alpha=0.3, hatch="x")
    # Format untuk y-scale
    plt2.set_yscale("log", base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ScalarFormatter())
    plt2.ticklabel_format(axis="y", style="plain")
    plt2.invert_yaxis()

    # Plot global wavelet spectrum
    plt3 = plt.subplot(gs[1, -1])
    plt.plot(global_wave_spec, period)
    plt.plot(global_signif, period, "--")
    plt.xlabel("Power (mm$^2$)")
    plt.title("c) Global Wavelet Spectrum")
    plt.xlim([0, 1.25 * np.max(global_wave_spec)])
    # Format untuk y-scale
    plt.ylim([np.min(period), np.max(period)])
    plt3.set_yscale("log", base=2, subs=None)
    ax = plt.gca().yaxis
    ax.set_major_formatter(ScalarFormatter())
    plt3.ticklabel_format(axis="y", style="plain")
    plt3.invert_yaxis()

    # Plot 2 - 8 tahun scale-average time series
    plt.subplot(gs[2, 0:3])
    plt.plot(t2, scale_avg, "k")
    plt.xticks(t2[19::20], t2[19::20], rotation=30)
    plt.xlabel("Waktu (Tahun-Bulan)")
    plt.ylabel("Average Variance (mm$^2$)")
    plt.title("d) Plot 0.5 - 8 Tahun Scale Average Time Series")
    plt.plot([t2[0], t2[-1]], scale_signif+[0,0], "--")

    suptitle = plt.suptitle("Wavelet " + mother + " Kota " + kota, y=1.01, weight="bold", fontsize=15)
    fig.savefig("Minggu (11)/hasil/wavelet_" + mom + "_" + kota + ".png", bbox_extra_artists=(suptitle,), bbox_inches="tight")
    