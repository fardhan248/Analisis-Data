def filter(fs, cutoff, order, data, tipe, f):
    """
    Fungsi Filter
    fs      : Frekuensi sampel (float)
    cutoff  : Cut off frekuensi batas (float/array 1D)
    order   : Order/kemiringan design filter (int)
    data    : Data time series (array 1D)
    tipe    : Tipe filter low/high/bandpass (str)
    f       : Frekuensi sampel DFT (array 1D)
    """
    from scipy.fftpack import fft
    from scipy.signal import butter, filtfilt
    import numpy as np

    # Filtering Data
    ## Low/High/Band pass
    nyquist = 1/2 * fs
    normal_cut = cutoff/nyquist # normal_cutoff = cutoff/(1/2 * fs)

    b, a = butter(order, normal_cut, analog=False, btype=tipe) # Design filter
    data_lhb = filtfilt(b, a, data, padlen=None)

    F_lhb = fft(data_lhb)
    F_lhb = abs(F_lhb[np.where(f > 0)])
    return F_lhb

def filterplot(x, yF, yFF, cutoff, filt, save, namfile):
    """
    Fungsi plot hasil filter dan fft
    x       : Data sumbu x (array 1D)
    yF      : Data sumbu y plot Fourier (array 1D)
    yFF     : Data sumbu y plot hasil filter (array 1D)
    cutoff  : Cut off frekuensi batas (float/array 1D)
    filt    : Tipe filter Lowpass/Highpass/Bandpass (str)
    save    : Save plot atau tidak (True/False)
    namfile : Nama file (dapat di directory tertentu) beserta ekstensinya (str)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(15,13))
    ax1 = fig.add_subplot(211)

    ax1.semilogx(x, yF, color="blue", label="FFT")
    ax1.set_ylabel("Amplitudo", fontsize=15)
    ax1.set_title("Periodogram FFT", fontsize=20, weight="bold")
    ax1.set_xticks(np.arange(12, 12*(10+1), 12), np.arange(1, 10+1, 1))
    ax1.set_xlabel("Periode (Tahun)", fontsize=15)
    if (filt == "Bandpass"):
        plt.axvline(x=1/cutoff[0], linestyle="dashed", color="r", label="Cut Off", alpha=0.7)
        plt.axvline(x=1/cutoff[1], linestyle="dashed", color="r", alpha=0.7)
    else: # (filt /= "Bandpass")
        plt.axvline(x=1/cutoff, linestyle="dashed", color="r", label="Cut Off", alpha=0.7)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()

    ax = fig.add_subplot(212)

    ax.semilogx(x, yFF, color="blue", label="FFT " + filt)
    ax.set_ylabel("Amplitudo", fontsize=15)
    ax.set_title("Periodogram FFT " + filt, fontsize=20, weight="bold")
    ax.set_xticks(np.arange(12, 12*(10+1), 12), np.arange(1, 10+1, 1))
    ax.set_xlabel("Periode (Tahun)", fontsize=15)
    if (filt == "Bandpass"):
        plt.axvline(x=1/cutoff[0], linestyle="dashed", color="r", label="Cut Off", alpha=0.7)
        plt.axvline(x=1/cutoff[1], linestyle="dashed", color="r", alpha=0.7)
    else: # (filt /= "Bandpass")
        plt.axvline(x=1/cutoff, linestyle="dashed", color="r", label="Cut Off", alpha=0.7)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()

    if (save == True):
        fig.savefig(namfile, dpi=300)
