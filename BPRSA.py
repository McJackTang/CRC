import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt
import os
from scipy.signal import hilbert
from antropy import sample_entropy  # pip install antropy

# Normalization function (-1, 1)
def normalize(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = x * 2 - 1
    return x

def plot_hr_with_anchors(hr, anchors, fs, title="HR Signal with Anchors"):
    t = np.arange(len(hr)) / fs
    plt.figure(figsize=(25, 6))
    plt.plot(t, hr, label='HR Signal')
    plt.plot(t[anchors], hr[anchors], 'ro', label='Anchors', markersize=3)
    plt.xlabel('Time (s)')
    plt.ylabel('HR (normalized)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_each_seg(segs, wL, wH, fs, title_prefix="Segment", save_dir=r"C:\Users\康萌\Desktop\PILab_Internship\呼吸信号片段"):
    # Create directory for saving images if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    t_win = np.arange(-wL, wH) / fs
    for i, seg in enumerate(segs):
        plt.figure(figsize=(6, 3))
        plt.plot(t_win, seg, color='C0')
        plt.plot(0, seg[wL], 'ro', markersize=6, label='Anchor')
        plt.axvline(0, color='k', ls='--', lw=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f"{title_prefix} #{i+1}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{title_prefix}_{i+1}.png"))
        plt.close()

def plot_segs_with_anchors(segs, wL, wH, fs, bprsa, title="All Segments with Anchors"):
    t_win = np.arange(-wL, wH) / fs
    plt.figure(figsize=(20, 6))
    # Only plot segments with index as multiples of 50
    for i in range(0, len(segs), 50):
        seg = segs[i]
        plt.plot(t_win, seg, alpha=0.7, color='gray', label=f'seg' if i == 0 else None)
        plt.plot(0, seg[wL], 'ro', markersize=5)
    if bprsa is not None:
        plt.plot(t_win, bprsa, color='C1', lw=2, label='BPRSA')
    plt.axvline(0, color='k', ls='--', lw=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def find_anchors(hr, T_pts, kind='dec'):
    """
    Find anchor points in HR sequence.
    hr: HR sequence (bpm)
    T_pts: window size (number of samples)
    kind: 'dec' for HR decrease, 'acc' for HR increase
    """
    N = len(hr)
    idxs = np.arange(T_pts, N - T_pts)
    back = np.array([hr[i-T_pts:i].mean() for i in idxs])
    forw = np.array([hr[i:i+T_pts].mean() for i in idxs])
    if kind == 'dec':
        mask = forw < back
    else:
        mask = forw > back
    plot_hr_with_anchors(hr, idxs[mask], fs=60, title=f"Anchors for {kind} events T={T_pts/60} s")
    return idxs[mask]


def bprsa_localmean(anchor_sig, target_sig, fs, T, L, H, direction='dec'):
    """
    BPRSA using local mean anchor detection.
    """
    T_pts = int(T * fs)
    anchors = find_anchors(anchor_sig, T_pts, kind=direction)
    wL, wH = int(L * fs), int(H * fs)
    segs = []
    for a in anchors:
        if a - wL >= 0 and a + wH < len(target_sig):
            segs.append(target_sig[a-wL:a+wH])
    segs = np.array(segs)
    if segs.size == 0:
        raise RuntimeError("No complete anchor segments found. Please check T, L, H settings.")
    print(f"Found {len(segs)} complete anchor segments.")
    bprsa_curve = segs.mean(axis=0)
    t_win = np.arange(-wL, wH) / fs
    plot_segs_with_anchors(segs, wL, wH, fs, normalize(bprsa_curve), title=f"BPRSA for {direction} events T={T} s")
    # Compute normalized PSD
    sig = bprsa_curve - bprsa_curve.mean()
    f, Pxx = welch(sig, fs=fs, nperseg=min(1024, len(sig)))
    mask = (f >= 0) & (f <= 3)
    f_0_3 = f[mask]
    Pxx_0_3 = Pxx[mask]
    Pxx_norm = Pxx_0_3 / np.trapz(Pxx_0_3, f_0_3) * 100
    return t_win, bprsa_curve, f_0_3, Pxx_norm

def plot_prsa(t, prsa, f, pxx_norm, kind='dec', signal='HR'):
    """
    Plot PRSA curve and its normalized PSD.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(t, prsa, lw=2, label='{} {} events'.format(signal, '↓' if kind == 'dec' else '↑'))
    axes[0].axvline(0, color='k', lw=1, ls='--')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel(f'{signal} deviation (bpm)')
    if signal == 'RESP':
        axes[0].set_title(f'PRSA: {signal} {"expiratory" if kind == "dec" else "inspiratory"}')
    else:
        axes[0].set_title(f'PRSA: {signal} {"dec" if kind == "dec" else "acc"}')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(f, pxx_norm, lw=2)
    axes[1].axvline(0, color='k', lw=1, ls='--')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Normalized PSD (%)')
    axes[1].set_title(f'PRSA: {signal} {kind}')
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

def plot_bprsa_axes(ax0, ax1, t1, bprsa, t2, prsa, kind='dec', signal='HR', target_signal='RESP'):
    """
    Plot BPRSA and PRSA curves on given axes.
    ax0: BPRSA curve
    ax1: PRSA curve
    """
    ax0.plot(t1, bprsa, lw=2)
    ax0.axvline(0, color='k', lw=1, ls='--')
    ax0.set_xlabel('Time (s)')
    ax0.grid(True)
    if target_signal == 'RESP':
        ax0.set_ylabel(f'{target_signal} amp(au)')
    if signal == 'RESP':
        ax0.set_title(f'BPRSA: {signal} {"expiratory" if kind == "dec" else "inspiratory"} → {target_signal} response')
    else:
        ax0.set_title(f'BPRSA: BVP {"dec" if kind == "dec" else "acc"} → {target_signal} response')
    ax1.plot(t2, prsa, lw=2, label='{} {} events'.format(signal, '↓' if kind == 'dec' else '↑'))
    ax1.axvline(0, color='k', lw=1, ls='--')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'{signal} deviation')
    ax1.set_title(f'PRSA: {signal} {kind}')
    ax1.legend()
    ax1.grid(True)

def plot_bprsa(t, prsa, f, pxx_norm, kind='dec', signal='HR', target_signal='RESP'):
    """
    Plot BPRSA curve and its normalized PSD.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(t, prsa, lw=2)
    axes[0].axvline(0, color='k', lw=1, ls='--')
    axes[0].set_xlabel('Time (s)')
    if target_signal == 'RESP':
        axes[0].set_ylabel(f'{target_signal} amp(au)')
    if signal == 'RESP':
        axes[0].set_title(f'BPRSA: {signal} {"expiratory" if kind == "dec" else "inspiratory"} → {target_signal} response')
    else:
        axes[0].set_title(f'BPRSA: {signal} {"dec" if kind == "dec" else "acc"} → {target_signal} response')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(f, pxx_norm, lw=2)
    axes[1].axvline(0, color='k', lw=1, ls='--')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Normalized PSD (%)')
    axes[1].set_title(f'PRSA: {signal} {kind}')
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    """
    Apply a bandpass filter to the data.
    """
    if fs is None or fs <= 0:
        return np.zeros_like(data)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_prsa_metrics(t_win, prsa_curve, fs, kind='dec', small_win=5, large_win=50):
    """
    Compute PRSA metrics:
      kind='dec' -> [IDR, SDR, ADR]
      kind='acc' -> [IAR, SAR, AAR]
    Parameters:
    - t_win:      time axis array, length == prsa_curve.size
    - prsa_curve: PRSA curve (deviation relative to anchor)
    - fs:         sampling rate (Hz)
    - kind:       'dec' for deceleration metrics; 'acc' for acceleration metrics
    - small_win:  "instantaneous" window size, in samples (default 5)
    - large_win:  "average" window size, in samples (default 50)
    Returns:
      If kind='dec', returns np.array([IDR, SDR, ADR])
      If kind='acc', returns np.array([IAR, SAR, AAR])
    """
    center_idx = np.argmin(np.abs(t_win))
    N = prsa_curve.size
    before_idx = np.arange(center_idx - small_win, center_idx)
    after_idx  = np.arange(center_idx,         center_idx + small_win)
    if before_idx[0] < 0 or after_idx[-1] >= N:
        raise ValueError("small_win too large, out of bounds")
    win_before = prsa_curve[before_idx]
    win_after  = prsa_curve[after_idx]
    # 1) IDR, IAR
    IDR = np.ptp(win_after)
    IAR = np.ptp(win_before)
    # 2) SDR, SAR
    idx_max_a = after_idx[np.argmax(win_after)]
    idx_min_a = after_idx[np.argmin(win_after)]
    SDR = (prsa_curve[idx_max_a] - prsa_curve[idx_min_a]) / (abs(idx_max_a - idx_min_a) / fs)
    idx_max_b = before_idx[np.argmax(win_before)]
    idx_min_b = before_idx[np.argmin(win_before)]
    SAR = (prsa_curve[idx_max_b] - prsa_curve[idx_min_b]) / (abs(idx_max_b - idx_min_b) / fs)
    # 3) ADR, AAR
    start_B = center_idx - large_win
    end_A   = center_idx + large_win
    if start_B < 0 or end_A > N:
        raise ValueError("large_win too large, out of bounds")
    mean_before = prsa_curve[start_B:center_idx].mean()
    mean_after  = prsa_curve[center_idx:center_idx + large_win].mean()
    ADR = mean_after - mean_before
    AAR = ADR  # Modify here if distinction is needed
    if kind == 'dec':
        return np.array([IDR, SDR, ADR])
    elif kind == 'acc':
        return np.array([IAR, SAR, AAR])
    else:
        raise ValueError("kind must be 'dec' or 'acc'")

def sample_entropy_custom(x, m, r):
    """
    Custom implementation of Sample Entropy (SampEn):
      x: 1D signal array
      m: embedding dimension
      r: tolerance (threshold)
    Returns SampEn value.
    """
    N = len(x)
    def _phi(m):
        X = np.array([x[i:i+m] for i in range(N-m+1)])
        C = []
        for i in range(len(X)):
            dist = np.max(np.abs(X - X[i]), axis=1)
            C.append((dist <= r).sum() - 1)
        return np.sum(C) / ((N-m+1)*(N-m))
    return -np.log(_phi(m+1) / _phi(m))

def compute_bprsa_metrics(t_win, bprsa_curve, fs, sampen_m=None):
    """
    Compute BPRSA metrics from respiratory curve:
      MRA, SAP, SampEn
    Parameters:
    - t_win:        time axis array, length == bprsa_curve.size
    - bprsa_curve:  BPRSA curve
    - fs:           sampling rate (Hz)
    - sampen_m:     SampEn embedding dimension, default is 0.5*fs
    Returns:
      metrics: shape=(3,) array, [MRA, SAP, SampEn]
    """
    # MRA = max−min
    MRA = bprsa_curve.max() - bprsa_curve.min()
    # SAP = instantaneous slope at center point, using central difference
    center_idx = np.argmin(np.abs(t_win))
    if center_idx == 0 or center_idx == len(bprsa_curve) - 1:
        raise ValueError("Curve too short to compute central difference")
    SAP = (bprsa_curve[center_idx+1] - bprsa_curve[center_idx-1]) / (2/fs)
    # SampEn
    if sampen_m is None:
        sampen_m = int(0.5 * fs)
    r = 0.2 * np.std(bprsa_curve)
    SampEn = sample_entropy_custom(bprsa_curve, sampen_m, r)
    return np.array([MRA, SAP, SampEn])
