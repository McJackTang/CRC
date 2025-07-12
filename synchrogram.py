import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import hilbert, find_peaks, welch, butter, filtfilt
from math import gcd
from scipy.fft import fft, fftfreq
from collections import Counter
import math

def simplify_ratio(n, m):
    """Simplify the ratio (n, m) by their greatest common divisor."""
    g = gcd(n, m)
    return (n // g, m // g)

def adjust_phase_to_0_mpi(phi_rr):
    """
    Unwrap phase to avoid phase jumps.
    """
    return np.unwrap(phi_rr)

def get_hr(y, fs=60, min=30, max=180):
    """
    Estimate heart rate (bpm) from signal using Welch's method.
    Args:
        y: Input signal.
        fs: Sampling frequency.
        min: Minimum heart rate (bpm).
        max: Maximum heart rate (bpm).
    Returns:
        Estimated heart rate in bpm.
    """
    p, q = welch(y, fs, nfft=int(1e6/fs), nperseg=np.min((len(y)-1, 512)))
    return p[(p > min/60) & (p < max/60)][np.argmax(q[(p > min/60) & (p < max/60)])] * 60

def calculate_dynamic_n_m_v2(hr_segment, rr_segment, fs=60):
    """
    Dynamically estimate n:m ratio:
      1) Estimate heart rate HR_bpm (30–180 bpm)
      2) Estimate respiratory rate VR_bpm (6–60 bpm)
      3) ratio = HR_bpm/VR_bpm, round to nearest integer for n, m=1

    Args:
        hr_segment: ECG or pulse signal segment (numpy array)
        rr_segment: Respiratory or volume signal segment (numpy array)
        fs: Sampling rate (Hz)
    Returns:
        ratio: Approximate integer ratio
    """
    HR_bpm = get_hr(hr_segment, fs, min=30, max=180)
    VR_bpm = get_hr(rr_segment, fs, min=6, max=30)
    ratio = HR_bpm / VR_bpm
    n = int(np.round(ratio))
    m = 1
    return ratio

def calculate_gamma(sync, ratio_0):
    """
    Calculate synchronization degree (gamma) for different n:m ratios.
    Args:
        sync: Phase array.
        ratio_0: Initial ratio.
    Returns:
        max_key: (n, m) with maximum gamma.
        max_gamma: Maximum gamma value.
    """
    n_m_sync_dict = {}
    n = ratio_0

    # m = 1
    mm = 1
    for nn in range(n - 1, n + 2):
        if nn <= 1:
            continue
        phi_rr_mod = np.mod(sync, 2 * np.pi * mm) / (2 * np.pi)
        window_psi = (2 * np.pi / mm) * (np.mod((phi_rr_mod * nn), mm))
        cos_sum = np.sum(np.cos(window_psi)) / len(window_psi)
        sin_sum = np.sum(np.sin(window_psi)) / len(window_psi)
        gamma = cos_sum**2 + sin_sum**2
        n_m_sync_dict[(nn, mm)] = gamma

    # m = 2
    mm = 2
    for nn in range((n - 1) * 2 + 1, (n + 1) * 2):
        phi_rr_mod = np.mod(sync, 2 * np.pi * mm) / (2 * np.pi)
        window_psi = (2 * np.pi / mm) * (np.mod((phi_rr_mod * nn), mm))
        cos_sum = np.sum(np.cos(window_psi)) / len(window_psi)
        sin_sum = np.sum(np.sin(window_psi)) / len(window_psi)
        gamma = cos_sum**2 + sin_sum**2
        n_m_sync_dict[(nn, mm)] = gamma

    # m = 3
    mm = 3
    for nn in range((n - 1) * 3 + 1, (n + 1) * 3):
        phi_rr_mod = np.mod(sync, 2 * np.pi * mm) / (2 * np.pi)
        window_psi = (2 * np.pi / mm) * (np.mod((phi_rr_mod * nn), mm))
        cos_sum = np.sum(np.cos(window_psi)) / len(window_psi)
        sin_sum = np.sum(np.sin(window_psi)) / len(window_psi)
        gamma = cos_sum**2 + sin_sum**2
        n_m_sync_dict[(nn, mm)] = gamma

    max_key, max_gamma = max(n_m_sync_dict.items(), key=lambda item: item[1])
    return max_key, max_gamma

def calculate_dynamic_n_m(hr_segment, rr_segment):
    """
    Dynamically adjust n:m based on peak detection.
    Args:
        hr_segment: Heart rate signal segment.
        rr_segment: Respiratory rate signal segment.
    Returns:
        n:m ratio (rounded).
    """
    hr_peaks, _ = find_peaks(hr_segment, width=10)
    rr_peaks, _ = find_peaks(rr_segment, height=0.5)
    n = len(hr_peaks)
    m = len(rr_peaks)
    return round(n / m)

def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    """
    Apply a bandpass filter to the data.
    Args:
        data: Input signal.
        lowcut: Low cutoff frequency.
        highcut: High cutoff frequency.
        fs: Sampling frequency.
        order: Filter order.
    Returns:
        Filtered signal.
    """
    if fs is None or fs <= 0:
        return np.zeros_like(data)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def synchrogram_1(hr_signal, rr_signal, title=""):
    """
    Main function to compute and plot synchrogram and synchronization metrics.
    Args:
        hr_signal: Heart rate signal.
        rr_signal: Respiratory rate signal.
        title: Plot title.
    Returns:
        Various synchronization metrics and results.
    """
    hr_min, hr_max = hr_signal.min(), hr_signal.max()
    rr_min, rr_max = rr_signal.min(), rr_signal.max()

    hr_norm = 2 * (hr_signal - hr_min) / (hr_max - hr_min) - 1
    rr_norm = 2 * (rr_signal - rr_min) / (rr_max - rr_min) - 1

    # Compute phase of the slower signal (rr_signal) using Hilbert transform
    phi_rr = np.angle(hilbert(rr_norm))
    phi_rr = adjust_phase_to_0_mpi(phi_rr)
    phi_rr_1 = np.mod(phi_rr, 2 * np.pi)
    phi_rr_2 = np.mod(phi_rr, 4 * np.pi)
    phi_rr_3 = np.mod(phi_rr, 6 * np.pi)

    peaks, _ = find_peaks(hr_norm, width=8)
    print(len(peaks))

    # Extract phase of rr_signal at peak positions
    sync = phi_rr[peaks]
    sync_1 = phi_rr_1[peaks]
    sync_2 = phi_rr_2[peaks]
    sync_3 = phi_rr_3[peaks]

    window_size = 600
    ratio_data = []
    for i in range(len(peaks)):
        p = peaks[i]
        start = max(0, p - window_size)
        end = min(len(hr_signal), p + window_size)
        hr_seg_local = hr_signal[start:end]
        rr_seg_local = rr_signal[start:end]
        local_n = calculate_dynamic_n_m(hr_seg_local, rr_seg_local)
        ratio_data.append((i, local_n))
    print(ratio_data)

    gamma_all = []
    max_n_m = []
    N = 50

    for k in range(N // 2, len(peaks) - N // 2):
        start = int(k - N // 2 + 1)
        end = int(k + N // 2 + 1)
        window_sync = sync[start:end]
        ratio_cur = ratio_data[k][1]
        n_m, gamma_cur = calculate_gamma(window_sync, ratio_cur)
        gamma_all.append(gamma_cur)
        max_n_m.append(n_m)

    print(f"len(psi_plus):{len(sync)}")
    max_n_m = [simplify_ratio(n, m) for n, m in max_n_m]

    # === Synchronization metrics ===
    fs = 60  # Actual sampling rate
    peaks_trimmed = peaks[N // 2:len(peaks) - N // 2]
    total_duration_sec = (peaks_trimmed[-1] - peaks_trimmed[0]) / fs
    threshold_sync = 0.1
    sync_flags = np.array(gamma_all) > threshold_sync

    # Find all synchronized segment index ranges
    sync_epochs = []
    start_idx = 0
    in_sync = False

    for i, flag in enumerate(sync_flags):
        if flag and not in_sync:
            start_idx = i
            in_sync = True
        elif not flag and in_sync:
            end_idx = i - 1
            sync_epochs.append((start_idx, end_idx))
            in_sync = False
    if in_sync:
        sync_epochs.append((start_idx, len(sync_flags) - 1))

    # Calculate duration (seconds) of each synchronized segment
    durations_sec = []
    valid_sync_epochs = []
    for start, end in sync_epochs:
        t_start = peaks_trimmed[start]
        t_end = peaks_trimmed[end]
        duration = (t_end - t_start) / fs
        if duration >= 5:
            durations_sec.append(duration)
            valid_sync_epochs.append((start, end))
    sync_epochs = valid_sync_epochs

    # 1. %Sync: Percentage of synchronized time
    sync_duration_sec = sum(durations_sec)
    percent_sync = (sync_duration_sec / total_duration_sec) * 100

    # 2. NumSync: Number of synchronized segments
    num_sync = len(durations_sec)

    # 3. AvgDurSync: Average duration of synchronization
    avg_dur_sync_sec = np.mean(durations_sec) if durations_sec else 0

    # 4. FreqRat: Frequency ratio (approximate by mean of (n, m))
    freq_ratio = calculate_dynamic_n_m_v2(hr_norm, rr_norm)

    # Segment heart rate for checking HR changes
    seg_hr_check = 1200
    hr_segments = [hr_norm[i:i + seg_hr_check] for i in range(0, len(hr_norm), seg_hr_check)]
    hr_per_segment = [get_hr(segment, fs, min=30, max=180) for segment in hr_segments]

    # === Print results ===
    print("\n===== Synchronization Metrics =====")
    print(f"1. %Sync: {percent_sync:.2f}%")
    print(f"2. NumSync: {num_sync}")
    print(f"3. AvgDurSync: {avg_dur_sync_sec:.2f} s")
    print(f"Durations: {durations_sec}")
    print(f"Sync segments: {sync_epochs}")
    print(f"Total count: {len(gamma_all)}")
    print(f"4. FreqRat (HR/RR): {freq_ratio:.2f}")
    print(f"HR changes: {hr_per_segment}")
    print("===================================\n")

    # Synchronization degree statistics
    min_gamma = min(gamma_all)
    max_gamma = max(gamma_all)
    mean_gamma = np.mean(gamma_all)
    std_gamma = np.std(gamma_all)
    print(f"Min gamma: {min_gamma:.2f}")
    print(f"Max gamma: {max_gamma:.2f}")
    print(f"Mean gamma: {mean_gamma:.2f}")
    print(f"Std gamma: {std_gamma:.2f}")

    fs = 60  # Sampling rate
    time = np.arange(len(hr_norm)) / fs
    plt.figure(figsize=(15, 3))
    plt.plot(time, hr_norm, 'r', label='hr_signal')
    plt.plot(time, rr_norm, 'b', label='rr_signal')
    plt.title("Signal HR and RR")
    plt.legend()
    plt.show()

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    time_peaks = peaks / fs
    axes[0].plot(time_peaks, sync_1 / (2 * np.pi), "o", markersize=5)
    axes[0].set_title(r'$\mathrm{\psi}_1(t_k)$', fontsize=13)
    axes[0].set_xlabel("Time(s)")
    axes[0].set_ylabel("respiratory cycles")
    axes[1].plot(time_peaks, sync_2 / (2 * np.pi), "o", markersize=5)
    axes[1].set_title(r'$\mathrm{\psi}_3(t_k)$', fontsize=13)
    axes[1].set_xlabel("Time(s)")
    axes[1].set_ylabel("respiratory cycles")
    axes[2].plot(time_peaks, sync_3 / (2 * np.pi), "o", markersize=5)
    axes[2].set_title(r'$\mathrm{\psi}_3(t_k)$', fontsize=13)
    axes[2].set_xlabel("Time(s)")
    axes[2].set_ylabel("respiratory cycles")
    plt.tight_layout()
    plt.show()

    min_val = min(gamma_all)
    max_val = max(gamma_all)
    start_tick = math.floor(min_val * 10) / 10
    end_tick = math.ceil(max_val * 10) / 10
    plt.figure(figsize=(10, 2))
    plt.ylim([start_tick, end_tick])
    plt.yticks(np.arange(start_tick, end_tick + 0.001, 0.1))
    time_peaks_trimmed = time_peaks[N // 2: len(peaks) - N // 2]
    plt.plot(time_peaks_trimmed, gamma_all, '-')
    plt.xlim([time_peaks[0], time_peaks[-1]])
    plt.title("Synchronization Degree over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Synchronization Degree")
    plt.show()

    n_vals, m_vals = zip(*max_n_m)
    max_n = max(n_vals)
    max_m = max(m_vals)
    heatmap = np.zeros((max_n + 1, max_m + 1))
    for n, m in max_n_m:
        heatmap[n, m] += 1

    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, cmap='Blues', origin='lower')
    for n in range(max_n + 1):
        for m in range(max_m + 1):
            if heatmap[n, m] > 0:
                plt.text(m, n, int(heatmap[n, m]), ha='center', va='center', color='black')
    plt.xlim(0, 3.5)
    plt.colorbar(label='Frequency')
    plt.title("Heatmap of (n, m) Frequency")
    plt.xlabel("m value")
    plt.ylabel("n value")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    n_vals, m_vals = zip(*max_n_m)
    time_points = peaks[N // 2:len(peaks) - N // 2] / fs

    plt.figure(figsize=(15, 6))
    sync_time_points = []
    sync_n_vals = []
    sync_m_vals = []
    for start, end in sync_epochs:
        for i in range(start, end + 1):
            sync_time_points.append(time_points[i])
            sync_n_vals.append(n_vals[i])
            sync_m_vals.append(m_vals[i])
    plt.plot(sync_time_points, sync_n_vals, marker='o', label='n (sync only)', linestyle='None', linewidth=2)
    plt.plot(sync_time_points, sync_m_vals, marker='s', label='m (sync only)', linestyle='None', linewidth=2)
    for x, y_n, y_m in zip(sync_time_points, sync_n_vals, sync_m_vals):
        plt.text(x, y_n + 0.2, f'{y_n}', ha='center', va='bottom', fontsize=6, color='blue')
        plt.text(x, y_m - 0.4, f'{y_m}', ha='center', va='top', fontsize=6, color='red')
    plt.title("Line Plot of (n, m) over Time (Sync Epochs Only)")
    plt.xlabel("Time Point")
    plt.xlim([time_peaks[0], time_peaks[-1]])
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Count frequency of n:m ratios in synchronized segments
    sync_ratio_counter = Counter()
    for start, end in sync_epochs:
        for i in range(start, end + 1):
            ratio = (n_vals[i], m_vals[i])
            sync_ratio_counter[ratio] += 1
    sorted_ratios = sorted(sync_ratio_counter.items(), key=lambda x: x[1], reverse=True)
    ratios = [f"{n}:{m}" for (n, m), _ in sorted_ratios]
    counts = [count for _, count in sorted_ratios]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(ratios, counts)
    plt.xlabel("n:m Ratio")
    plt.ylabel("Frequency")
    plt.title("Frequency of n:m Ratios in Sync Epochs")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return (percent_sync, num_sync, avg_dur_sync_sec, freq_ratio, min_gamma, max_gamma,
            mean_gamma, std_gamma, sync_ratio_counter, gamma_all, time_peaks_trimmed)
