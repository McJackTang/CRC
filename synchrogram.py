import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import hilbert, find_peaks
from scipy.signal import  welch, butter, filtfilt
from math import gcd
def simplify_ratio(n, m):
    g = gcd(n, m)
    return (n // g, m // g)  # 返回化简后的(n, m)   
def adjust_phase_to_0_mpi(phi_rr):
    # 初始化一个数组来存储连续的相位
    """continuous_phi = np.zeros_like(phi_rr)
    
    # 第一个点的相位设置为 0
    continuous_phi[0] = 0
    
    # 计算相位的变化量并累加
    for i in range(1, len(phi_rr)):
        delta_phi = phi_rr[i] - phi_rr[i-1]
    
        if delta_phi > np.pi:
            delta_phi -= 2 * np.pi
        elif delta_phi < -np.pi:
            delta_phi += 2 * np.pi

        continuous_phi[i] = continuous_phi[i-1] + delta_phi
    
    # 映射到 [0, 2*pi*m] 范围内
    #continuous_phi = np.mod(continuous_phi, 2 * np.pi * m)"""
    
    return np.unwrap(phi_rr)  # 使用 np.unwrap() 函数来处理相位跳变问题
def get_hr(y, fs=60, min=30, max=180):
    #fs: 采样频率
    p, q = welch(y, fs, nfft=int(1e6/fs), nperseg=np.min((len(y)-1, 512)))
    return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60 
from scipy.fft import fft, fftfreq
def calculate_dynamic_n_m_v2(hr_segment, rr_segment, fs=60):
    """
    动态计算 n:m：
      1) 用 get_hr 估计心率 HR_bpm（30–180 bpm）
      2) 用 get_hr 估计呼吸率 VR_bpm（6–60 bpm）
      3) ratio = HR_bpm/VR_bpm，向最近整数取整为 n，m=1
    
    Arguments:
      hr_segment -- 一段心电或脉搏信号（numpy 数组）
      rr_segment -- 一段呼吸或体积信号（numpy 数组）
      fs         -- 采样率 (Hz)
    返回:
      (n, m)     -- 近似的整数比
    """
    # 1) 心率：30–180 bpm
    HR_bpm = get_hr(hr_segment, fs, min=30, max=180)
    # 2) 呼吸率：6–60  bpm
    VR_bpm = get_hr(rr_segment, fs, min=6,  max=30)
    
    # 3) 比值四舍五入
    ratio = HR_bpm / VR_bpm
    n = int(np.round(ratio))
    m = 1
    
    return ratio
def calculate_gamma(sync,ratio_0):
    # Initialize n_m_sync_dict as an empty dictionary
    n_m_sync_dict = {}
    
    n=ratio_0
    #m=1时
    mm=1
    for nn in range((n-1),(n+1)+1):
        if nn <= 1:
            continue #跳过nn=1的情况
        phi_rr_mod=np.mod(sync,2*np.pi*mm)/(2*np.pi)
        window_psi=(2*np.pi/mm) *(np.mod((phi_rr_mod * nn), mm))
        
        cos_sum = (np.sum((np.cos(window_psi)))) / len(window_psi)
        sin_sum = (np.sum((np.sin(window_psi)))) / len(window_psi)
    # 计算同步度
        gamma = cos_sum**2 + sin_sum**2
        n_m_sync_dict[(nn, mm)] = gamma 
    
    #mm=2
    mm=2
    for nn in range((n-1)*2+1,(n+1)*2):
        phi_rr_mod=np.mod(sync,2*np.pi*mm)/(2*np.pi)
        window_psi=(2*np.pi/mm) *(np.mod((phi_rr_mod * nn), mm))
        
        cos_sum = (np.sum((np.cos(window_psi)))) / len(window_psi)
        sin_sum = (np.sum((np.sin(window_psi)))) / len(window_psi)
    # 计算同步度
        gamma = cos_sum**2 + sin_sum**2
        n_m_sync_dict[(nn, mm)] = gamma 
    
    #mm=3
    mm=3
    for nn in range((n-1)*3+1,(n+1)*3):
        phi_rr_mod=np.mod(sync,2*np.pi*mm)/(2*np.pi)
        window_psi=(2*np.pi/mm) *(np.mod((phi_rr_mod * nn), mm))
        
        cos_sum = (np.sum((np.cos(window_psi)))) / len(window_psi)
        sin_sum = (np.sum((np.sin(window_psi)))) / len(window_psi)
    # 计算同步度
        gamma = cos_sum**2 + sin_sum**2
        n_m_sync_dict[(nn, mm)] = gamma 
    
    max_key, max_gamma = max(n_m_sync_dict.items(), key=lambda item: item[1])
    
    return max_key, max_gamma
def calculate_dynamic_n_m(hr_segment, rr_segment):
    """根据峰值检测动态调整 n:m"""
    # 获取 hr_segment 和 rr_segment 中的峰值
    hr_peaks, _ = find_peaks(hr_segment, width=10)
    rr_peaks, _ = find_peaks(rr_segment, height=0.5)
    
    # 根据峰值数目或其他标准动态调整 n:m
    n = len(hr_peaks)  # 假设 n 是 HR 信号的峰值数目
    m = len(rr_peaks)  # 假设 m 是 RR 信号的峰值数目
    
    # 根据需求自定义动态调整规则
    return round(n/m)
def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    """Apply a bandpass filter to the data."""
    
    if fs is None or fs <= 0:
        return np.zeros_like(data)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)
def synchrogram_1(hr_signal, rr_signal, title=""):
    
    hr_min, hr_max = hr_signal.min(), hr_signal.max()
    rr_min, rr_max = rr_signal.min(), rr_signal.max()

    hr_norm = 2*(hr_signal - hr_min) / (hr_max - hr_min)-1
    rr_norm = 2*(rr_signal - rr_min) / (rr_max - rr_min)-1
    # 计算较慢信号（rr_signal）的相位，使用Hilbert变换
    phi_rr = np.angle(hilbert(rr_norm))
    phi_rr = adjust_phase_to_0_mpi(phi_rr)  
    phi_rr_1 = np.mod(phi_rr, 2 * np.pi )  
    phi_rr_2 = np.mod(phi_rr, 4 * np.pi) 
    phi_rr_3 = np.mod(phi_rr, 6 * np.pi )  
    
    peaks, _ = find_peaks(hr_norm,width=8)  
    print(len(peaks))

    # 在峰值位置提取 rr_signal 的相位
    sync= phi_rr[peaks]  # 提取 rr_signal 的相位
    sync_1 = phi_rr_1[peaks]
    sync_2 = phi_rr_2[peaks]  
    sync_3 = phi_rr_3[peaks] 
    
    window_size = 600
    ratio_data = []  # 用来保存 (峰值位置, (n, m))
    for i in range(0,len(peaks)): #计算每个峰值位置的 初步猜测的 n:m 比例
        p=peaks[i]
        start = max(0, p - window_size)
        end = min(len(hr_signal), p + window_size)
        hr_seg_local = hr_signal[start:end]
        rr_seg_local = rr_signal[start:end]
        local_n= calculate_dynamic_n_m(hr_seg_local, rr_seg_local)
        ratio_data.append((i, local_n ))
    print(ratio_data)
    gamma_all = []
    max_n_m=[]
    
    N=50
    
    for k in range(N//2,len(peaks)-N//2):
        # 取滑动窗口内的相对相位
        start = int( k - N//2+1)
        end = int(k + N//2+1)
        window_sync = sync[start:end]
        ratio_cur=ratio_data[k][1]
        n_m,gamma_cur=calculate_gamma(window_sync,ratio_cur)
        gamma_all.append(gamma_cur)
        max_n_m.append(n_m)
        
    
    print(f"len(psi_plus):{len(sync)}")
    max_n_m = [simplify_ratio(n, m) for n, m in max_n_m]
    # === 计算同步指标 ===
    
    fs = 60  # 实际采样率
    
    peaks_trimmed = peaks[N//2:len(peaks)-N//2]  # gamma_all 对应的峰值位置
    total_duration_sec = (peaks_trimmed[-1] - peaks_trimmed[0]) / fs  # 总持续时间（秒）
    threshold_sync = 0.1  # 同步度阈值
    sync_flags = np.array(gamma_all) > threshold_sync #
    # 找出所有同步段的索引范围
    sync_epochs = []
    start_idx = 0
    in_sync = False

    for i, flag in enumerate(sync_flags):
        if flag and not in_sync:
            # 新的同步段开始
            start_idx = i
            in_sync = True
        elif not flag and in_sync:
            # 同步段结束
            end_idx = i - 1
            sync_epochs.append((start_idx, end_idx))
            in_sync = False
    # 最后一段是同步状态
    if in_sync:
        sync_epochs.append((start_idx, len(sync_flags) - 1))

    # 计算每段同步持续时间（秒）
    durations_sec = []
    valid_sync_epochs = []
    for start, end in sync_epochs:
        t_start = peaks_trimmed[start]
        t_end = peaks_trimmed[end]
        duration = (t_end - t_start) / fs
        #设置同步时间阈值
        if duration>=5:
            durations_sec.append(duration)
            valid_sync_epochs.append((start, end))
    sync_epochs = valid_sync_epochs  # 更新同步段列表

    # 1. %Sync：同步时间百分比
    sync_duration_sec = sum(durations_sec)
    percent_sync = (sync_duration_sec / total_duration_sec) * 100

    # 2. NumSync：同步片段次数（同步段数量）
    num_sync = len(durations_sec)

    # 3. AvgDurSync：同步持续时间的平均值
    avg_dur_sync_sec = np.mean(durations_sec) if durations_sec else 0

    # 4. FreqRat：频率比（用(n, m)的平均来近似）
    freq_ratio=calculate_dynamic_n_m_v2(hr_norm, rr_norm)
    #n_vals, m_vals = zip(*max_n_m)
    #avg_n = np.mean(n_vals)
    #avg_m = np.mean(m_vals)
    #freq_ratio = avg_n / avg_m if avg_m != 0 else np.nan
    # 分段查看心率变化
    seg_hr_check=1200
    #将hr_norm分段
    hr_segments = [hr_norm[i:i + seg_hr_check] for i in range(0, len(hr_norm), seg_hr_check)]
    #计算每段上的心率
    hr_per_segment = [get_hr(segment, fs, min=30, max=180) for segment in hr_segments]
    # === 打印输出结果 ===
    print("\n===== Synchronization Metrics =====")
    print(f"1. %Sync (同步百分比): {percent_sync:.2f}%")
    print(f"2. NumSync (同步段数): {num_sync}")
    print(f"3. AvgDurSync (同步持续时间平均值): {avg_dur_sync_sec:.2f}秒")
    print(f"各个持续时间: {durations_sec}")
    print(f"各个同步段: {sync_epochs}")
    print(f"总数量: {len(gamma_all)}")
    print(f"4. FreqRat (心跳/呼吸频率比): {freq_ratio:.2f}")
    print(f"心率变化: {hr_per_segment}")

    print("===================================\n")

    #同步度的最小值，最大值，均值，标准差
    min_gamma = min(gamma_all)
    max_gamma = max(gamma_all)
    mean_gamma = np.mean(gamma_all)
    std_gamma = np.std(gamma_all)
    print(f"最小同步度: {min_gamma:.2f}")
    print(f"最大同步度: {max_gamma:.2f}")
    print(f"平均同步度: {mean_gamma:.2f}")
    print(f"同步度标准差: {std_gamma:.2f}")
    fs = 60  # 假设采样率为60Hz
    time = np.arange(len(hr_norm)) / fs 
    plt.figure(figsize=(15, 3))
    plt.plot(time,hr_norm, 'r', label='hr_signal')
    plt.plot(time,rr_norm, 'b', label='rr_signal')
    plt.title(f"Signal HR and RR")
    plt.legend()
    plt.show()
    # 绘制结果
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))  
    #plt.suptitle(f"segment{title}")
    # 绘制 hr_signal 和 rr_signal 信号
    """axes[0].plot(hr_norm, 'r', label='hr_signal')
    axes[0].plot(rr_norm, 'b', label='rr_signal')
    axes[0].set_title(f" Signal HR and RR") 
    axes[0].legend()"""
 # 在第一个图中标记 hr_signal 的峰值位置
    """axes[0].plot(peaks, hr_norm[peaks] ,"ro")  # 红色标记峰值位置
    for i in peaks:
        axes[0].annotate(f'{hr_norm[i]:.2f}', (i, hr_norm[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center')"""
                         
    # 将 peaks 转成以秒为单位
    time_peaks = peaks / fs
    # 绘制同步图
    axes[0].plot(time_peaks, sync_1/(2*np.pi), "o",markersize=5)
    axes[0].set_title(r'$\mathrm{\psi}_1(t_k)$', fontsize=13)
    axes[0].set_xlabel("Time(s)")
    axes[0].set_ylabel("respiratory cycles")
    axes[1].plot(time_peaks, sync_2/(2*np.pi), "o",markersize=5)
    axes[1].set_title(r'$\mathrm{\psi}_3(t_k)$', fontsize=13)
    axes[1].set_xlabel("Time(s)")
    axes[1].set_ylabel("respiratory cycles")
    axes[2].plot(time_peaks, sync_3/(2*np.pi), "o",markersize=5)
    axes[2].set_title(r'$\mathrm{\psi}_3(t_k)$', fontsize=13)
    axes[2].set_xlabel("Time(s)")
    axes[2].set_ylabel("respiratory cycles")
    plt.tight_layout()
    plt.show()
    min_val = min(gamma_all)
    max_val = max(gamma_all)

# 向下/向上取整到 0.1 的倍数
    import math
    start_tick = math.floor(min_val * 10) / 10
    end_tick = math.ceil(max_val * 10) / 10
    plt.figure(figsize=(10, 2))
    plt.ylim([start_tick, end_tick])
    plt.yticks(np.arange(start_tick, end_tick + 0.001, 0.1))
    # 绘制同步度变化图
    time_peaks_trimmed = time_peaks[N // 2 : len(peaks) - N // 2]
    plt.plot(time_peaks_trimmed, gamma_all, '-')
    plt.xlim([time_peaks[0], time_peaks[-1]])
    plt.title(f"Synchronization Degree over Time")
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

    plt.xlim(0, 3.5)  # 
    #plt.ylim(0, 10) # 纵轴 n 从 0 到 10
    plt.colorbar(label='Frequency')
    plt.title("Heatmap of (n, m) Frequency")
    plt.xlabel("m value")
    plt.ylabel("n value")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    

    n_vals, m_vals = zip(*max_n_m)
    time_points = peaks[N//2:len(peaks)-N//2]/fs  # 取中间的时间点


    plt.figure(figsize=(15, 6))
    # 筛选同步段内的时间点和对应的n, m
    sync_time_points = []
    sync_n_vals = []
    sync_m_vals = []

    for start, end in sync_epochs:
        for i in range(start, end + 1):  # 包含末端点
            sync_time_points.append(time_points[i])
            sync_n_vals.append(n_vals[i])
            sync_m_vals.append(m_vals[i])

    # 绘制 n 和 m 折线图（只在同步段内）
    plt.plot(sync_time_points, sync_n_vals, marker='o', label='n (sync only)', linestyle='None', linewidth=2)
    plt.plot(sync_time_points, sync_m_vals, marker='s', label='m (sync only)', linestyle='None', linewidth=2)

    # 加上数值标签
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
    
    from collections import Counter

    # === 统计同步段上的同步比出现频率 ===
    sync_ratio_counter = Counter()
    for start, end in sync_epochs:
        for i in range(start, end + 1):
            ratio = (n_vals[i], m_vals[i])
            sync_ratio_counter[ratio] += 1

    # 排序输出（按出现次数降序）
    sorted_ratios = sorted(sync_ratio_counter.items(), key=lambda x: x[1], reverse=True)
    ratios = [f"{n}:{m}" for (n, m), _ in sorted_ratios]
    counts = [count for _, count in sorted_ratios]

    # === 绘制柱状图（同步比出现的频率） ===
    plt.figure(figsize=(12, 6))
    bars = plt.bar(ratios, counts)
    plt.xlabel("n:m Ratio")
    plt.ylabel("Frequency")
    plt.title("Frequency of n:m Ratios in Sync Epochs")

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    
    return percent_sync, num_sync, avg_dur_sync_sec, freq_ratio, min_gamma, max_gamma, mean_gamma, std_gamma,sync_ratio_counter,gamma_all,time_peaks_trimmed
