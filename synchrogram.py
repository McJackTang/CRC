import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch, butter, filtfilt

def adjust_phase_to_0_mpi(phi_rr): #相位展开
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

def calculate_dynamic_n_m(hr_segment, rr_segment):
    """根据峰值检测动态调整 n:m"""
    hr_peaks, _ = find_peaks(hr_segment, height=0.5)
    rr_peaks, _ = find_peaks(rr_segment, height=0.5)

    n = len(hr_peaks)  
    m = len(rr_peaks)  

    return round(n/m)

def get_hr(y, fs=60, min=30, max=180):
    #fs: 采样频率
    p, q = welch(y, fs, nfft=int(1e5/fs), nperseg=np.min((len(y)-1, 512)))
    return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60 

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
    
    return n

def calculate_gamma(sync,ratio_0):
    # Initialize n_m_sync_dict as an empty dictionary
    n_m_sync_dict = {}
    
    n=ratio_0
    #m=1时
    mm=1
    for nn in range((n-1),(n+1)+1):
        if nn==1:
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
    #print(f"最大 γ 对应的 (n, m) = {max_key}, γ = {max_gamma}")
    
    return max_key, max_gamma

def simplify_ratio(n, m):
    g = gcd(n, m)
    return (n // g, m // g)    
def synchrogram_1(hr_signal, rr_signal, title=""):
    
    hr_min, hr_max = hr_signal.min(), hr_signal.max()
    rr_min, rr_max = rr_signal.min(), rr_signal.max()

    hr_norm = 2*(hr_signal - hr_min) / (hr_max - hr_min)-1
    rr_norm = 2*(rr_signal - rr_min) / (rr_max - rr_min)-1
    # 计算较慢信号（rr_signal）的相位，使用Hilbert变换
    phi_rr = np.angle(hilbert(rr_norm))
    phi_rr = adjust_phase_to_0_mpi(phi_rr)  # 调整相位到 [0, 2πm] 范围
    phi_rr_1 = np.mod(phi_rr, 2 * np.pi )  # 将相位限制在 [0, 2π] 范围内
    phi_rr_2 = np.mod(phi_rr, 4 * np.pi)  # 将相位限制在 [0, 4π] 范围内
    phi_rr_3 = np.mod(phi_rr, 6 * np.pi )  # 将相位限制在 [0, 6π] 范围内
    # 找到较快信号（hr_signal）的峰值位置
    peaks, _ = find_peaks(hr_norm)  # 找到 hr_signal 的峰值位置
    print(len(peaks))
    #peaks_rr = find_peaks(rr_norm)[0]  # 找到 rr_signal 的峰值位置

    # 在峰值位置提取 rr_signal 的相位
    sync= phi_rr[peaks]  # 提取 rr_signal 的相位
    sync_1 = phi_rr_1[peaks]
    sync_2 = phi_rr_2[peaks]  # 提取 rr_signal 的相位
    sync_3 = phi_rr_3[peaks]  # 提取 rr_signal 的相位
    
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
        start = int(max(0, k - N//2+1))
        end = int(min(len(peaks), k + N//2+1))
        window_sync = sync[start:end]
        ratio_cur=ratio_data[k][1]
        n_m,gamma_cur=calculate_gamma(window_sync,ratio_cur)
        gamma_all.append(gamma_cur)
        max_n_m.append(n_m)
        
    
    print(f"len(psi_plus):{len(sync)}")
    max_n_m = [simplify_ratio(n, m) for n, m in max_n_m]
    # === 计算同步指标 ===
    # 1. %Sync：同步时间百分比
    threshold_sync = 0.1  # 同步度阈值
    sync_flags = np.array(gamma_all) > threshold_sync #
    percent_sync = np.sum(sync_flags) / len(sync_flags) * 100  # 百分比

    # 2. NumSync：同步片段次数（同步段数量）
    from itertools import groupby
    sync_epochs = [list(g) for k, g in groupby(sync_flags) if k == True]  # 连续的同步段
    num_sync = len(sync_epochs)

    # 3. AvgDurSync：同步持续时间的平均值
    if num_sync > 0:
        avg_dur_sync = np.mean([len(epoch) for epoch in sync_epochs])
    else:
        avg_dur_sync = 0

    # 4. FreqRat：频率比（用(n, m)的平均来近似）
    n_vals, m_vals = zip(*max_n_m)
    avg_n = np.mean(n_vals)
    avg_m = np.mean(m_vals)
    freq_ratio = avg_n / avg_m if avg_m != 0 else np.nan

    # === 打印输出结果 ===
    print("\n===== Synchronization Metrics =====")
    print(f"1. %Sync (同步百分比): {percent_sync:.2f}%")
    print(f"2. NumSync (同步段数): {num_sync}")
    print(f"3. AvgDurSync (同步持续时间平均值): {avg_dur_sync:.2f} 窗口数")
    print(f"4. FreqRat (心跳/呼吸频率比): {freq_ratio:.2f}")
    print("===================================\n")

    # 绘制结果
    fig, axes = plt.subplots(5, 1, figsize=(10, 10))
    
    # 绘制 hr_signal 和 rr_signal 信号
    axes[0].plot(hr_norm, 'r', label='hr_signal')
    axes[0].plot(rr_norm, 'b', label='rr_signal')
    axes[0].set_title(f" Signal HR and RR") 
    axes[0].legend()
 # 在第一个图中标记 hr_signal 的峰值位置
    """axes[0].plot(peaks, hr_norm[peaks] ,"ro")  # 红色标记峰值位置
    for i in peaks:
        axes[0].annotate(f'{hr_norm[i]:.2f}', (i, hr_norm[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center')"""
                         
    # 绘制同步图
    axes[1].plot(peaks, sync_1/(2*np.pi), "o")
    axes[1].set_title(f"m=1")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Phase")
    axes[2].plot(peaks, sync_2/(2*np.pi), "o")
    axes[2].set_title(f"m=2")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Phase")
    axes[3].plot(peaks, sync_3/(2*np.pi), "o")
    axes[3].set_title(f"m=3")
    axes[3].set_xlabel("Time")
    axes[3].set_ylabel("Phase")
    min_val = min(gamma_all)
    max_val = max(gamma_all)

# 向下/向上取整到 0.1 的倍数
    import math
    start_tick = math.floor(min_val * 10) / 10
    end_tick = math.ceil(max_val * 10) / 10
    axes[4].set_ylim([start_tick, end_tick])
    axes[4].set_yticks(np.arange(start_tick, end_tick + 0.001, 0.1))
    # 绘制同步度变化图
    axes[4].plot(peaks[N//2:len(sync)-N//2], gamma_all, '-')
    axes[4].set_title(f"Synchronization Degree over Time")
    axes[4].set_xlabel("Time")
    axes[4].set_ylabel("Synchronization Degree")

    plt.tight_layout()
    plt.show()

    #绘制n:m热图
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

    #绘制n:m折线图
    n_vals, m_vals = zip(*max_n_m)
    time_points = np.arange(len(max_n_m))

    plt.figure(figsize=(15, 6))
    plt.plot(time_points, n_vals, marker='o', label='n', linestyle='-', linewidth=2)
    plt.plot(time_points, m_vals, marker='s', label='m', linestyle='--', linewidth=2)
    #
    for i, (x, y_n, y_m) in enumerate(zip(time_points, n_vals, m_vals)):
        plt.text(x, y_n + 0.2, f'{y_n}', ha='center', va='bottom', fontsize=6, color='blue')
        plt.text(x, y_m - 0.4, f'{y_m}', ha='center', va='top', fontsize=6, color='red')
    plt.title("Line Plot of (n, m) over Time")
    plt.xlabel("Time Point")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
