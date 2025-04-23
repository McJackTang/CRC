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
    """根据峰值检测猜测第一个 n:m"""
    hr_peaks, _ = find_peaks(hr_segment, height=0.5)
    rr_peaks, _ = find_peaks(rr_segment, height=0.5)
  
    n = len(hr_peaks)
    m = len(rr_peaks)  

    return n, m

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
    m = 1
    
    return n, m

def calculate_gamma(hr_seg,rr_seg):
    """
    得到段内所有可能的n:m对应的同步度表
    以字典形式记录
    """
    phi_rr = np.angle(hilbert(rr_seg))
    phi_rr = adjust_phase_to_0_mpi(phi_rr) 
    peaks,_ = find_peaks(hr_seg)
    sync_seg=phi_rr[peaks]
    n,m=calculate_dynamic_n_m_v2(hr_seg,rr_seg)  #得到第一个猜测的同步比，可使用峰值数目比和频率比两种方法
    n=round(n/m)
    m=1
    print('n:',n)
    print('m:',m)
    n_m_sync_dict = {}
    #m=1时
    mm=1
    for nn in range((n-1),(n+1)+1):
        if nn==1:
            continue #跳过nn=1的情况
        phi_rr_mod=np.mod(sync_seg,2*np.pi*mm)/(2*np.pi)
        psi_plus=(2*np.pi/mm) *(np.mod((phi_rr_mod * nn), mm))
        gamma_sync = []
        N=3*nn
        for k in range(0, len(psi_plus)):  # 遍历所有时间点，使用滑动窗口
        # 取滑动窗口内的相对相位
            start=max(0, k - N//2+1)
            end=min(len(psi_plus), k + N//2+1)
            window_psi = psi_plus[start:end]  # 窗口内的相对相位
        #window_psi = psi_plus[k-N//2+1:k+N//2+1]  # 窗口内的相对相位

    # 计算余弦和正弦的和
            cos_sum = (np.sum((np.cos(window_psi)))) / len(window_psi)
            sin_sum = (np.sum((np.sin(window_psi)))) / len(window_psi)
    # 计算同步度
            gamma = cos_sum**2 + sin_sum**2
            gamma_sync.append(gamma)
        n_m_sync_dict[(nn, mm)] = gamma_sync 
    
    #mm=2
    mm=2
    for nn in range((n-1)*2+1,(n+1)*2):
        phi_rr_mod=np.mod(sync_seg,2*np.pi*mm)/(2*np.pi)
        psi_plus=(2*np.pi/mm) *(np.mod((phi_rr_mod * nn), mm))
        gamma_sync = []
        N=3*nn
        for k in range(0, len(psi_plus)):  # 遍历所有时间点，使用滑动窗口
        # 取滑动窗口内的相对相位
            start=max(0, k - N//2+1)  
            end=min(len(psi_plus), k + N//2+1)
            window_psi = psi_plus[start:end]  # 窗口内的相对相位
        #window_psi = psi_plus[k-N//2+1:k+N//2+1]  # 窗口内的相对相位

    # 计算余弦和正弦的和
            cos_sum = (np.sum((np.cos(window_psi)))) / len(window_psi)
            sin_sum = (np.sum((np.sin(window_psi)))) / len(window_psi)
    # 计算同步度
            gamma = cos_sum**2 + sin_sum**2
            gamma_sync.append(gamma)
        n_m_sync_dict[(nn, mm)] = gamma_sync
    
    #mm=3
    mm=3
    for nn in range((n-1)*3+1,(n+1)*3):
        phi_rr_mod=np.mod(sync_seg,2*np.pi*mm)/(2*np.pi)
        psi_plus=(2*np.pi/mm) *(np.mod((phi_rr_mod * nn), mm))
        gamma_sync = []
        N=3*nn
        for k in range(0, len(psi_plus)): 
            start=max(0, k - N//2+1)
            end=min(len(psi_plus), k + N//2+1)
            window_psi = psi_plus[start:end]  # 窗口内的相对相位
            #window_psi = psi_plus[k-N//2+1:k+N//2+1]  # 窗口内的相对相位
            #圆方差
            cos_sum = (np.sum((np.cos(window_psi)))) / len(window_psi)
            sin_sum = (np.sum((np.sin(window_psi)))) / len(window_psi)
            gamma = cos_sum**2 + sin_sum**2
            gamma_sync.append(gamma)
        n_m_sync_dict[(nn, mm)] = gamma_sync
    #print(n_m_sync_dict[(3,1)])
    #print(n_m_sync_dict[(4,1)])
    return n_m_sync_dict,peaks  

def simplify_ratio(n, m):
    g = gcd(n, m)
    return (n // g, m // g)    
def synchrogram(hr_signal, rr_signal, title=1):
    
    hr_min, hr_max = hr_signal.min(), hr_signal.max()
    rr_min, rr_max = rr_signal.min(), rr_signal.max()

    hr_norm = 2*(hr_signal - hr_min) / (hr_max - hr_min)-1
    rr_norm = 2*(rr_signal - rr_min) / (rr_max - rr_min)-1
    
    phi_rr = np.angle(hilbert(rr_norm))
    phi_rr = adjust_phase_to_0_mpi(phi_rr)  
    phi_rr_1 = np.mod(phi_rr, 2 * np.pi )  
    phi_rr_2 = np.mod(phi_rr, 4 * np.pi)  
    phi_rr_3 = np.mod(phi_rr, 6 * np.pi )  
   
    peaks, _ = find_peaks(hr_norm,height=0)  
    print(len(peaks))
    
    sync= phi_rr[peaks]  
    sync_1 = phi_rr_1[peaks]
    sync_2 = phi_rr_2[peaks]  
    sync_3 = phi_rr_3[peaks]  

    seg_len = 1200  # 每段长度
    hr_segs = []
    rr_segs = []
    for i in range(0, len(hr_signal), seg_len):
        hr_segs.append(hr_signal[i:i + seg_len])
        rr_segs.append(rr_signal[i:i + seg_len])
    # 计算同步度
    gamma_sync_all = []
    sync_max_n_m = []  # 存储每段的最大同步度对应的n:m比例
    peaks_all = []  # 存储所有段的峰值位置

    for idx, (hr_seg, rr_seg) in enumerate(zip(hr_segs, rr_segs)):
        segment_start_index = idx * seg_len  # 当前段的起始位置
        # 对每段信号计算n:m同步度
        n_m_sync_dict,peak_seg = calculate_gamma(hr_seg, rr_seg)
        global_peaks = [p + segment_start_index for p in peak_seg] # 将当前段的峰值位置转换为全局峰值位置

        # 比较多个n:m比例的同步度，选择最大同步度的n:m
        max_gamma_mean = -1  # 初始化最大同步度
        max_sync_n_m = None  # 存储对应的n:m比例
        max_gamma_list = []

        for (n, m), gamma_list in n_m_sync_dict.items():
            mean_gamma_in_segment =np.mean(gamma_list)  # 当前段同步度均值
            cur_gamma_list= gamma_list  # 当前n:m的同步度列表
        # 如果当前n:m的同步度更大，则更新最大同步度和比例
            if (mean_gamma_in_segment) > max_gamma_mean:
                max_gamma_list = cur_gamma_list  # 更新当前n:m的同步度列表
                max_gamma_mean = mean_gamma_in_segment  # 更新最大同步度
               # max_gamma_list =  cur_gamma_list  # 更新当前n:m的同步度列表
                max_sync_n_m = (n, m)  # 更新最大同步度和比例
        # 如果同步度相同，取m较大的比例
            elif (mean_gamma_in_segment) == max_gamma_mean:
                n_val, m_val = max_sync_n_m
                new_n, new_m = n, m
                if new_m > m_val:
                    max_sync_n_m = (new_n, new_m)  # 更新最大同步度和比例
                    max_gamma_list = cur_gamma_list  # 更新当前n:m的同步度列表
                    

        gamma_sync_all.extend(max_gamma_list)  # 将当前段的同步度添加到总列表中
        sync_max_n_m.append(max_sync_n_m)
        peaks_all.extend(global_peaks)  # 存储当前段的峰值位置
    
    sync_max_n_m = [simplify_ratio(n, m) for n, m in sync_max_n_m]
    print(sync_max_n_m)  #打印每段检测到的同步比

    # 绘制结果
    fig, axes = plt.subplots(5, 1, figsize=(10, 10))

    # 绘制 hr_signal 和 rr_signal 信号
    axes[0].plot(hr_norm, 'r', label='hr_signal')
    axes[0].plot(rr_norm, 'b', label='rr_signal')
    axes[0].set_title(f" Signal HR and RR") 
    axes[0].legend()
                         
    # 绘制同步图 m=1 m=2 m=3
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
    
    # 绘制同步度变化图
    axes[4].plot(peaks_all, gamma_sync_all, '-')
    axes[4].set_title(f"Synchronization Degree over Time")
    axes[4].set_xlabel("Time")
    axes[4].set_ylabel("Synchronization Degree")
