import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
#归一化函数（-1,1）
def normalize(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = x * 2 - 1
    return x

def find_anchors(hr, T_pts, kind='dec'):
    """
    hr: 心率序列 (bpm)
    T_pts: 前后窗口大小 (样本点数)
    kind: 'dec' 找 HR↓，'acc' 找 HR↑
    """
    N = len(hr)
    idxs = np.arange(T_pts, N - T_pts)
    back = np.array([hr[i-T_pts:i].mean() for i in idxs])
    forw = np.array([hr[i:i+T_pts].mean() for i in idxs])
    if kind == 'dec':
        mask = forw < back
    else:
        mask = forw > back
    return idxs[mask]

def bprsa_localmean(anchor_sig, target_sig, fs, T, L, H, direction='dec'):
    
    #  转换 T 为样本点数
    T_pts = int(T * fs)
    anchors = find_anchors(anchor_sig, T_pts, kind=direction)

    #  截取并平均 target_sig 片段
    wL, wH = int(L * fs), int(H * fs)
    segs = []
    for a in anchors:
        if a - wL >= 0 and a + wH < len(target_sig):
            segs.append(target_sig[a-wL:a+wH])
    segs = np.array(segs)
    if segs.size == 0:
        raise RuntimeError("没有找到任何完整的锚点片段，请检查 T, L, H 设置。")

    bprsa_curve = segs.mean(axis=0)
    t_win = np.arange(-wL, wH) / fs

    # 计算归一化 PSD
    # 去直流
    sig = bprsa_curve - bprsa_curve.mean()
    f, Pxx = welch(sig, fs=fs, nperseg=min(1024, len(sig)))
    mask = (f >= 0) & (f <= 3)
    f_0_3 = f[mask]
    Pxx_0_3 = Pxx[mask]
    Pxx_norm = Pxx_0_3 / np.trapz(Pxx_0_3, f_0_3) * 100
    #Pxx_norm = Pxx / np.trapz(Pxx, f) * 100
    return t_win, bprsa_curve, f_0_3, Pxx_norm

## 通用绘图函数
def plot_bprsa(t, prsa, f, pxx_norm, kind='dec', signal='HR', target_signal='RESP'):
   
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 左侧：时间域曲线
    #label='{} {} events'.format(signal, '↓' if kind == 'dec' else '↑')
    axes[0].plot(t, prsa, lw=2)
    axes[0].axvline(0, color='k', lw=1, ls='--')
    axes[0].set_xlabel('Time (s)')
    if target_signal == 'RESP':
        axes[0].set_ylabel(f'{target_signal} amp(au)')
    #axes[0].set_ylabel(f'{signal} deviation (bpm)')
    if signal == 'RESP':
        axes[0].set_title(f'BPRSA: {signal} {"expiratory" if kind == "dec" else "inspiratory"} → {target_signal} response')
    else:
        axes[0].set_title(f'BPRSA: {signal} {"dec" if kind == "dec" else "acc"} → {target_signal} response')
    axes[0].legend()
    axes[0].grid(True)
    
    # 右侧：频率域曲线 (PSD)
    axes[1].plot(f, pxx_norm, lw=2)
    axes[1].axvline(0, color='k', lw=1, ls='--')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Normalized PSD (%)')
    axes[1].set_title(f'PRSA: {signal} {kind}')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
