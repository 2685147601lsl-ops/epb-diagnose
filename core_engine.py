import numpy as np
import pywt
import os
import joblib
from scipy.signal import hilbert, fftconvolve
from scipy.stats import kurtosis, skew, pearsonr
import scipy.io as sio
import random
from collections import Counter

class EPBCore:
    def __init__(self):
        self.fs = 51200
        self.raw_signal = None
        self.wpt_signal = None
        self.final_signal = None
        
        # 加载模型和标准化器
        self.model = None
        self.scaler = None
        # 设置为你个人的 EPB 诊断分类标签
        self.class_names = ['正常 (Normal)', '故障 (Fault)']
        
        try:
            # 使用脚本所在目录构建绝对路径，避免 CWD 问题
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_path, 'svm_model.pkl')
            scaler_path = os.path.join(base_path, 'scaler.pkl')
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Warning: Could not load SVM model: {e}")

    def load_data(self, file_path):
        if file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True).item()
            self.raw_signal = data['signal']
            self.fs = data.get('fs', 51200)
        elif file_path.endswith('.mat'):
            # 处理西储大学 MAT 格式
            data = sio.loadmat(file_path)
            # 自动搜索包含 _DE_time 的键
            de_key = [k for k in data.keys() if '_DE_time' in k]
            if de_key:
                self.raw_signal = data[de_key[0]].flatten()
                self.fs = 12000 # CWRU 数据集基准
                print(f"Loaded CWRU Mat file, setting fs=12000Hz, key={de_key[0]}")
            else:
                raise ValueError("Could not find Drive End (_DE_time) signal in MAT file.")
        else:
            # 处理 TXT 格式
            try:
                self.raw_signal = np.loadtxt(file_path, skiprows=19, usecols=(1,))
            except Exception:
                self.raw_signal = np.loadtxt(file_path)
            self.raw_signal = self.raw_signal - np.mean(self.raw_signal)
            self.fs = 51200 # EPB 实测数据默认采样率
            print(f"Loaded TXT file, setting fs=51200Hz")
            
        return self.raw_signal

    def wpt_denoise(self, signal, level=4, wavelet='coif5'):
        wp = pywt.WaveletPacket(signal, wavelet=wavelet, mode='symmetric', maxlevel=level)
        nodes = [n for n in wp.get_level(level, 'natural')]
        node_metrics = []
        zero_wp = pywt.WaveletPacket(data=np.zeros_like(signal), wavelet=wavelet, mode='symmetric', maxlevel=level)
        
        for node in nodes:
            path = node.path
            zero_wp[path].data = node.data
            rec_node = zero_wp.reconstruct(update=False)[:len(signal)]
            cc, _ = pearsonr(signal, rec_node) if np.std(rec_node) > 0 else (0, 0)
            node_metrics.append({'path': path, 'data': node.data, 'cc': abs(cc), 'kurt': kurtosis(rec_node)})
            zero_wp[path].data = np.zeros_like(node.data)

        max_cc = max([m['cc'] for m in node_metrics]) if node_metrics else 1
        max_kurt = max([m['kurt'] for m in node_metrics]) if node_metrics else 1
        for m in node_metrics:
            m['score'] = (m['cc'] / max_cc) * 1.0 + (m['kurt'] / max_kurt) * 1.5
            
        selected_nodes = sorted(node_metrics, key=lambda x: x['score'], reverse=True)[:8]
        final_wp = pywt.WaveletPacket(data=np.zeros_like(signal), wavelet=wavelet, mode='symmetric', maxlevel=level)
        for item in selected_nodes:
            final_wp[item['path']].data = item['data']
        
        self.wpt_signal = final_wp.reconstruct(update=False)[:len(signal)]
        return self.wpt_signal

    def fast_mckd(self, x, filter_len, T, M, max_iter=20):
        L, T, M = int(filter_len), int(T), int(M)
        N = len(x)
        f = np.zeros(L); f[L//2] = 1
        for _ in range(max_iter):
            y = fftconvolve(x, f, mode='same')
            target = y**3 
            for m in range(1, M+1):
                shift = m * T
                if shift < N:
                    y_shifted = np.roll(y, shift)
                    target += (1.0/(1+0.5*m)) * y_shifted * y_shifted * y 
            grad = fftconvolve(x, target[::-1], mode='same')
            center = len(grad)//2
            f_new = grad[center - L//2 : center + L//2 + (L%2)]
            norm = np.linalg.norm(f_new)
            if norm == 0: break
            f_new /= norm
            if np.linalg.norm(f - f_new[:L]) < 0.001: break
            f = f_new[:L]
        return fftconvolve(x, f, mode='same')

    def get_spectrum(self, sig):
        env = np.abs(hilbert(sig))
        env -= np.mean(env)
        n = len(env)
        freqs = np.fft.rfftfreq(n, 1/self.fs)
        fft_val = np.abs(np.fft.rfft(env)) / n * 2
        return freqs, fft_val
        
    def envelope_entropy(self, x):
        env = np.abs(hilbert(x))
        env = env / (np.sum(env) + 1e-8)
        env = env[env > 0]
        return -np.sum(env * np.log(env))

    def calculate_fitness(self, signal):
        k = kurtosis(signal)
        e = self.envelope_entropy(signal)
        if e == 0: return 0
        return k / (e + 1e-5)

    def optimize_mckd_params(self, signal, progress_callback=None):
        # progress_callback 留作更新UI进度条使用
        # 限制 T 的搜索范围以避免进入低频噪声的局部最优区
        T_min = int(self.fs / 400.0) 
        T_max = int(self.fs / 10.0)
        bounds = {'L': (50, 200), 'M': (1, 6), 'T': (T_min, T_max)}
        
        # 为了提高初始寻优概率，给一个较合理的起始 T 参数猜测
        curr = {'L': 100, 'M': 3, 'T': random.randint(*bounds['T'])}
        res = self.fast_mckd(signal, curr['L'], curr['T'], curr['M'], max_iter=15)
        curr_fit = self.calculate_fitness(res)
        best, best_fit = curr.copy(), curr_fit
        
        # 恢复缓慢的退火速率以保证找到真正的全局最优点
        Temp, alpha = 100.0, 0.9 
        step, total_steps = 0, 0
        
        # 预先计算总步数用于进度条
        temp_calc = Temp
        while temp_calc > 1.0:
            total_steps += 5
            temp_calc *= alpha
            
        while Temp > 1.0:
            for _ in range(5):
                step += 1
                if progress_callback:
                    progress_callback(int(step / total_steps * 100))
                    
                new_s = curr.copy()
                key = random.choice(['L', 'M', 'T', 'ALL'])
                if key in ['L', 'ALL']: new_s['L'] = np.clip(curr['L'] + random.randint(-20, 20), *bounds['L'])
                if key in ['M', 'ALL']: new_s['M'] = np.clip(curr['M'] + random.randint(-1, 1), *bounds['M'])
                if key in ['T', 'ALL']: new_s['T'] = np.clip(curr['T'] + random.randint(-20, 20), *bounds['T'])
                
                for k in new_s: new_s[k] = int(new_s[k])
                if new_s == curr: continue
                
                processed = self.fast_mckd(signal, new_s['L'], new_s['T'], new_s['M'], max_iter=15)
                new_fit = self.calculate_fitness(processed)
                
                if new_fit - curr_fit > 0 or random.random() < np.exp((new_fit - curr_fit) / Temp):
                    curr, curr_fit = new_s, new_fit
                    if curr_fit > best_fit:
                        best_fit = curr_fit
                        best = curr.copy()
            Temp *= alpha
            
        return best['L'], best['M'], best['T']

    def extract_features(self, segment):
        features = []
        # 1. 时域 (4)
        rms = np.sqrt(np.mean(segment**2))
        kur = kurtosis(segment)
        pp = np.max(segment) - np.min(segment)
        sk = skew(segment)
        features.extend([rms, kur, pp, sk])
        
        # 2. 频域 (5)
        mag = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1/self.fs)
        P = mag / (np.sum(mag) + 1e-8)
        fc = np.sum(freqs * P)
        f_rms = np.sqrt(np.sum((freqs**2) * P))
        f_std = np.sqrt(np.sum(((freqs - fc)**2) * P))
        f_kur = np.sum(((freqs - fc)**4) * P) / (f_std**4 + 1e-8)
        f_skew = np.sum(((freqs - fc)**3) * P) / (f_std**3 + 1e-8)
        features.extend([fc, f_rms, f_std, f_kur, f_skew])
        
        # 3. WPT 能量 (16)
        wp = pywt.WaveletPacket(segment, wavelet='coif5', mode='symmetric', maxlevel=4)
        nodes = [n for n in wp.get_level(4, 'natural')]
        energy = [np.sum(n.data ** 2) for n in nodes]
        total_e = sum(energy) + 1e-8
        features.extend([e / total_e for e in energy])
        
        return np.array(features)

    def diagnose(self, signal):
        if self.model is None or self.scaler is None:
            return "模型未加载 (Model Not Loaded)", 0.0
            
        if len(signal) < 2048:
            return "信号长度不足 (Signal Too Short)", 0.0

        # === 鲁棒性优化：滑动窗口投票机制 ===
        window_size = 2048
        step_size = 1000
        n_windows = (len(signal) - window_size) // step_size + 1
        
        # 如果信号不够长进行一次滑动，就退化为单次诊断
        if n_windows < 1:
            n_windows = 1
            step_size = 0
            
        # 限制最大窗口数以防止计算过久
        n_windows = min(n_windows, 50) 
        
        predictions = []
        confidences = []
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            segment = signal[start_idx:end_idx]
            
            feat = self.extract_features(segment)
            feat_scaled = self.scaler.transform(feat.reshape(1, -1))
            
            prob = self.model.predict_proba(feat_scaled)[0]
            label_idx = np.argmax(prob)
            
            predictions.append(label_idx)
            confidences.append(prob[label_idx])
            
        # 投票统计
        counter = Counter(predictions)
        # 获取得票最多的标签
        final_label_idx = counter.most_common(1)[0][0]
        
        # 计算该标签的平均置信度
        label_confs = [confidences[i] for i in range(len(predictions)) if predictions[i] == final_label_idx]
        final_confidence = np.mean(label_confs)
        
        print(f"--- Sliding Window Diagnosis (Voting) ---")
        print(f"Windows Total: {n_windows}")
        print(f"Votes: {dict(counter)}")
        print(f"Final Selection: {final_label_idx} with Confidence: {final_confidence:.4f}")

        # 支持多分类：安全获取类别名称
        if final_label_idx < len(self.class_names):
            class_name = self.class_names[final_label_idx]
        else:
            class_name = f"未定义类别 (Class {final_label_idx})"
            
        return class_name, final_confidence
