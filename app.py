import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import time
from core_engine import EPBCore

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="EPB 状态智能监测云系统",
    page_icon="⚙️",
    layout="wide"
)

# 解决 plt 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 初始化会话状态缓存与核心引擎
# ==========================================
@st.cache_resource
def load_engine():
    # 只需要加载一次核心算法引擎
    return EPBCore()

engine = load_engine()

if 'raw_signal' not in st.session_state:
    st.session_state.raw_signal = None
if 'wpt_signal' not in st.session_state:
    st.session_state.wpt_signal = None
if 'final_signal' not in st.session_state:
    st.session_state.final_signal = None

# ==========================================
# 辅助绘图函数
# ==========================================
def plot_signal(sig, title, color='black', fs=51200):
    fig, ax = plt.subplots(figsize=(10, 3))
    t = np.arange(len(sig)) / fs
    ax.plot(t, sig, color=color, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    return fig

def plot_spectrum(sig, title, color='red', fs=51200):
    fig, ax = plt.subplots(figsize=(10, 3))
    freqs, vals = engine.get_spectrum(sig)
    ax.plot(freqs, vals, color=color, linewidth=0.8)
    ax.set_xlim(0, 400)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    return fig

# ==========================================
# 侧边栏：操作控制
# ==========================================
with st.sidebar:
    st.title("🎛️ 控制面板")
    st.markdown("---")
    
    # 1. 加载数据
    st.header("步骤 1: 数据接入")
    uploaded_file = st.file_uploader("上传诊断信号数据", type=['txt', 'npy', 'mat'])
    
    if uploaded_file is not None:
        if st.button("开始解析文件"):
            # 保存临时文件让 np.loadtxt 读取
            with tempfile.NamedTemporaryFile(delete=False, suffix='.'+uploaded_file.name.split('.')[-1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
                
            try:
                sig = engine.load_data(tmp_path)
                st.session_state.raw_signal = sig
                st.session_state.wpt_signal = None
                st.session_state.final_signal = None
                st.success(f"成功解析 {len(sig)} 点信号数据！")
            except Exception as e:
                st.error(f"解析失败: {e}")
            finally:
                os.unlink(tmp_path)

    st.markdown("---")
    
    # 2. 信号处理
    st.header("步骤 2: 降噪与增强")
    if st.button("执行 WPT 信号降噪", disabled=st.session_state.raw_signal is None):
        with st.spinner('正在执行小波包分解与自适应节点选取...'):
            sig = engine.wpt_denoise(st.session_state.raw_signal)
            st.session_state.wpt_signal = sig
            st.success("WPT 降噪完成！")

    if st.button("执行 SA-MCKD 特征增强", disabled=st.session_state.wpt_signal is None):
        # 建立进度条回调
        progress_bar = st.progress(0, text="初始化模拟退火算法...")
        
        def update_pb(pct):
            progress_bar.progress(pct / 100.0, text=f"自适应寻优中... ({pct}%)")
            
        best_L, best_M, best_T = engine.optimize_mckd_params(st.session_state.wpt_signal, progress_callback=update_pb)
        progress_bar.progress(1.0, text="参数锁定，最终解卷积运算中...")
        
        enhanced = engine.fast_mckd(st.session_state.wpt_signal, best_L, best_T, best_M, max_iter=20)
        st.session_state.final_signal = enhanced
        progress_bar.empty()
        st.success(f"增强成功！参数 [L:{best_L}, M:{best_M}, T:{best_T}]")

    st.markdown("---")
    
    # 3. 智能诊断
    st.header("步骤 3: 一键出具报告")
    if st.button("启动智能诊断引擎", disabled=st.session_state.raw_signal is None):
        st.session_state.run_diagnosis = True

# ==========================================
# 主界面：可视化数据大屏
# ==========================================
st.title("EPB 电子驻车制动器智能监测大屏")

# 根据是否有数据渲染不同的视图
if st.session_state.raw_signal is None:
    st.info("👈 请从左侧边栏上传待诊断设备的振动/声压数据 (.txt, .npy, .mat)")
else:
    # --- 顶栏：诊断报告 ---
    if getattr(st.session_state, 'run_diagnosis', False):
        with st.spinner("AI 诊断中..."):
            res, conf = engine.diagnose(st.session_state.raw_signal)
            # 使用容器高亮展示诊断结果
            success_color = "🟢" if "正常" in res else ("🔴" if "故障" in res else "🟡")
            st.markdown(f"### {success_color} 诊断系统结论：**{res}**")
            st.progress(float(conf), text=f"专家系统置信度评估：{conf * 100:.2f}%")
            st.caption("注：为防止信号增益特征偏移，报告分析均建立在全息原始特征采样基础之上。")
            st.markdown("---")
        # 清除触发状态
        st.session_state.run_diagnosis = False

    # --- 下方：波形对比图 ---
    col1, col2 = st.columns(2)
    fs = engine.fs

    with col1:
        st.subheader("时域波形图 (Waveforms)")
        st.pyplot(plot_signal(st.session_state.raw_signal, "1. 原始信号 (Raw Signal)", color='gray', fs=fs))
        
        if st.session_state.wpt_signal is not None:
            st.pyplot(plot_signal(st.session_state.wpt_signal, "2. WPT 降噪 (Denoised)", color='green', fs=fs))
            
        if st.session_state.final_signal is not None:
            st.pyplot(plot_signal(st.session_state.final_signal, "3. AMCKD 冲击增强 (Enhanced)", color='blue', fs=fs))

    with col2:
        st.subheader("包络谱特征 (Envelope Spectrums)")
        st.pyplot(plot_spectrum(st.session_state.raw_signal, "1. 原始包络谱", color='#FF9999', fs=fs))
        
        if st.session_state.wpt_signal is not None:
            st.pyplot(plot_spectrum(st.session_state.wpt_signal, "2. WPT 包络谱", color='#FF6666', fs=fs))
            
        if st.session_state.final_signal is not None:
            st.pyplot(plot_spectrum(st.session_state.final_signal, "3. 最终分析诊断图 (AMCKD)", color='#CC0000', fs=fs))
