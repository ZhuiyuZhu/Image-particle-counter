"""
荧光颗粒检测器 v3.3 (Streamlit Web版)
功能完全对应桌面版，支持批量上传、在线预览、CSV导出
"""

import streamlit as st
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import blob_log, peak_local_max
import pandas as pd
from PIL import Image, ImageDraw
import io
from datetime import datetime

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="荧光颗粒检测器 v3.3",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 核心函数 ====================
def load_image(file_obj):
    """兼容文件路径和Streamlit上传的文件对象"""
    if isinstance(file_obj, str):
        # 本地路径模式（保留原逻辑）
        image_path = os.path.normpath(file_obj)
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    else:
        # Streamlit UploadedFile 对象
        file_bytes = np.frombuffer(file_obj.read(), dtype=np.uint8)
    
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("无法解码图片，可能格式不支持或文件损坏")
    
    # 位深度转换（保留原逻辑）
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype == np.uint32:
        img = (img / 16777216).astype(np.uint8)
    
    # 统一为3通道BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    return img

def detect_particles(img_bgr, params):
    """核心检测逻辑（从原DetectionWorker提取）"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    channel_map = {'红色': 0, '绿色': 1, '蓝色': 2}
    ch_idx = channel_map.get(params['color'], 0)
    
    if len(img_rgb.shape) == 3 and img_rgb.shape[2] >= 3:
        channel = img_rgb[:, :, ch_idx].astype(float)
    else:
        channel = img_rgb.astype(float)
    
    # 根据模式设置参数
    if params['mode'] == '高精度':
        log_threshold, local_threshold, min_distance = 0.08, 0.2, 8
    elif params['mode'] == '高召回':
        log_threshold, local_threshold, min_distance = 0.03, 0.1, 4
    else:
        log_threshold, local_threshold, min_distance = 0.05, 0.15, 6
    
    # 根据是否等大调整
    if params['uniform_size']:
        min_sigma, max_sigma = 1.5, 3
    else:
        min_sigma, max_sigma = 0.8, 5
    
    results = {'img_rgb': img_rgb}
    
    # LoG检测
    if params['use_log']:
        red_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        blobs = blob_log(
            red_norm,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=10,
            threshold=log_threshold,
            overlap=0.3 if params['uniform_size'] else 0.5
        )
        if len(blobs) > 0:
            blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
        results['log'] = {
            'count': len(blobs),
            'blobs': blobs,
        }
    
    # 局部最大值检测
    if params['use_local']:
        bg = ndimage.uniform_filter(channel, size=50)
        red_clean = ndimage.gaussian_filter((channel - bg).clip(0), sigma=2)
        red_norm2 = (red_clean - red_clean.min()) / (red_clean.max() - red_clean.min() + 1e-8)
        
        coords = peak_local_max(
            red_norm2,
            min_distance=min_distance,
            threshold_abs=local_threshold,
            exclude_border=True
        )
        results['local'] = {
            'count': len(coords),
            'coords': coords,
        }
    
    return results

def draw_results(results, params):
    """绘制检测结果，返回PIL图像列表"""
    img_rgb = results['img_rgb']
    images = []
    captions = []
    
    # 原始图像
    images.append(Image.fromarray(img_rgb))
    captions.append(f"原始图像 ({img_rgb.shape[1]}×{img_rgb.shape[0]})")
    
    # LoG结果
    if params['use_log'] and 'log' in results:
        img_log = Image.fromarray(img_rgb.copy())
        draw = ImageDraw.Draw(img_log)
        for y, x, r in results['log']['blobs']:
            # 绿色圆圈
            draw.ellipse(
                [x - r, y - r, x + r, y + r],
                outline=(0, 255, 0), width=2
            )
            # 蓝色中心点
            draw.ellipse(
                [x - 2, y - 2, x + 2, y + 2],
                fill=(255, 0, 0)
            )
        images.append(img_log)
        captions.append(f"LoG: {results['log']['count']} 个颗粒")
    else:
        images.append(None)
        captions.append("LoG: 未启用")
    
    # 局部最大值结果
    if params['use_local'] and 'local' in results:
        img_local = Image.fromarray(img_rgb.copy())
        draw = ImageDraw.Draw(img_local)
        coords = results['local']['coords']
        for y, x in coords:
            # 绿色填充圆
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(0, 255, 0))
            # 白色外圈
            draw.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(255, 255, 255), width=2)
        images.append(img_local)
        captions.append(f"局部最大值: {results['local']['count']} 个颗粒")
    else:
        images.append(None)
        captions.append("局部最大值: 未启用")
    
    return images, captions

def create_downloadable_figure(images, captions):
    """将三张图拼接为一张用于下载"""
    valid_images = [img for img in images if img is not None]
    if not valid_images:
        return None
    
    widths, heights = zip(*(i.size for i in valid_images))
    total_width = sum(widths)
    max_height = max(heights)
    
    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for img in valid_images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return combined

def build_csv_data(results):
    """构建CSV数据"""
    rows = []
    if 'log' in results:
        for i, (y, x, r) in enumerate(results['log']['blobs']):
            rows.append({
                'method': 'LoG',
                'id': i + 1,
                'x': int(x),
                'y': int(y),
                'radius': round(r, 2)
            })
    if 'local' in results:
        for i, (y, x) in enumerate(results['local']['coords']):
            rows.append({
                'method': 'LocalMax',
                'id': i + 1,
                'x': int(x),
                'y': int(y),
                'radius': 'N/A'
            })
    return pd.DataFrame(rows)

# ==================== Streamlit UI ====================
st.sidebar.title("🔬 荧光颗粒检测器")
st.sidebar.caption("v3.3 Web版 | 支持中文路径和TIF格式")

# ---- 参数设置 ----
st.sidebar.markdown("### 检测参数")

color = st.sidebar.selectbox("颗粒颜色", ['红色', '绿色', '蓝色'], index=0)
mode = st.sidebar.selectbox(
    "检测模式", 
    ['平衡', '高精度', '高召回'], 
    index=0,
    help="高精度=减少误检，高召回=检测更多"
)
uniform_size = st.sidebar.checkbox("颗粒大小均匀", value=False, 
                                   help="如果颗粒大小相似建议勾选")

st.sidebar.markdown("### 检测方法")
use_log = st.sidebar.checkbox("LoG斑点检测", value=True, 
                              help="提供颗粒半径信息")
use_local = st.sidebar.checkbox("局部最大值检测", value=True, 
                                help="速度更快")

if not use_log and not use_local:
    st.sidebar.error("请至少选择一种检测方法！")

params = {
    'color': color,
    'mode': mode,
    'uniform_size': uniform_size,
    'use_log': use_log,
    'use_local': use_local
}

# ---- 文件上传 ----
st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader(
    "上传图片（支持批量）",
    type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'gif'],
    accept_multiple_files=True
)

# ==================== 主界面 ====================
st.title("荧光颗粒检测结果")

if not uploaded_files:
    st.info("👈 请在左侧上传图片开始检测（支持拖拽）")
    st.stop()

# 批量处理进度
progress_bar = st.progress(0)
status_text = st.empty()

all_summaries = []

for idx, uploaded_file in enumerate(uploaded_files):
    status_text.text(f"正在处理: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")
    
    try:
        # 读取并检测
        img_bgr = load_image(uploaded_file)
        results = detect_particles(img_bgr, params)
        
        # 绘制结果
        images, captions = draw_results(results, params)
        
        # 展示结果（三列布局）
        st.markdown(f"### 📄 {uploaded_file.name}")
        cols = st.columns(3)
        
        for col, img, cap in zip(cols, images, captions):
            with col:
                if img is not None:
                    st.image(img, caption=cap, use_container_width=True)
                else:
                    st.info(cap)
        
        # 统计信息
        stats = []
        stats.append(f"**尺寸:** {results['img_rgb'].shape[1]} × {results['img_rgb'].shape[0]} 像素")
        if 'log' in results:
            stats.append(f"**LoG检测:** {results['log']['count']} 个颗粒")
        if 'local' in results:
            stats.append(f"**局部最大值:** {results['local']['count']} 个颗粒")
        if 'log' in results and 'local' in results:
            avg = (results['log']['count'] + results['local']['count']) // 2
            stats.append(f"**建议计数:** <span style='color:green; font-size:1.2em'>{avg}</span> 个")
        
        st.markdown(" | ".join(stats), unsafe_allow_html=True)
        
        # 下载按钮区域
        dcol1, dcol2 = st.columns(2)
        
        # 下载结果图
        combined_img = create_downloadable_figure(images, captions)
        if combined_img is not None:
            buf = io.BytesIO()
            combined_img.save(buf, format='PNG')
            dcol1.download_button(
                label="💾 保存结果图",
                data=buf.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_result.png",
                mime="image/png",
                key=f"img_{idx}"
            )
        
        # 下载CSV
        df = build_csv_data(results)
        if not df.empty:
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False, encoding='utf-8-sig')
            dcol2.download_button(
                label="📊 导出CSV坐标",
                data=csv_buf.getvalue().encode('utf-8-sig'),
                file_name=f"{uploaded_file.name.split('.')[0]}_coords.csv",
                mime="text/csv",
                key=f"csv_{idx}"
            )
        
        # 汇总数据
        all_summaries.append({
            'filename': uploaded_file.name,
            'log_count': results.get('log', {}).get('count', 0),
            'local_count': results.get('local', {}).get('count', 0),
            'recommended': (results.get('log', {}).get('count', 0) + 
                           results.get('local', {}).get('count', 0)) // 2
        })
        
        st.markdown("---")
        
    except Exception as e:
        st.error(f"处理 {uploaded_file.name} 时出错: {str(e)}")
    
    progress_bar.progress((idx + 1) / len(uploaded_files))

status_text.empty()
progress_bar.empty()

# 批量汇总导出
if len(all_summaries) > 1:
    st.sidebar.markdown("---")
    st.sidebar.subheader("批量汇总")
    summary_df = pd.DataFrame(all_summaries)
    csv_summary = io.StringIO()
    summary_df.to_csv(csv_summary, index=False, encoding='utf-8-sig')
    st.sidebar.download_button(
        label="📥 下载汇总CSV",
        data=csv_summary.getvalue().encode('utf-8-sig'),
        file_name=f"batch_summary_{datetime.now().strftime('%H%M%S')}.csv",
        mime="text/csv"
    )
    st.sidebar.dataframe(summary_df, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.success("检测完成！")
