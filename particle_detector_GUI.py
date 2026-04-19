"""
荧光颗粒检测器 v3.3
修复：中文路径支持、TIF格式兼容
"""

import sys
import os
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import blob_log, peak_local_max
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QFileDialog,
    QMessageBox, QProgressBar, QGroupBox, QTextEdit, QSplitter,
    QFrame
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QFont
import pandas as pd
from datetime import datetime

# 支持的图片格式
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')


def load_image(image_path):
    """
    加载图片，支持中文路径和特殊字符

    参数:
        image_path: 图片路径
    返回:
        img: OpenCV格式的图片数组
    """
    # 标准化路径
    image_path = os.path.normpath(image_path)

    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"文件不存在: {image_path}")

    # 方法1：cv2.imread（英文路径）
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        # 如果是16位或32位图像，转换为8位
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        elif img.dtype == np.uint32:
            img = (img / 16777216).astype(np.uint8)

        # 确保是3通道BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img

    # 方法2：使用imdecode（中文路径兼容）
    try:
        with open(image_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

            if img is not None:
                # 同样处理位深度
                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)
                elif img.dtype == np.uint32:
                    img = (img / 16777216).astype(np.uint8)

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                return img
    except Exception as e:
        raise ValueError(f"无法解码图片: {image_path}, 错误: {str(e)}")

    raise ValueError(f"不支持的图片格式或损坏的文件: {image_path}")


class DetectionWorker(QThread):
    """后台检测线程"""
    progress = Signal(int)
    result = Signal(dict)
    log = Signal(str)

    def __init__(self, image_path, params, auto_save_dir=None):
        super().__init__()
        self.image_path = image_path
        self.params = params
        self.auto_save_dir = auto_save_dir
        self.is_batch = isinstance(image_path, list)

    def run(self):
        try:
            if self.is_batch:
                self.process_batch()
            else:
                self.process_single(self.image_path)
        except Exception as e:
            self.log.emit(f"错误: {str(e)}")
            import traceback
            self.log.emit(traceback.format_exc())

    def process_batch(self):
        """批量处理"""
        total = len(self.image_path)
        all_results = []

        for i, path in enumerate(self.image_path):
            self.log.emit(f"[{i + 1}/{total}] 处理: {os.path.basename(path)}")
            try:
                result = self.process_single(path, emit_signal=False)
                all_results.append({'file': path, 'result': result})
            except Exception as e:
                self.log.emit(f"  跳过: {os.path.basename(path)} - {str(e)}")
            self.progress.emit(int((i + 1) / total * 100))

        self.result.emit({
            'type': 'batch',
            'data': all_results,
            'auto_save_dir': self.auto_save_dir
        })

    def process_single(self, image_path, emit_signal=True):
        """单张处理"""
        # 使用修复后的加载函数
        img = load_image(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 选择颜色通道
        channel_map = {'红色': 0, '绿色': 1, '蓝色': 2}
        ch_idx = channel_map.get(self.params['color'], 0)

        # 处理多通道图像
        if len(img_rgb.shape) == 3 and img_rgb.shape[2] >= 3:
            channel = img_rgb[:, :, ch_idx].astype(float)
        else:
            channel = img_rgb.astype(float)

        # 根据模式设置参数
        if self.params['mode'] == '高精度':
            log_threshold, local_threshold, min_distance = 0.08, 0.2, 8
        elif self.params['mode'] == '高召回':
            log_threshold, local_threshold, min_distance = 0.03, 0.1, 4
        else:
            log_threshold, local_threshold, min_distance = 0.05, 0.15, 6

        # 根据是否等大调整
        if self.params['uniform_size']:
            min_sigma, max_sigma = 1.5, 3
        else:
            min_sigma, max_sigma = 0.8, 5

        results = {'img_rgb': img_rgb, 'file': image_path}

        # LoG检测
        if self.params['use_log']:
            self.log.emit("  正在运行LoG检测...")
            red_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            blobs = blob_log(
                red_norm,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=10,
                threshold=log_threshold,
                overlap=0.3 if self.params['uniform_size'] else 0.5
            )
            if len(blobs) > 0:
                blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
            results['log'] = {
                'count': len(blobs),
                'blobs': blobs,
            }
            self.log.emit(f"  LoG检测到 {len(blobs)} 个颗粒")

        # 局部最大值检测
        if self.params['use_local']:
            self.log.emit("  正在运行局部最大值检测...")
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
            self.log.emit(f"  局部最大值检测到 {len(coords)} 个颗粒")

        if emit_signal:
            self.result.emit({
                'type': 'single',
                'file': image_path,
                'data': results
            })

        return results


class DropArea(QFrame):
    """拖拽区域"""
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        self.setMaximumHeight(150)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.icon_label = QLabel("📁")
        self.icon_label.setStyleSheet("font-size: 40px;")
        self.icon_label.setAlignment(Qt.AlignCenter)

        self.text_label = QLabel("拖拽图片到这里，或点击选择")
        self.text_label.setStyleSheet("font-size: 14px; color: #333333;")
        self.text_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)

        self.setStyleSheet("""
            DropArea {
                background-color: #f0f0f0;
                border: 2px dashed #aaaaaa;
                border-radius: 10px;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                DropArea {
                    background-color: #d4edda;
                    border: 2px dashed #28a745;
                    border-radius: 10px;
                }
            """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            DropArea {
                background-color: #f0f0f0;
                border: 2px dashed #aaaaaa;
                border-radius: 10px;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("""
            DropArea {
                background-color: #f0f0f0;
                border: 2px dashed #aaaaaa;
                border-radius: 10px;
            }
        """)
        urls = event.mimeData().urls()
        files = []

        for url in urls:
            path = url.toLocalFile()
            path = os.path.normpath(path)  # 标准化路径

            if os.path.isfile(path):
                if path.lower().endswith(SUPPORTED_FORMATS):
                    files.append(path)
            elif os.path.isdir(path):
                for root, dirs, filenames in os.walk(path):
                    for f in filenames:
                        if f.lower().endswith(SUPPORTED_FORMATS):
                            files.append(os.path.normpath(os.path.join(root, f)))

        if files:
            self.files_dropped.emit(files)

    def mousePressEvent(self, event):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFiles)

        # 构建文件过滤器
        filter_str = "Images ("
        filter_str += " ".join([f"*{ext}" for ext in SUPPORTED_FORMATS])
        filter_str += ");;All Files (*)"

        dialog.setNameFilter(filter_str)

        if dialog.exec():
            files = [os.path.normpath(f) for f in dialog.selectedFiles()]
            self.files_dropped.emit(files)

    def set_files(self, count, name=None):
        if count == 1 and name:
            self.icon_label.setText("✓")
            self.text_label.setText(f"已选择: {name}")
        else:
            self.icon_label.setText("✓")
            self.text_label.setText(f"已选择 {count} 张图片")


class ResultCanvas(QWidget):
    """结果展示组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(5, 5, 5, 5)

        self.labels = []
        self.titles = []

        titles = ['原始图像', 'LoG检测', '局部最大值']
        for title in titles:
            container = QFrame()
            container.setStyleSheet("background-color: #ffffff; border: 1px solid #dddddd;")
            container_layout = QVBoxLayout(container)
            container_layout.setSpacing(5)
            container_layout.setContentsMargins(5, 5, 5, 5)

            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #000000;")
            container_layout.addWidget(title_label)

            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setMinimumSize(300, 300)
            img_label.setStyleSheet("background-color: #f5f5f5;")
            img_label.setText("等待检测...")
            container_layout.addWidget(img_label)

            self.labels.append(img_label)
            self.titles.append(title_label)
            self.layout.addWidget(container)

    def array_to_qpixmap(self, img_array, max_size=350):
        """numpy数组转QPixmap"""
        if img_array is None:
            return None

        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        h, w = img_array.shape[:2]
        scale = min(max_size / w, max_size / h, 1.0)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(img_array.shape) == 3:
            bytes_per_line = 3 * img_array.shape[1]
            q_image = QImage(img_array.data, img_array.shape[1], img_array.shape[0],
                             bytes_per_line, QImage.Format_RGB888)
        else:
            bytes_per_line = img_array.shape[1]
            q_image = QImage(img_array.data, img_array.shape[1], img_array.shape[0],
                             bytes_per_line, QImage.Format_Grayscale8)

        return QPixmap.fromImage(q_image)

    def plot_results(self, img_rgb, log_data=None, local_data=None):
        """绘制结果"""
        # 原始图像
        pixmap0 = self.array_to_qpixmap(img_rgb)
        if pixmap0:
            self.labels[0].setPixmap(pixmap0)
            self.titles[0].setText(f"原始图像 ({img_rgb.shape[1]}x{img_rgb.shape[0]})")

        # LoG结果
        if log_data and 'blobs' in log_data and len(log_data['blobs']) > 0:
            img_log = img_rgb.copy()
            for y, x, r in log_data['blobs']:
                cv2.circle(img_log, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.circle(img_log, (int(x), int(y)), 2, (255, 0, 0), -1)

            pixmap1 = self.array_to_qpixmap(img_log)
            if pixmap1:
                self.labels[1].setPixmap(pixmap1)
                self.titles[1].setText(f"LoG: {log_data['count']} 个颗粒")
        else:
            self.labels[1].setText("未启用")
            self.titles[1].setText("LoG检测")

        # 局部最大值结果
        if local_data and 'coords' in local_data and len(local_data['coords']) > 0:
            img_local = img_rgb.copy()
            coords = local_data['coords']
            for y, x in coords:
                cv2.circle(img_local, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.circle(img_local, (int(x), int(y)), 7, (255, 255, 255), 2)

            pixmap2 = self.array_to_qpixmap(img_local)
            if pixmap2:
                self.labels[2].setPixmap(pixmap2)
                self.titles[2].setText(f"局部最大值: {local_data['count']} 个颗粒")
        else:
            self.labels[2].setText("未启用")
            self.titles[2].setText("局部最大值检测")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("荧光颗粒检测器 v3.3")
        self.setGeometry(100, 100, 1400, 800)

        self.current_files = []
        self.current_results = []
        self.default_output_dir = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        # 标题
        title = QLabel("荧光颗粒检测器")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #1a5490;")
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        # 版本信息
        version = QLabel("v3.3 - 支持中文路径和TIF格式")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet("color: #666666; font-size: 11px;")
        left_layout.addWidget(version)

        # 拖拽区域
        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self.handle_files)
        left_layout.addWidget(self.drop_area)

        # 参数设置
        params_group = QGroupBox("检测参数")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(10)

        # 颗粒颜色
        color_layout = QHBoxLayout()
        color_label = QLabel("颗粒颜色:")
        color_label.setStyleSheet("color: #000000;")
        color_layout.addWidget(color_label)
        self.color_combo = QComboBox()
        self.color_combo.addItems(['红色', '绿色', '蓝色'])
        self.color_combo.setCurrentText('红色')
        color_layout.addWidget(self.color_combo)
        params_layout.addLayout(color_layout)

        # 检测模式
        mode_layout = QHBoxLayout()
        mode_label = QLabel("检测模式:")
        mode_label.setStyleSheet("color: #000000;")
        mode_layout.addWidget(mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['平衡', '高精度', '高召回'])
        self.mode_combo.setToolTip("高精度=减少误检, 高召回=检测更多")
        mode_layout.addWidget(self.mode_combo)
        params_layout.addLayout(mode_layout)

        # 颗粒均匀
        self.uniform_check = QCheckBox("颗粒大小均匀（推荐勾选如果颗粒大小相似）")
        self.uniform_check.setChecked(False)
        self.uniform_check.setStyleSheet("color: #000000;")
        params_layout.addWidget(self.uniform_check)

        # 检测方法
        method_label = QLabel("检测方法（至少选择一项）:")
        method_label.setStyleSheet("color: #000000; font-weight: bold;")
        params_layout.addWidget(method_label)

        self.log_check = QCheckBox("LoG斑点检测（提供颗粒半径信息）")
        self.log_check.setChecked(True)
        self.log_check.setStyleSheet("color: #000000;")
        params_layout.addWidget(self.log_check)

        self.local_check = QCheckBox("局部最大值检测（速度更快）")
        self.local_check.setChecked(True)
        self.local_check.setStyleSheet("color: #000000;")
        params_layout.addWidget(self.local_check)

        # 批量处理
        self.batch_check = QCheckBox("批量处理（自动保存结果）")
        self.batch_check.setChecked(False)
        self.batch_check.setStyleSheet("color: #000000;")
        params_layout.addWidget(self.batch_check)

        left_layout.addWidget(params_group)

        # 运行按钮
        self.run_btn = QPushButton("开始检测")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover { background-color: #218838; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        self.run_btn.clicked.connect(self.start_detection)
        left_layout.addWidget(self.run_btn)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 3px;
            }
        """)
        left_layout.addWidget(self.progress)

        # 日志
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                color: #212529;
                font-family: Consolas, monospace;
                font-size: 11px;
                border: 1px solid #ced4da;
                border-radius: 4px;
            }
        """)
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # 右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        # 结果展示
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout(result_group)

        self.result_canvas = ResultCanvas()
        result_layout.addWidget(self.result_canvas)

        right_layout.addWidget(result_group)

        # 统计信息
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_label = QLabel("等待检测开始...")
        self.stats_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                color: #000000;
                padding: 15px;
                border-radius: 6px;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        self.stats_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.stats_label.setWordWrap(True)
        self.stats_label.setMinimumHeight(80)
        stats_layout.addWidget(self.stats_label)

        right_layout.addWidget(stats_group)

        # 操作按钮
        btn_layout = QHBoxLayout()

        self.save_btn = QPushButton("保存结果图")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover { background-color: #0056b3; }
            QPushButton:disabled { background-color: #6c757d; color: #cccccc; }
        """)
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)

        self.export_btn = QPushButton("导出CSV坐标")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover { background-color: #5a32a3; }
            QPushButton:disabled { background-color: #6c757d; color: #cccccc; }
        """)
        self.export_btn.clicked.connect(self.export_csv)
        self.export_btn.setEnabled(False)
        btn_layout.addWidget(self.export_btn)

        right_layout.addLayout(btn_layout)

        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1050])

    def handle_files(self, files):
        self.current_files = files

        if len(files) == 1:
            self.drop_area.set_files(1, os.path.basename(files[0]))
            self.log(f"加载: {files[0]}")
        else:
            self.drop_area.set_files(len(files))
            self.log(f"批量加载: {len(files)} 张图片")

    def get_params(self):
        return {
            'color': self.color_combo.currentText(),
            'mode': self.mode_combo.currentText(),
            'uniform_size': self.uniform_check.isChecked(),
            'use_log': self.log_check.isChecked(),
            'use_local': self.local_check.isChecked()
        }

    def log(self, message):
        time_str = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{time_str}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_detection(self):
        if not self.current_files:
            QMessageBox.warning(self, "提示", "请先选择图片！")
            return

        params = self.get_params()

        if not params['use_log'] and not params['use_local']:
            QMessageBox.warning(self, "提示", "请至少选择一种检测方法！")
            return

        # 批量处理时选择输出目录
        auto_save_dir = None
        if self.batch_check.isChecked() and len(self.current_files) > 1:
            auto_save_dir = QFileDialog.getExistingDirectory(self, "选择结果保存文件夹")
            if not auto_save_dir:
                return
            self.log(f"批量结果将保存到: {auto_save_dir}")

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.save_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.log("=" * 40)
        self.log("开始检测...")

        self.worker = DetectionWorker(self.current_files, params, auto_save_dir)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log)
        self.worker.result.connect(self.handle_result)
        self.worker.finished.connect(self.detection_finished)
        self.worker.start()

    def handle_result(self, result):
        if result['type'] == 'single':
            self.current_results = [result]
            data = result['data']
            file = result['file']

            self.result_canvas.plot_results(
                data['img_rgb'],
                data.get('log'),
                data.get('local')
            )

            stats_text = f"<b>文件:</b> {os.path.basename(file)}<br>"
            stats_text += f"<b>尺寸:</b> {data['img_rgb'].shape[1]} x {data['img_rgb'].shape[0]} 像素<br>"

            if 'log' in data:
                stats_text += f"<b>LoG检测:</b> {data['log']['count']} 个颗粒<br>"
            if 'local' in data:
                stats_text += f"<b>局部最大值:</b> {data['local']['count']} 个颗粒<br>"

            if 'log' in data and 'local' in data:
                avg = (data['log']['count'] + data['local']['count']) // 2
                stats_text += f"<b>建议计数:</b> <span style='color: #28a745; font-size: 15px;'>{avg}</span> 个"

            self.stats_label.setText(stats_text)
            self.save_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

        elif result['type'] == 'batch':
            self.current_results = result['data']
            self.log(f"批量完成: {len(result['data'])} 张图片")

            if result['data']:
                first = result['data'][0]
                self.result_canvas.plot_results(
                    first['result']['img_rgb'],
                    first['result'].get('log'),
                    first['result'].get('local')
                )

            # 自动保存
            auto_save_dir = result.get('auto_save_dir')
            if auto_save_dir:
                self.save_batch_results(auto_save_dir)
            else:
                # 询问是否保存
                reply = QMessageBox.question(self, "保存结果",
                                             f"批量处理完成！是否保存结果？",
                                             QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    output_dir = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
                    if output_dir:
                        self.save_batch_results(output_dir)

    def detection_finished(self):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.log("检测完成！")
        self.log("=" * 40)

    def save_result(self):
        if not self.current_results:
            return

        file, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "result.png",
            "PNG Images (*.png);;JPEG Images (*.jpg)"
        )
        if file:
            import matplotlib.pyplot as plt
            data = self.current_results[0]['data']
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

            img_rgb = data['img_rgb']
            axes[0].imshow(img_rgb)
            axes[0].set_title('Original')
            axes[0].axis('off')

            if 'log' in data and len(data['log']['blobs']) > 0:
                axes[1].imshow(img_rgb)
                for y, x, r in data['log']['blobs']:
                    circle = plt.Circle((x, y), r, color='lime', fill=False)
                    axes[1].add_patch(circle)
                axes[1].set_title(f"LoG: {data['log']['count']}")
            else:
                axes[1].set_title('LoG: 未启用')
            axes[1].axis('off')

            if 'local' in data and len(data['local']['coords']) > 0:
                axes[2].imshow(img_rgb)
                coords = data['local']['coords']
                axes[2].scatter(coords[:, 1], coords[:, 0], c='lime', s=20)
                axes[2].set_title(f"LocalMax: {data['local']['count']}")
            else:
                axes[2].set_title('LocalMax: 未启用')
            axes[2].axis('off')

            plt.tight_layout()
            fig.savefig(file, dpi=200, bbox_inches='tight')
            plt.close()

            self.log(f"结果已保存: {file}")
            QMessageBox.information(self, "完成", f"结果已保存到:\\n{file}")

    def export_csv(self):
        if not self.current_results:
            return

        file, _ = QFileDialog.getSaveFileName(
            self, "导出CSV", "coordinates.csv",
            "CSV Files (*.csv)"
        )
        if file:
            data = self.current_results[0]['data']

            rows = []
            if 'log' in data:
                for i, (y, x, r) in enumerate(data['log']['blobs']):
                    rows.append({
                        'method': 'LoG',
                        'id': i + 1,
                        'x': int(x),
                        'y': int(y),
                        'radius': round(r, 2)
                    })
            if 'local' in data:
                coords = data['local']['coords']
                for i, (y, x) in enumerate(coords):
                    rows.append({
                        'method': 'LocalMax',
                        'id': i + 1,
                        'x': int(x),
                        'y': int(y),
                        'radius': 'N/A'
                    })

            df = pd.DataFrame(rows)
            df.to_csv(file, index=False, encoding='utf-8-sig')
            self.log(f"CSV已导出: {file}")
            QMessageBox.information(self, "完成", f"坐标已导出:\\n{file}\\n共 {len(rows)} 个颗粒")

    def save_batch_results(self, output_dir):
        import matplotlib.pyplot as plt

        summary = []
        self.log(f"保存到: {output_dir}")

        for item in self.current_results:
            file = item['file']
            data = item['result']
            basename = os.path.splitext(os.path.basename(file))[0]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
            img_rgb = data['img_rgb']

            axes[0].imshow(img_rgb)
            axes[0].set_title('Original')
            axes[0].axis('off')

            if 'log' in data and len(data['log']['blobs']) > 0:
                axes[1].imshow(img_rgb)
                for y, x, r in data['log']['blobs']:
                    circle = plt.Circle((x, y), r, color='lime', fill=False)
                    axes[1].add_patch(circle)
                axes[1].set_title(f"LoG: {data['log']['count']}")
            else:
                axes[1].set_title('LoG: 未启用')
            axes[1].axis('off')

            if 'local' in data and len(data['local']['coords']) > 0:
                axes[2].imshow(img_rgb)
                coords = data['local']['coords']
                axes[2].scatter(coords[:, 1], coords[:, 0], c='lime', s=15)
                axes[2].set_title(f"LocalMax: {data['local']['count']}")
            else:
                axes[2].set_title('LocalMax: 未启用')
            axes[2].axis('off')

            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{basename}_result.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

            summary.append({
                'filename': basename,
                'log_count': data.get('log', {}).get('count', 0),
                'local_count': data.get('local', {}).get('count', 0),
                'recommended': (data.get('log', {}).get('count', 0) +
                                data.get('local', {}).get('count', 0)) // 2
            })

        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(output_dir, "summary.csv"), index=False, encoding='utf-8-sig')

        self.log(f"批量保存完成！")
        QMessageBox.information(self, "完成",
                                f"处理了 {len(summary)} 张图片\\n"
                                f"保存到: {output_dir}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 简洁的白色主题
    app.setStyleSheet("""
        QMainWindow {
            background-color: #ffffff;
        }
        QWidget {
            font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #cccccc;
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 8px;
            background-color: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #000000;
        }
        QLabel {
            color: #000000;
        }
        QComboBox {
            padding: 5px;
            border: 1px solid #aaaaaa;
            border-radius: 4px;
            background-color: #ffffff;
            color: #000000;
        }
        QCheckBox {
            color: #000000;
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QPushButton {
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }
        QTextEdit {
            border: 1px solid #cccccc;
            border-radius: 4px;
        }
        QProgressBar {
            border: 1px solid #aaaaaa;
            border-radius: 4px;
            text-align: center;
        }
        QSplitter::handle {
            background-color: #dddddd;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
