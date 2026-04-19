"""
荧光颗粒检测器 - 双方法对比完整版
支持 LoG斑点检测 和 局部最大值检测 两种算法
日期: 2026-04-17
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import blob_log, peak_local_max
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime


class FluorescentParticleDetector:
    """荧光颗粒检测器类"""

    def __init__(self, image_path):
        """初始化并加载图像"""
        self.image_path = image_path
        self.load_image()

        # 存储结果
        self.log_blobs = None
        self.log_params = None
        self.local_max_coords = None
        self.clustered_coords = None
        self.local_params = None

    def load_image(self):
        """加载图像"""
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"找不到图片: {self.image_path}\n请检查路径是否正确！")

        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.red_channel = self.img_rgb[:, :, 0].astype(float)
        self.green_channel = self.img_rgb[:, :, 1].astype(float)
        self.blue_channel = self.img_rgb[:, :, 2].astype(float)

        print(f"✓ 图像加载成功: {self.img_rgb.shape}")

    def detect_log(self, min_sigma=1, max_sigma=4, num_sigma=10,
                   threshold=0.05, overlap=0.5):
        """
        LoG (Laplacian of Gaussian) 斑点检测

        原理: 多尺度高斯拉普拉斯滤波，检测圆形亮斑
        适用: 颗粒圆形、大小不一、需要半径信息
        """
        print(f"\n{'=' * 60}")
        print("[LoG斑点检测]")
        print(f"{'=' * 60}")
        print(f"参数设置:")
        print(f"  min_sigma = {min_sigma} (最小半径)")
        print(f"  max_sigma = {max_sigma} (最大半径)")
        print(f"  num_sigma = {num_sigma} (尺度数量)")
        print(f"  threshold = {threshold} (检测阈值)")
        print(f"  overlap   = {overlap} (重叠容忍度)")

        # 归一化
        red_norm = (self.red_channel - self.red_channel.min()) / \
                   (self.red_channel.max() - self.red_channel.min())

        # LoG检测
        blobs = blob_log(
            red_norm,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap
        )

        # 计算实际半径
        if len(blobs) > 0:
            blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

        self.log_blobs = blobs
        self.log_params = {
            'min_sigma': min_sigma,
            'max_sigma': max_sigma,
            'num_sigma': num_sigma,
            'threshold': threshold,
            'overlap': overlap
        }

        print(f"\n✓ 检测完成: 发现 {len(blobs)} 个颗粒")
        return blobs

    def detect_local_max(self, min_distance=6, threshold_abs=0.15,
                         sigma=2, bg_size=50,
                         use_clustering=False, dbscan_eps=4):
        """
        局部最大值检测（带背景减除）

        原理: 背景减除后寻找亮度局部峰值
        适用: 颗粒密集、背景不均匀、需要快速检测
        """
        print(f"\n{'=' * 60}")
        print("[局部最大值检测]")
        print(f"{'=' * 60}")
        print(f"参数设置:")
        print(f"  min_distance  = {min_distance} (最小点间距)")
        print(f"  threshold_abs = {threshold_abs} (亮度阈值)")
        print(f"  sigma         = {sigma} (平滑程度)")
        print(f"  bg_size       = {bg_size} (背景窗口)")
        print(f"  use_clustering= {use_clustering} (是否聚类)")
        if use_clustering:
            print(f"  dbscan_eps    = {dbscan_eps} (聚类半径)")

        # 背景减除
        background = ndimage.uniform_filter(self.red_channel, size=bg_size)
        bg_subtracted = self.red_channel - background
        bg_subtracted = np.clip(bg_subtracted, 0, None)

        # 归一化
        processed = (bg_subtracted - bg_subtracted.min()) / \
                    (bg_subtracted.max() - bg_subtracted.min() + 1e-8)

        # 高斯平滑
        smoothed = ndimage.gaussian_filter(processed, sigma=sigma)
        self.smoothed_image = smoothed

        # 局部最大值检测
        coordinates = peak_local_max(
            smoothed,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            exclude_border=True
        )

        self.local_max_coords = coordinates
        final_coords = coordinates

        # DBSCAN聚类去重
        if use_clustering and len(coordinates) > 0:
            clustering = DBSCAN(eps=dbscan_eps, min_samples=1).fit(coordinates)
            cluster_centers = []

            for label in set(clustering.labels_):
                if label != -1:
                    points = coordinates[clustering.labels_ == label]
                    center = points.mean(axis=0).astype(int)
                    cluster_centers.append(center)

            self.clustered_coords = np.array(cluster_centers) if cluster_centers else np.array([])
            final_coords = self.clustered_coords

            print(f"\n  初步检测: {len(coordinates)} 个点")
            print(f"  聚类合并: {len(self.clustered_coords)} 个颗粒")
            print(f"  合并比例: {(1 - len(self.clustered_coords) / len(coordinates)) * 100:.1f}%")

        self.local_params = {
            'min_distance': min_distance,
            'threshold_abs': threshold_abs,
            'sigma': sigma,
            'bg_size': bg_size,
            'use_clustering': use_clustering,
            'dbscan_eps': dbscan_eps if use_clustering else None
        }

        print(f"\n✓ 检测完成: 发现 {len(final_coords)} 个颗粒")
        return final_coords

    def visualize_comparison(self, save_path=None, show_radius_hist=True):
        """对比显示两种方法的检测结果"""
        fig = plt.figure(figsize=(20, 12))

        # 创建子图布局
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # ========== 第1行: LoG方法 ==========
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.img_rgb)
        ax1.set_title('原始图像', fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.img_rgb)
        if self.log_blobs is not None and len(self.log_blobs) > 0:
            for y, x, r in self.log_blobs:
                circle = plt.Circle((x, y), r, color='lime',
                                    linewidth=1.5, fill=False)
                ax2.add_patch(circle)
                ax2.plot(x, y, 'r.', markersize=3)
        ax2.set_title(f'LoG检测: {len(self.log_blobs) if self.log_blobs is not None else 0} 个颗粒',
                      fontsize=12, fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        if self.log_blobs is not None and len(self.log_blobs) > 0 and show_radius_hist:
            ax3.hist(self.log_blobs[:, 2], bins=20, color='green',
                     alpha=0.7, edgecolor='black')
            ax3.set_title('LoG颗粒半径分布', fontsize=12, fontweight='bold')
            ax3.set_xlabel('半径 (像素)')
            ax3.set_ylabel('数量')
        else:
            ax3.text(0.5, 0.5, '无LoG数据', ha='center', va='center')
            ax3.set_title('半径分布', fontsize=12)
        ax3.axis('off' if self.log_blobs is None or len(self.log_blobs) == 0 else 'on')

        # ========== 第2行: 局部最大值方法 ==========
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(self.img_rgb)
        ax4.set_title('原始图像', fontsize=12, fontweight='bold')
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(self.img_rgb)
        if self.local_max_coords is not None:
            coords = self.clustered_coords if self.clustered_coords is not None else self.local_max_coords
            ax5.scatter(coords[:, 1], coords[:, 0], c='lime', s=25,
                        alpha=0.7, edgecolors='black', linewidths=0.5)
            title = f'局部最大值: {len(coords)} 个'
            if self.clustered_coords is not None:
                title += f'\n(原始{len(self.local_max_coords)}个)'
            ax5.set_title(title, fontsize=12, fontweight='bold')
        else:
            ax5.set_title('局部最大值检测', fontsize=12)
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 2])
        if hasattr(self, 'smoothed_image'):
            im = ax6.imshow(self.smoothed_image, cmap='hot')
            ax6.set_title('预处理后图像\n(背景减除+高斯平滑)', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        else:
            ax6.text(0.5, 0.5, '无预处理数据', ha='center', va='center')
            ax6.set_title('预处理图像', fontsize=12)

        plt.suptitle('荧光颗粒检测对比结果', fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"\n✓ 对比图已保存: {save_path}")

        plt.show()

    def print_report(self, save_json=None):
        """打印详细统计报告"""
        print(f"\n{'=' * 70}")
        print("                    荧光颗粒检测统计报告")
        print(f"{'=' * 70}")
        print(f"图像文件: {self.image_path}")
        print(f"图像尺寸: {self.img_rgb.shape}")
        print(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report_data = {
            'image_path': self.image_path,
            'image_shape': self.img_rgb.shape,
            'timestamp': datetime.now().isoformat(),
            'methods': {}
        }

        # LoG结果
        if self.log_blobs is not None:
            print(f"\n{'─' * 70}")
            print("[LoG斑点检测]")
            print(f"{'─' * 70}")
            print(f"  颗粒总数:     {len(self.log_blobs)}")

            if len(self.log_blobs) > 0:
                radii = self.log_blobs[:, 2]
                print(f"  平均半径:     {radii.mean():.2f} 像素")
                print(f"  半径标准差:   {radii.std():.2f} 像素")
                print(f"  半径范围:     {radii.min():.2f} - {radii.max():.2f} 像素")
                print(f"  中位半径:     {np.median(radii):.2f} 像素")

                # 半径分布
                small = np.sum(radii < 2)
                medium = np.sum((radii >= 2) & (radii < 4))
                large = np.sum(radii >= 4)
                print(f"  颗粒大小分布:")
                print(f"    小颗粒 (<2px):    {small} ({small / len(radii) * 100:.1f}%)")
                print(f"    中颗粒 (2-4px):   {medium} ({medium / len(radii) * 100:.1f}%)")
                print(f"    大颗粒 (>4px):    {large} ({large / len(radii) * 100:.1f}%)")

            report_data['methods']['log'] = {
                'count': int(len(self.log_blobs)),
                'params': self.log_params,
                'radius_mean': float(radii.mean()) if len(self.log_blobs) > 0 else 0,
                'radius_std': float(radii.std()) if len(self.log_blobs) > 0 else 0
            }

        # 局部最大值结果
        if self.local_max_coords is not None:
            print(f"\n{'─' * 70}")
            print("[局部最大值检测]")
            print(f"{'─' * 70}")

            raw_count = len(self.local_max_coords)
            final_count = len(self.clustered_coords) if self.clustered_coords is not None else raw_count

            if self.clustered_coords is not None:
                print(f"  原始检测数:   {raw_count}")
                print(f"  聚类后数量:   {final_count}")
                print(f"  合并点数:     {raw_count - final_count}")
                print(f"  合并比例:     {(1 - final_count / raw_count) * 100:.1f}%")
            else:
                print(f"  颗粒总数:     {final_count}")

            # 计算密度
            img_area = self.img_rgb.shape[0] * self.img_rgb.shape[1]
            density = final_count / img_area * 1000000  # 每百万像素
            print(f"  颗粒密度:     {density:.2f} 个/百万像素")

            report_data['methods']['local_max'] = {
                'raw_count': int(raw_count),
                'final_count': int(final_count),
                'params': self.local_params,
                'density_per_megapixel': float(density)
            }

        # 方法对比
        if self.log_blobs is not None and self.local_max_coords is not None:
            print(f"\n{'─' * 70}")
            print("[方法对比]")
            print(f"{'─' * 70}")

            log_count = len(self.log_blobs)
            local_count = len(self.clustered_coords) if self.clustered_coords is not None else len(
                self.local_max_coords)

            diff = abs(log_count - local_count)
            diff_percent = diff / max(log_count, local_count) * 100

            print(f"  LoG检测:        {log_count:6d} 个")
            print(f"  局部最大值:     {local_count:6d} 个")
            print(f"  绝对差异:       {diff:6d} 个")
            print(f"  相对差异:       {diff_percent:5.1f}%")

            if log_count > local_count:
                print(f"\n  → LoG检测到更多颗粒 (可能包含更多小颗粒或噪点)")
            elif local_count > log_count:
                print(f"\n  → 局部最大值检测到更多颗粒 (可能对密集区域更敏感)")
            else:
                print(f"\n  → 两种方法结果一致")

            # 推荐最终计数
            avg_count = (log_count + local_count) // 2
            print(f"\n  建议最终计数:   {avg_count} 个 (平均值)")

            report_data['comparison'] = {
                'log_count': int(log_count),
                'local_max_count': int(local_count),
                'difference': int(diff),
                'difference_percent': float(diff_percent),
                'recommended_count': int(avg_count)
            }

        print(f"\n{'=' * 70}")

        # 保存JSON报告
        if save_json:
            with open(save_json, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"✓ 报告已保存: {save_json}")

        return report_data

    def export_coordinates(self, log_path=None, local_path=None):
        """导出颗粒坐标到CSV"""
        import pandas as pd

        if log_path and self.log_blobs is not None:
            df_log = pd.DataFrame({
                'y': self.log_blobs[:, 0],
                'x': self.log_blobs[:, 1],
                'radius': self.log_blobs[:, 2]
            })
            df_log.to_csv(log_path, index=False)
            print(f"✓ LoG坐标已导出: {log_path}")

        if local_path and self.local_max_coords is not None:
            coords = self.clustered_coords if self.clustered_coords is not None else self.local_max_coords
            df_local = pd.DataFrame({
                'y': coords[:, 0],
                'x': coords[:, 1]
            })
            df_local.to_csv(local_path, index=False)
            print(f"✓ 局部最大值坐标已导出: {local_path}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='荧光颗粒检测器 - 支持LoG和局部最大值两种方法',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python particle_detector.py image.jpg
  python particle_detector.py image.jpg --log-threshold 0.03 --local-distance 4
  python particle_detector.py image.jpg -o results.png --export-csv
        """
    )

    # 输入输出
    parser.add_argument('image', help='输入图像路径')
    parser.add_argument('-o', '--output', help='保存结果图像路径', default='detection_result.png')
    parser.add_argument('--report', help='保存JSON报告路径', default='report.json')
    parser.add_argument('--export-csv', action='store_true', help='导出坐标到CSV')

    # LoG参数
    log_group = parser.add_argument_group('LoG斑点检测参数')
    log_group.add_argument('--log-min-sigma', type=float, default=1, help='最小半径 (默认: 1)')
    log_group.add_argument('--log-max-sigma', type=float, default=4, help='最大半径 (默认: 4)')
    log_group.add_argument('--log-num-sigma', type=int, default=10, help='尺度数量 (默认: 10)')
    log_group.add_argument('--log-threshold', type=float, default=0.05, help='检测阈值 (默认: 0.05)')
    log_group.add_argument('--log-overlap', type=float, default=0.5, help='重叠容忍度 (默认: 0.5)')

    # 局部最大值参数
    local_group = parser.add_argument_group('局部最大值检测参数')
    local_group.add_argument('--local-distance', type=int, default=6, help='最小点间距 (默认: 6)')
    local_group.add_argument('--local-threshold', type=float, default=0.15, help='亮度阈值 (默认: 0.15)')
    local_group.add_argument('--local-sigma', type=float, default=2, help='平滑程度 (默认: 2)')
    local_group.add_argument('--local-bg-size', type=int, default=50, help='背景窗口 (默认: 50)')
    local_group.add_argument('--local-cluster', action='store_true', help='启用DBSCAN聚类')
    local_group.add_argument('--local-eps', type=float, default=4, help='聚类半径 (默认: 4)')

    # 其他
    parser.add_argument('--no-vis', action='store_true', help='不显示图像')
    parser.add_argument('--channel', type=int, default=0, choices=[0, 1, 2],
                        help='颜色通道: 0=红, 1=绿, 2=蓝 (默认: 0)')

    args = parser.parse_args()

    # 创建检测器
    detector = FluorescentParticleDetector(args.image)

    # 运行LoG检测
    detector.detect_log(
        min_sigma=args.log_min_sigma,
        max_sigma=args.log_max_sigma,
        num_sigma=args.log_num_sigma,
        threshold=args.log_threshold,
        overlap=args.log_overlap
    )

    # 运行局部最大值检测
    detector.detect_local_max(
        min_distance=args.local_distance,
        threshold_abs=args.local_threshold,
        sigma=args.local_sigma,
        bg_size=args.local_bg_size,
        use_clustering=args.local_cluster,
        dbscan_eps=args.local_eps
    )

    # 可视化
    if not args.no_vis:
        detector.visualize_comparison(save_path=args.output)

    # 打印报告
    detector.print_report(save_json=args.report)

    # 导出CSV
    if args.export_csv:
        detector.export_coordinates(
            log_path='log_coordinates.csv',
            local_path='local_max_coordinates.csv'
        )

    print("\n✓ 全部完成！")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 方式1: 直接修改这里运行 (简单使用)
    IMAGE_PATH = r"G:\tools\red_sub.jpg"

    # 创建检测器
    detector = FluorescentParticleDetector(IMAGE_PATH)

    # ========== 场景A: 高召回率 (检测尽可能多的颗粒) ==========
    # detector.detect_log(min_sigma=0.8, max_sigma=4, num_sigma=15,
    #                    threshold=0.03, overlap=0.3)
    # detector.detect_local_max(min_distance=4, threshold_abs=0.1,
    #                          sigma=1.5, use_clustering=True, dbscan_eps=3)

    # ========== 场景B: 高精度 (减少误检) ==========
    # detector.detect_log(min_sigma=1.2, max_sigma=4, num_sigma=10,
    #                    threshold=0.1, overlap=0.2)
    # detector.detect_local_max(min_distance=8, threshold_abs=0.2,
    #                          sigma=2.5, use_clustering=False)

    # ========== 场景C: 平衡模式 (推荐) ==========
    detector.detect_log(min_sigma=1, max_sigma=4, num_sigma=10,
                        threshold=0.05, overlap=0.5)
    detector.detect_local_max(min_distance=6, threshold_abs=0.15,
                              sigma=2, use_clustering=False)

    # 显示结果
    detector.visualize_comparison(save_path="detection_result.png")

    # 打印报告
    detector.print_report(save_json="detection_report.json")

    # 导出坐标 (可选)
    # detector.export_coordinates("log_coords.csv", "local_coords.csv")

    print("\n" + "=" * 70)
    print("检测完成！请查看:")
    print("  - detection_result.png (对比图)")
    print("  - detection_report.json (详细报告)")
    print("=" * 70)

    # 方式2: 命令行运行 (高级使用)
    # 在终端输入: python particle_detector.py image.jpg --help 查看所有选项
