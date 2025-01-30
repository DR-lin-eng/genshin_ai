import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from collections import deque
import random
import os
from glob import glob
from tqdm import tqdm
import json
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import pyautogui
import keyboard
import threading
import queue
from torchvision import transforms
import time
import win32gui
import win32con
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import warnings
warnings.filterwarnings('ignore')

# 全局配置
CONFIG = {
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'BATCH_SIZE': 32,
    'SEQUENCE_LENGTH': 32,
    'FRAME_SIZE': (256, 256),
    'NUM_WORKERS': min(4, os.cpu_count()),
    'GPU_MEMORY_FRACTION': 0.8,
    'ENABLE_CUDA_OPTIMIZATION': True
}

def setup_environment():
    """设置运行环境"""
    print("\n正在设置运行环境...")
    
    # CUDA设置
    if torch.cuda.is_available():
        print(f"检测到CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        
        # 启用CUDA优化
        if CONFIG['ENABLE_CUDA_OPTIMIZATION']:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print("CUDA优化已启用")
            
        # 设置GPU内存使用限制
        try:
            torch.cuda.set_per_process_memory_fraction(
                CONFIG['GPU_MEMORY_FRACTION'], 0
            )
            print(f"GPU内存限制设置为: {CONFIG['GPU_MEMORY_FRACTION']*100}%")
        except Exception as e:
            print(f"设置GPU内存限制失败: {str(e)}")
    else:
        print("警告: 未检测到CUDA设备，将使用CPU运行")
        CONFIG['DEVICE'] = torch.device('cpu')
    
    # 创建必要的目录
    required_dirs = ['./videos', './models', './visualizations']
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("目录结构已创建")
    
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print("随机种子已设置")
    
    print("环境设置完成\n")

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
        self.memory_usage = []
    
    def checkpoint(self, name):
        """记录检查点"""
        self.checkpoints[name] = time.time() - self.start_time
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**2
            self.memory_usage.append((name, memory_used))
    
    def report(self):
        """生成性能报告"""
        print("\n性能报告:")
        print("-" * 50)
        
        # 时间统计
        for name, time_taken in self.checkpoints.items():
            print(f"{name}: {time_taken:.2f} 秒")
        
        # GPU内存使用
        if self.memory_usage:
            print("\nGPU内存使用:")
            for name, memory in self.memory_usage:
                print(f"{name}: {memory:.2f} MB")
        
        print("-" * 50)

def cleanup():
    """清理资源"""
    print("\n清理资源...")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU缓存已清理")
    
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("OpenCV窗口已关闭")
    
    # 强制垃圾回收
    gc.collect()
    print("垃圾回收已执行")
    
    print("资源清理完成")

# 初始化性能监控器
monitor = PerformanceMonitor()

class GPUSceneDetector:
    """GPU加速的场景检测器"""
    def __init__(self, batch_size=32):
        print("初始化GPU场景检测器...")
        self.device = CONFIG['DEVICE']
        self.batch_size = batch_size
        self.use_cuda_flow = False
        self.last_frames = None
        
        # 尝试初始化CUDA光流计算器
        try:
            if hasattr(cv2.cuda, 'FarnebackOpticalFlow'):
                self.flow_computer = cv2.cuda.FarnebackOpticalFlow.create(
                    numLevels=5,
                    pyrScale=0.5,
                    fastPyramids=False,
                    winSize=21,
                    numIters=3,
                    polyN=7,
                    polySigma=1.5,
                    flags=0
                )
                self.use_cuda_flow = True
                print("使用CUDA光流计算")
            else:
                print("CUDA光流不可用，使用CPU光流计算")
        except Exception as e:
            print(f"初始化CUDA光流失败: {str(e)}")
            print("使用CPU光流计算")
        
        # 创建特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ).to(self.device)
        
        # 创建场景分类器
        self.scene_classifier = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 8)  # 8个场景类型的概率
        ).to(self.device)
        
        # 创建动作分类器
        self.motion_classifier = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)  # 7种动作类型的概率
        ).to(self.device)
        
        print("GPU场景检测器初始化完成")
    
    @torch.no_grad()
    def process_batch(self, frames):
        """批量处理帧"""
        try:
            # 将帧转换为tensor并移到GPU
            frame_tensors = []
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frame_tensors.append(frame)
            
            batch = torch.stack(frame_tensors).to(self.device)
            batch_size = batch.size(0)
            
            # 使用自动混合精度
            with torch.cuda.amp.autocast():
                # 提取特征
                features = self.feature_extractor(batch)
                
                # 场景分类
                scene_logits = self.scene_classifier(features)
                scene_probs = torch.sigmoid(scene_logits)
                
                # 动作分类
                motion_logits = self.motion_classifier(features)
                motion_probs = torch.sigmoid(motion_logits)
                
                # 计算光流
                flows = self._compute_batch_flow(batch)
            
            # 处理结果
            results = []
            for i in range(batch_size):
                scene_info = {
                    'is_menu': scene_probs[i, 0].item() > 0.5,
                    'is_map': scene_probs[i, 1].item() > 0.5,
                    'is_dialog': scene_probs[i, 2].item() > 0.5,
                    'is_loading': scene_probs[i, 3].item() > 0.5,
                    'is_combat': scene_probs[i, 4].item() > 0.5,
                    'is_swimming': scene_probs[i, 5].item() > 0.5,
                    'is_flying': scene_probs[i, 6].item() > 0.5,
                    'is_climbing': scene_probs[i, 7].item() > 0.5,
                    
                    'is_walking': motion_probs[i, 0].item() > 0.5,
                    'is_running': motion_probs[i, 1].item() > 0.5,
                    'is_jumping': motion_probs[i, 2].item() > 0.5,
                    'is_fighting': motion_probs[i, 3].item() > 0.5,
                    'is_swimming_motion': motion_probs[i, 4].item() > 0.5,
                    'is_flying_motion': motion_probs[i, 5].item() > 0.5,
                    'is_climbing_motion': motion_probs[i, 6].item() > 0.5,
                }
                
                # 添加光流信息
                if flows is not None and i < len(flows):
                    flow = flows[i]
                    scene_info.update({
                        'has_motion': flow.abs().mean().item() > 0.1,
                        'movement_direction': self._get_direction_from_flow(flow),
                        'vertical_speed': flow[..., 1].mean().item(),
                        'horizontal_speed': flow[..., 0].mean().item()
                    })
                
                results.append(scene_info)
            
            # 更新last_frames
            self.last_frames = batch.clone()
            
            return results
            
        except Exception as e:
            print(f"批处理错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return [{}] * len(frames)
    
    def _compute_batch_flow(self, batch):
        """计算批量光流"""
        try:
            if self.last_frames is None:
                self.last_frames = batch
                return None
            
            flows = []
            for i in range(len(batch)):
                flow = self._compute_flow(
                    self.last_frames[i],
                    batch[i]
                )
                flows.append(flow)
            
            return flows
            
        except Exception as e:
            print(f"批量光流计算错误: {str(e)}")
            return None
    
    def _compute_flow(self, frame1, frame2):
        """计算单对帧之间的光流"""
        try:
            # 转换为灰度图
            gray1 = frame1.mean(dim=0).cpu().numpy()
            gray2 = frame2.mean(dim=0).cpu().numpy()
            
            # 确保数据类型正确
            gray1 = (gray1 * 255).astype(np.uint8)
            gray2 = (gray2 * 255).astype(np.uint8)
            
            # 计算光流 (CPU光流)
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                None,
                pyr_scale=0.5,
                levels=5,
                winsize=21,
                iterations=3,
                poly_n=7,
                poly_sigma=1.5,
                flags=0
            )
            
            return torch.from_numpy(flow).to(self.device)
            
        except Exception as e:
            print(f"光流计算错误: {str(e)}")
            return torch.zeros((frame1.shape[1], frame1.shape[2], 2)).to(self.device)
    
    def _get_direction_from_flow(self, flow):
        """从光流计算运动方向"""
        try:
            mean_x = flow[..., 0].mean().item()
            mean_y = flow[..., 1].mean().item()
            
            if abs(mean_x) < 0.05 and abs(mean_y) < 0.05:
                return "静止"
            
            angle = np.arctan2(mean_y, mean_x)
            directions = ["右", "右下", "下", "左下", "左", "左上", "上", "右上"]
            index = int((angle + np.pi) / (2 * np.pi / 8)) % 8
            
            return directions[index]
            
        except Exception as e:
            print(f"方向计算错误: {str(e)}")
            return "未知"


class ParallelVideoLoader:
    """并行视频加载器"""
    def __init__(self, num_workers=4, batch_size=32):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.frame_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        self.scene_detector = GPUSceneDetector(batch_size=batch_size)
        self.workers = []
        self.is_running = False
        print(f"并行加载器初始化完成 (workers: {num_workers}, batch_size: {batch_size})")
    
    def start(self):
        """启动并行处理"""
        self.is_running = True
        
        # 启动GPU处理线程
        self.gpu_thread = threading.Thread(
            target=self._gpu_process_loop,
            name="GPU-Processor",
            daemon=True
        )
        self.gpu_thread.start()
        
        # 启动工作线程
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"启动了 {self.num_workers} 个工作线程和 1 个GPU处理线程")
    
    def stop(self):
        """停止所有处理"""
        print("\n正在停止并行处理...")
        self.is_running = False
        
        # 等待所有线程完成
        self.gpu_thread.join()
        for worker in self.workers:
            worker.join()
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        print("并行处理已停止")
    
    def add_video(self, video_path):
        """添加视频到处理队列"""
        if not os.path.exists(video_path):
            print(f"错误: 视频文件不存在 - {video_path}")
            return
            
        self.frame_queue.put(video_path)
        print(f"已添加视频到队列: {os.path.basename(video_path)}")
    
    def _worker_loop(self):
        """工作线程循环"""
        while self.is_running:
            try:
                video_path = self.frame_queue.get_nowait()
                self._process_video(video_path)
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"工作线程错误: {str(e)}")
    
    def _process_video(self, video_path):
        """处理单个视频文件"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频: {video_path}")
                return
                
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 调整帧大小并转换颜色空间
                frame = cv2.resize(frame, CONFIG['FRAME_SIZE'])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
                if len(frames) >= self.batch_size:
                    self.result_queue.put(frames)
                    frames = []
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"已处理 {frame_count} 帧 - {os.path.basename(video_path)}")
            
            # 处理剩余的帧
            if frames:
                self.result_queue.put(frames)
                
        except Exception as e:
            print(f"视频处理错误: {str(e)}")
        finally:
            cap.release()
    
    def _gpu_process_loop(self):
        """GPU处理循环"""
        batch_count = 0
        while self.is_running:
            try:
                frames = self.result_queue.get_nowait()
                self.scene_detector.process_batch(frames)
                batch_count += 1
                
                if batch_count % 10 == 0:
                    print(f"已处理 {batch_count} 批次")
                    
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"GPU处理错误: {str(e)}")

class EnhancedDataset(Dataset):
    """使用临时文件缓存的数据集实现"""
    def __init__(self, video_folder, sequence_length=32, frame_size=(256, 256)):
        print("\n初始化数据集...")
        self.video_paths = self._scan_video_folder(video_folder)
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.cache_dir = "./tmp/sequences"
        self.sequence_paths = []
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"创建缓存目录: {self.cache_dir}")
        
        # 加载或处理数据
        self._prepare_data()
        print(f"数据集初始化完成，包含 {len(self.sequence_paths)} 个序列")
    
    def _scan_video_folder(self, folder):
        """扫描视频文件夹"""
        video_files = []
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        print(f"扫描文件夹: {folder}")
        for ext in video_extensions:
            found_files = glob(os.path.join(folder, f'*{ext}'))
            video_files.extend(found_files)
            if found_files:
                print(f"发现 {len(found_files)} 个{ext}文件")
        
        return video_files
    
    def _prepare_data(self):
        """准备数据"""
        print("\n开始处理视频数据...")
        
        for video_idx, video_path in enumerate(self.video_paths):
            print(f"\n处理视频 [{video_idx + 1}/{len(self.video_paths)}]: {os.path.basename(video_path)}")
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"无法打开视频")
                    continue
                
                # 获取视频信息
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                print(f"视频信息: {total_frames} 帧, {fps} FPS")
                
                frames = []
                frame_count = 0
                sequence_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 调整帧大小并转换颜色空间
                    frame = cv2.resize(frame, self.frame_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    frame_count += 1
                    
                    # 当收集足够的帧时，保存序列
                    if len(frames) >= self.sequence_length:
                        sequence = np.array(frames)
                        sequence_path = self._save_sequence(sequence, video_idx, sequence_count)
                        self.sequence_paths.append(sequence_path)
                        # 滑动窗口，这里可以改成不同策略
                        frames = frames[self.sequence_length//2:]  
                        sequence_count += 1
                        
                        # 清理内存
                        if sequence_count % 100 == 0:
                            frames = frames[-self.sequence_length:]
                            gc.collect()
                    
                    # 显示进度
                    if frame_count % 100 == 0:
                        print(f"\r已处理: {frame_count}/{total_frames} 帧, "
                              f"创建了 {sequence_count} 个序列", end="")
                
                cap.release()
                print(f"\n视频处理完成: 创建了 {sequence_count} 个序列")
                
            except Exception as e:
                print(f"处理视频时出错: {str(e)}")
                continue
    
    def _save_sequence(self, sequence, video_idx, sequence_idx):
        """保存序列到临时文件"""
        filename = f"sequence_{video_idx}_{sequence_idx}.npy"
        filepath = os.path.join(self.cache_dir, filename)
        np.save(filepath, sequence)
        return filepath
    
    def _load_sequence(self, filepath):
        """从临时文件加载序列"""
        try:
            sequence = np.load(filepath)
            return sequence / 255.0  # 归一化到0-1
        except Exception as e:
            print(f"加载序列错误: {str(e)}")
            return np.zeros((self.sequence_length, *self.frame_size, 3))
    
    def __len__(self):
        """返回数据集长度"""
        return len(self.sequence_paths)
    
    def __getitem__(self, idx):
        """获取数据项"""
        try:
            # 加载序列
            sequence = self._load_sequence(self.sequence_paths[idx])
            
            # 数据增强(可以自行扩充)
            if random.random() < 0.5:
                sequence = self._augment_sequence(sequence)
            
            # 转换为tensor, (N,H,W,C) -> (N,C,H,W)
            sequence = torch.from_numpy(sequence).float().permute(0, 3, 1, 2)
            
            # 假设我们要预测序列最后一帧
            # frames: (seq_len-1, 3, H, W), target: (1, 3, H, W)
            frames = sequence[:-1]
            target = sequence[-1:].clone()  # 最后一帧作为预测目标
            
            return frames, target
            
        except Exception as e:
            print(f"获取数据项错误: {str(e)}")
            # 返回空序列
            empty_sequence = torch.zeros(
                (self.sequence_length-1, 3, *self.frame_size),
                dtype=torch.float32
            )
            empty_target = torch.zeros(
                (1, 3, *self.frame_size),
                dtype=torch.float32
            )
            return empty_sequence, empty_target
    
    def _augment_sequence(self, sequence):
        """数据增强简单示例"""
        try:
            augmented = sequence.copy()
            
            # 随机亮度调整
            if random.random() < 0.5:
                factor = random.uniform(0.8, 1.2)
                augmented = augmented * factor
            
            # 随机对比度调整
            if random.random() < 0.5:
                factor = random.uniform(0.8, 1.2)
                mean = augmented.mean()
                augmented = np.clip((augmented - mean) * factor + mean, 0, 1)
            
            return augmented
            
        except Exception as e:
            print(f"数据增强错误: {str(e)}")
            return sequence
    
    def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            print(f"清理缓存目录: {self.cache_dir}")
        except Exception as e:
            print(f"清理缓存错误: {str(e)}")

def show_sequence(sequence, title="Sequence visualization"):
    """可视化序列"""
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.permute(0, 2, 3, 1).cpu().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle(title)
    
    for i, ax in enumerate(axes.flat):
        if i < len(sequence):
            ax.imshow(sequence[i])
            ax.axis('off')
            ax.set_title(f'Frame {i+1}')
    
    plt.tight_layout()
    plt.show()

def adjust_brightness(frame, factor):
    """调整亮度"""
    return np.clip(frame * factor, 0, 255).astype(np.uint8)

def adjust_contrast(frame, factor):
    """调整对比度"""
    mean = np.mean(frame, axis=(0, 1), keepdims=True)
    return np.clip((frame - mean) * factor + mean, 0, 255).astype(np.uint8)

def adjust_hue(frame, factor):
    """调整色相"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + factor * 180) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

class DataPrefetcher:
    """数据预取器，可选加速"""
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_data = None
        self._preload()
    
    def _preload(self):
        """预加载下一批数据"""
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        
        with torch.cuda.stream(self.stream):
            if isinstance(self.next_data, dict):
                for k, v in self.next_data.items():
                    if isinstance(v, torch.Tensor):
                        self.next_data[k] = v.to(self.device, non_blocking=True)
            else:
                self.next_data = [
                    t.to(self.device, non_blocking=True) 
                    for t in self.next_data 
                    if isinstance(t, torch.Tensor)
                ]
    
    def next(self):
        """获取下一批数据"""
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self._preload()
        return data
    
    def __iter__(self):
        return self
    
    def __next__(self):
        data = self.next()
        if data is None:
            raise StopIteration
        return data

class ResBlock(nn.Module):
    """简单残差块示例"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果通道或步幅不一致，需要下采样
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DeconvBlock(nn.Module):
    """反卷积块，用于上采样"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2,
            padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))

class SelfSupervisedPathfinder(nn.Module):
    """自监督寻路模型"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        print("\n初始化神经网络模型...")
        
        # 视觉特征提取器（ResNet风格），将尺寸从 (3,256,256) 缩减
        self.encoder = nn.Sequential(
            ResBlock(3, 32, stride=2),    # -> (32, 128, 128)
            ResBlock(32, 64, stride=2),   # -> (64, 64, 64)
            ResBlock(64, 128, stride=2),  # -> (128, 32, 32)
            ResBlock(128, 256, stride=1), # -> (256, 32, 32)
            nn.AdaptiveAvgPool2d((8, 8))  # -> (256, 8, 8)
        )
        print("特征提取器已创建")
        
        # 运动编码器（双向LSTM）
        self.motion_encoder = nn.LSTM(
            input_size=256 * 8 * 8,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        print("运动编码器已创建")
        
        # 动作预测器 (示例: 最终输出8维，假设做一些动作分类或数值预测)
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 8)  
        )
        print("动作预测器已创建")
        
        # 帧预测器（反卷积网络），输入是最终隐藏向量 -> 还原到图像
        self.frame_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256 * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)),
            DeconvBlock(256, 128),  # -> (128,16,16)
            DeconvBlock(128, 64),   # -> (64,32,32)
            DeconvBlock(64, 32),    # -> (32,64,64)
            DeconvBlock(32, 16),    # -> (16,128,128)
            # 这里把 stride=4 改为 stride=2, 并加 padding=1 => 输出 (3,256,256)
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        print("帧预测器已创建")
        
        # 初始化权重
        self.apply(self._init_weights)
        print("权重初始化完成")
        
        # 打印模型结构信息（可选）
        self._print_model_info()
    
    def _init_weights(self, m):
        """初始化模型权重"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def _print_model_info(self):
        """打印模型信息，用于调试"""
        print("\n模型结构信息 (仅测试打印，不影响训练):")
        
        x = torch.randn(2, 31, 3, 256, 256)  # (batch, seq_len, channels, H, W)
        print("1. 输入尺寸:", x.shape)
        
        batch_size, seq_len = x.size(0), x.size(1)
        x_reshaped = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        features = self.encoder(x_reshaped)
        print("2. 编码器输出:", features.shape)
        
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.motion_encoder(features)
        print("3. LSTM输出:", lstm_out.shape)
        
        final_hidden = lstm_out[:, -1, :]
        actions = self.action_predictor(final_hidden)
        print("4. 动作预测输出:", actions.shape)
        
        pred_frame = self.frame_predictor(final_hidden)
        print("5. 帧预测输出:", pred_frame.shape, "(期望: [B, 3, 256, 256])")
    
    def forward(self, x):
        """
        x: [B, seq_len, 3, 256, 256]
        return: actions, pred_frame
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 1. 特征提取
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))  
        features = self.encoder(x)  # -> [B*seq_len, 256, 8, 8]
        features = features.view(batch_size, seq_len, -1)  # -> [B, seq_len, 256*8*8]
        
        # 2. LSTM 编码
        lstm_out, _ = self.motion_encoder(features)  # -> [B, seq_len, hidden_dim*2]
        final_hidden = lstm_out[:, -1, :]            # 取最后时刻隐藏向量
        
        # 3. 动作预测
        actions = self.action_predictor(final_hidden)
        
        # 4. 帧预测 (还原一帧)
        pred_frame = self.frame_predictor(final_hidden)  # -> [B, 3, 256, 256]
        
        return actions, pred_frame


def create_model(pretrained_path=None):
    """创建模型实例"""
    try:
        print("\n创建模型...")
        model = SelfSupervisedPathfinder()
        
        # 如果有预训练权重，可在此加载
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"加载预训练权重: {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path))
        
        # 移动到GPU
        model = model.to(CONFIG['DEVICE'])
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型信息:")
        print(f"- 总参数量: {total_params:,}")
        print(f"- 可训练参数量: {trainable_params:,}")
        print(f"- 设备: {next(model.parameters()).device}")
        
        return model
        
    except Exception as e:
        print(f"创建模型错误: {str(e)}")
        raise e

def save_model(model, path, optimizer=None, epoch=None, loss=None):
    """保存模型"""
    try:
        save_dict = {
            'model_state_dict': model.state_dict(),
        }
        
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            save_dict['epoch'] = epoch
        if loss is not None:
            save_dict['loss'] = loss
        
        torch.save(save_dict, path)
        print(f"模型已保存: {path}")
        
    except Exception as e:
        print(f"保存模型错误: {str(e)}")

def load_model(path, model=None):
    """加载模型"""
    try:
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return None, None
            
        checkpoint = torch.load(path, map_location=CONFIG['DEVICE'])
        
        if model is None:
            model = SelfSupervisedPathfinder()
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(CONFIG['DEVICE'])
        
        return model, checkpoint
        
    except Exception as e:
        print(f"加载模型错误: {str(e)}")
        return None, None

class Trainer:
    """训练器"""
    def __init__(self, model, train_loader, val_loader=None, 
                 learning_rate=1e-3, weight_decay=1e-4):
        self.device = CONFIG['DEVICE']
        print(f"\n使用设备: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器 (可选)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, 
            patience=5, verbose=True
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        # 你可以根据需求调整各项损失的权重
        self.loss_weights = {
            'frame': 1.0,
            'action': 0.1,
            'temporal': 0.05
        }
        
        print(f"\n训练器初始化完成:")
        print(f"- 学习率: {learning_rate}")
        print(f"- 权重衰减: {weight_decay}")
        print(f"- 损失权重: {self.loss_weights}")
        if torch.cuda.is_available():
            print(f"- GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"- GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (frames, target) in enumerate(progress):
            try:
                frames = frames.to(self.device)
                target = target.to(self.device)
                
                loss = self._train_step(frames, target)
                total_loss += loss
                
                # 更新进度条信息
                progress.set_postfix({
                    'loss': f"{loss:.4f}",
                    'avg_loss': f"{total_loss/(batch_idx+1):.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
            except Exception as e:
                print(f"\n批次 {batch_idx} 训练错误: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return total_loss / num_batches
    
    def _train_step(self, frames, target):
        """训练一个批次"""
        # 清零梯度
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # 前向传播
            actions, pred_frame = self.model(frames)
            
            # 计算损失
            frame_loss = F.mse_loss(pred_frame, target.squeeze(1))  # target: [B,1,3,256,256]-> squeeze(1)-> [B,3,256,256]
            action_smooth_loss = self._compute_action_smoothness(actions)
            temporal_loss = self._compute_temporal_consistency(frames, pred_frame)
            
            loss = (self.loss_weights['frame'] * frame_loss 
                    + self.loss_weights['action'] * action_smooth_loss
                    + self.loss_weights['temporal'] * temporal_loss)
        
        # 反向传播
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def _compute_action_smoothness(self, actions):
        """动作平滑度损失: 简单示例"""
        # actions shape: [B, 8]，这里仅做示例
        # 若想要时序层面的动作平滑，需要对LSTM输出中间序列
        # 目前只是一个demo
        return torch.mean(torch.abs(actions))
    
    def _compute_temporal_consistency(self, frames, pred_frame):
        """时序一致性损失：对比最后一帧与预测帧"""
        # frames[:, -1] 是倒数第一帧  shape: [B, 3, 256,256]
        # pred_frame shape: [B, 3, 256,256]
        # 如果 frames 的时序维度是 axis=1，就要 frames[:, -1, ...]
        # 但是 frames 在外部维度 (batch_size, seq_len, 3, 256,256)
        # 这里 frames 传进来后 shape是 [B, 31,3,256,256]
        # frames[:, -1] => [B, 3,256,256]
        last_real_frame = frames[:, -1]  
        return F.mse_loss(pred_frame, last_real_frame)
    
    def validate(self):
        """验证模型(可选)"""
        if self.val_loader is None:
            return None
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for frames, target in tqdm(self.val_loader, desc="验证"):
                try:
                    frames = frames.to(self.device)
                    target = target.to(self.device)
                    
                    actions, pred_frame = self.model(frames)
                    
                    loss = F.mse_loss(pred_frame, target.squeeze(1))
                    total_loss += loss.item()
                    
                except Exception as e:
                    print(f"验证错误: {str(e)}")
                    continue
        
        return total_loss / len(self.val_loader)

def train_model(model, train_dataset, val_dataset=None, 
                num_epochs=5, batch_size=32, save_dir="./models"):
    """训练主函数"""
    print("\n开始训练...")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader)
    
    best_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        for epoch in range(num_epochs):
            # 训练
            train_loss = trainer.train_epoch(epoch)
            
            # 验证
            val_loss = trainer.validate() if val_loader else None
            
            # 打印信息
            print(f"\nEpoch {epoch+1}/{num_epochs} - 训练loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"              验证loss: {val_loss:.4f}")
            
            # 调整学习率
            if val_loss is not None:
                trainer.scheduler.step(val_loss)
            
            # 保存最优模型
            if val_loss is not None and val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print("保存最佳模型")
            
            # 定期保存检查点
            if (epoch + 1) % 2 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': train_loss,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                print(f"保存检查点 checkpoint_epoch_{epoch+1}.pth")
            
    except KeyboardInterrupt:
        print("\n训练被中断 (Ctrl+C)")
    except Exception as e:
        print(f"\n训练错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 保存最终模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': train_loss,
        }, os.path.join(save_dir, 'final_model.pth'))
        print("保存最终模型\n训练结束")

# 这里的推理示例中使用了 Inferencer，但代码里并未定义该类，如需推理可自行实现：
def start_inference(model, video_source=None):
    """仅做演示，推理部分需要你自己实现 Inferencer"""
    print("\n开始推理... (请注意：Inferencer 未定义，仅示例)")
    # 伪代码
    # inferencer = Inferencer(model)
    # ...
    pass

def choose_mode():
    """选择运行模式"""
    print("\n选择运行模式:")
    print("1. 视频文件训练")
    print("2. 实时截屏训练 (示例，未完整实现)")
    print("3. 实时推理测试 (Inferencer未实现)")
    print("4. 视频文件推理测试 (Inferencer未实现)")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            print("无效选择，请重试")
        except Exception as e:
            print(f"输入错误: {str(e)}")
            print("请重新选择")

def setup_video_training():
    """设置视频训练"""
    print("\n设置视频训练...")
    
    # 检查视频文件夹
    video_dir = "./videos"
    if not os.path.exists(video_dir):
        print(f"创建视频文件夹: {video_dir}")
        os.makedirs(video_dir)
    
    video_files = glob(os.path.join(video_dir, "*.*"))
    if not video_files:
        print(f"\n错误: 未找到视频文件")
        print(f"请将训练视频放入 {video_dir} 文件夹")
        return None
    
    print(f"\n找到 {len(video_files)} 个视频文件:")
    for file in video_files:
        print(f"- {os.path.basename(file)}")
    
    # 创建数据集
    try:
        dataset = EnhancedDataset(
            video_folder=video_dir,
            sequence_length=CONFIG['SEQUENCE_LENGTH'],
            frame_size=CONFIG['FRAME_SIZE']
        )
        return dataset
    except Exception as e:
        print(f"\n创建数据集错误: {str(e)}")
        return None

def setup_realtime_training():
    """设置实时训练(仅示例，未实现)"""
    print("\n设置实时训练 (示例) ...")
    print("此功能尚未完整实现，仅做占位。")
    return None

def main():
    """主函数"""
    try:
        # 设置环境
        setup_environment()
        
        # 选择模式
        mode = choose_mode()
        
        # 创建必要的目录
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./visualizations", exist_ok=True)
        
        # 创建模型
        model = create_model()
        
        if mode in [1, 2]:  # 训练模式
            if mode == 1:
                # 视频训练
                dataset = setup_video_training()
            else:
                # 实时训练 (示例)
                dataset = setup_realtime_training()
            
            if dataset is None:
                print("\n错误: 无法创建数据集")
                return
            
            # 这里仅演示训练，不拆分验证集
            train_model(
                model=model,
                train_dataset=dataset,
                val_dataset=None,
                num_epochs=5,  # 迭代次数可自行调整
                batch_size=CONFIG['BATCH_SIZE'],
                save_dir="./models"
            )
            
        else:  # 推理模式
            # 加载最新模型
            model_path = "./models/final_model.pth"
            if not os.path.exists(model_path):
                print(f"\n错误: 未找到模型文件 {model_path}")
                return
            
            model, _ = load_model(model_path, model)
            if model is None:
                print("\n错误: 无法加载模型")
                return
            
            # 开始推理 (Inferencer 未实现)
            if mode == 3:
                # 实时推理
                start_inference(model)
            else:
                # 视频推理
                video_files = glob("./videos/*.*")
                if not video_files:
                    print("\n错误: 未找到视频文件")
                    return
                
                print("\n选择视频文件:")
                for i, file in enumerate(video_files):
                    print(f"{i+1}. {os.path.basename(file)}")
                
                while True:
                    try:
                        choice = int(input(f"\n请选择视频文件 (1-{len(video_files)}): "))
                        if 1 <= choice <= len(video_files):
                            video_path = video_files[choice-1]
                            break
                        print("无效选择，请重试")
                    except ValueError:
                        print("请输入数字")
                
                start_inference(model, video_path)
    
    except KeyboardInterrupt:
        print("\n程序被中断 (Ctrl+C)")
    except Exception as e:
        print(f"\n程序错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        cleanup()
        print("\n程序结束")

if __name__ == "__main__":
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    main()
