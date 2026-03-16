"""
功能实现模块
包含所有UI功能的真实实现逻辑
"""

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageOps, ImageDraw
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import io
from skimage import io as skio
import ttkbootstrap as ttk
from tkinter import messagebox, filedialog, simpledialog
import pandas as pd
import glob
import re
import time
import webbrowser
from ultralytics import YOLO
from cellpose import models, io
from sam2.build_sam import build_sam2_video_predictor
import sys
import importlib
from pathlib import Path
import imagej
import csv
import math
import vtk
# 新增:直接分析cellpose掩膜,避免ImageJ的Find Edges失真
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellpose_direct_analysis import analyze_cellpose_mask
import tkinter as tk
from shapely.geometry import Polygon, Point
from scipy.stats import norm
from matplotlib.projections.polar import PolarAxes
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import math
import csv
import re
Image.MAX_IMAGE_PIXELS = None
# VTK错误抑制
try:
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
except Exception:
    pass
try:
    vtk.vtkObject.GlobalWarningDisplayOff()
except Exception:
    pass
try:
    sink = vtk.vtkFileOutputWindow()
    sink.SetFileName("NUL")  # 若想写文件改成 "vtk_errors.log"
    vtk.vtkOutputWindow.SetInstance(sink)
except Exception:
    pass

class ThreeDViewer_VTK:
    def __init__(self, parent, width=400, height=320):
        self.parent = parent
        self.width = width
        self.height = height
        self.ac_dir = None

        # Tkinter Frame
        self.frame = tk.Frame(parent, width=width, height=height, bg="white")
        self.frame.propagate(False)
        self.frame.update()  # 确保 winfo_id 可用

        # 渲染器
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1, 1, 1)

        # 渲染窗口
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(width, height)
        self.render_window.SetWindowInfo(str(int(self.frame.winfo_id())))

        # 交互器
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.Initialize()

        # 绑定 Tk 鼠标事件到 VTK
        self._bind_events()

    def place(self, **kwargs):
        self.frame.place(**kwargs)

    def _tk_to_vtk_coords(self, event):
        """Tk 坐标系 → VTK 坐标系"""
        return event.x, self.frame.winfo_height() - event.y

    def _bind_events(self):
        f = self.frame
        f.bind("<ButtonPress-1>", self._on_button_press)
        f.bind("<ButtonPress-2>", self._on_button_press)
        f.bind("<ButtonPress-3>", self._on_button_press)
        f.bind("<ButtonRelease-1>", self._on_button_release)
        f.bind("<ButtonRelease-2>", self._on_button_release)
        f.bind("<ButtonRelease-3>", self._on_button_release)
        f.bind("<B1-Motion>", self._on_mouse_move)
        f.bind("<B2-Motion>", self._on_mouse_move)
        f.bind("<B3-Motion>", self._on_mouse_move)
        f.bind("<Motion>", self._on_mouse_move)
        f.bind("<MouseWheel>", self._on_mouse_wheel)

    def _set_event_info(self, event):
        x, y = self._tk_to_vtk_coords(event)
        ctrl = bool(event.state & 0x4)
        shift = bool(event.state & 0x1)
        # key="", repeat=False, keysym=None
        self.interactor.SetEventInformation(x, y, ctrl, shift, "", False, None)

    def _on_button_press(self, event):
        self._set_event_info(event)
        if event.num == 1:
            self.interactor.InvokeEvent("LeftButtonPressEvent")
        elif event.num == 2:
            self.interactor.InvokeEvent("MiddleButtonPressEvent")
        elif event.num == 3:
            self.interactor.InvokeEvent("RightButtonPressEvent")

    def _on_button_release(self, event):
        self._set_event_info(event)
        if event.num == 1:
            self.interactor.InvokeEvent("LeftButtonReleaseEvent")
        elif event.num == 2:
            self.interactor.InvokeEvent("MiddleButtonReleaseEvent")
        elif event.num == 3:
            self.interactor.InvokeEvent("RightButtonReleaseEvent")

    def _on_mouse_move(self, event):
        self._set_event_info(event)
        self.interactor.InvokeEvent("MouseMoveEvent")

    def _on_mouse_wheel(self, event):
        self._set_event_info(event)
        if event.delta > 0:
            self.interactor.InvokeEvent("MouseWheelForwardEvent")
        else:
            self.interactor.InvokeEvent("MouseWheelBackwardEvent")

    def resample_edge_points(self, points, n_samples=200):
        """按弧长重采样轮廓点，保证点数一致"""
        if len(points) == 0:
            return np.zeros((n_samples, 2), dtype=np.float32)
        if len(points) == 1:
            return np.repeat(points, n_samples, axis=0)

        diffs = np.diff(points, axis=0)
        dist = np.hypot(diffs[:, 0], diffs[:, 1])
        cumdist = np.hstack([[0], np.cumsum(dist)])
        total_length = cumdist[-1]
        target_dist = np.linspace(0, total_length, n_samples)

        new_points = np.zeros((n_samples, 2), dtype=np.float32)
        for i, td in enumerate(target_dist):
            idx = np.searchsorted(cumdist, td)
            if idx == 0:
                new_points[i] = points[0]
            elif idx >= len(points):
                new_points[i] = points[-1]
            else:
                ratio = (td - cumdist[idx-1]) / (cumdist[idx] - cumdist[idx-1])
                new_points[i] = points[idx-1] + ratio * (points[idx] - points[idx-1])
        return new_points

    def preprocess_mask(self, img_path):
        """转8bit灰度→threshold=248→闭运算→开运算→最大连通域→上边界点集（重采样）"""
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 248, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        tissue_mask = np.uint8(labels == largest_label) * 255
        contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        top_edge = {}
        for p in cnt:
            x, y = p[0]
            if x not in top_edge or y < top_edge[x]:
                top_edge[x] = y
        edge_points = np.array([[x, top_edge[x]] for x in sorted(top_edge.keys())], dtype=np.float32)

        # 新增：重采样
        edge_points = self.resample_edge_points(edge_points, n_samples=200)

        return tissue_mask, edge_points

    def rigid_transform(self, A, B):
        """Kabsch 刚性配准"""
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_B - R @ centroid_A
        return R, t

    def load_3d_data(
        self,
        n_colors=7,
        z_step=70,
        r_min=10, r_max=40,
        sphere_res=16
    ):
        pattern = os.path.join(self.ac_dir, "cellinfo__*.csv")
        files = glob.glob(pattern)

        def _extract_idx(path):
            m = re.search(r'cellinfo__([0-9]+)', os.path.basename(path))
            return int(m.group(1)) if m else -1

        files = sorted(files, key=_extract_idx)
        if not files:
            print("No matching CSV files were found")
            return

        xs, ys, zs, areas = [], [], [], []

        # 参考切片
        ref_idx = _extract_idx(files[0])
        ref_mask_path = os.path.join(self.ac_dir, f"mask_{ref_idx:05d}.jpg")
        _, ref_edge = self.preprocess_mask(ref_mask_path)

        for i, f in enumerate(files):
            idx = _extract_idx(f)
            mask_path = os.path.join(self.ac_dir, f"mask_{idx:05d}.jpg")

            try:
                df = pd.read_csv(f, header=0)
                if df.shape[0] < 2:
                    continue
                x = df.iloc[1:, 7].astype(float).values
                y = df.iloc[1:, 8].astype(float).values
                a = df.iloc[1:, 2].astype(float).values
            except Exception as e:
                print(f"Failed to read file: {f}, error: {e}")
                continue

            _, edge_points = self.preprocess_mask(mask_path)
            R, t = self.rigid_transform(edge_points, ref_edge)

            # 点云配准
            coords = np.vstack([x, y]).T.astype(np.float32)
            coords_transformed = (R @ coords.T).T + t

            z = np.full_like(x, fill_value=i * z_step, dtype=float)
            xs.append(coords_transformed[:, 0])
            ys.append(coords_transformed[:, 1])
            zs.append(z)
            areas.append(a)

        if not xs:
            print("CSV has no valid data")
            return

        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        z_all = np.concatenate(zs)
        area_all = np.concatenate(areas)

        a_min = float(np.min(area_all))
        a_max = float(np.max(area_all))
        if np.isclose(a_min, a_max):
            n_colors = 1
            bounds = np.array([a_min, a_max + 1.0])
        else:
            bounds = np.linspace(a_min, a_max, n_colors + 1)

        cmap = cm.get_cmap('jet', n_colors)
        norm = BoundaryNorm(boundaries=bounds, ncolors=n_colors, clip=True)
        point_colors = (np.array(cmap(norm(area_all)))[:, :3] * 255).astype(np.uint8)

        if np.isclose(a_min, a_max):
            radii = np.full_like(area_all, (r_min + r_max) / 2.0, dtype=float)
        else:
            radii = r_min + (area_all - a_min) / (a_max - a_min) * (r_max - r_min)

        # 清空渲染器
        self.renderer.RemoveAllViewProps()

        # 添加球体
        for x0, y0, z0, r, c in zip(x_all, y_all, z_all, radii, point_colors):
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(r)
            sphere.SetThetaResolution(sphere_res)
            sphere.SetPhiResolution(sphere_res)
            sphere.SetCenter(x0, y0, z0)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(c[0]/255, c[1]/255, c[2]/255)

            self.renderer.AddActor(actor)

        self.renderer.ResetCamera()
        self.render_window.Render()

    def create_renderer(self, frame, xs, ys, zs, areas, colors, sphere_res=12, group_name=""):
        frame.update()
        if len(xs) == 0 or len(areas) == 0:
            print("No valid data, skipping render")
            return None, None

        a_min = float(np.min(areas)) if len(areas) else 0.0
        a_max = float(np.max(areas)) if len(areas) else 1.0
        denom = (a_max - a_min) if (a_max - a_min) > 0 else 1e-6
        if group_name:
            label = tk.Label(frame, text=group_name, 
                            font=("Arial", 12, "bold"),
                            bg="#ffffff", fg="#000000",
                            relief=tk.RAISED, bd=1)
            label.pack(side=tk.TOP, pady=(5,0))
        ren = vtk.vtkRenderer()
        ren.SetBackground(1.0, 1.0, 1.0)

        rw = vtk.vtkRenderWindow()
        rw.SwapBuffersOn()
        try:
            rw.SetDoubleBuffer(True)
        except Exception:
            pass
        try:
            rw.SetMultiSamples(0)  # 关闭多重采样可减少闪烁
        except Exception:
            pass
        rw.AddRenderer(ren)
        rw.SetSize(max(frame.winfo_width(), 1), max(frame.winfo_height(), 1))

        # 嵌入到 Tk frame（Windows）
        try:
            hwnd = frame.winfo_id()
            try:
                rw.SetWindowInfo(str(hwnd))
            except Exception:
                rw.SetParentId(int(hwnd))
        except Exception as e:
            print(f"Failed to embed window: {e}")
            rw.SetOffScreenRendering(False)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(rw)
        iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        iren.Initialize()

        for x0, y0, z0, a, c in zip(xs, ys, zs, areas, colors):
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(10.0 + (float(a) - a_min) / denom * 30.0)
            sphere.SetThetaResolution(sphere_res)
            sphere.SetPhiResolution(sphere_res)
            sphere.SetCenter(float(x0), float(y0), float(z0))

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
            ren.AddActor(actor)

        ren.ResetCamera()
        rw.Render()
        def _initial_render():
            try:
                ren.ResetCameraClippingRange()
                rw.Render()
            except:
                pass
        frame.after(0, _initial_render)
        # 状态与引用
        frame._vtk_rw = rw
        frame._vtk_iren = iren
        frame._vtk_ren = ren
        frame._vtk_running = True
        frame._vtk_after_id = None
        frame._vtk_ignore_until = 0
        frame._vtk_last_size = (frame.winfo_width(), frame.winfo_height())

        def _on_enter(_e=None):
            # 进入后短时间忽略 Configure 触发
            frame._vtk_ignore_until = time.time() + 0.15  # 150ms

        def _on_leave(_e=None):
            # 离开后短时间忽略 Configure 触发
            frame._vtk_ignore_until = time.time() + 0.15  # 150ms

        frame._vtk_events = [
            "<ButtonPress-1>", "<ButtonPress-2>", "<ButtonPress-3>",
            "<ButtonRelease-1>", "<ButtonRelease-2>", "<ButtonRelease-3>",
            "<B1-Motion>", "<B2-Motion>", "<B3-Motion>",
            "<MouseWheel>", "<Configure>", "<Destroy>"
        ]

        # 事件辅助
        def _tk_to_vtk_coords(event):
            return event.x, frame.winfo_height() - event.y

        def _set_event_info(event):
            if not getattr(frame, "_vtk_running", False) or not frame.winfo_exists():
                return False
            x, y = _tk_to_vtk_coords(event)
            ctrl = bool(event.state & 0x4)
            shift = bool(event.state & 0x1)
            iren.SetEventInformation(x, y, ctrl, shift, "", False, None)
            return True

        def _on_button_press(event):
            if not _set_event_info(event):
                return
            if event.num == 1:
                iren.InvokeEvent("LeftButtonPressEvent")
            elif event.num == 2:
                iren.InvokeEvent("MiddleButtonPressEvent")
            elif event.num == 3:
                iren.InvokeEvent("RightButtonPressEvent")

        def _on_button_release(event):
            if not _set_event_info(event):
                return
            if event.num == 1:
                iren.InvokeEvent("LeftButtonReleaseEvent")
            elif event.num == 2:
                iren.InvokeEvent("MiddleButtonReleaseEvent")
            elif event.num == 3:
                iren.InvokeEvent("RightButtonReleaseEvent")

        def _on_mouse_move(event):
            if not (event.state & (0x100 | 0x200 | 0x400)):
                return
            if not _set_event_info(event):
                return
            iren.InvokeEvent("MouseMoveEvent")

        def _on_mouse_wheel(event):
            if not _set_event_info(event):
                return
            if getattr(event, "delta", 0) > 0:
                iren.InvokeEvent("MouseWheelForwardEvent")
            else:
                iren.InvokeEvent("MouseWheelBackwardEvent")
        _resize_after = {"id": None}
        SIZE_DELTA_THRESHOLD = 2  # 小于等于 2px 的尺寸变化忽略

        def _on_resize(event):
            try:
                if not (getattr(frame, "_vtk_running", False) and frame.winfo_exists()):
                    return

                # 进入/离开短时间内忽略（避免闪）
                if time.time() < getattr(frame, "_vtk_ignore_until", 0):
                    return

                new_size = (max(event.width, 1), max(event.height, 1))
                last_size = getattr(frame, "_vtk_last_size", None)

                # 尺寸未变化或变化很小 → 忽略
                if last_size is not None:
                    if abs(new_size[0] - last_size[0]) <= SIZE_DELTA_THRESHOLD and \
                    abs(new_size[1] - last_size[1]) <= SIZE_DELTA_THRESHOLD:
                        return

                # 防抖：合并快速连续的 resize
                if _resize_after["id"] is not None:
                    frame.after_cancel(_resize_after["id"])

                def _do():
                    try:
                        rw.SetSize(*new_size)
                        frame._vtk_last_size = new_size
                        ren.ResetCameraClippingRange()
                        rw.Render()
                    except Exception as e:
                        print(f"Resize error: {e}")

                _resize_after["id"] = frame.after(40, _do)  # 40ms 略微更稳
            except Exception as e:
                print(f"Resize error: {e}")

        # 绑定
        frame.bind("<ButtonPress-1>", _on_button_press)
        frame.bind("<ButtonPress-2>", _on_button_press)
        frame.bind("<ButtonPress-3>", _on_button_press)
        frame.bind("<ButtonRelease-1>", _on_button_release)
        frame.bind("<ButtonRelease-2>", _on_button_release)
        frame.bind("<ButtonRelease-3>", _on_button_release)
        frame.bind("<B1-Motion>", _on_mouse_move)
        frame.bind("<B2-Motion>", _on_mouse_move)
        frame.bind("<B3-Motion>", _on_mouse_move)
        frame.bind("<MouseWheel>", _on_mouse_wheel)
        frame.bind("<Configure>", _on_resize)
        frame.bind("<Enter>", _on_enter)
        frame.bind("<Leave>", _on_leave)

        # frame 销毁兜底（先置 running=False，避免 Destroy 过程里的 Configure 触发 Render）
        def _on_destroy(_event=None):
            frame._vtk_running = False
            try:
                self._cleanup_vtk_frame(frame)
            except Exception as e:
                print(f"Destroy cleanup error: {e}")

        frame.bind("<Destroy>", _on_destroy)

        # 自动渲染
        def _auto_render():
            if not iren.GetEnabled(): 
                return
            try:
                if frame.winfo_exists() and getattr(frame, "_vtk_running", False):
                    rw.Render()
                    frame._vtk_after_id = frame.after(30, _auto_render)
            except Exception as e:
                print(f"Auto render error: {e}")

        frame._vtk_after_id = frame.after(30, _auto_render)

        return rw, iren
    def clear_3d_data(self):
        """清除所有三维数据"""
        self.renderer.RemoveAllViewProps()
        self.render_window.Render()
    def cleanup(self):
        """彻底清理VTK资源"""
        try:
            # 停止交互器
            if hasattr(self, 'interactor'):
                self.interactor.TerminateApp()
                self.interactor.SetRenderWindow(None)
            
            # 清理渲染器
            if hasattr(self, 'renderer'):
                self.renderer.RemoveAllViewProps()
            
            # 清理渲染窗口
            if hasattr(self, 'render_window'):
                self.render_window.Finalize()
                
            # 解绑事件
            if hasattr(self, 'frame'):
                for event in ["<ButtonPress-1>", "<ButtonPress-2>", "<ButtonPress-3>",
                            "<ButtonRelease-1>", "<ButtonRelease-2>", "<ButtonRelease-3>",
                            "<B1-Motion>", "<B2-Motion>", "<B3-Motion>",
                            "<Motion>", "<MouseWheel>"]:
                    try:
                        self.frame.unbind(event)
                    except:
                        pass
        except Exception as e:
            print(f"Error cleaning VTK resources: {e}")
    def _cleanup_vtk_frame(self, frame):
        # 1) 停止 after
        try:
            if getattr(frame, "_vtk_after_id", None):
                try:
                    frame.after_cancel(frame._vtk_after_id)
                except Exception:
                    pass
                frame._vtk_after_id = None
        except Exception:
            pass
        frame._vtk_running = False

        # 2) 解绑事件
        try:
            if getattr(frame, "_vtk_events", None):
                for ev in frame._vtk_events:
                    try:
                        frame.unbind(ev)
                    except Exception:
                        pass
                frame._vtk_events = []
        except Exception:
            pass

        rw = getattr(frame, "_vtk_rw", None)
        iren = getattr(frame, "_vtk_iren", None)
        ren = getattr(frame, "_vtk_ren", None)

        # 3) 先关 interactor（断开与 RenderWindow 的关联）
        try:
            if iren is not None:
                try:
                    iren.Disable()
                except Exception:
                    pass
                try:
                    iren.TerminateApp()
                except Exception:
                    pass
                try:
                    iren.RemoveAllObservers()
                except Exception:
                    pass
                try:
                    iren.SetRenderWindow(None)
                except Exception:
                    pass
        except Exception:
            pass

        # 4) 切到离屏并渲染一次（此时 HWND 仍有效），然后彻底脱钩 HWND
        try:
            if rw is not None:
                try:
                    rw.SwapBuffersOff()  # 若版本无此方法会抛异常，忽略即可
                except Exception:
                    pass
                try:
                    rw.SetOffScreenRendering(True)
                except Exception:
                    pass
                try:
                    # 这次 Render 目的是让 VTK 切换到离屏上下文，后续析构不再依赖 HWND
                    rw.Render()
                except Exception:
                    pass

                # 移除 renderer 与一切观察者
                try:
                    if ren is not None:
                        try:
                            ren.RemoveAllProps()
                        except Exception:
                            pass
                        rw.RemoveRenderer(ren)
                except Exception:
                    pass
                try:
                    rw.RemoveAllObservers()
                except Exception:
                    pass

                # 与宿主 HWND 解绑
                try:
                    rw.SetWindowInfo("0")
                except Exception:
                    pass
                try:
                    rw.SetParentId(0)
                except Exception:
                    pass
                # 切记：不调用 MakeCurrent / ReleaseGraphicsResources / Finalize
        except Exception:
            pass

        # 5) 断引用
        for attr in ("_vtk_rw", "_vtk_iren", "_vtk_ren"):
            if hasattr(frame, attr):
                setattr(frame, attr, None)
class FunctionalityManager:
    """功能管理器，包含所有真实功能的实现"""
    
    def __init__(self, ui_manager):
        self.ui_manager = ui_manager
        self.output_dir = None
        self.imageshow = None
        self.imageshow2 = None
        self.tk_image = None
        self.firstframe = None
        self.global_label = "HE"
        self.points = []
        self.labels = []
        self.mask_files = []
        self.current_mask_index = 0
        self.ac_dir = None
        self.gp_dir = None
        self.nucleus_not_found = False
        self.all_data = {}
        self.analysis_win = None
        self.current_utils = None
        self._flash_widget = None
        
        # 初始化其他属性
        self.ratio = 2/5  # 默认缩放比例
        self.roi_drawing = False
        self.roi_points = []
        self.roi_id = None
        self.temp_line_id = None
        self.roi_flashing = False
        self.background_adjusted = False
        self.background_blurred = False
        self.blurred_image = None
        self.segview = None
        self.count = 0
        self.image_history = []
        self.annotations = []
        self.latest_mask = None
        self.is_in_mask_editing = False
        # 初始化模型
        self._init_yolo_model()
        
        # 初始化ImageJ
        self._init_imagej()
        
    def _init_cellpose_models(self):
        """初始化必要的模型"""
        try:
            if self.ui_manager.selected_animal == "mouse" and self.global_label == "HE":
                self.model1 = models.CellposeModel(gpu=True, pretrained_model="cpsam_HE")
                self.model2 = models.CellposeModel(gpu=True, pretrained_model="cpsam_Mouse_HE_nucleus")
            elif self.ui_manager.selected_animal == "mouse" and self.global_label == "SafraninO":
                self.model1 = models.CellposeModel(gpu=True, pretrained_model="cpsam_Mouse_SO")
                self.model2 = models.CellposeModel(gpu=True, pretrained_model="cpsam_Mouse_SO_nucleus")
            elif self.ui_manager.selected_animal == "rat" or self.ui_manager.selected_animal == "hamster" and self.global_label == "SafraninO":
                self.model1 = models.CellposeModel(gpu=True, pretrained_model="cpsam_HE")
                self.model2 = models.CellposeModel(gpu=True, pretrained_model="Rat_HE_Nucleus")
            else:
                self.model1 = models.CellposeModel(gpu=True, pretrained_model="cpsam_HE")
                self.model2 = models.CellposeModel(gpu=True, pretrained_model="Rat_HE_Nucleus")
            print("Cellpose model initialized successfully")
        except Exception as e:
            print(f"Model initialization failed: {e}")
            messagebox.showerror("Error", f"Model initialization failed: {e}")
    def _init_yolo_model(self):
        """初始化必要的模型"""
        try:
            # 初始化YOLO模型
            self.yolo_model = YOLO(r'C:\Users\hantang\runs\detect\train4\weights\last.pt')
            print("YOLO model initialized successfully")
        except Exception as e:
            print(f"Model initialization failed: {e}")
            messagebox.showerror("Error", f"Model initialization failed: {e}")
    def _init_imagej(self):
        """初始化ImageJ"""
        try:
            # 优先使用带有 IJ1 Legacy 的完整 Fiji 发行版
            self.ij = imagej.init('C:/fiji/Fiji.app', mode='interactive')

        except Exception as e:
            print(f"ImageJ initialization failed: {e}")
            messagebox.showerror("Error", f"Failed to initialize ImageJ: {e}")

    def _ask_localization_mode(self):
        """弹出对话框让用户选择自动定位或手动定位，返回 'auto' / 'manual' / None"""
        result = [None]
        dialog = tk.Toplevel(self.ui_manager.root)
        dialog.title("Select Localization Mode")
        dialog.resizable(False, False)

        # 配色方案（基于 ttkbootstrap superhero）
        bg_dark = "#1c1e35"
        fg_light = "#f0f0f0"
        fg_muted = "#a0a0a0"
        color_auto = "#0088cc"    # 蓝色 (info)
        color_manual = "#ff9800"  # 橙色 (warning)

        dialog.configure(bg=bg_dark)

        # 窗口大小和位置
        dw, dh = 750, 450
        dialog.update_idletasks()
        sx = (dialog.winfo_screenwidth() - dw) // 2
        sy = (dialog.winfo_screenheight() - dh) // 2
        dialog.geometry(f"{dw}x{dh}+{sx}+{sy}")
        dialog.transient(self.ui_manager.root)
        dialog.grab_set()

        # ─── 标题区域 ───────────────────────────
        title_frame = tk.Frame(dialog, bg=bg_dark)
        title_frame.pack(fill='x', padx=0, pady=(50, 15))

        title = tk.Label(
            title_frame,
            text="Select Localization Mode",
            font=("Segoe UI", 34, "bold"),
            fg=fg_light,
            bg=bg_dark
        )
        title.pack()

        # 副标题
        subtitle = tk.Label(
            title_frame,
            text="Choose how to position tissue regions on each image",
            font=("Segoe UI", 12),
            fg=fg_muted,
            bg=bg_dark
        )
        subtitle.pack(pady=(8, 0))

        # ─── 按钮容器 ──────────────────────────
        btn_frame = tk.Frame(dialog, bg=bg_dark)
        btn_frame.pack(fill='both', expand=True, padx=50, pady=(40, 60))

        # 按钮状态管理
        button_state = {'hover_auto': False, 'hover_manual': False}

        # ─── 自动定位按钮 ──────────────────────
        auto_container = tk.Frame(btn_frame, bg=bg_dark)
        auto_container.pack(side='left', expand=True, fill='both', padx=25)

        auto_btn_canvas = tk.Canvas(
            auto_container,
            width=240,
            height=110,
            bg=bg_dark,
            highlightthickness=0
        )
        auto_btn_canvas.pack(pady=(0, 16))
        auto_btn_id = auto_btn_canvas.create_rectangle(
            6, 6, 234, 104,
            fill=color_auto, outline=color_auto, width=0
        )
        auto_text_id = auto_btn_canvas.create_text(
            120, 55,
            text="Automatic (YOLO)",
            font=("Segoe UI", 17, "bold"),
            fill="white"
        )

        def on_auto_enter(e):
            button_state['hover_auto'] = True
            auto_btn_canvas.itemconfig(auto_btn_id, fill="#0077bb", outline="#0077bb")

        def on_auto_leave(e):
            button_state['hover_auto'] = False
            auto_btn_canvas.itemconfig(auto_btn_id, fill=color_auto, outline=color_auto)

        def on_auto_click(e):
            result[0] = "auto"
            dialog.destroy()

        auto_btn_canvas.bind("<Enter>", on_auto_enter)
        auto_btn_canvas.bind("<Leave>", on_auto_leave)
        auto_btn_canvas.bind("<Button-1>", on_auto_click)
        auto_btn_canvas.config(cursor="hand2")

        # 自动定位描述
        auto_desc = tk.Label(
            auto_container,
            text="AI detects tissue regions,\nyou review & adjust",
            font=("Segoe UI", 11),
            fg=fg_muted,
            bg=bg_dark,
            wraplength=220,
            justify='center'
        )
        auto_desc.pack()

        # ─── 手动定位按钮 ──────────────────────
        manual_container = tk.Frame(btn_frame, bg=bg_dark)
        manual_container.pack(side='left', expand=True, fill='both', padx=25)

        manual_btn_canvas = tk.Canvas(
            manual_container,
            width=240,
            height=110,
            bg=bg_dark,
            highlightthickness=0
        )
        manual_btn_canvas.pack(pady=(0, 16))
        manual_btn_id = manual_btn_canvas.create_rectangle(
            6, 6, 234, 104,
            fill=color_manual, outline=color_manual, width=0
        )
        manual_text_id = manual_btn_canvas.create_text(
            120, 55,
            text="Manual",
            font=("Segoe UI", 17, "bold"),
            fill="white"
        )

        def on_manual_enter(e):
            button_state['hover_manual'] = True
            manual_btn_canvas.itemconfig(manual_btn_id, fill="#ff7700", outline="#ff7700")

        def on_manual_leave(e):
            button_state['hover_manual'] = False
            manual_btn_canvas.itemconfig(manual_btn_id, fill=color_manual, outline=color_manual)

        def on_manual_click(e):
            result[0] = "manual"
            dialog.destroy()

        manual_btn_canvas.bind("<Enter>", on_manual_enter)
        manual_btn_canvas.bind("<Leave>", on_manual_leave)
        manual_btn_canvas.bind("<Button-1>", on_manual_click)
        manual_btn_canvas.config(cursor="hand2")

        # 手动定位描述
        manual_desc = tk.Label(
            manual_container,
            text="You drag the selection box\non each image",
            font=("Segoe UI", 11),
            fg=fg_muted,
            bg=bg_dark,
            wraplength=220,
            justify='center'
        )
        manual_desc.pack()

        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.wait_window()
        return result[0]

    # ------------------------------------------------------------------
    # Crop preview window (shared by auto & manual localization)
    # ------------------------------------------------------------------
    def _open_crop_preview(self, img, img_name, crop_w, crop_h,
                           initial_left_x, initial_top_y,
                           current_idx, total_count):
        """全屏预览窗口：在原图上拖动固定尺寸方框选取ROI。

        Args:
            img          : 原始图像 (numpy BGR)
            img_name     : 文件名
            crop_w/crop_h: 裁剪尺寸（原始像素）
            initial_left_x/initial_top_y: 初始裁剪位置（原始像素）
            current_idx  : 当前图像序号 (0-based)
            total_count  : 图像总数
        Returns:
            (left_x, top_y) 或 None（跳过）
        """
        result = [None]
        h_img, w_img = img.shape[:2]

        # --- window ---
        preview = tk.Toplevel(self.ui_manager.root)
        preview.title(f"Localization Preview  —  Image {current_idx + 1} / {total_count}:  {img_name}")
        preview.state('zoomed')
        preview.configure(bg='#1a1a2e')
        preview.transient(self.ui_manager.root)
        preview.grab_set()
        preview.update_idletasks()

        screen_w = preview.winfo_screenwidth()
        screen_h = preview.winfo_screenheight()

        # 预留控制栏高度
        ctrl_h = 100
        canvas_max_w = screen_w - 40
        canvas_max_h = screen_h - ctrl_h - 60
        scale = min(canvas_max_w / w_img, canvas_max_h / h_img)
        disp_w = int(w_img * scale)
        disp_h = int(h_img * scale)

        # --- 缩放显示图像 ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((disp_w, disp_h), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)

        canvas = tk.Canvas(preview, width=disp_w, height=disp_h,
                           bg='#1a1a2e', highlightthickness=0)
        canvas.pack(pady=(10, 0))
        canvas.create_image(0, 0, anchor='nw', image=img_tk, tags="bg")

        # --- 裁剪框（显示坐标） ---
        rect_dw = int(crop_w * scale)
        rect_dh = int(crop_h * scale)
        rx = max(0, min(int(initial_left_x * scale), disp_w - rect_dw))
        ry = max(0, min(int(initial_top_y * scale), disp_h - rect_dh))

        state = {'dragging': False, 'ox': 0, 'oy': 0, 'rx': rx, 'ry': ry}

        # 半透明遮罩（stipple 实现）
        dim_kw = dict(fill='black', stipple='gray50', outline='')
        dim_top = canvas.create_rectangle(0, 0, disp_w, ry, **dim_kw)
        dim_bot = canvas.create_rectangle(0, ry + rect_dh, disp_w, disp_h, **dim_kw)
        dim_lft = canvas.create_rectangle(0, ry, rx, ry + rect_dh, **dim_kw)
        dim_rgt = canvas.create_rectangle(rx + rect_dw, ry, disp_w, ry + rect_dh, **dim_kw)

        rect_id = canvas.create_rectangle(rx, ry, rx + rect_dw, ry + rect_dh,
                                          outline='#00ffff', width=3, tags='sel')

        def _update_overlay():
            _rx, _ry = state['rx'], state['ry']
            canvas.coords(dim_top, 0, 0, disp_w, _ry)
            canvas.coords(dim_bot, 0, _ry + rect_dh, disp_w, disp_h)
            canvas.coords(dim_lft, 0, _ry, _rx, _ry + rect_dh)
            canvas.coords(dim_rgt, _rx + rect_dw, _ry, disp_w, _ry + rect_dh)
            canvas.coords(rect_id, _rx, _ry, _rx + rect_dw, _ry + rect_dh)

        # --- 拖动逻辑 ---
        def on_press(e):
            _rx, _ry = state['rx'], state['ry']
            if _rx <= e.x <= _rx + rect_dw and _ry <= e.y <= _ry + rect_dh:
                state['dragging'] = True
                state['ox'] = e.x
                state['oy'] = e.y

        def on_drag(e):
            if not state['dragging']:
                return
            dx = e.x - state['ox']
            dy = e.y - state['oy']
            nx = max(0, min(state['rx'] + dx, disp_w - rect_dw))
            ny = max(0, min(state['ry'] + dy, disp_h - rect_dh))
            state['rx'], state['ry'] = nx, ny
            state['ox'], state['oy'] = e.x, e.y
            _update_overlay()

        def on_release(e):
            state['dragging'] = False

        canvas.bind("<Button-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)

        # --- 控制栏 ---
        ctrl = ttk.Frame(preview)
        ctrl.pack(pady=10)

        info_lbl = ttk.Label(ctrl,
            text=f"Drag the cyan rectangle to select the ROI.  "
                 f"({crop_w}×{crop_h} px)   Image {current_idx+1}/{total_count}: {img_name}",
            font=("Segoe UI", 12))
        info_lbl.pack(pady=(0, 8))

        btn_row = ttk.Frame(ctrl)
        btn_row.pack()

        def confirm():
            lx = int(state['rx'] / scale)
            ty = int(state['ry'] / scale)
            lx = max(0, min(lx, w_img - crop_w))
            ty = max(0, min(ty, h_img - crop_h))
            result[0] = (lx, ty)
            preview.destroy()

        def skip():
            preview.destroy()

        ttk.Button(btn_row, text="  Confirm  ", command=confirm,
                   bootstyle="success", width=18).pack(side='left', padx=20)
        ttk.Button(btn_row, text="  Skip  ", command=skip,
                   bootstyle="danger-outline", width=18).pack(side='left', padx=20)

        # 防止引用被回收
        preview._img_ref = img_tk
        preview.protocol("WM_DELETE_WINDOW", skip)
        preview.wait_window()
        return result[0]

    # ------------------------------------------------------------------
    # Manual localization
    # ------------------------------------------------------------------
    def _process_manual_localization(self, file_path, CROP_W, CROP_H):
        """手动定位：逐张打开预览窗口，用户拖动方框选取ROI"""
        try:
            img_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
            img_files = sorted(
                [f for f in os.listdir(file_path)
                 if f.lower().endswith(img_exts) and not f.startswith('.')])

            if not img_files:
                messagebox.showwarning("Warning", "No image files found in the selected directory")
                return

            count = 0
            total = len(img_files)

            for idx, fname in enumerate(img_files):
                img_path = os.path.join(file_path, fname)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                h_img, w_img = img.shape[:2]
                if w_img < CROP_W or h_img < CROP_H:
                    messagebox.showwarning(
                        "Warning",
                        f"Image {fname} too small ({w_img}×{h_img} < {CROP_W}×{CROP_H}). Skipping.")
                    continue

                init_lx = max(0, (w_img - CROP_W) // 2)
                init_ty = max(0, (h_img - CROP_H) // 2)

                crop_result = self._open_crop_preview(
                    img, fname, CROP_W, CROP_H,
                    init_lx, init_ty, idx, total)

                if crop_result is not None:
                    lx, ty = crop_result
                    crop_img = img[ty:ty + CROP_H, lx:lx + CROP_W]
                    save_name = str(count).zfill(5) + ".jpg"
                    cv2.imwrite(os.path.join(self.output_dir, save_name),
                                crop_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    count += 1

            self._precompute_label(file_path)

            if count == 0:
                messagebox.showwarning("Warning", "No images were processed")
            else:
                messagebox.showinfo("Done", f"Successfully processed {count} images")

        except Exception as e:
            print(f"Manual localization failed: {e}")
            messagebox.showerror("Error", f"Manual localization error: {str(e)}")

    def open_image(self):
        """打开图像功能 - 修正版"""
        try:
            # 检查是否选择了动物类型
            if not hasattr(self.ui_manager, 'selected_animal') or not self.ui_manager.selected_animal:
                messagebox.showerror("Error", "Please select the animal type first")
                return
            # 设置裁剪尺寸
            if self.ui_manager.selected_animal == "rat":
                CROP_W, CROP_H = 5500, 2970
            elif self.ui_manager.selected_animal == "hamster":
                CROP_W, CROP_H = 4500, 2250
            else:
                CROP_W, CROP_H = 2500, 1250
            # 清空检查标记
            if hasattr(self.ui_manager, 'rectangle2'):
                self.ui_manager.rectangle2.delete("check")
            # 询问是否进行图像定位
            response = messagebox.askyesno("Localization", "Do you want to perform image localization?")
            loc_mode = None
            if response:
                loc_mode = self._ask_localization_mode()
                if loc_mode is None:
                    return  # 用户关闭了选择窗口
            # 选择目录
            file_path = filedialog.askdirectory(title="选择图像目录")
            if not file_path:
                return
            # 打开新数据前，重置与撤销/注释/历史相关的所有状态，避免跨目录污染
            try:
                self.points = []
                self.labels = []
                self.count = 0
                # 重置掩码序列状态
                self.mask_files = []
                self.current_mask_index = 0
                if hasattr(self, 'image_history'):
                    self.image_history.clear()
                if hasattr(self, 'annotations'):
                    self.annotations.clear()
                # 删除画布上旧的提示点
                if hasattr(self.ui_manager, 'point'):
                    self.ui_manager.point.delete("circles")
                # 清理临时显示图层
                if hasattr(self, 'segview') and self.segview:
                    try:
                        self.ui_manager.point.delete(self.segview)
                    except Exception:
                        pass
                # 退出掩码编辑模式并恢复界面（若卡在编辑模式）
                if hasattr(self, 'editing_mode') and self.editing_mode:
                    self._restore_original_interface()
            except Exception:
                pass
            # 创建输出目录
            self.output_dir = os.path.join(file_path, "out")
            os.makedirs(self.output_dir, exist_ok=True)
            if response and loc_mode:
                if loc_mode == "auto":
                    # 使用YOLO进行定位（含预览确认/调整）
                    self._process_with_yolo(file_path, CROP_W, CROP_H)
                else:
                    # 手动定位
                    self._process_manual_localization(file_path, CROP_W, CROP_H)
            # 不定位时，直接用out/00000.jpg，不强制要求tif存在
            # tif仅用于label预计算
            # 预计算label（单线程处理第一个tif文件）
            tif_files = [f for f in os.listdir(file_path) if f.lower().endswith('.tif')]
            if len(tif_files) > 0:
                first_file = tif_files[0]
                pil_image = Image.open(os.path.join(file_path, first_file))
                small_img = pil_image.resize((100, 100))
                average_color = np.mean(np.array(small_img), axis=(0, 1))
                self.global_label = "HE" if abs(average_color[0] - 240) < abs(average_color[0] - 170) else "SafraninO"
            else:
                self.global_label = "HE"  # 默认值
            # 直接显示out/00000.jpg
            jpg_path = os.path.join(self.output_dir, "00000.jpg")
            if not os.path.exists(jpg_path):
                messagebox.showerror("Error", f"File not found: {jpg_path}. Please run localization or ensure an image exists in the 'out' folder")
                return
            self.imageshow = cv2.imread(jpg_path)
            output_folder = os.path.join(self.output_dir, "1st")
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, "00000.jpg")
            cv2.imwrite(output_path, self.imageshow)
            imageshow = cv2.cvtColor(self.imageshow, cv2.COLOR_BGR2RGB)
            imageshow2 = imageshow.copy()
            if self.ui_manager.selected_animal == "rat":
                self.ratio = 2/11
            elif self.ui_manager.selected_animal == "hamster":
                self.ratio = 2/9
            else:
                self.ratio = 2/5
            imageshow2 = cv2.resize(imageshow2, dsize=None, fx=self.ratio, fy=self.ratio)
            self.imageshow2 = Image.fromarray(imageshow2)
            self.tk_image = ImageTk.PhotoImage(self.imageshow2)
            if hasattr(self.ui_manager, 'point'):
                self.ui_manager.point.delete("circles")
                self.firstframe = self.ui_manager.point.create_image(500, 400, anchor='center', image=self.tk_image)
                # 进入新图后，确保撤销历史以当前帧为起点
                self.image_history = []
                self.annotations = []
                self.count = 0
            self.ui_manager.viewer.ac_dir = os.path.join(self.output_dir, "AC")
            self.SAMbanned = False
            # 确保三维可视化窗口有效
            if hasattr(self.ui_manager, 'viewer') and self.ui_manager.viewer:
                # 如果失败，重新创建viewer
                if hasattr(self.ui_manager, 'viewer'):
                    try:
                        self.ui_manager.viewer.cleanup()
                    except:
                        pass
                self.ui_manager.viewer = ThreeDViewer_VTK(self.ui_manager.main_window, width=400, height=320)
                self.ui_manager.viewer.place(relx=0.872, rely=0.362, anchor="center")
                self.ui_manager.viewer.ac_dir = os.path.join(self.output_dir, "AC")
            # 检查当前文件夹下 cellpose 分割与 Post_analysis 状态
            self.cellpose_ready = False
            self.post_analysis_ready = False
            folder = self.output_dir if hasattr(self, 'output_dir') else None
            if folder and os.path.isdir(folder):
                # 递归检查 cellpose 分割结果（如 *_cellfill 文件）
                cellfill_files = glob.glob(os.path.join(folder, '**', '*cellfill*'), recursive=True)
                self.cellpose_ready = bool(cellfill_files)
                # 递归检查 Post_analysis 结果（如 analysis_*.csv 或 analysis_*.jpg）
                analysis_csv = glob.glob(os.path.join(folder, '**', 'analysis_*.csv'), recursive=True)
                analysis_img = glob.glob(os.path.join(folder, '**', 'analysis_*.jpg'), recursive=True)
                self.post_analysis_ready = bool(analysis_csv or analysis_img)
                self.ui_manager.viewer.load_3d_data()
            else:
                # 如果viewer不存在，重新创建它
                self.ui_manager.viewer = ThreeDViewer_VTK(self.ui_manager.main_window, width=400, height=320)
                self.ui_manager.viewer.place(relx=0.872, rely=0.362, anchor="center")
                self.ui_manager.viewer.ac_dir = os.path.join(self.output_dir, "AC")
                self.ui_manager.viewer.load_3d_data()
        except Exception as e:
            messagebox.showerror("Error", f"Error opening images: {str(e)}")
            print(f"Error opening images: {e}")
    
    def _process_with_yolo(self, file_path, CROP_W, CROP_H):
        """使用YOLO模型处理图像（含逐图预览确认/拖动调整）"""
        try:
            count = 0

            # ── YOLO 推理 ──────────────────────────────────────────
            results = self.yolo_model.predict(
                source=file_path,
                save_txt=True,
                imgsz=1216,
                conf=0.25,
                max_det=2
            )

            total_count = len(results)

            for r_idx, r in enumerate(results):
                img_path = r.path
                img_name = os.path.basename(img_path)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                h_img, w_img = img.shape[:2]
                boxes = r.boxes

                # ── 计算 YOLO 建议的裁剪位置 ──────────────────────
                proposed_lx = max(0, (w_img - CROP_W) // 2)   # 默认居中（无检测时）
                proposed_ty = max(0, (h_img - CROP_H) // 2)

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    xyxy = xyxy[confs >= 0.25]

                    if len(xyxy) == 2:
                        centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in xyxy]
                        mid_y = (centers[0][1] + centers[1][1]) / 2
                        widths = [x2 - x1 for x1, _, x2, _ in xyxy]
                        bi = 0 if widths[0] >= widths[1] else 1
                        x1b, _, x2b, _ = xyxy[bi]
                        proposed_lx = int(x1b) if centers[bi][0] < centers[1 - bi][0] else int(x2b) - CROP_W
                        proposed_ty = int(mid_y - CROP_H / 2)
                    elif len(xyxy) == 1:
                        x1, y1, x2, y2 = xyxy[0]
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        if cx < w_img / 2:
                            proposed_lx = int(x1)
                            proposed_ty = int(y2 - CROP_H / 2)
                        else:
                            proposed_lx = int(x2) - CROP_W
                            proposed_ty = int(cy - CROP_H / 2)

                # 边界限制
                proposed_lx = max(0, min(proposed_lx, w_img - CROP_W))
                proposed_ty = max(0, min(proposed_ty, h_img - CROP_H))

                # ── 打开预览窗口（自动定位结果可拖动微调）─────────
                crop_result = self._open_crop_preview(
                    img, img_name, CROP_W, CROP_H,
                    proposed_lx, proposed_ty, r_idx, total_count)

                if crop_result is not None:
                    lx, ty = crop_result
                    crop_img = img[ty:ty + CROP_H, lx:lx + CROP_W]
                    filename = str(count).zfill(5) + ".jpg"
                    cv2.imwrite(os.path.join(self.output_dir, filename),
                                crop_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    count += 1

            # 预计算标签
            self._precompute_label(file_path)

        except Exception as e:
            print(f"YOLO processing failed: {e}")
            # 回退到TIF居中裁剪
            self._process_tif_files(file_path, CROP_W, CROP_H)
    
    def _process_tif_files(self, file_path, CROP_W, CROP_H):
        """处理TIF文件"""
        try:
            # 获取TIF文件列表
            tif_files = [f for f in os.listdir(file_path) if f.endswith(".tif")]
            
            if len(tif_files) == 0:
                messagebox.showwarning("Warning", "No TIF files were found")
                return
            
            # 预计算标签
            self._precompute_label(file_path)
            
            # 处理文件
            count = 0
            for idx, tif_file in enumerate(tif_files):
                try:
                    # 读取图像
                    img_path = os.path.join(file_path, tif_file)
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    
                    if img is None:
                        continue
                    
                    h_img, w_img = img.shape[:2]
                    
                    # 简单的居中裁剪
                    left_x = max(0, (w_img - CROP_W) // 2)
                    top_y = max(0, (h_img - CROP_H) // 2)
                    
                    # 边界检查
                    left_x = min(left_x, w_img - CROP_W)
                    top_y = min(top_y, h_img - CROP_H)
                    
                    # 裁剪图像
                    crop_img = img[top_y:top_y+CROP_H, left_x:left_x+CROP_W]
                    
                    # 保存图像
                    filename = str(count).zfill(5) + ".jpg"
                    save_path = os.path.join(self.output_dir, filename)
                    cv2.imwrite(save_path, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    count += 1
                    
                except Exception as e:
                    print(f"Error processing file {tif_file}: {e}")
                    continue
            
            messagebox.showinfo("Done", f"Successfully processed {count} files")
            
        except Exception as e:
            print(f"TIF processing failed: {e}")
            messagebox.showerror("Error", f"Error while processing TIF files: {str(e)}")
    
    def _process_no_detection_files(self, no_detection_files, file_path, start_count):
        """处理未检测到的文件"""
        try:
            # 并行处理所有文件
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = []
                for idx, tif_file in enumerate(no_detection_files, start=start_count+1):
                    future = executor.submit(
                        self._process_single_file,
                        tif_file,
                        file_path,
                        self.output_dir,
                        idx
                    )
                    futures.append(future)
                
                # 处理结果
                success_count = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            success_count += 1
                    except Exception as e:
                        print(f"Task raised an exception: {str(e)}")
                
                print(f"Successfully processed {success_count}/{len(no_detection_files)} files")
                
        except Exception as e:
            print(f"Parallel processing failed: {e}")
    
    def _process_single_file(self, tif_file, file_path, output_dir, idx):
        """处理单个文件"""
        try:
            # 读取图像
            img_path = os.path.join(file_path, tif_file)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                return False
            
            h_img, w_img = img.shape[:2]
            
            # 居中裁剪
            left_x = max(0, (w_img - 2500) // 2)
            top_y = max(0, (h_img - 1250) // 2)
            
            # 边界检查
            left_x = min(left_x, w_img - 2500)
            top_y = min(top_y, h_img - 1250)
            
            # 裁剪图像
            crop_img = img[top_y:top_y+1250, left_x:left_x+2500]
            
            # 保存图像
            filename = str(idx).zfill(5) + ".jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            return True
            
        except Exception as e:
            print(f"Error processing file {tif_file}: {e}")
            return False
    
    def _precompute_label(self, file_path):
        """预计算标签"""
        try:
            tif_files = [f for f in os.listdir(file_path) if f.endswith(".tif")]
            if len(tif_files) > 0:
                first_file = tif_files[0]
                pil_image = Image.open(os.path.join(file_path, first_file))
                small_img = pil_image.resize((100, 100))
                average_color = np.mean(np.array(small_img), axis=(0, 1))
                self.global_label = "HE" if abs(average_color[0] - 250) < abs(average_color[0] - 170) else "SafraninO"
            else:
                self.global_label = "HE"
        except Exception as e:
            print(f"Precomputing label failed: {e}")
            self.global_label = "HE"
    
    def _display_first_image(self):
        """显示第一张图像"""
        try:
            if not self.output_dir or not os.path.exists(self.output_dir):
                return
            
            # 查找第一张图像
            first_image_path = os.path.join(self.output_dir, "00000.jpg")
            if not os.path.exists(first_image_path):
                # 查找其他图像文件
                image_files = [f for f in os.listdir(self.output_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not image_files:
                    return
                first_image_path = os.path.join(self.output_dir, image_files[0])
            
            # 读取并显示图像
            self.imageshow = cv2.imread(first_image_path)
            if self.imageshow is None:
                return
            
            # 创建1st目录并保存
            output_folder = os.path.join(self.output_dir, "1st")
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, "00000.jpg")
            cv2.imwrite(output_path, self.imageshow)
            
            # 转换颜色空间
            imageshow = cv2.cvtColor(self.imageshow, cv2.COLOR_BGR2RGB)
            imageshow2 = imageshow.copy()
            
            # 根据动物类型设置缩放比例
            if self.ui_manager.selected_animal == "rat":
                self.ratio = 2/11
            elif self.ui_manager.selected_animal == "hamster":
                self.ratio = 2/9
            else:
                self.ratio = 2/5
            
            # 缩放图像
            imageshow2 = cv2.resize(imageshow2, dsize=None, fx=self.ratio, fy=self.ratio)
            self.imageshow2 = Image.fromarray(imageshow2)
            
            # 转换为Tkinter格式
            self.tk_image = ImageTk.PhotoImage(self.imageshow2)
            
            # 在画布上显示图像
            if hasattr(self.ui_manager, 'point'):
                self.ui_manager.point.delete("circles")
                self.firstframe = self.ui_manager.point.create_image(
                    500, 400, anchor='center', image=self.tk_image
                )
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def start_sam(self):
        """启动SAM功能 - 完整实现"""
        try:
            if not self.output_dir or not os.path.exists(self.output_dir):
                messagebox.showerror("Error", "Please open an image first")
                return
            
            if not self.points or not self.labels:
                messagebox.showwarning("Warning", "Please add some points first")
                return
            
            # 初始化SAM模型
            device = torch.device("cuda")
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            
            # 启用tfloat32
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # 解析 SAM2 资源路径（兼容打包后运行）
            def _resolve_sam2_paths():
                # 首选硬编码路径（开发环境）
                p_ckpt = Path("C:/Users/hantang/anaconda3/envs/pytorch/lib/site-packages/sam2/samgp.pt")
                p_cfg = Path("C:/Users/hantang/anaconda3/envs/pytorch/lib/site-packages/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml")
                if p_ckpt.exists() and p_cfg.exists():
                    return str(p_ckpt), str(p_cfg)

                # 备选1：PyInstaller 临时目录
                meipass = getattr(sys, "_MEIPASS", None)
                if meipass:
                    base = Path(meipass) / "sam2"
                    c1 = base / "samgp.pt"
                    f1 = base / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml"
                    if c1.exists() and f1.exists():
                        return str(c1), str(f1)

                # 备选2：通过已安装包定位
                try:
                    sam2_mod = importlib.import_module("sam2")
                    base2 = Path(sam2_mod.__file__).resolve().parent
                    c2 = base2 / "samgp.pt"
                    f2 = base2 / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml"
                    if c2.exists() and f2.exists():
                        return str(c2), str(f2)
                except Exception:
                    pass

                # 最后返回硬编码（可能不存在，便于错误提示）
                return str(p_ckpt), str(p_cfg)

            sam2_checkpoint, model_cfg = _resolve_sam2_paths()
            predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
            
            # 初始化推理状态
            inference_state = predictor.init_state(video_path=self.output_dir)
            predictor.reset_state(inference_state)
            self.SAMbanned = True
            # 准备点数据
            pointslist = np.array(self.points, np.float32)
            labelslist = np.array(self.labels, np.int32)
            
            # 添加新的点或框
            ann_frame_idx = 0
            ann_obj_id = 1
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=pointslist,
                labels=labelslist,
            )
            
            # 创建matplotlib图像显示结果
            frame_names = [
                p for p in os.listdir(self.output_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg"] 
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            
            # 显示第一帧结果
            fig = plt.figure(figsize=(3, 3))
            plt.imshow(Image.open(os.path.join(self.output_dir, frame_names[ann_frame_idx])))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            self._show_points(pointslist, labelslist, plt.gca())
            self._show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
            plt.axis('off')
            
            # 传播到视频中的所有帧
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            # 保存所有掩码
            color_num = 0
            for out_frame_idx in range(0, len(frame_names)):
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    now_img = self._add_mask2(frame_names, out_mask, color_num)
                color_num += 1
                
                # 根据模式创建目录
                if self.ui_manager.selected_mode == "Growth Plate":
                    self.gp_dir = os.path.join(self.output_dir, "GP")
                    os.makedirs(self.gp_dir, exist_ok=True)
                    mask_out_name = os.path.join(self.output_dir, "GP", f"mask_{str(out_frame_idx).zfill(5)}.jpg")
                elif self.ui_manager.selected_mode == "Articular Cartilage":
                    self.ac_dir = os.path.join(self.output_dir, "AC")
                    os.makedirs(self.ac_dir, exist_ok=True)
                    mask_out_name = os.path.join(self.output_dir, "AC", f"mask_{str(out_frame_idx).zfill(5)}.jpg")
                
                # 保存掩码图像
                now_img = cv2.cvtColor(now_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(mask_out_name, now_img)
            
            # 更新掩码文件列表
            if self.ac_dir:
                self.mask_files = [
                    os.path.join(self.ac_dir, f) for f in os.listdir(self.ac_dir)
                    if os.path.splitext(f)[-1].lower() in [".jpg", ".jpeg", ".png"]
                ]
                self.mask_files.sort()
                self.current_mask_index = 0
                self._display_current_mask()
            
            # 保存第一张图像
            save_path = os.path.join(self.output_dir, "00000.jpg")
            save_path2 = os.path.join(self.output_dir, "1st", "00000.jpg")
            cv2.imwrite(save_path, self.imageshow)
            cv2.imwrite(save_path2, self.imageshow)
            
            # 清空点数据
            self.points.clear()
            self.labels.clear()
            
            messagebox.showinfo("Done", "SAM processing completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error when starting SAM: {str(e)}")
            print(f"Error starting SAM: {e}")
    
    def _show_points(self, points, labels, ax):
        """显示点"""
        try:
            for point, label in zip(points, labels):
                color = 'red' if label == 1 else 'blue'
                ax.plot(point[0], point[1], 'o', color=color, markersize=10)
        except Exception as e:
            print(f"Failed to display points: {e}")
    
    def _show_mask(self, mask, ax, obj_id):
        """显示掩码"""
        try:
            ax.imshow(mask, alpha=0.5, cmap='jet')
        except Exception as e:
            print(f"Failed to display mask: {e}")
    
    def _add_mask2(self, frame_names, mask, color_num):
        """添加掩码到图像"""
        try:
            # 1. 加载原始图像并确保RGB格式
            img_path = os.path.join(self.output_dir, frame_names[color_num])
            img = np.array(Image.open(img_path)).astype(np.uint8)
            
            if len(img.shape) == 2:  # 如果是灰度图
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # 如果有alpha通道
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # 2. 处理mask（确保与图像尺寸匹配）
            if len(mask.shape) > 2:
                mask = mask[0]
            mask = mask.astype(bool)  # 转换为布尔掩码
            
            # 3. 创建透明背景结果图像
            result = np.zeros_like(img)
            result[mask] = img[mask]  # 只保留mask区域的像素
            
            # 4. 白色背景替代
            result[~mask] = 255  # 非mask区域设为白色
            
            return result
            
        except Exception as e:
            print(f"Failed to add mask: {e}")
            return self.imageshow
    
    def _display_current_mask(self):
        """显示当前掩码"""
        try:
            if self.current_mask_index < len(self.mask_files):
                # 清除画布
                self.ui_manager.rectangle2.delete("check")
                
                # 加载掩码与对应原图
                img_path = self.mask_files[self.current_mask_index]
                mask_pil = Image.open(img_path).convert("RGB")

                # 映射回原图路径（与 show_on_point 一致的规则）
                dir_path, file_name = os.path.split(img_path)
                dir_parts = dir_path.split(os.sep)
                if len(dir_parts) > 1:
                    dir_parts.pop(-1)  # 去掉 AC
                original_name = file_name.replace("mask_", "")
                original_path = os.path.join(os.sep.join(dir_parts), original_name)
                if os.path.exists(original_path):
                    orig_pil = Image.open(original_path).convert("RGB")
                else:
                    # 回退：若找不到原图，就把掩码当作原图
                    orig_pil = mask_pil.copy()

                # 转为 Numpy
                mask_np = np.array(mask_pil)
                orig_np = np.array(orig_pil)

                # 生成布尔掩码（白色背景为 False）
                if mask_np.ndim == 3:
                    mask_bool = ~(
                        (mask_np[:, :, 0] > 250) &
                        (mask_np[:, :, 1] > 250) &
                        (mask_np[:, :, 2] > 250)
                    )
                else:
                    mask_bool = mask_np > 0

                # 调整原图与掩码尺寸一致
                if (orig_np.shape[1], orig_np.shape[0]) != (mask_np.shape[1], mask_np.shape[0]):
                    orig_np = np.array(Image.fromarray(orig_np).resize((mask_np.shape[1], mask_np.shape[0]), Image.Resampling.LANCZOS))

                # 合成（参考 update_segmentation_display）
                darkened = (orig_np * 0.6).astype(np.uint8)
                fg_enhanced = np.array(ImageEnhance.Brightness(Image.fromarray(orig_np)).enhance(1.5))
                combined = darkened.copy()
                combined[mask_bool] = fg_enhanced[mask_bool]

                # 边缘强化（蓝色）
                mask_uint8 = (mask_bool.astype(np.uint8) * 255)
                kernel3 = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(mask_uint8, kernel3, iterations=1)
                eroded = cv2.erode(mask_uint8, kernel3, iterations=1)
                edge = cv2.subtract(dilated, eroded)
                edge = (edge > 0).astype(np.uint8)
                edge_blue = np.zeros_like(combined)
                kernel5 = np.ones((5, 5), np.uint8)
                dilated_edge = cv2.dilate(edge.astype(np.uint8), kernel5, iterations=2)
                edge_blue[dilated_edge == 1] = [0, 120, 255]
                edge_img = Image.fromarray(edge_blue).filter(ImageFilter.GaussianBlur(radius=4))
                combined_pil = Image.fromarray(combined)
                combined_pil = Image.blend(combined_pil, edge_img, alpha=0.7)

                # 掩膜区域整体轻微加蓝
                blue_layer = np.zeros_like(combined, dtype=np.uint8)
                blue_layer[:, :] = [0, 0, 255]
                blue_layer = cv2.bitwise_and(blue_layer, blue_layer, mask=mask_uint8)
                combined_with_blue = cv2.addWeighted(np.array(combined_pil), 0.9, blue_layer, 0.1, 0)

                # 缩放至侧栏展示宽度 398，保持比例
                h0, w0 = combined_with_blue.shape[:2]
                new_w = 398
                new_h = max(1, int((new_w / w0) * h0))
                preview_pil = Image.fromarray(combined_with_blue).resize((new_w, new_h), Image.Resampling.LANCZOS)

                # 绘制到侧栏
                self.current_mask_img = ImageTk.PhotoImage(preview_pil)
                self.ui_manager.rectangle2.create_image(204, 164, image=self.current_mask_img, tags="check")

                # 文件名
                filename = os.path.basename(img_path)
                self.ui_manager.rectangle2.create_text(204, 15, text=filename, fill="red", tags="check")
                
        except Exception as e:
            print(f"Failed to display current mask: {e}")
    
    def startcellpose(self):
        """启动Cellpose功能 - 完整实现"""
        try:
            if not self.output_dir or not os.path.exists(self.output_dir):
                messagebox.showerror("Error", "Please open an image first")
                return
            self._init_cellpose_models()
            # 设置输入路径
            self.output_dir = self.output_dir.replace("\\", "/")
            if self.ui_manager.selected_mode == "Growth Plate":
                input_path = os.path.join(self.output_dir, "GP")
            elif self.ui_manager.selected_mode == "Articular Cartilage":
                input_path = os.path.join(self.output_dir, "AC")
            else:
                input_path = os.path.join(self.output_dir, "Meniscus")
            
            input_path = input_path.replace("\\", "/")
            
            # 获取掩码文件 - 只处理原始mask文件,排除cellfill/nucleusfill等已处理文件
            # 重要:按文件名排序确保从00000开始处理
            growthplate_files = sorted([file for file in os.listdir(input_path) 
                                if 'mask' in file 
                                and not file.startswith('cellfill')
                                and not file.startswith('nucleusfill')
                                and not '_cp_masks' in file
                                and not file.endswith('.csv')])
            
            print(f"Found {len(growthplate_files)} files to process")
            if len(growthplate_files) > 0:
                print(f"Processing order: {growthplate_files[0]} ... {growthplate_files[-1]}")
            
            # 创建输出目录
            nucleus_output = os.path.join(self.output_dir, "nucleus_output")
            nucleus_output = nucleus_output.replace("\\", "/")
            os.makedirs(nucleus_output, exist_ok=True)
            
            # 处理每个文件
            if self.ui_manager.selected_mode == "Growth Plate":
                for file in growthplate_files:
                    self._process_growthplate_file(file, input_path)
            elif self.ui_manager.selected_mode == "Articular Cartilage":
                for file in growthplate_files:
                    self._process_articular_cartilage_file(file, input_path, nucleus_output)
            else:
                # 其他模式按相同流程处理
                for file in growthplate_files:
                    self._process_articular_file(file, input_path)
            
            # AC-only：directest分析已在_run_ac_imagej_pipeline中逐个处理完成
            # 不需要再次批量处理 (会等待已删除的cellfill文件导致卡住)
            # if self.ui_manager.selected_mode == "Articular Cartilage":
            #     try:
            #         self.run_cellpose_post_analysis(target_dir=input_path, max_wait_per_file=180)
            #     except Exception as e:
            #         print(f"AC post-analysis skipped: {e}")

            print("\n" + "="*60)
            print("✅ Cellpose processing completed successfully!")
            print(f"   Processed {len(growthplate_files)} files")
            print("="*60 + "\n")
            
            messagebox.showinfo("Done", "Cellpose processing completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error when starting Cellpose: {str(e)}")
            print(f"Error starting Cellpose: {e}")
    
    def _process_growthplate_file(self, file, input_path):
        """处理生长板文件"""
        try:
            # 读取图像
            growthplate = io.imread(os.path.join(input_path, file))
            file_names = file.replace("mask", "cell")
            file_names = os.path.join(input_path, file_names)
            
            # 根据标签选择模型
            if self.global_label == "HE":
                masks, flows, styles = self.model2.eval(growthplate, flow_threshold=0.4, cellprob_threshold=-1)
            else:
                masks, flows, styles = self.model1.eval(growthplate, flow_threshold=0.4, cellprob_threshold=-1)
            
            # 保存结果
            io.masks_flows_to_seg(growthplate, masks, flows, file_names)
            io.save_masks(growthplate, masks, flows, file_names, png=True)
            
            # 运行与原始脚本一致的宏：对原始 mask 进行缩放与阈值、生成 gpfilled 与 Results CSV
            gp_path = f"{input_path}/{file}"
            gp_path2 = gp_path.replace("mask", "gpfilled")
            gp_path3 = file.replace("mask", "gpfilled")
            gp_path4 = gp_path.replace("mask", "Results").replace("tif", "csv")
            macro_code = f"""
                open("{gp_path}");
                selectImage("{file}");
                run("Scale...", "x=0.05 y=0.05 width=175 height=75 interpolation=Bilinear average create");
                run("8-bit");
                setAutoThreshold("Default no-reset");
                setThreshold(0, 242, "raw");
                run("Convert to Mask");
                saveAs("png", "{gp_path2}");
            """
            self.ij.py.run_macro(macro_code)

            params_centerline = {
                "method": "Skeletonize",
                "interpolation": 1,
                "min_radius": 1,
                "min_distance": 1,
                "samples": 200,
                "max_trials": 200
            }
            self.current_utils.run_ijm_macro("select centerline", params_centerline, image_title = gp_path3)
            params_width = {
                "sample": 5,
                "left_offset": 0,
                "right_offset": 0,
                "radius": 15,
                "line_color": "pink",
                "show": ""
            }
            self.current_utils.run_ijm_macro("width profile perpendicular to centerline", params_width, image_title = gp_path3)
            # 保存 Results 为 CSV
            macro_code2 = f"""
                selectWindow('Results');
                saveAs("Results", "{gp_path4}");
                run("Close All");
            """
            try:
                self.ij.py.run_macro(macro_code2)
            except Exception:
                pass
            try:
                if os.path.exists(gp_path2):
                    os.remove(gp_path2)
            except Exception:
                pass

            # 遍历生成的 _cp_masks.png，制备 cellfill、Analyze Particles 生成 cellinfo，并进一步绘制参数热图
            for image_file in os.listdir(input_path):
                if image_file.endswith(".png") and "_cp_masks" in image_file:
                    image_path = f"{input_path}/{image_file}"
                    image_file2 = image_file.replace("_cp_masks", "_cp_masks-1")
                    image_file3 = "Result of " + image_file2
                    mask_rename = image_file.replace("_cp_masks", "").replace("cell", "cellfill")
                    image_process_path = f"{input_path}/{mask_rename}"
                    cellinfo_path = f"{input_path}/cellinfo_{mask_rename.replace('cellfill', '').split('.')[0]}.csv"

                    macro_m = f"""
                        open("{image_path}");
                        selectImage("{image_file}");
                        run("Duplicate...", " ");
                        selectImage("{image_file}");
                        run("Find Edges");
                        run("8-bit");
                        setAutoThreshold("Default no-reset");
                        setThreshold(1, 255, "raw");
                        setOption("BlackBackground", true);
                        run("Convert to Mask");
                        selectImage("{image_file2}");
                        run("8-bit");
                        setAutoThreshold("Default no-reset");
                        setThreshold(1, 255, "raw");
                        setOption("BlackBackground", true);
                        run("Convert to Mask");
                        imageCalculator("Subtract create", "{image_file2}","{image_file}");
                        selectImage("{image_file3}");
                        saveAs("PNG", "{image_process_path}");
                        selectImage("{mask_rename}");
                        run("Analyze Particles...", "size=0-500 circularity=0.20-1.00 display exclude clear include composite");
                        selectWindow("Results");
                        saveAs("Results", "{cellinfo_path}");
                        run("Close All");
                    """
                    self.ij.py.run_macro(macro_m)

                    # 读取两个 CSV：cellinfo 与方向 width profile 结果（gp Results CSV）
                    if not (os.path.exists(cellinfo_path) and os.path.exists(gp_path4)):
                        continue
                    with open(cellinfo_path, 'r', newline='') as file1:
                        reader1 = csv.reader(file1)
                        try:
                            next(reader1)
                        except StopIteration:
                            continue
                        labels1, areas, x_values, y_values, circ, Ar, roundn = [], [], [], [], [], [], []
                        for row in reader1:
                            labels1.append(row[0])
                            areas.append(row[2])
                            x_values.append(float(row[3]))
                            y_values.append(float(row[4]))
                            circ.append(float(row[5]))
                            Ar.append(float(row[6]))
                            roundn.append(float(row[7]))

                    with open(gp_path4, 'r', newline='') as file2:
                        reader2 = csv.reader(file2)
                        try:
                            next(reader2)
                        except StopIteration:
                            continue
                        labels2, x_values2, y_values2, angles, lengths = [], [], [], [], []
                        for row in reader2:
                            labels2.append(row[0])
                            x_values2.append(float(row[3]))
                            y_values2.append(float(row[4]))
                            angles.append(float(row[5]))
                            # 注意：原文件第12列（索引12）
                            if len(row) > 12:
                                lengths.append(float(row[12]))
                            else:
                                lengths.append(0.0)

                    # 放缩
                    x_values2 = [x * 20 for x in x_values2]
                    y_values2 = [y * 20 for y in y_values2]
                    lengths = [l * 20 for l in lengths]

                    # 最近配对与区间计数
                    ratiolist, count_of_labels1 = {}, {}
                    for label1, x1, y1 in zip(labels1, x_values, y_values):
                        min_distance = float('inf')
                        nearest_label = None
                        for label2, x2, y2 in zip(labels2, x_values2, y_values2):
                            distance = math.hypot(x2 - x1, y2 - y1)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_label = label2
                        if nearest_label is None:
                            continue
                        idx = labels2.index(nearest_label)
                        dx = x_values2[idx] - x1
                        dy = y_values2[idx] - y1
                        distance2 = math.hypot(dx, dy)
                        angleb = 0 if x_values2[idx] == x1 else math.atan(dy / (x_values2[idx] - x1))
                        if x_values2[idx] > x1:
                            ratio = (lengths[idx]/2 - distance2 * math.sin(-math.pi / 2 + angleb + math.radians(angles[idx]))) / max(lengths[idx], 1e-6)
                        else:
                            ratio = (lengths[idx]/2 + distance2 * math.sin(-math.pi / 2 + angleb + math.radians(angles[idx]))) / max(lengths[idx], 1e-6)
                        ratiolist[label1] = ratio

                        count = 0
                        for label3, x3, y3 in zip(labels1, x_values, y_values):
                            if -50 <= x3-x1 <= 50 and -85 <= y3-y1 <= 85:
                                count += 1
                        count_of_labels1[label1] = count

                    # 分类
                    label_categories = []
                    sorted_areas = sorted(map(float, areas))
                    desired_area = sorted_areas[int(len(sorted_areas) * 0.7)] if sorted_areas else 0.0
                    for label in labels1:
                        ratio = ratiolist.get(label, 0.5)
                        count = count_of_labels1.get(label, 0)
                        area = areas[labels1.index(label)]
                        cir = circ[labels1.index(label)]
                        ar = Ar[labels1.index(label)]
                        if ratio <= 0.2:
                            label_categories.append('a')
                        elif ratio >= 0.8:
                            label_categories.append('c')
                        elif 0.35 <= ratio <= 0.65:
                            label_categories.append('b')
                        elif 0.2 < ratio < 0.35:
                            label_categories.append('a' if count < 4 else 'b')
                        elif 0.65 < ratio < 0.8:
                            if float(area) < desired_area and cir < 0.75:
                                label_categories.append('b')
                            else:
                                label_categories.append('c')

                    # 统计与热图绘制
                    import numpy as np
                    from matplotlib.colors import Normalize
                    import matplotlib.pyplot as plt
                    image = Image.open(image_process_path).convert('RGB')
                    image_np = np.array(image)
                    contours, _ = cv2.findContours(image_np[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    copied_np = image_np.copy(); copied_np2 = image_np.copy(); copied_np3 = image_np.copy(); copied_np4 = image_np.copy()
                    colors_map = {'c': (255, 0, 0), 'b': (0, 255, 0), 'a': (0, 0, 255)}

                    # 计算颜色映射
                    def get_percentile_limits(data, lower_percentile=5, upper_percentile=95):
                        lower_limit = np.percentile(data, lower_percentile) if data else 0
                        upper_limit = np.percentile(data, upper_percentile) if data else 1
                        return lower_limit, upper_limit
                    vmin1, vmax1 = get_percentile_limits(list(map(float, areas)))
                    vmin2, vmax2 = get_percentile_limits(list(map(float, circ)))
                    vmin3, vmax3 = get_percentile_limits(list(map(float, Ar)))
                    vmin4, vmax4 = get_percentile_limits(list(map(float, roundn)))
                    norm1 = Normalize(vmin=vmin1, vmax=vmax1, clip=True)
                    norm2 = Normalize(vmin=vmin2, vmax=vmax2, clip=True)
                    norm3 = Normalize(vmin=vmin3, vmax=vmax3, clip=True)
                    norm4 = Normalize(vmin=vmin4, vmax=vmax4, clip=True)
                    cmap = plt.cm.coolwarm

                    def rgba_to_bgr(color):
                        r, g, b, a = color
                        return (int(b * 255), int(g * 255), int(r * 255))
                    colors1 = list(map(rgba_to_bgr, [cmap(norm1(float(a))) for a in areas]))
                    colors2 = list(map(rgba_to_bgr, [cmap(norm2(float(c))) for c in circ]))
                    colors3 = list(map(rgba_to_bgr, [cmap(norm3(float(x))) for x in Ar]))
                    colors4 = list(map(rgba_to_bgr, [cmap(norm4(float(x))) for x in roundn]))

                    centers = list(zip(x_values, y_values))
                    def draw_contours_on_image(centers, contours, colors, target_image):
                        for center, color in zip(centers, colors):
                            for contour in contours:
                                if cv2.pointPolygonTest(contour, center, False) >= 0:
                                    cv2.drawContours(target_image, [contour], -1, color, thickness=cv2.FILLED)
                                    break

                    # 主图按分类上色
                    for center in centers:
                        for contour in contours:
                            if cv2.pointPolygonTest(contour, center, False) >= 0:
                                category = label_categories[centers.index(center)] if centers.index(center) < len(label_categories) else 'b'
                                color = colors_map.get(category, (0, 255, 0))
                                cv2.drawContours(image_np, [contour], -1, color, thickness=cv2.FILLED)
                                break
                    for colors_list, target in zip([colors1, colors2, colors3, colors4], [copied_np, copied_np2, copied_np3, copied_np4]):
                        draw_contours_on_image(centers, contours, colors_list, target)

                    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
                    axs[0, 0].imshow(cv2.cvtColor(copied_np, cv2.COLOR_BGR2RGB)); axs[0, 0].set_title("Size")
                    axs[0, 1].imshow(cv2.cvtColor(copied_np2, cv2.COLOR_BGR2RGB)); axs[0, 1].set_title("Circularity")
                    axs[1, 0].imshow(cv2.cvtColor(copied_np3, cv2.COLOR_BGR2RGB)); axs[1, 0].set_title("AR")
                    axs[1, 1].imshow(cv2.cvtColor(copied_np4, cv2.COLOR_BGR2RGB)); axs[1, 1].set_title("Roundness")
                    for ax in axs.ravel():
                        ax.axis('off')
                    cellpara_heatmap_path = image_process_path.replace("cellfill", "cellpara_heatmap")
                    plt.savefig(cellpara_heatmap_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
        except Exception as e:
            print(f"Failed to process growth plate file: {e}")

    def _process_articular_cartilage_file(self, file, input_path, nucleus_output):
        """处理关节软骨(AC): 细胞与细胞核两套掩码，生成 cellinfo/nuinfo,再进行配对与过滤。"""
        try:
            # 读取图像
            ac_img = io.imread(os.path.join(input_path, file))
            # 输出基名
            file_names = os.path.join(input_path, file.replace("mask", "cell"))
            nucleus_names = os.path.join(nucleus_output, file.replace("mask", "nucleus"))

            # 推理：细胞与细胞核
            masks_cell, flows_cell, _ = self.model1.eval(ac_img, flow_threshold=3, cellprob_threshold=-1)
            masks_nu, flows_nu, _ = self.model2.eval(ac_img, flow_threshold=3, cellprob_threshold=-1)

            io.masks_flows_to_seg(ac_img, masks_cell, flows_cell, file_names)
            io.save_masks(ac_img, masks_cell, flows_cell, file_names, png=True)
            if 'masks_nu' in locals():
                io.masks_flows_to_seg(ac_img, masks_nu, flows_nu, nucleus_names)
                io.save_masks(ac_img, masks_nu, flows_nu, nucleus_names, png=True)

            # 调用 AC 的 ImageJ 管线，生成 cellinfo 和 nuinfo
            self._run_ac_imagej_pipeline(input_path, nucleus_output, file_names, nucleus_names)
        except Exception as e:
            print(f"Failed to process articular cartilage file: {e}")
    
    def _run_imagej_macros(self, file, input_path):
        """运行ImageJ宏,基于Cellpose输出生成 cellinfo__*.csv"""
        try:
            # 懒加载 ImageJ
            if not hasattr(self, 'ij') or self.ij is None:
                self._init_imagej()

            # 解析索引（如 mask_00012.* -> 00012）
            try:
                m = re.search(r"(\d+)", file)
                index_str = m.group(1) if m else "00000"
            except Exception:
                index_str = "00000"

            # Cellpose 保存的掩码文件名：<file_names>_cp_masks.png
            cell_base = os.path.join(input_path, file.replace("mask", "cell"))
            mask_png = f"{cell_base}_cp_masks.png"  # e.g., cell_00012_cp_masks.png
            output_csv = os.path.join(input_path, f"cellinfo__{index_str}.csv")

            # ImageJ Macro：打开掩码图，测量并导出 CSV
            mask_png_norm = mask_png.replace('\\', '/')
            output_csv_norm = output_csv.replace('\\', '/')
            macro = f"""
            open("{mask_png_norm}" );
            run("8-bit");
            setAutoThreshold("Default");
            // 确保是二值
            run("Make Binary");
            // 填充孔洞，平滑
            run("Fill Holes");
            run("Watershed");
            // 设置测量指标
            run("Set Measurements...", "area mean min centroid perimeter shape redirect=None decimal=3");
            // 分析粒子，尺寸阈值可按需调整（单位：像素^2)
            run("Analyze Particles...", "size=50-Infinity show=Nothing display clear include summarize");
            // 保存结果
            saveAs("Results", "{output_csv_norm}" );
            // 关闭图像与结果
            close();
            run("Clear Results");
            """

            self.ij.py.run_macro(macro)
        except Exception as e:
            print(f"Failed to run ImageJ macro: {e}")

    def _run_ac_imagej_pipeline(self, input_path: str, nucleus_output: str, file_names: str, nucleus_names: str) -> None:
        """AC处理:直接从Cellpose掩膜分析,替代ImageJ的Find Edges处理"""

        # 规范路径与派生文件名
        input_path = input_path.replace('\\', '/')
        nucleus_output = nucleus_output.replace('\\', '/')
        file_names = file_names.replace('\\', '/')
        nucleus_names = nucleus_names.replace('\\', '/')

        image_file = os.path.basename(file_names.replace('.jpg', '_cp_masks.png'))
        image_path = f"{input_path}/{image_file}"
        mask_rename = image_file.replace("_cp_masks", "").replace("cell", "cellfill")
        image_process_path = f"{input_path}/{mask_rename}"
        cellinfo_path = f"{input_path}/cellinfo_{mask_rename.replace('cellfill', '').split('.')[0]}.csv"

        nu_file = os.path.basename(nucleus_names.replace('.jpg', '_cp_masks.png'))
        nu_path = f"{nucleus_output}/{nu_file}"
        numask_rename = nu_file.replace("_cp_masks", "").replace("nucleus", "nucleusfill")
        nu_process_path = f"{nucleus_output}/{numask_rename}"
        nuinfo_path = f"{nucleus_output}/nuinfo_{numask_rename.replace('nucleusfill', '').split('.')[0]}.csv"

        # ===== 使用直接分析替代ImageJ宏 =====
        try:
            # 分析细胞掩膜 (暂时保存cellfill.png用于directest分析,后续会删除)
            print(f"Processing cell mask: {image_path}")
            cells_df, cellfill_mask = analyze_cellpose_mask(
                image_path,
                cellinfo_path,
                output_mask_path=image_process_path,  # 暂时保存用于directest分析
                min_area=40,  # 注意:functionality中使用40-500而不是10-1000
                max_area=500,
                min_circularity=0.20,
                max_circularity=1.00
            )
            print(f"  Found {len(cells_df)} cells")

            # 分析细胞核掩膜 (不保存nucleusfill.png,仅生成CSV)
            print(f"Processing nucleus mask: {nu_path}")
            nuclei_df, nufill_mask = analyze_cellpose_mask(
                nu_path,
                nuinfo_path,
                output_mask_path=None,  # 不保存nucleusfill.png
                min_area=2,
                max_area=50,
                min_circularity=0.10,
                max_circularity=1.00
            )
            print(f"  Found {len(nuclei_df)} nuclei")

            # 保留_cp_masks.png用于验证 (不删除)
            # 如果需要清理,可以取消注释:
            # if os.path.exists(image_path):
            #     os.remove(image_path)
            # if os.path.exists(nu_path):
            #     os.remove(nu_path)

        except Exception as e:
            print(f"Error in AC direct analysis: {e}")
            import traceback
            traceback.print_exc()
            raise

        # ===== Nucleus-Cell配对:直接从cellpose实例掩膜提取每个细胞的轮廓 =====
        # 读取cellpose实例分割掩膜 (每个细胞有唯一label)
        cell_cp_mask = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if cell_cp_mask is None:
            raise RuntimeError(f"Cannot read cellpose mask: {image_path}")
        
        nuclei_df = pd.read_csv(nuinfo_path)
        cells_df = pd.read_csv(cellinfo_path)

        paired_results = nuinfo_path.replace("nuinfo", "pairinfo")
        matched_data = []
        
        # 获取所有细胞labels (排除背景0)
        cell_labels = np.unique(cell_cp_mask)
        cell_labels = cell_labels[cell_labels > 0]
        
        print(f"  Processing {len(cell_labels)} cell labels for pairing...")
        
        for cell_label in cell_labels:
            # 提取单个细胞的二值掩膜
            cell_binary = (cell_cp_mask == cell_label).astype(np.uint8) * 255
            
            # 查找该细胞的轮廓
            contours, _ = cv2.findContours(cell_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            
            contour = contours[0]  # 每个label应该只有一个连通区域
            if len(contour) < 3:
                continue
            
            try:
                poly = Polygon(contour.squeeze())
            except Exception:
                continue

            # 找到CSV中对应的细胞 (通过质心距离匹配)
            cell_x = cells_df.iloc[:, 7].values
            cell_y = cells_df.iloc[:, 8].values
            centroid = contour.mean(axis=0)[0]
            distances = np.sqrt((cell_x - centroid[0])**2 + (cell_y - centroid[1])**2)
            if len(distances) == 0:
                continue
            closest_cell_idx = int(np.argmin(distances))

            cell_area = float(cells_df.iloc[closest_cell_idx, 2])
            cell_circularity = float(cells_df.iloc[closest_cell_idx, 9]) if cells_df.shape[1] > 9 else 0.0
            cell_aspect_ratio = float(cells_df.iloc[closest_cell_idx, 12]) if cells_df.shape[1] > 12 else 0.0

            # 检查哪些nucleus在这个细胞内
            for _, nucleus_row in nuclei_df.iterrows():
                nucleus_x = float(nucleus_row[7])
                nucleus_y = float(nucleus_row[8])
                if poly.contains(Point(nucleus_x, nucleus_y)):
                    nucleus_area = float(nucleus_row[2])
                    nucleus_circularity = float(nucleus_row[9]) if len(nucleus_row) > 9 else 0.0
                    nucleus_aspect_ratio = float(nucleus_row[12]) if len(nucleus_row) > 12 else 0.0
                    area_ratio = nucleus_area / cell_area if cell_area > 0 else np.nan
                    circ_ratio = nucleus_circularity / cell_circularity if cell_circularity > 0 else np.nan
                    ar_ratio = nucleus_aspect_ratio / cell_aspect_ratio if cell_aspect_ratio > 0 else np.nan
                    matched_data.append({
                        'Cell_ID': closest_cell_idx,
                        'Nucleus_X': nucleus_x,
                        'Nucleus_Y': nucleus_y,
                        'Area_Ratio': area_ratio,
                        'Circularity_Ratio': circ_ratio,
                        'Aspect_Ratio_Ratio': ar_ratio,
                    })

        matched_df = pd.DataFrame(matched_data)
        print(f"  Found {len(matched_df)} nucleus-cell pairs")
        
        # 过滤Area_Ratio > 0.85的nucleus (可能是分割错误)
        remove_ids = matched_df.loc[matched_df['Area_Ratio'] > 0.85, ['Nucleus_X', 'Nucleus_Y']].round(2)
        matched_df = matched_df[matched_df['Area_Ratio'] <= 0.85]

        nuclei_df_filtered = nuclei_df[
            ~nuclei_df[['X','Y']].round(2).apply(tuple, axis=1).isin(remove_ids.apply(tuple, axis=1))
        ]

        nuclei_df_filtered.to_csv(nuinfo_path, index=False)
        matched_df.to_csv(paired_results, index=False)
        print(f"  Saved {len(matched_df)} valid pairs to {os.path.basename(paired_results)}")
        
        # NOTE: Do NOT copy or create `cellinfo__*.png` files here.
        # ac-cc can now accept `cell_XXXXX_cp_masks.png` (matching by embedded index)
        # so we avoid creating duplicate files to save disk and avoid surprising side effects.
        
        # 追加：对当前切片尝试执行 post Cellpose 分析（directest 逻辑）
        # 需要创建临时cellfill用于directest分析
        cellfill_created = False
        try:
            base = os.path.basename(image_process_path)
            m = re.search(r"(\d+)", base)
            idx_str = m.group(1) if m else None
            if idx_str is not None:
                # 从cellpose实例掩膜生成二值cellfill用于directest
                temp_cellfill = (cell_cp_mask > 0).astype(np.uint8) * 255
                cv2.imwrite(image_process_path, temp_cellfill)
                cellfill_created = True
                
                rows = self._run_single_directest_like_analysis(
                    mask_path=image_process_path.replace("cellfill", "mask").replace(".png", ".jpg"),
                    cellfill_path=image_process_path,
                    slice_index=int(idx_str)
                )
                if rows:
                    self._append_results_csv(rows)
        except Exception as e:
            print(f"  Warning: directest analysis failed: {e}")
        finally:
            # 确保删除临时cellfill (即使出错也要删除)
            if cellfill_created and os.path.exists(image_process_path):
                try:
                    os.remove(image_process_path)
                    print(f"  Cleaned up temporary cellfill: {os.path.basename(image_process_path)}")
                except Exception as e:
                    print(f"  Warning: Could not remove {image_process_path}: {e}")
    
    def cellinfoplot(self):
        """细胞信息绘图功能 - 完全按照原文件实现"""
        # 关闭选择窗口
        if hasattr(self, 'analysis_win') and self.analysis_win.winfo_exists():
            self.analysis_win.destroy()
        
        # 让用户选择分析模式
        self.select_analysis_mode()

    # ====== directest 核心算法移植 BEGIN ======
    def _fill_holes(self, binary: np.ndarray) -> np.ndarray:
        binary = binary.copy().astype(np.uint8)
        im_floodfill = binary.copy()
        h, w = binary.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        filled = binary | im_floodfill_inv
        return filled

    def _bfs_farthest_point(self, skel: np.ndarray, start: tuple) -> list:
        h, w = skel.shape
        visited = np.zeros_like(skel, dtype=bool)
        queue = [(start, [start])]
        visited[start] = True
        farthest_path = []
        while queue:
            (y, x), path = queue.pop(0)
            if len(path) > len(farthest_path):
                farthest_path = path
            for ny in range(y - 1, y + 2):
                for nx in range(x - 1, x + 2):
                    if (ny, nx) != (y, x) and 0 <= ny < h and 0 <= nx < w:
                        if skel[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append(((ny, nx), path + [(ny, nx)]))
        return farthest_path

    def _compute_centerline(self, mask: np.ndarray):
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_bin = (mask > 0).astype(np.uint8)
        skel = skeletonize(mask_bin).astype(np.uint8)

        coords = np.column_stack(np.where(skel > 0))
        if coords.size == 0:
            return (skel * 255).astype(np.uint8), []

        endpoints = []
        for y, x in coords:
            if np.sum(skel[y - 1:y + 2, x - 1:x + 2]) == 2:
                endpoints.append((y, x))
        if not endpoints:
            return (skel * 255).astype(np.uint8), []

        path1 = self._bfs_farthest_point(skel, endpoints[0])
        path2 = self._bfs_farthest_point(skel, path1[-1])

        main_skel = np.zeros_like(skel, dtype=np.uint8)
        for y, x in path2:
            if 0 <= y < main_skel.shape[0] and 0 <= x < main_skel.shape[1]:
                main_skel[y, x] = 255

        return main_skel, path2

    def _build_centerline_orientation(self, path: list, shape: tuple, k: int = 3) -> np.ndarray:
        h, w = shape
        normal_angle_map = np.full((h, w), np.nan, dtype=np.float32)
        if len(path) < 2:
            return normal_angle_map
        for i in range(len(path)):
            i0 = max(0, i - k)
            i1 = min(len(path) - 1, i + k)
            y0, x0 = path[i0]
            y1, x1 = path[i1]
            dy = float(y1 - y0)
            dx = float(x1 - x0)
            if dx == 0 and dy == 0:
                continue
            tangent_angle = (np.degrees(np.arctan2(dy, dx)) % 180.0)
            normal_angle = (tangent_angle + 90.0) % 180.0
            y, x = path[i]
            normal_angle_map[y, x] = normal_angle
        return normal_angle_map

    def _ray_length(self, mask: np.ndarray, start_yx: tuple, dir_yx: tuple, max_steps: int = 2000, step: float = 1.0) -> float:
        h, w = mask.shape
        y, x = float(start_yx[0]), float(start_yx[1])
        dy, dx = float(dir_yx[0]), float(dir_yx[1])
        length = 0.0
        for _ in range(max_steps):
            y += dy * step
            x += dx * step
            iy, ix = int(round(y)), int(round(x))
            if iy < 0 or iy >= h or ix < 0 or ix >= w:
                break
            if mask[iy, ix] == 0:
                break
            length += step
        return length

    def _unit_vec_from_angle_deg(self, theta_deg: float) -> tuple:
        rad = math.radians(theta_deg)
        dx = math.cos(rad)
        dy = math.sin(rad)
        norm = math.hypot(dx, dy)
        if norm == 0:
            return (0.0, 0.0)
        return (dy / norm, dx / norm)

    def _cell_depth_fraction_fixed_surface(self, cartilage_mask: np.ndarray, centerline_path: list,
                                           normal_angle_map: np.ndarray, cell_center_yx: tuple,
                                           closest_path_idx: int, surface_side: str = "up", step: float = 1.0):
        h, w = cartilage_mask.shape
        cy, cx = cell_center_yx
        py, px = centerline_path[closest_path_idx]

        normal_angle = normal_angle_map[int(round(py)), int(round(px))]
        if np.isnan(normal_angle):
            return None, None

        n_dy, n_dx = self._unit_vec_from_angle_deg(normal_angle)
        L_plus = self._ray_length(cartilage_mask, (py, px), (n_dy, n_dx), step=step)
        L_minus = self._ray_length(cartilage_mask, (py, px), (-n_dy, -n_dx), step=step)
        T = L_plus + L_minus
        if T <= 1e-6:
            return None, None

        if surface_side == "up":
            surface_is_plus = (n_dy < 0)
        elif surface_side == "down":
            surface_is_plus = (n_dy > 0)
        else:
            raise ValueError("surface_side 必须是 'up' 或 'down'")

        v_dy = cy - py
        v_dx = cx - px
        d = v_dy * n_dy + v_dx * n_dx

        if surface_is_plus:
            depth_frac = (L_plus - d) / T
            side = +1 if d >= 0 else -1
        else:
            depth_frac = (L_minus + d) / T
            side = +1 if d <= 0 else -1

        depth_frac = float(np.clip(depth_frac, 0.0, 1.0))
        return depth_frac, side

    def _depth_to_layer(self, depth_frac: float, n_layers: int = 3) -> int:
        thresholds = [0.2, 0.5, 1.0]  # 定义新的深度比例阈值
        for idx, threshold in enumerate(thresholds):
            if depth_frac < threshold:
                return idx
        return n_layers - 1

    def _wait_for_file(self, path: str, max_wait: int = 180, read_flag: int = cv2.IMREAD_GRAYSCALE):
        start = time.time()
        while time.time() - start < max_wait:
            if os.path.exists(path):
                img = cv2.imread(path, read_flag)
                if img is not None:
                    return img
            time.sleep(0.5)
        return None

    def _run_single_directest_like_analysis(self, mask_path: str, cellfill_path: str, slice_index: int,
                                            surface_side: str = "up", ray_step: float = 1.0, k_window: int = 3,
                                            save_vis: bool = True):
        # 读取软骨掩膜
        cartilage_img = cv2.imread(mask_path)
        if cartilage_img is None:
            return None
        gray = cv2.cvtColor(cartilage_img, cv2.COLOR_BGR2GRAY)
        _, cartilage_binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        kernel = np.ones((8, 8), np.uint8)
        cartilage_binary = cv2.morphologyEx(cartilage_binary, cv2.MORPH_CLOSE, kernel)
        cartilage_binary = cv2.bitwise_not(cartilage_binary)
        # 扩充四周2像素
        cartilage_binary = np.pad(cartilage_binary, pad_width=2, mode='constant', constant_values=0)
        cartilage_binary = self._fill_holes(cartilage_binary)
        cartilage_binary = cartilage_binary[2:-2, 2:-2]
        # 读取细胞掩膜
        cell_img = cv2.imread(cellfill_path, cv2.IMREAD_GRAYSCALE)
        if cell_img is None:
            return None
        _, cell_binary = cv2.threshold(cell_img, 127, 255, cv2.THRESH_BINARY)

        # 中心线与法线角
        centerline, centerline_path = self._compute_centerline(cartilage_binary)
        normal_angle_map = self._build_centerline_orientation(centerline_path, centerline.shape, k=k_window)

        if len(centerline_path) > 0:
            path_xy = np.array([(p[1], p[0]) for p in centerline_path], dtype=np.float32)
            kdtree = cKDTree(path_xy)
        else:
            kdtree = None

        vis = cv2.cvtColor(cartilage_binary, cv2.COLOR_GRAY2BGR)
        vis[np.rint(centerline).astype(np.uint8) > 0] = (255, 255, 255)

        contours, _ = cv2.findContours(cell_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for i, cnt in enumerate(contours):
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if len(cnt) >= 5:
                (cxy, (width, height), angle_deg) = cv2.fitEllipse(cnt)
                if width < height:
                    ellipse_angle = angle_deg % 180.0
                    major_len = height / 2.0
                else:
                    ellipse_angle = (angle_deg + 90.0) % 180.0
                    major_len = width / 2.0
            else:
                ellipse_angle = 0.0
                major_len = 20.0

            if kdtree is not None and len(centerline_path) > 0:
                dist, idx = kdtree.query([cx, cy])
                py, px = centerline_path[idx]
                iy, ix = int(round(py)), int(round(px))
                closest_point = (ix, iy)
                normal_angle = normal_angle_map[iy, ix]
                if np.isnan(normal_angle):
                    normal_angle = 0.0
            else:
                closest_point = None
                normal_angle = 0.0

            angle_diff = abs((ellipse_angle - normal_angle + 90) % 180 - 90)

            if kdtree is not None and closest_point is not None:
                depth_frac, side = self._cell_depth_fraction_fixed_surface(
                    cartilage_mask=cartilage_binary,
                    centerline_path=centerline_path,
                    normal_angle_map=normal_angle_map,
                    cell_center_yx=(cy, cx),
                    closest_path_idx=idx,
                    surface_side=surface_side,
                    step=ray_step,
                )
            else:
                depth_frac, side = None, None

            layer = self._depth_to_layer(depth_frac, 3) if depth_frac is not None else None

            # 可视化
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
            if closest_point is not None:
                cv2.circle(vis, closest_point, 3, (0, 255, 255), -1)
            L = max(20.0, major_len)
            rad = math.radians(ellipse_angle)
            p1 = (int(cx - L * math.cos(rad)), int(cy - L * math.sin(rad)))
            p2 = (int(cx + L * math.cos(rad)), int(cy + L * math.sin(rad)))
            cv2.line(vis, p1, p2, (0, 255, 0), 1)
            rad_n = math.radians(normal_angle)
            pn1 = (int(cx - L * math.cos(rad_n)), int(cy - L * math.sin(rad_n)))
            pn2 = (int(cx + L * math.cos(rad_n)), int(cy + L * math.sin(rad_n)))
            cv2.line(vis, pn1, pn2, (255, 0, 0), 1)
            if depth_frac is not None:
                cv2.putText(vis, f"L{layer + 1}:{angle_diff:.1f}", (cx + 6, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            results.append({
                "slice": slice_index,
                "cell_index": i + 1,
                "cx": cx,
                "cy": cy,
                "ellipse_angle": float(ellipse_angle),
                "normal_angle": float(normal_angle),
                "angle_diff": float(angle_diff),
                "depth_frac": None if depth_frac is None else float(depth_frac),
                "layer": None if depth_frac is None else int(layer),
            })

        # 保存可视化
        if save_vis and hasattr(self, 'output_dir') and self.output_dir:
            out_dir = self.output_dir
            try:
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"analysis_{slice_index:05d}.jpg")
                cv2.imwrite(out_path, vis)
            except Exception:
                pass

        return results

    def _append_results_csv(self, rows: list):
        if not rows:
            return
        if not hasattr(self, 'output_dir') or not self.output_dir:
            return
        csv_path = os.path.join(self.output_dir, "directest_summary.csv")
        header = ["slice", "cell_index", "cx", "cy", "ellipse_angle", "normal_angle", "angle_diff", "depth_frac", "layer"]
        file_exists = os.path.exists(csv_path)
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if not file_exists:
                    writer.writeheader()
                for r in rows:
                    writer.writerow({k: r.get(k, None) for k in header})
        except Exception:
            pass

    def run_cellpose_post_analysis(self, target_dir: str = None, max_wait_per_file: int = 180):
        """在 Cellpose+ImageJ 全流程结束后调用：
        - 匹配 mask_XXXXX.jpg 与 cellfill_XXXXX.png
        - 等待文件可读
        - 对每一对执行 directest-like 分析
        - 将所有结果累积写入 self.output_dir/directest_summary.csv
        - 保存每张 analysis_XXXXX.jpg 便于核查
        """
        if target_dir is None:
            # 仅在 AC 目录下运行，不对 GP 执行
            target_dir = self.ac_dir
        if not target_dir or not os.path.isdir(target_dir):
            return
        # 如果明确传入的是 GP 目录，则直接跳过
        try:
            if getattr(self, 'gp_dir', None) and os.path.abspath(target_dir) == os.path.abspath(self.gp_dir or ''):
                return
        except Exception:
            pass

        mask_files = sorted(glob.glob(os.path.join(target_dir, "mask_*.jpg")))
        results_all = []
        for mpath in mask_files:
            base = os.path.basename(mpath)
            m = re.search(r"(\d+)", base)
            if not m:
                continue
            idx = int(m.group(1))
            cellfill = os.path.join(target_dir, f"cellfill_{idx:05d}.png")
            # 等待 cellfill 就绪
            cf = self._wait_for_file(cellfill, max_wait=max_wait_per_file, read_flag=cv2.IMREAD_GRAYSCALE)
            if cf is None:
                continue
            # mask 也要确认可读
            mf = self._wait_for_file(mpath, max_wait=max_wait_per_file, read_flag=cv2.IMREAD_COLOR)
            if mf is None:
                continue
            rows = self._run_single_directest_like_analysis(mask_path=mpath, cellfill_path=cellfill, slice_index=idx)
            if rows:
                results_all.extend(rows)

        # 累计写入一次 CSV
        self._append_results_csv(results_all)

    # ====== directest 核心算法移植 END ======
    
    def select_analysis_mode(self):
        """让用户选择分析模式（单组或双组）"""
        self.analysis_win = ttk.Toplevel(self.ui_manager.main_window)
        self.analysis_win.title("选择分析模式")
        self.analysis_win.geometry("300x200")
        self.analysis_win.resizable(False, False)
        
        # 居中显示
        screen_width = self.ui_manager.main_window.winfo_screenwidth()
        screen_height = self.ui_manager.main_window.winfo_screenheight()
        x = (screen_width - 300) // 2
        y = (screen_height - 200) // 2
        self.analysis_win.geometry(f"+{x}+{y}")
        
        # 添加标题
        title_label = ttk.Label(
            self.analysis_win, 
            text="Select Cellinfo Mode",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=20)
        
        # 单组分析按钮
        single_btn = ttk.Button(
            self.analysis_win,
            text="Single Group",
            width=15,
            command=lambda: self.analyze_cell_parameters("single"),
        )
        single_btn.pack(pady=10)
        
        # 双组分析按钮
        two_groups_btn = ttk.Button(
            self.analysis_win,
            text="Two Groups",
            width=15,
            command=lambda: self.analyze_cell_parameters("two_groups"),
        )
        two_groups_btn.pack(pady=10)
    
    def analyze_cell_parameters(self, mode):
        """分析细胞参数分布"""
        # 关闭选择窗口
        if hasattr(self, 'analysis_win') and self.analysis_win.winfo_exists():
            self.analysis_win.destroy()
        
        # 存储数据
        data = {}
        group_names = {}
        data_type = "cell"
        
        if mode == "single":
            # 让用户选择文件夹
            folder_path = self.ui_manager.ask_directory("选择包含cellinfo文件的文件夹")
            if not folder_path:
                return
            if self.ui_manager.selected_mode == "Growth Plate":
                nu_path = folder_path.replace("GP", "nucleus_output")
            elif self.ui_manager.selected_mode == "Articular Cartilage":
                nu_path = folder_path.replace("AC", "nucleus_output")
            else:
                nu_path = folder_path.replace("Meniscus", "nucleus_output")
            if not os.path.exists(nu_path):
                self.nucleus_not_found = True
                data["cell"] = self.read_info_files(folder_path, "cellinfo")
            # 读取数据
            elif os.path.exists(nu_path):
                self.nucleus_not_found = False
                data["cell"] = self.read_info_files(folder_path, "cellinfo")
                data["nucleus"] = self.read_info_files(nu_path, "nuinfo")
                data["pair"] = self.read_pair_files(nu_path)
            group_names["Group1"] = os.path.basename(os.path.dirname(os.path.dirname(folder_path)))
            
        elif mode == "two_groups":
            # 选择第一组文件夹
            folder1 = self.ui_manager.ask_directory("Choosing first cellinfo folder")
            if not folder1:
                return
            if self.ui_manager.selected_mode == "Growth Plate":
                nu_path1 = folder1.replace("GP", "nucleus_output")
            elif self.ui_manager.selected_mode == "Articular Cartilage":
                nu_path1 = folder1.replace("AC", "nucleus_output")
            else:
                nu_path1 = folder1.replace("Meniscus", "nucleus_output")
            # 选择第二组文件夹
            folder2 = self.ui_manager.ask_directory("Choosing second cellinfo folder")
            if not folder2:
                return
            if self.ui_manager.selected_mode == "Growth Plate":
                nu_path2 = folder2.replace("GP", "nucleus_output")
            elif self.ui_manager.selected_mode == "Articular Cartilage":
                nu_path2 = folder2.replace("AC", "nucleus_output")
            else:
                nu_path2 = folder2.replace("Meniscus", "nucleus_output")
            if not os.path.exists(nu_path1) and not os.path.exists(nu_path2):
                self.nucleus_not_found = True
                data["cell"] = {
                    "Group1": self.read_info_files(folder1, "cellinfo"),
                    "Group2": self.read_info_files(folder2, "cellinfo")
                }
            else:
                self.nucleus_not_found = False
                data["cell"] = {
                    "Group1": self.read_info_files(folder1, "cellinfo"),
                    "Group2": self.read_info_files(folder2, "cellinfo")
                }
                data["nucleus"] = {
                    "Group1": self.read_info_files(nu_path1, "nuinfo"),
                    "Group2": self.read_info_files(nu_path2, "nuinfo")
                }
                data["pair"] = {
                    "Group1": self.read_pair_files(nu_path1),
                    "Group2": self.read_pair_files(nu_path2)
                }
            # 获取文件夹名称作为组名
            group_names["Group1"] = os.path.basename(os.path.dirname(os.path.dirname(folder1)))
            group_names["Group2"] = os.path.basename(os.path.dirname(os.path.dirname(folder2)))
        
        # 创建分析窗口
        self.create_analysis_window(data, group_names, mode, data_type)
    
    def read_pair_files(self, folder_path):
        """读取文件夹中的所有pairinfo文件"""
        # 查找所有CSV文件
        csv_files = glob.glob(os.path.join(folder_path, "*pairinfo*.csv"))
        if not csv_files:
            print(f"No pairinfo files found in '{folder_path}'")
            return None
        
        # 创建数据存储结构
        ratio_data = {
            "Area_Ratio": [],
            "Circularity_Ratio": [],
            "Aspect_Ratio_Ratio": []
        }
        
        # 读取每个文件的数据
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                
                column_names = df.columns.tolist()
                
                if "Area_Ratio" in column_names:
                    ratio_data["Area_Ratio"].extend(df["Area_Ratio"].dropna().values)
                elif len(column_names) > 3:
                    ratio_data["Area_Ratio"].extend(df.iloc[:, 3].dropna().values)
                
                if "Circularity_Ratio" in column_names:
                    ratio_data["Circularity_Ratio"].extend(df["Circularity_Ratio"].dropna().values)
                elif len(column_names) > 4:
                    ratio_data["Circularity_Ratio"].extend(df.iloc[:, 4].dropna().values)
                
                if "Aspect_Ratio_Ratio" in column_names:
                    ratio_data["Aspect_Ratio_Ratio"].extend(df["Aspect_Ratio_Ratio"].dropna().values)
                elif len(column_names) > 5:
                    ratio_data["Aspect_Ratio_Ratio"].extend(df.iloc[:, 5].dropna().values)
                
                print(f"Read pairinfo file successfully: {os.path.basename(file)}, found {len(ratio_data['Area_Ratio'])} ratios")
                    
            except Exception as e:
                print(f"Error reading pairinfo file {file}: {e}")
                import traceback
                traceback.print_exc()
        
        if not any(ratio_data.values()):
            print(f"No valid ratio data found in pairinfo files under '{folder_path}'")
            return None
        
        return ratio_data
    
    def read_info_files(self, folder_path, file_type):
        """读取文件夹中的所有cellinfo或nuinfo文件"""
        if file_type == "cellinfo":
            pattern = "*cellinfo*.csv"
        else:
            pattern = "*nuinfo*.csv"
        
        csv_files = glob.glob(os.path.join(folder_path, pattern))
        if not csv_files:
            if file_type != "nuinfo":
                self.ui_manager.show_info_message("信息", f"在 '{folder_path}' 中未找到包含'{file_type}'的CSV文件")
            return None
        
        # 创建数据存储结构
        data = {
            "Area": [],
            "Circ": [],
            "AR": [],
            "Round": [],
            "Solidity": []
        }
        
        # 读取每个文件的数据
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                
                # 检查列是否存在
                if len(df.columns) >= 15:
                    # 第3列: Area (索引2)
                    data["Area"].extend(df.iloc[:, 2].dropna().values)
                    
                    # 第10列: Circularity (索引9)
                    data["Circ"].extend(df.iloc[:, 9].dropna().values)
                    
                    # 第13列: Aspect Ratio (索引12)
                    data["AR"].extend(df.iloc[:, 12].dropna().values)
                    
                    # 第14列: Roundness (索引13)
                    data["Round"].extend(df.iloc[:, 13].dropna().values)
                    
                    # 第15列: Solidity (索引14)
                    data["Solidity"].extend(df.iloc[:, 14].dropna().values)
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        
        return data
    
    def create_analysis_window(self, data, group_names, mode, data_type):
        """创建参数分析窗口"""
        # 创建新窗口
        self.analysis_win = ttk.Toplevel(self.ui_manager.main_window)
        title = "Cell Parameters Analysis" if data_type == "cell" else "Nucleus Parameters Analysis"
        self.analysis_win.title(title if mode == "single" else f"{title} Comparison")
        self.analysis_win.state('zoomed')
        self.analysis_win.resizable(True, True)
        
        # 存储数据
        self.all_data = data  # 存储所有数据（细胞和细胞核）
        self.data_type = data_type  # 当前数据类型
        self.analysis_mode = mode
        self.group_names = group_names
        self.current_param = "Area"
        self.current_plot_type = "hist"
        self.scatter_params = {"x": "Area", "y": "Circ"}
        self.radar_params = {"type": "Normal"}
        
        # 根据当前数据类型获取数据
        if mode == "single":
            self.analysis_data = {
                "Group1": self.all_data[data_type]
            }
        else:
            self.analysis_data = self.all_data[data_type]

        if self.nucleus_not_found == True:
            self.pair_data = None
        else:
            self.pair_data = self.all_data["pair"]
        
        # 创建左右布局
        left_frame = ttk.Frame(self.analysis_win)
        left_frame.pack(side=ttk.LEFT, fill=ttk.BOTH, expand=True, padx=10, pady=10)
        
        right_frame = ttk.Frame(self.analysis_win)
        right_frame.pack(side=ttk.RIGHT, fill=ttk.Y, padx=10, pady=10)
        
        # 在左侧创建绘图区域
        fig_frame = ttk.Frame(left_frame)
        fig_frame.pack(fill=ttk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill=ttk.BOTH, expand=True)
        
        # 添加统计信息标签
        stats_frame = ttk.Frame(left_frame)
        stats_frame.pack(fill=ttk.X, pady=5)
        
        self.stats_label = ttk.Label(
            stats_frame, 
            text="Summary",
            font=("Arial", 10),
            justify=ttk.LEFT
        )
        self.stats_label.pack(anchor="w")
        
        # 在右侧创建参数按钮
        param_buttons = [
            ("Area", "Area"),
            ("Circ", "Circularity"),
            ("AR", "Aspect Ratio"),
            ("Round", "Roundness"),
            ("Solidity", "Solidity"),
        ]
        
        for param, label in param_buttons:
            btn = ttk.Button(
                right_frame,
                text=label,
                width=12,
                command=lambda p=param: self.update_plot(p)
            )
            btn.pack(pady=8, fill=ttk.X)

        # 添加散点图按钮
        scatter_btn = ttk.Button(
            right_frame,
            text="Scatter",
            width=12,
            command=self.show_scatter_dialog
        )
        scatter_btn.pack(pady=8, fill=ttk.X)
        
        # 添加雷达图按钮
        radar_btn = ttk.Button(
            right_frame,
            text="Radar",
            width=12,
            command=self.show_radar_dialog
        )
        radar_btn.pack(pady=8, fill=ttk.X)

        ratio_btn = ttk.Button(
            right_frame,
            text="paired ratio",
            width=12,
            command=self.draw_ratio_violin_plot,
        )
        ratio_btn.pack(pady=8, fill=ttk.X)
        
        # 添加数据源切换按钮
        if self.nucleus_not_found == False:
            data_source_text = "switch nucleus" if self.data_type == "cell" else "switch cell"
            self.data_source_btn = ttk.Button(
                right_frame,
                text=data_source_text,
                width=12,
                command=self.toggle_data_source,
                bootstyle="info"
            )
            self.data_source_btn.pack(pady=8, fill=ttk.X)
        
        # 添加导出按钮
        export_btn = ttk.Button(
            right_frame,
            text="Export",
            width=12,
            command=self.export_all_plots,
            bootstyle="success"
        )
        export_btn.pack(pady=8, fill=ttk.X)
        
        exit_btn = ttk.Button(
            right_frame,
            text="Exit",
            width=12,
            command=self.analysis_win.destroy,
            bootstyle="danger"
        )
        exit_btn.pack(pady=8, fill=ttk.X)
        
        # 初始化显示第一个参数的图表
        self.update_plot("Area")
    
    def toggle_data_source(self):
        """切换细胞/细胞核数据源"""
        # 切换数据类型
        if self.data_type == "cell":
            new_data_type = "nucleus"
            new_title = "Nucleus Parameters Analysis"
        else:
            new_data_type = "cell"
            new_title = "Cell Parameters Analysis"
        
        # 更新窗口标题
        if self.analysis_mode == "single":
            self.analysis_win.title(new_title)
        else:
            self.analysis_win.title(f"{new_title} Comparison")
        
        # 更新按钮文本
        self.data_source_btn.config(
            text="switch cell" if new_data_type == "nucleus" else "switch nucleus"
        )
        
        # 更新数据
        self.data_type = new_data_type
        
        if self.analysis_mode == "single":
            self.analysis_data = {
                "Group1": self.all_data[new_data_type]
            }
        else:
            self.analysis_data = self.all_data[new_data_type]
        
        # 重新绘制当前视图
        if self.current_plot_type == "scatter":
            self.draw_scatter_plot()
        elif self.current_plot_type == "radar":
            self.show_radar_plot()
        else:
            self.update_plot(self.current_param)
    
    def draw_ratio_violin_plot(self):
        """绘制比值参数的小提琴图"""
        self.current_plot_type = "ratio_violin"
        self.fig.clear()
        
        # 根据分析模式确定组数和数据结构
        if self.analysis_mode == "single":
            groups = ["Group1"]
            # 单组模式下直接使用self.pair_data
            plot_data = {"Group1": self.pair_data}
        else:
            groups = ["Group1", "Group2"]
            # 双组模式下使用self.pair_data字典
            plot_data = self.pair_data
        
        # 获取比值参数数据
        ratio_params = ["Area_Ratio", "Circularity_Ratio", "Aspect_Ratio_Ratio"]
        param_labels = ["Area_Ratio", "Circularity_Ratio", "Aspect_Ratio_Ratio"]
        
        # 检查数据是否存在
        for group in groups:
            if group not in plot_data or plot_data[group] is None:
                self.ui_manager.show_error_message("数据缺失", f"没有找到{group}的配对信息数据")
                return
            # 进一步检查是否有实际数据
            has_data = False
            for param in ratio_params:
                if param in plot_data[group] and len(plot_data[group][param]) > 0:
                    has_data = True
                    break
            if not has_data:
                self.ui_manager.show_error_message("数据缺失", f"{group}的配对信息中没有有效比值数据")
                return
        
        # 创建子图 - 修复figsize问题
        self.ax = self.fig.subplots(1, 3)  # 创建1行3列的子图
        
        # 设置颜色
        colors = ['#1f77b4', '#ff7f0e']  # 蓝橙色系
        
        # 遍历每个参数
        for i, param in enumerate(ratio_params):
            ax = self.ax[i]
            
            # 收集所有组的数据
            data_to_plot = []
            for group in groups:
                if param in plot_data[group]:
                    data_to_plot.append(plot_data[group][param])
                else:
                    data_to_plot.append([])  # 空列表
            
            # 绘制小提琴图
            if any(len(d) > 0 for d in data_to_plot):  # 确保有数据可绘制
                parts = ax.violinplot(
                    data_to_plot, 
                    showmeans=False, 
                    showmedians=True,
                    showextrema=True
                )
                
                # 设置颜色
                for j, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[j % len(colors)])
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
                
                # 设置中位数颜色
                parts['cmedians'].set_edgecolor('black')
                parts['cmedians'].set_linewidth(2)
                
                # 设置箱线图颜色
                parts['cbars'].set_edgecolor('black')
                parts['cmins'].set_edgecolor('black')
                parts['cmaxes'].set_edgecolor('black')
            
            # 设置标题和标签
            ax.set_title(param_labels[i], fontsize=12)
            ax.set_ylabel("ratio", fontsize=10)
            
            # 设置x轴标签
            if self.analysis_mode == "single":
                ax.set_xticks([1])
                ax.set_xticklabels([self.group_names["Group1"]])
            else:
                ax.set_xticks([1, 2])
                ax.set_xticklabels([self.group_names["Group1"], self.group_names["Group2"]])
            
            # 添加网格
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # 添加参考线
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        
        # 设置整体标题
        self.fig.suptitle("nucleus/cell parameters ratio", fontsize=16)
        
        # 调整布局
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 计算并显示统计信息
        stats_text = "ratio summary:\n"
        
        for group in groups:
            stats_text += f"{self.group_names[group]}:\n"
            for param in ratio_params:
                if param in plot_data[group] and len(plot_data[group][param]) > 0:
                    data = plot_data[group][param]
                    mean_val = np.mean(data)
                    median_val = np.median(data)
                    stats_text += f"{param}: mean={mean_val:.3f}, median={median_val:.3f}\n"
                else:
                    stats_text += f"{param}: no data\n"
            stats_text += "\n"
        
        self.stats_label.config(text=stats_text)
        
        # 刷新画布
        self.canvas.draw()
    
    def show_scatter_dialog(self):
        """显示散点图参数选择对话框"""
        dialog = tk.Toplevel(self.analysis_win)
        dialog.title("选择散点图参数")
        dialog.geometry("300x150")
        dialog.resizable(False, False)
        dialog.transient(self.analysis_win)
        dialog.grab_set()
        
        # 居中显示
        screen_width = self.analysis_win.winfo_screenwidth()
        screen_height = self.analysis_win.winfo_screenheight()
        x = (screen_width - 300) // 2
        y = (screen_height - 150) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # 添加X轴参数选择
        tk.Label(dialog, text="X轴参数:", font=("Arial", 10)).pack(pady=(10, 0))
        x_options = ["Area", "Circ", "AR", "Round", "Solidity"]
        x_var = tk.StringVar(value=self.scatter_params["x"])
        x_combobox = ttk.Combobox(dialog, textvariable=x_var, values=x_options, state="readonly")
        x_combobox.pack(pady=5)
        
        # 添加Y轴参数选择
        tk.Label(dialog, text="Y轴参数:", font=("Arial", 10)).pack(pady=(5, 0))
        y_var = tk.StringVar(value=self.scatter_params["y"])
        y_combobox = ttk.Combobox(dialog, textvariable=y_var, values=x_options, state="readonly")
        y_combobox.pack(pady=5)
        
        # 确认按钮
        def on_confirm():
            self.scatter_params["x"] = x_var.get()
            self.scatter_params["y"] = y_var.get()
            self.draw_scatter_plot()
            dialog.destroy()
        
        confirm_btn = tk.Button(dialog, text="Next", width=10, command=on_confirm)
        confirm_btn.pack(pady=2)
    
    def show_radar_dialog(self):
        """显示雷达图参数选择对话框"""
        dialog = tk.Toplevel(self.analysis_win)
        dialog.title("Choose radar plot type")
        dialog.geometry("300x150")
        dialog.resizable(False, False)
        dialog.transient(self.analysis_win)
        dialog.grab_set()
        
        # 居中显示
        screen_width = self.analysis_win.winfo_screenwidth()
        screen_height = self.analysis_win.winfo_screenheight()
        x = (screen_width - 300) // 2
        y = (screen_height - 150) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # 添加雷达图类型选择
        tk.Label(dialog, text="Radar plot type", font=("Arial", 10)).pack(pady=(10, 0))
        x_options = ["Normal", "Difference"]
        x_var = tk.StringVar(value=self.radar_params["type"])
        x_combobox = ttk.Combobox(dialog, textvariable=x_var, values=x_options, state="readonly")
        x_combobox.pack(pady=10)
        
        # 确认按钮
        def on_confirm2():
            self.radar_params["type"] = x_var.get()
            self.show_radar_plot()
            dialog.destroy()
        
        confirm_btn = tk.Button(dialog, text="Next", width=10, command=on_confirm2)
        confirm_btn.pack(pady=10)
    
    def draw_scatter_plot(self):
        """绘制散点图"""
        self.current_plot_type = "scatter"
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        # 获取参数
        x_param = self.scatter_params["x"]
        y_param = self.scatter_params["y"]
        
        # 颜色设置
        colors = ['#1f77b4', '#ff7f0e']  # 蓝橙色系
        alpha = 0.3  # 透明度
        
        # 单组分析
        if self.analysis_mode == "single":
            group = "Group1"
            x_data = np.array(self.analysis_data[group][x_param])
            y_data = np.array(self.analysis_data[group][y_param])
            
            # 绘制散点图
            self.ax.scatter(x_data, y_data, color=colors[0], alpha=alpha, s=2)
            
            # 计算相关系数
            correlation = np.corrcoef(x_data, y_data)[0, 1]
            
            # 更新统计信息
            stats_text = (
                f"{self.group_names[group]} scatter plot:\n"
                f"X: {x_param}, Y: {y_param}\n"
                f"Sample size: {len(x_data)}\n"
                f"Correlation coefficient: {correlation:.4f}"
            )
        
        # 双组分析
        elif self.analysis_mode == "two_groups":
            stats_text = ""
            correlations = []
            
            # 绘制两组数据
            for i, group in enumerate(["Group1", "Group2"]):
                if group not in self.analysis_data or self.analysis_data[group] is None:
                    continue
                
                x_data = np.array(self.analysis_data[group][x_param])
                y_data = np.array(self.analysis_data[group][y_param])
                
                # 绘制散点图
                self.ax.scatter(x_data, y_data, color=colors[i], alpha=alpha, label=self.group_names[group], s=2)
                
                # 计算相关系数
                if len(x_data) > 1 and len(y_data) > 1:
                    correlation = np.corrcoef(x_data, y_data)[0, 1]
                    correlations.append(correlation)
                    stats_text += (
                        f"{self.group_names[group]}:\n"
                        f"Correlation coefficient: {correlation:.4f}\n"
                        f"Sample size: {len(x_data)}\n\n"
                    )
                else:
                    stats_text += (
                        f"{self.group_names[group]}:\n"
                        f"Sample size not enough\n"
                        f"Sample size: {len(x_data)}\n\n"
                    )
            
            # 添加图例
            self.ax.legend()
        
        # 添加标题和标签
        param_names = {
            "Area": "Area",
            "Circ": "Circularity",
            "AR": "Aspect Ratio",
            "Round": "Roundness",
            "Solidity": "Solidity"
        }
        
        title = f"{param_names.get(x_param, x_param)} vs {param_names.get(y_param, y_param)}"
        if self.analysis_mode == "two_groups":
            title += " (Comparison)"
            
        self.ax.set_title(title, fontsize=14)
        self.ax.set_xlabel(param_names.get(x_param, x_param), fontsize=12)
        self.ax.set_ylabel(param_names.get(y_param, y_param), fontsize=12)
        
        # 添加网格
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        # 更新统计信息标签
        self.stats_label.config(text=stats_text)
        
        # 刷新画布
        self.canvas.draw()
    
    def show_radar_plot(self):
        self.current_plot_type = "radar"
        self.fig.clear()

        # 参数列表
        params = ["Area", "Circ", "AR", "Round", "Solidity"]
        num_vars = len(params)
        if self.radar_params["type"] == "Difference":
            # 收集所有组的 Area 和 AR 原始值，用于归一化
            all_area = []
            all_ar   = []
            for grp in ["Group1", "Group2"]:
                data = self.analysis_data.get(grp)
                if data is not None:
                    all_area += data["Area"]
                    all_ar   += data["AR"]
            if not all_area or not all_ar:
                return

            # 计算 Area 和 AR 的 global min/max/range
            min_area, max_area = min(all_area), max(all_area)
            min_ar,   max_ar   = min(all_ar),   max(all_ar)
            range_area = max_area - min_area or 1.0
            range_ar   = max_ar   - min_ar   or 1.0

            # 计算各组在每个参数上的归一化均值
            group_means = {}
            for grp in ["Group1", "Group2"]:
                data = self.analysis_data.get(grp)
                if data is None:
                    return
                means = []
                for p in params:
                    m = np.mean(data[p])
                    if p == "Area":
                        m = (m - min_area) / range_area
                    elif p == "AR":
                        m = (m - min_ar)   / range_ar
                    means.append(m)
                group_means[grp] = means

            # 只做两组差值
            g1 = group_means["Group1"]
            g2 = group_means["Group2"]
            delta = [b - a for a, b in zip(g1, g2)]
            delta += delta[:1]  # 闭合

            # 创建极坐标轴
            self.ax: PolarAxes = self.fig.add_subplot(111, projection='polar')

            # 计算各维度对应的角度
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]

            # 把 0° 定位到正上方，顺时针方向绘制
            self.ax.set_theta_offset(np.pi / 2)
            self.ax.set_theta_direction(-1)

            # 设置参数标签
            self.ax.set_xticks(angles[:-1])
            self.ax.set_xticklabels(params)

            # 计算 delta 的最小/最大值并留出 10% 边距
            dmin, dmax = min(delta), max(delta)
            pad = max(abs(dmin), abs(dmax)) * 0.1
            self.ax.set_ylim(dmin - pad, dmax + pad)

            # 自定义 y 轴刻度为增量百分比
            ticks = np.linspace(dmin, dmax, 5)
            self.ax.set_yticks(ticks)
            self.ax.set_yticklabels([f"{t*100:.1f}%" for t in ticks],
                                    color="grey", size=8)
            self.ax.set_rlabel_position(0)
            self.ax.grid(True)
            theta_circle = np.linspace(0, 2 * np.pi, 360)
            r_zero = [0.0] * len(theta_circle)
            self.ax.plot(theta_circle, r_zero, color="black", linestyle="--", linewidth=2)
            # 绘制差值雷达
            self.ax.plot(angles, delta,
                        color="#2ca02c", linewidth=2)
            self.ax.fill(angles, delta,
                        color="#2ca02c", alpha=0.25)

            # 标题与网格
            title = f"Difference Radar Plot between {self.group_names['Group2']} and {self.group_names['Group1']}"
            self.ax.set_title(title, size=16, y=1.1)

            # 更新统计信息：显示每个参数的增量
            stats = ["Increment (%):"]
            for p, d in zip(params, delta[:-1]):
                stats.append(f"{p}: {d*100:.2f}%")
            self.stats_label.config(text="\n".join(stats))

            # 刷新画布
            self.canvas.draw()
        else:
            all_area_values = [self.analysis_data[group]["Area"] for group in self.analysis_data if self.analysis_data[group] is not None]
            all_area_values = [val for sublist in all_area_values for val in sublist]  # 扁平化
            min_area = min(all_area_values)
            max_area = max(all_area_values)
            range_area = max_area - min_area
            all_AR_values = [self.analysis_data[group]["AR"] for group in self.analysis_data if self.analysis_data[group] is not None]
            all_AR_values = [val for sublist in all_AR_values for val in sublist]  # 扁平化
            min_AR = min(all_AR_values)
            max_AR = max(all_AR_values)
            range_AR = max_AR - min_AR
            # 避免除以零
            if range_area == 0:
                range_area = 1
            if range_AR == 0:
                range_AR = 1
            # 计算每个参数的平均值
            group_means = {}
            
            if self.analysis_mode == "single":
                group = "Group1"
                means = []
                for param in params:
                    mean_val = np.mean(self.analysis_data[group][param])
                    if param == "Area":  # 对 "Area" 参数进行归一化
                        mean_val = (mean_val - min_area) / range_area
                    elif param == "AR":  # 对 "AR" 参数进行归一化
                        mean_val = (mean_val - min_AR) / range_AR
                    means.append(mean_val)
                group_means[group] = means

            elif self.analysis_mode == "two_groups":
                for group in ["Group1", "Group2"]:
                    if group in self.analysis_data and self.analysis_data[group] is not None:
                        means = []
                        for param in params:
                            mean_val = np.mean(self.analysis_data[group][param])
                            if param == "Area":  # 对 "Area" 参数进行归一化
                                mean_val = (mean_val - min_area) / range_area
                            elif param == "AR":  # 对 "AR" 参数进行归一化
                                mean_val = (mean_val - min_AR) / range_AR
                            means.append(mean_val)
                        group_means[group] = means
            
            # 如果没有数据，直接返回
            if not group_means:
                return
            
            # 创建极坐标
            self.ax: PolarAxes = self.fig.add_subplot(111, polar=True)
            
            # 计算雷达图的角度
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            
            # 设置雷达图坐标
            self.ax.set_theta_offset(np.pi / 2)
            self.ax.set_theta_direction(-1)
            
            # 设置标签
            self.ax.set_xticks(angles[:-1])
            self.ax.set_xticklabels(params)
            
            # 颜色设置
            colors = ['#1f77b4', '#ff7f0e']  # 蓝橙色系
            
            # 设置y轴标签
            self.ax.set_rlabel_position(0)
            self.ax.set_ylim(0, 1.1)
            self.ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            self.ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
            
            # 绘制每组数据
            for i, (group, means) in enumerate(group_means.items()):
                # 归一化数据（使所有参数值在0-1范围内）
                values = means + [means[0]]  # 闭合
                
                # 绘制雷达图轮廓
                self.ax.plot(angles, values, color=colors[i], linewidth=2, linestyle='solid', 
                            label=f"{self.group_names[group]}")
                
                # 填充颜色
                self.ax.fill(angles, values, color=colors[i], alpha=0.1)
            
            # 添加图例
            self.ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
            
            # 添加标题
            self.ax.set_title("radar plot (mean)", size=16, color='black', y=1.1)
            
            # 添加网格
            self.ax.grid(True)
            
            # 更新统计信息标签
            stats_text = "radar (mean):\n"
            for group, means in group_means.items():
                stats_text += f"{self.group_names[group]}:\n"
                for param, mean in zip(params, means):
                    stats_text += f"{param}: {mean:.4f}\n"
                stats_text += "\n"
            
            self.stats_label.config(text=stats_text)
            
            # 刷新画布
            self.canvas.draw()
    
    def update_plot(self, param):
        """更新当前显示的参数图表"""
        self.current_param = param
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        # 颜色设置
        colors = ['#1f77b4', '#ff7f0e']  # 蓝橙色系
        alpha = 0.3  # 透明度
        
        # 单组分析
        if self.analysis_mode == "single":
            group = "Group1"
            param_data = np.array(self.analysis_data[group][param])
            
            # 计算基本统计量
            mean = np.mean(param_data)
            std = np.std(param_data)
            median = np.median(param_data)
            min_val = np.min(param_data)
            max_val = np.max(param_data)
            
            # 绘制直方图
            n, bins, patches = self.ax.hist(
                param_data, 
                bins=30, 
                density=True, 
                alpha=alpha, 
                color=colors[0],
                edgecolor='none',
                label=self.group_names[group]
            )
            
            # 计算并绘制最大似然高斯分布
            xmin, xmax = min_val - 0.1*(max_val-min_val), max_val + 0.1*(max_val-min_val)
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mean, std)
            self.ax.plot(x, p, color=colors[0], linewidth=2)
            
            # 更新统计信息
            stats_text = (
                f"{self.group_names[group]} Summary ({param}):\n"
                f"Mean: {mean:.4f}\n"
                f"SD: {std:.4f}\n"
                f"Median: {median:.4f}\n"
                f"Min: {min_val:.4f}\n"
                f"Max: {max_val:.4f}\n"
                f"Sample size: {len(param_data)}"
            )
        
        # 双组分析
        elif self.analysis_mode == "two_groups":
            stats_text = ""
            
            # 绘制两组数据
            for i, group in enumerate(["Group1", "Group2"]):
                if group not in self.analysis_data or self.analysis_data[group] is None:
                    continue
                
                param_data = np.array(self.analysis_data[group][param])
                
                # 计算统计量
                mean = np.mean(param_data)
                std = np.std(param_data)
                median = np.median(param_data)
                min_val = np.min(param_data)
                max_val = np.max(param_data)
                
                # 绘制直方图
                n, bins, patches = self.ax.hist(
                    param_data, 
                    bins=30, 
                    density=True, 
                    alpha=alpha, 
                    color=colors[i],
                    edgecolor='none',
                    label=f"{self.group_names[group]} (n={len(param_data)})"
                )
                
                # 计算并绘制高斯分布
                xmin, xmax = min_val - 0.1*(max_val-min_val), max_val + 0.1*(max_val-min_val)
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mean, std)
                self.ax.plot(x, p, color=colors[i], linewidth=2, linestyle='--')
                
                # 添加统计信息
                stats_text += (
                    f"{self.group_names[group]} Summary:\n"
                    f"Mean: {mean:.4f}\n"
                    f"SD: {std:.4f}\n"
                    f"Sample size: {len(param_data)}\n\n"
                )
        
        # 添加标题和标签
        param_names = {
            "Area": "Area",
            "Circ": "Circularity",
            "AR": "Aspect Ratio",
            "Round": "Roundness",
            "Solidity": "Solidity"
        }
        
        title = f"{param_names.get(param, param)} Distribution"
        if self.analysis_mode == "two_groups":
            title += " (Comparison)"
            
        self.ax.set_title(title, fontsize=14)
        self.ax.set_xlabel(param_names.get(param, param), fontsize=12)
        self.ax.set_ylabel("Probability density", fontsize=12)
        
        # 添加图例
        self.ax.legend()
        
        # 添加网格
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        # 更新统计信息标签
        self.stats_label.config(text=stats_text)
        
        # 刷新画布
        self.canvas.draw()
    
    def export_all_plots(self):
        """导出所有参数的图表为PDF"""
        # 让用户选择保存位置
        save_path = self.ui_manager.ask_file("Save All Plots as PDF", [("PDF 文件", "*.pdf"), ("所有文件", "*.*")])
        
        if not save_path:
            return
        
        # 创建多页PDF
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(save_path) as pdf:
            # 添加标题页
            title_fig = plt.figure(figsize=(11, 8.5))
            title_fig.suptitle(
                f"Cell Parameters Analysis" if self.data_type == "cell" else "Nucleus Parameters Analysis", 
                fontsize=20
            )
            plt.figtext(0.5, 0.5, 
                    f"Target: {'Cell' if self.data_type == 'cell' else 'Nucleus'}\n"
                    f"Analysis mode: {'Single group' if self.analysis_mode == 'single' else 'Two groups'}\n"
                    f"Group name: {', '.join(self.group_names.values())}",
                    ha="center", va="center", fontsize=16)
            pdf.savefig(title_fig)
            plt.close(title_fig)
            
            # 导出各参数直方图
            for param in ["Area", "Circ", "AR", "Round", "Solidity"]:
                # 创建新图表
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 颜色设置
                colors = ['#1f77b4', '#ff7f0e']  # 蓝橙色系
                alpha = 0.3  # 透明度
                
                # 单组分析
                if self.analysis_mode == "single":
                    group = "Group1"
                    param_data = np.array(self.analysis_data[group][param])
                    
                    # 计算统计量
                    mean = np.mean(param_data)
                    std = np.std(param_data)
                    
                    # 绘制直方图
                    n, bins, patches = ax.hist(
                        param_data, 
                        bins=30, 
                        density=True, 
                        alpha=alpha, 
                        color=colors[0],
                        edgecolor='none',
                        label=self.group_names[group]
                    )
                    
                    # 绘制高斯分布
                    xmin, xmax = np.min(param_data), np.max(param_data)
                    x = np.linspace(xmin, xmax, 100)
                    p = norm.pdf(x, mean, std)
                    ax.plot(x, p, color=colors[0], linewidth=2)
                    
                    # 添加统计信息
                    stats_text = (
                        f"{self.group_names[group]}\n"
                        f"Mean: {mean:.4f}\n"
                        f"SD: {std:.4f}\n"
                        f"Sample size: {len(param_data)}"
                    )
                    ax.text(0.95, 0.95, stats_text, 
                            transform=ax.transAxes, 
                            verticalalignment='top', 
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # 双组分析
                elif self.analysis_mode == "two_groups":
                    # 绘制两组数据
                    for i, group in enumerate(["Group1", "Group2"]):
                        if group not in self.analysis_data or self.analysis_data[group] is None:
                            continue
                        
                        param_data = np.array(self.analysis_data[group][param])
                        
                        # 计算统计量
                        mean = np.mean(param_data)
                        std = np.std(param_data)
                        
                        # 绘制直方图
                        n, bins, patches = ax.hist(
                            param_data, 
                            bins=30, 
                            density=True, 
                            alpha=alpha, 
                            color=colors[i],
                            edgecolor='none',
                            label=f"{self.group_names[group]} (n={len(param_data)})"
                        )
                        
                        # 绘制高斯分布
                        xmin, xmax = np.min(param_data), np.max(param_data)
                        x = np.linspace(xmin, xmax, 100)
                        p = norm.pdf(x, mean, std)
                        ax.plot(x, p, color=colors[i], linewidth=2, linestyle='--')
                    
                    # 添加统计信息
                    stats_text = ""
                    for i, group in enumerate(["Group1", "Group2"]):
                        if group not in self.analysis_data or self.analysis_data[group] is None:
                            continue
                        
                        param_data = np.array(self.analysis_data[group][param])
                        mean = np.mean(param_data)
                        std = np.std(param_data)
                        
                        stats_text += (
                            f"{self.group_names[group]}\n"
                            f"Mean: {mean:.4f}\n"
                            f"SD: {std:.4f}\n"
                            f"Sample size: {len(param_data)}\n\n"
                        )
                    
                    ax.text(0.95, 0.95, stats_text, 
                            transform=ax.transAxes, 
                            verticalalignment='top', 
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # 添加标题和标签
                param_names = {
                    "Area": "Area",
                    "Circ": "Circularity",
                    "AR": "Aspect Ratio",
                    "Round": "Roundness",
                    "Solidity": "Solidity"
                }
                
                title = f"{param_names.get(param, param)} distribution"
                if self.analysis_mode == "two_groups":
                    title += " (Comparison)"
                
                ax.set_title(title, fontsize=14)
                ax.set_xlabel(param_names.get(param, param), fontsize=12)
                ax.set_ylabel("probability density", fontsize=12)
                
                # 添加网格和图例
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend()
                
                # 添加到PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        self.ui_manager.show_info_message("Export successful", f"All plots were saved in:\n{save_path}")
    
    # 其他功能方法
    # === ROI绘制与背景调整（1334-1522）完全还原 ===
    def show_temporary_message(self, message, duration):
        temp_label = ttk.Label(self.ui_manager.main_window, text=message, style='RedTlabel.TLabel')
        temp_label.place(relx=0.65, rely=0.85, anchor="center")
        self.ui_manager.main_window.after(duration, temp_label.destroy)

    def start_roi_flash(self):
        # 启动红框闪烁，避免重复启动
        self.roi_flashing = True
        # 先取消旧的定时任务（如果有）
        if hasattr(self, '_roi_flash_job') and self._roi_flash_job:
            try:
                self.ui_manager.main_window.after_cancel(self._roi_flash_job)
            except Exception:
                pass
            self._roi_flash_job = None
        # 立即执行一次
        self._toggle_roi_border()

    def stop_roi_flash(self):
        # 停止闪烁并恢复样式
        self.roi_flashing = False
        # 取消未执行的 after 任务
        if hasattr(self, '_roi_flash_job') and self._roi_flash_job:
            try:
                self.ui_manager.main_window.after_cancel(self._roi_flash_job)
            except Exception:
                pass
            self._roi_flash_job = None
        target = getattr(self, '_flash_widget', None) or getattr(self.ui_manager, 'draw_btn', None)
        try:
            if target and hasattr(target, 'winfo_exists') and target.winfo_exists():
                target.configure(style='draw.TButton')
        except Exception:
            # 忽略目标已销毁或不可配置的异常
            pass

    def _toggle_roi_border(self):
        # 定时切换按钮样式，增加存在性与异常保护
        if not getattr(self, 'roi_flashing', False):
            return
        target = getattr(self, '_flash_widget', None) or getattr(self.ui_manager, 'draw_btn', None)
        try:
            if not target or not hasattr(target, 'winfo_exists') or not target.winfo_exists():
                # 目标不存在，停止闪烁
                self.stop_roi_flash()
                return
            current = target.cget('style')
            next_style = 'draw.TButton' if current == 'draw_active.TButton' else 'draw_active.TButton'
            target.configure(style=next_style)
        except Exception:
            # 访问失败，终止闪烁
            self.stop_roi_flash()
            return
        # 重新调度下一次切换，并保存任务ID以便取消
        try:
            self._roi_flash_job = self.ui_manager.main_window.after(300, self._toggle_roi_border)
        except Exception:
            self._roi_flash_job = None

    def toggle_draw_mode(self):
        self.draw_mode = not getattr(self, 'draw_mode', False)
        if self.draw_mode:
            self.start_roi_flash()
            self.ui_manager.style.configure('blur.TButton', background="#E5DDC5", takefocus=False, state='normal')
        else:
            self.stop_roi_flash()
            self.ui_manager.style.configure('blur.TButton', background="#E5DDC5", takefocus=False, state='disabled')
        if getattr(self, 'roi_id', None):
            self.ui_manager.point.delete(self.roi_id)
            self.roi_id = None
        self.roi_points = []

    def delete_roi(self):
        # 取消当前ROI多边形
        if hasattr(self, 'roi_id') and self.roi_id:
            self.ui_manager.point.delete(self.roi_id)
            self.roi_id = None
        self.roi_points = []
        self.ui_manager.style.configure('blur.TButton', background="ivory", borderwidth=0, takefocus=False, state='disabled')

    def on_click(self, event):
        # 只处理ROI多边形绘制
        if getattr(self, 'draw_mode', False):
            self.roi_drawing = True
            self.roi_points.append((event.x, event.y))
            if not hasattr(self, 'temp_line_id') or self.temp_line_id is None:
                self.temp_line_id = self.ui_manager.point.create_line(
                    event.x, event.y, event.x, event.y, fill="cyan", width=2, tags="roi"
                )

    def on_mouse_move(self, event):
        if getattr(self, 'draw_mode', False) and getattr(self, 'roi_drawing', False):
            self.roi_points.append((event.x, event.y))
            coords = [coord for point in self.roi_points for coord in point]
            self.ui_manager.point.coords(self.temp_line_id, *coords)

    def on_mouse_up(self, event):
        if getattr(self, 'draw_mode', False) and getattr(self, 'roi_drawing', False):
            self.roi_drawing = False
            self.roi_points.append((event.x, event.y))
            if len(self.roi_points) < 2:
                self.ui_manager.point.delete(self.temp_line_id)
                self.temp_line_id = None
                self.roi_points = []
                return
            self.roi_points.append(self.roi_points[0])
            coords = [coord for point in self.roi_points for coord in point]
            self.ui_manager.point.coords(self.temp_line_id, *coords)
            self.roi_id = self.temp_line_id
            self.temp_line_id = None
            self.ui_manager.style.configure('blur.TButton', background="ivory", borderwidth=0, takefocus=False, state='normal')

    def apply_brightness_adjustment(self):
        if not hasattr(self, 'roi_points') or len(self.roi_points) < 3:
            return
        canvas_width = 1000
        canvas_height = 800
        if self.ui_manager.selected_animal == "rat":
            scaled_width = int(5500 * self.ratio)
            scaled_height = int(2970 * self.ratio)
        elif self.ui_manager.selected_animal == "hamster":
            scaled_width = int(4500 * self.ratio)
            scaled_height = int(2250 * self.ratio)
        else:
            scaled_width = int(2500 * self.ratio)
            scaled_height = int(1250 * self.ratio)
        image_x0 = (canvas_width - scaled_width) // 2
        image_y0 = (canvas_height - scaled_height) // 2
        img_path = os.path.join(self.output_dir, "00000.jpg")
        orig_img_tmp = cv2.imread(img_path)
        if orig_img_tmp is None:
            print("Failed to read the original image.")
            return
        orig_H, orig_W = orig_img_tmp.shape[:2]
        roi_coords_orig = []
        for x, y in self.roi_points[:-1]:
            dx = int(x - image_x0)
            dy = int(y - image_y0)
            dx = max(0, min(dx, scaled_width - 1))
            dy = max(0, min(dy, scaled_height - 1))
            ox = int(round(dx * (orig_W / max(1, scaled_width))))
            oy = int(round(dy * (orig_H / max(1, scaled_height))))
            ox = max(0, min(ox, orig_W - 1))
            oy = max(0, min(oy, orig_H - 1))
            roi_coords_orig.append((ox, oy))
        if len(roi_coords_orig) < 3:
            return
        preview_bgr = self.adjust_background_brightness(roi_coords_orig, (scaled_width, scaled_height))
        if preview_bgr is None:
            return
        preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(preview_rgb)
        self.tk_image2 = ImageTk.PhotoImage(result_pil)
        if hasattr(self, "blurred_image") and self.blurred_image:
            self.ui_manager.point.itemconfig(self.blurred_image, image=self.tk_image2)
        else:
            self.blurred_image = self.ui_manager.point.create_image(500, 400, anchor="center", image=self.tk_image2)
        self.background_adjusted = True
        self.show_temporary_message("Background adjusted", 1000)
        if getattr(self, "roi_id", None):
            self.ui_manager.point.delete(self.roi_id)
            self.roi_id = None
        self.roi_points = []
        self.ui_manager.style.configure('blur.TButton', background="ivory", borderwidth=0, takefocus=False, state='disabled')

    def adjust_background_brightness(self, roi_coords_orig, disp_size):
        img_path = os.path.join(self.output_dir, "00000.jpg")
        img_path2 = os.path.join(self.output_dir, "1st", "00000.jpg")
        if not os.path.exists(img_path):
            print("Original image not found. Please open an image first.")
            return None
        orig_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if orig_img is None:
            print("Failed to read the original image.")
            return None
        orig_H, orig_W = orig_img.shape[:2]
        disp_W, disp_H = disp_size
        mask = np.zeros((orig_H, orig_W), dtype=np.uint8)
        poly = np.array([roi_coords_orig], dtype=np.int32)
        cv2.fillPoly(mask, poly, 255)
        background_mask = mask == 0
        result_orig = orig_img.copy()
        alpha = 1.6
        beta = -100
        for c in range(result_orig.shape[2]):
            channel = result_orig[:, :, c]
            temp = cv2.convertScaleAbs(channel[background_mask], alpha=alpha, beta=beta)
            channel[background_mask] = temp.ravel()
            result_orig[:, :, c] = channel
        cv2.imwrite(img_path, result_orig)
        cv2.imwrite(img_path2, result_orig)
        preview = cv2.resize(result_orig, (disp_W, disp_H), interpolation=cv2.INTER_AREA)
        return preview

    def undo_action(self):
        if getattr(self, 'count', 0) > 0:
            self.count -= 1
            if hasattr(self, 'annotations') and self.annotations:
                latest_annotation = self.annotations.pop()
                for item in latest_annotation:
                    self.ui_manager.point.delete(item)
            if hasattr(self, 'segview'):
                self.ui_manager.point.delete(self.segview)
            if self.count == 0:
                self.points.clear()
                self.labels.clear()
                if hasattr(self, 'image_history'):
                    self.image_history.clear()
                if hasattr(self, 'annotations'):
                    self.annotations.clear()
                self.ui_manager.point.delete("circles")
                self.firstframe = self.ui_manager.point.create_image(500, 400, anchor="center", image=self.tk_image)
            else:
                if self.points:
                    self.points.pop()
                if self.labels:
                    self.labels.pop()
                if hasattr(self, 'image_history') and self.image_history:
                    self.image_history.pop()
                    if self.image_history:
                        self.his = ImageTk.PhotoImage(self.image_history[self.count - 1])
                        self.segview = self.ui_manager.point.create_image(500, 400, anchor="center", image=self.his)
            self.ui_manager.point.tag_raise("circles")
        else:
            self.show_temporary_message("No actions to undo.", duration=1200)

    def redo_action(self):
        did_redo = False
        if hasattr(self, "blurred_image") and self.blurred_image:
            self.ui_manager.point.delete(self.blurred_image)
            self.blurred_image = None
            self.background_blurred = False
            did_redo = True
        self.firstframe = self.ui_manager.point.create_image(500, 400, anchor='center', image=self.tk_image)
        save_path = os.path.join(self.output_dir, "00000.jpg")
        save_path2 = os.path.join(self.output_dir, "1st", "00000.jpg")
        cv2.imwrite(save_path, self.imageshow)
        cv2.imwrite(save_path2, self.imageshow)
        if getattr(self, 'count', 0) > 0:
            self.count = 0
            self.points.clear()
            self.labels.clear()
            if hasattr(self, 'image_history'):
                self.image_history.clear()
            if hasattr(self, 'annotations'):
                self.annotations.clear()
            self.ui_manager.point.delete("circles")
            if hasattr(self, 'segview'):
                self.ui_manager.point.delete(self.segview)
            did_redo = True
        if not did_redo:
            self.show_temporary_message("No actions to redo.", duration=1200)
    
    def next_mask(self):
        """下一个掩码"""
        try:
            # 若处于掩码编辑模式且有未保存修改，阻止切换
            if getattr(self, 'is_in_mask_editing', True):
                messagebox.showwarning("Unsaved changes", "Please save current mask edits (click the green check) before switching.")
                return
            if self.mask_files and self.current_mask_index < len(self.mask_files) - 1:
                self.current_mask_index += 1
                self._display_current_mask()
            else:
                messagebox.showinfo("Mask", "All mask images have been processed")
                # 若处于编辑模式，恢复到原有界面
                try:
                    self._restore_original_interface()
                except Exception:
                    pass
        except Exception as e:
            print(f"Error switching to next mask: {e}")
    
    def show_on_point(self):
        """显示点 - 增强版，支持笔刷和橡皮擦编辑"""
        try:
            # 若处于掩码编辑模式且有未保存修改，阻止切换
            if getattr(self, 'is_in_mask_editing', True):
                messagebox.showwarning("Unsaved changes", "Please save current mask edits (click the green check) before switching.")
                return
            if self.mask_files and self.current_mask_index < len(self.mask_files):
                # 加载原始图像和掩码
                img_path = self.mask_files[self.current_mask_index]
                self.current_mask_path = img_path

                # 分离目录和文件名
                dir_path, file_name = os.path.split(img_path)

                # 分离目录的各部分
                dir_parts = dir_path.split(os.sep)

                # 移除倒数第二部分
                if len(dir_parts) > 1:
                    dir_parts.pop(-1)  # 移除倒数第二部分（AC）

                # 移除文件名前缀 "mask_"
                file_name = file_name.replace("mask_", "")

                # 重新拼接路径
                original_path = os.path.join(os.sep.join(dir_parts), file_name)
                if os.path.exists(original_path):
                    self.original_img = Image.open(original_path).convert("RGB")
                else:
                    # 如果没有找到原始图像，使用掩码图像
                    self.original_img = Image.open(img_path).convert("RGB")
                
                # 加载掩码图像
                self.mask_img = Image.open(img_path).convert("RGB")
                
                # 调整尺寸以适应point画布
                w, h = self.original_img.size
                if self.ui_manager.selected_animal == "rat":
                    ratio = 2/11
                elif self.ui_manager.selected_animal == "hamster":
                    ratio = 2/9
                else:
                    ratio = 2/5
                
                w2 = int(w * ratio)
                h2 = int(h * ratio)
                
                # 调整原始图像和掩码图像尺寸（并缓存为 NumPy 提升速度）
                self.original_img_resized = self.original_img.resize((w2, h2), Image.Resampling.LANCZOS)
                self.mask_img_resized = self.mask_img.resize((w2, h2), Image.Resampling.LANCZOS)
                self._orig_disp_np = np.array(self.original_img_resized)
                self._mask_disp_np = np.array(self.mask_img_resized)
                
                if self._mask_disp_np.ndim == 3:
                    self.orig_mask_bool = ~(
                        (self._mask_disp_np[:, :, 0] > 250) &
                        (self._mask_disp_np[:, :, 1] > 250) &
                        (self._mask_disp_np[:, :, 2] > 250)
                    )
                else:
                    self.orig_mask_bool = self._mask_disp_np > 0

                # 用户编辑掩码初始值 = 原始掩码
                self.edit_mask = self.orig_mask_bool.copy()
                
                # 创建编辑模式
                self._setup_mask_editing_mode()
                
                # 显示图像
                self._display_mask_with_background()
                
        except Exception as e:
            print(f"Error in show_on_point: {e}")

    def _display_mask_with_background(self):
        # 背景（原图 50% 透明）
        bg_np = (self._orig_disp_np * 0.5).astype(np.uint8)
        combined = bg_np.copy()

        # 用户编辑区域
        edit_area = self.edit_mask

        # 原始掩码的前景区域
        fg_area = self.orig_mask_bool

        # 在编辑区域里，如果原始是前景 → 显示彩色前景
        combined[edit_area & fg_area] = self._mask_disp_np[edit_area & fg_area]

        # 在编辑区域里，如果原始是背景 → 显示原图
        combined[edit_area & ~fg_area] = self._orig_disp_np[edit_area & ~fg_area]

        # 显示
        composite_img = Image.fromarray(combined)
        self.point_image = ImageTk.PhotoImage(composite_img)
        self.ui_manager.point.delete("circles")
        self.ui_manager.point.create_image(500, 400, anchor='center', image=self.point_image)

        # 提升方形光标覆盖层到最上层（若存在）
        if hasattr(self, '_overlay_square') and getattr(self, '_overlay_square', None):
            try:
                self.ui_manager.point.tag_raise(self._overlay_square)
            except Exception:
                pass

        if getattr(self, '_overlay_visible', False) and self.last_x is not None and self.last_y is not None:
            self._update_cursor_overlay_radius()

    def _draw_on_mask(self, event):
        """在掩码上绘制"""
        if not self.is_drawing or not self.editing_mode:
            return
        
        # 计算图像在画布上的位置和缩放比例
        canvas_width = self.ui_manager.point.winfo_width()
        canvas_height = self.ui_manager.point.winfo_height()
        img_width = self.original_img_resized.width
        img_height = self.original_img_resized.height
        
        # 计算图像在画布上的位置（居中显示）
        img_x0 = (canvas_width - img_width) // 2
        img_y0 = (canvas_height - img_height) // 2
        
        # 检查鼠标是否在图像范围内
        if not (img_x0 <= event.x <= img_x0 + img_width and 
                img_y0 <= event.y <= img_y0 + img_height):
            return
        
        # 转换为图像坐标
        img_x = event.x - img_x0
        img_y = event.y - img_y0
        
        # 确保坐标在图像范围内
        img_x = max(0, min(img_x, img_width - 1))
        img_y = max(0, min(img_y, img_height - 1))
        
        # 设置笔刷大小（转换为显示尺寸的比例）
        display_ratio = img_width / self.original_img.width
        brush_size = int(self.brush_size * display_ratio)
        
        if self.editing_mode == 'brush':
            self._add_to_mask(img_x, img_y, brush_size)
        elif self.editing_mode == 'eraser':
            self._remove_from_mask(img_x, img_y, brush_size)
        
        # 更新显示
        self._display_mask_with_background()
        
        # 更新光标位置
        self.last_x = event.x
        self.last_y = event.y
        if hasattr(self, '_overlay_square'):
            self._update_cursor_overlay_radius()

    def _add_to_mask(self, x, y, size):
        """向掩码添加方形区域"""
        h, w = self.edit_mask.shape
        half = size
        
        # 计算方形范围并裁剪到掩码边界
        x0 = max(0, x - half)
        x1 = min(w, x + half + 1)
        y0 = max(0, y - half)
        y1 = min(h, y + half + 1)
        
        # 应用方形笔刷
        self.edit_mask[y0:y1, x0:x1] = True

    def _remove_from_mask(self, x, y, size):
        """从掩码移除方形区域"""
        h, w = self.edit_mask.shape
        half = size
        
        # 计算方形范围并裁剪到掩码边界
        x0 = max(0, x - half)
        x1 = min(w, x + half + 1)
        y0 = max(0, y - half)
        y1 = min(h, y + half + 1)
        
        # 应用方形橡皮擦
        self.edit_mask[y0:y1, x0:x1] = False

    def _save_edited_mask(self):
        """保存编辑后的掩码"""
        try:
            if hasattr(self, 'current_mask_path') and self.current_mask_path:
                # 将编辑掩码放大到原尺寸
                full_w, full_h = self.mask_img.size
                edit_full = cv2.resize(
                    self.edit_mask.astype(np.uint8) * 255, 
                    (full_w, full_h), 
                    interpolation=cv2.INTER_NEAREST
                )
                
                # 获取原始图像
                original_np = np.array(self.original_img)
                
                # 创建新的mask图像
                if original_np.ndim == 3:  # RGB图像
                    # 创建一个全白的RGB图像
                    new_mask = np.ones_like(original_np) * 255
                    # 将mask区域设置为原始图像内容
                    mask_area = edit_full > 128
                    new_mask[mask_area] = original_np[mask_area]
                else:  # 灰度图像
                    new_mask = np.ones_like(original_np) * 255
                    mask_area = edit_full > 128
                    new_mask[mask_area] = original_np[mask_area]
                    
                # 保存图像
                Image.fromarray(new_mask).save(self.current_mask_path)
                print(f"Mask saved to: {self.current_mask_path}")
                
                # 恢复原有界面
                self._restore_original_interface()
                
                # 显示成功消息
                self.ui_manager.show_info_message("成功", "掩码已保存并更新")
                
        except Exception as e:
            print(f"Error saving mask: {e}")
            self.ui_manager.show_error_message("错误", f"保存掩码失败: {e}")

    # 保留原有的其他方法
    def _setup_mask_editing_mode(self):
        """设置掩码编辑模式"""
        # 隐藏原有按钮
        self.ui_manager.draw_btn.place_forget()
        self.ui_manager.blur_btn.place_forget()
        self.ui_manager.btn_undo.place_forget()
        self.ui_manager.btn_redo.place_forget()
        
        # 创建笔刷按钮
        self.brush_btn = ttk.Button(
            self.ui_manager.main_window,
            image=self.ui_manager.pen_img,
            command=self._activate_brush_mode,
            style='draw.TButton'
        )
        self.brush_btn.place(relx=0.69, rely=0.15, anchor="center")
        
        # 创建橡皮擦按钮
        self.eraser_btn = ttk.Button(
            self.ui_manager.main_window,
            image=self.ui_manager.eraser_img,
            command=self._activate_eraser_mode,
            style='draw.TButton'
        )
        self.eraser_btn.place(relx=0.73, rely=0.15, anchor="center")
        
        # 创建保存按钮（放在撤销和重做按钮中间位置）
        self.save_mask_btn = ttk.Button(
            self.ui_manager.main_window,
            image=self.ui_manager.success_img,
            command=lambda: (self._save_edited_mask(), self.next_mask()),
            style='draw.TButton'
        )
        self.save_mask_btn.place(relx=0.5, rely=0.85, anchor="center")

        # 自定义笔刷大小滑条（横向 Canvas 实现）
        self.brush_size = getattr(self, 'brush_size', 30)
        self._init_brush_slider()
        
        # 初始化编辑状态
        self.editing_mode = None  # None, 'brush', 'eraser'
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        # 进入掩码编辑模式与脏标志
        self.is_in_mask_editing = True
        
        # 绑定鼠标事件
        self.ui_manager.point.bind("<Button-1>", self._start_drawing)
        self.ui_manager.point.bind("<B1-Motion>", self._draw_on_mask)
        self.ui_manager.point.bind("<ButtonRelease-1>", self._stop_drawing)

        # 自定义光标预览
        self.ui_manager.point.bind("<Enter>", self._on_point_enter)
        self.ui_manager.point.bind("<Leave>", self._on_point_leave)
        self.ui_manager.point.bind("<Motion>", self._on_point_motion)

    def _activate_brush_mode(self):
        """激活笔刷模式"""
        self.editing_mode = 'brush'
        self.brush_btn.config(style='draw_active.TButton')
        self.eraser_btn.config(style='blur.TButton')
        # 刷新红框闪烁到 brush 按钮
        self._flash_widget = self.brush_btn
        if not self.roi_flashing:
            self.start_roi_flash()
        # 同时隐藏系统光标，避免打包后出现黑色小方块
        self.ui_manager.point.config(cursor="none")
        self._hide_system_cursor()

    def _activate_eraser_mode(self):
        """激活橡皮擦模式"""
        self.editing_mode = 'eraser'
        self.brush_btn.config(style='draw.TButton')
        self.eraser_btn.config(style='blur.TButton')
        # 停止闪烁（橡皮擦与 blur 样式一致，不闪烁）
        self.stop_roi_flash()
        self._flash_widget = None
        self.ui_manager.point.config(cursor="none")
        self._hide_system_cursor()

    def _start_drawing(self, event):
        """开始绘制"""
        if self.editing_mode:
            self.is_drawing = True
            self.last_x = event.x
            self.last_y = event.y
            self._draw_on_mask(event)

    def _stop_drawing(self, event):
        """停止绘制"""
        self.is_drawing = False
        self.last_x = None
        self.last_y = None

    def _restore_original_interface(self):
        """恢复原有界面"""
        # 移除编辑按钮
        if hasattr(self, 'brush_btn'):
            self.brush_btn.destroy()
        if hasattr(self, 'eraser_btn'):
            self.eraser_btn.destroy()
        if hasattr(self, 'save_mask_btn'):
            self.save_mask_btn.destroy()
        if hasattr(self, 'brush_canvas'):
            self.brush_canvas.destroy()
        
        # 恢复原有按钮
        self.ui_manager.draw_btn.place(relx=0.69, rely=0.15, anchor="center")
        self.ui_manager.blur_btn.place(relx=0.73, rely=0.15, anchor="center")
        self.ui_manager.btn_undo.place(relx=0.45, rely=0.85, anchor="center")
        self.ui_manager.btn_redo.place(relx=0.55, rely=0.85, anchor="center")
        
        # 解绑编辑模式绑定，并恢复到 UIManager 的默认交互绑定
        try:
            self.ui_manager.point.unbind("<Button-1>")
            self.ui_manager.point.unbind("<B1-Motion>")
            self.ui_manager.point.unbind("<ButtonRelease-1>")
            self.ui_manager.point.unbind("<Enter>")
            self.ui_manager.point.unbind("<Leave>")
            self.ui_manager.point.unbind("<Motion>")
        except Exception:
            pass
        # 恢复默认事件绑定
        try:
            self.ui_manager.bind_canvas_events()
        except Exception:
            pass
        
        # 重置光标
        self.ui_manager.point.config(cursor="")
        self._show_system_cursor()
        # 隐藏并删除编辑时的光标覆盖层，并清理属性，避免保留失效的 Canvas item id
        if hasattr(self, '_overlay_square'):
            try:
                self.ui_manager.point.delete(self._overlay_square)
            except Exception:
                pass
            try:
                delattr(self, '_overlay_square')
            except Exception:
                pass
        
        # 清理编辑状态
        self.editing_mode = None
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        self._overlay_visible = False
        # 退出编辑模式并清空脏标记
        self.is_in_mask_editing = False

    def _on_brush_size_change(self, *_):
        try:
            self.brush_size = int(self._brush_var.get())
        except Exception:
            self.brush_size = 30

    # ===== 自定义滑条实现（Canvas） =====
    def _init_brush_slider(self):
        try:
            # 规格
            self._slider_w = 180
            self._slider_h = 8
            self._knob_r = 12
            self._slider_min = 5
            self._slider_max = 150
            pad = 16
            # 画布
            bg_color = 'ivory'
            self.brush_canvas = ttk.Canvas(self.ui_manager.main_window, width=self._slider_w + pad*2, height=32)
            self.brush_canvas.config(bg=bg_color)
            self.brush_canvas.place(relx=0.7, rely=0.198, anchor="center")
            self.board = self.ui_manager.round_rect(self.brush_canvas, 0, 0, 211, 31, 6, fill='white', outline = "#e8e8e8")
            # 不调用 tkraise 以避免与 Canvas 的 tag_raise 冲突
            x0 = pad
            x1 = pad + self._slider_w
            y = 16
            # 轨道与激活轨道
            self._track = self.brush_canvas.create_line(x0, y, x1, y, width=self._slider_h, capstyle=tk.ROUND, fill="#bfbfbf")
            self._active = self.brush_canvas.create_line(x0, y, x0, y, width=self._slider_h, capstyle=tk.ROUND, fill="#0e6aba")
            # 初始位置
            knob_x = self._value_to_x(self.brush_size, x0, x1)
            self._knob = self.brush_canvas.create_oval(knob_x-self._knob_r, y-self._knob_r, knob_x+self._knob_r, y+self._knob_r, fill="#ffffff", outline="#e8e8e8", width=2)
            self._knob_inner = self.brush_canvas.create_oval(knob_x-self._knob_r//2, y-self._knob_r//2, knob_x+self._knob_r//2, y+self._knob_r//2, fill="#0e6aba", outline='#0e6aba')
            # 滑条上不显示预览圆，改为主窗口覆盖层显示
            self._show_overlay_center()
            # 事件
            self.brush_canvas.tag_bind(self._knob, "<B1-Motion>", self._on_brush_drag)
            self.brush_canvas.tag_bind(self._knob_inner, "<B1-Motion>", self._on_brush_drag)
        except Exception as e:
            print(f"Failed to initialize brush slider: {e}")

    def _on_brush_drag(self, event):
        pad = 16
        x0 = pad
        x1 = pad + self._slider_w
        y = 16
        x = max(x0, min(event.x, x1))
        # knob
        self.brush_canvas.coords(self._knob, x-self._knob_r, y-self._knob_r, x+self._knob_r, y+self._knob_r)
        self.brush_canvas.coords(self._knob_inner, x-self._knob_r//2, y-self._knob_r//2, x+self._knob_r//2, y+self._knob_r//2)
        self.brush_canvas.coords(self._active, x0, y, x, y)
        # 值
        self.brush_size = int(self._slider_min + (x-x0)/(x1-x0) * (self._slider_max - self._slider_min))
        self._show_overlay_center()

    def _value_to_x(self, val, x0, x1):
        val = max(self._slider_min, min(val, self._slider_max))
        ratio = (val - self._slider_min) / (self._slider_max - self._slider_min)
        return x0 + ratio * (x1 - x0)

    def _ensure_cursor_overlay(self):
        # 若属性不存在、为 None，或 Canvas 中对应图元已失效，则重新创建覆盖层
        needs_create = True
        try:
            if hasattr(self, '_overlay_square') and getattr(self, '_overlay_square', None):
                # 检查该 id 是否仍然存在于 Canvas（type 返回空字符串表示不存在）
                needs_create = (self.ui_manager.point.type(self._overlay_square) == "")
            else:
                needs_create = True
        except Exception:
            needs_create = True

        if needs_create:
            self._overlay_square = self.ui_manager.point.create_rectangle(
                0, 0, 0, 0,
                outline="black", width=2, fill="white", tags="cursor_overlay"
            )
            try:
                self.ui_manager.point.tag_raise(self._overlay_square)
            except Exception:
                pass

    def _update_cursor_overlay_radius(self):
        if self.last_x is None or self.last_y is None:
            return

        # 计算缩放比例
        img_width = self.original_img_resized.width
        display_ratio = img_width / self.original_img.width
        r = max(3, int(self.brush_size * display_ratio))

        self.ui_manager.point.coords(
            self._overlay_square,
            self.last_x - r, self.last_y - r,
            self.last_x + r, self.last_y + r
        )
        self.ui_manager.point.tag_raise(self._overlay_square)

    def _on_point_enter(self, event):
        """鼠标进入画布事件"""
        self._ensure_cursor_overlay()
        self._overlay_visible = True
        self.ui_manager.point.config(cursor="none")
        self._hide_system_cursor()

    def _on_point_leave(self, event):
        """鼠标离开画布事件"""
        self._overlay_visible = False
        # 恢复系统光标并隐藏覆盖层
        self.ui_manager.point.config(cursor="arrow")
        self._show_system_cursor()
        if hasattr(self, '_overlay_square') and getattr(self, '_overlay_square', None):
            try:
                self.ui_manager.point.coords(self._overlay_square, 0, 0, 0, 0)
            except Exception:
                pass

    def _on_point_motion(self, event):
        """鼠标在画布上移动事件"""
        # 更新光标位置
        self.last_x = event.x
        self.last_y = event.y
        
        # 计算图像在画布上的位置和缩放比例
        canvas_width = self.ui_manager.point.winfo_width()
        canvas_height = self.ui_manager.point.winfo_height()
        img_width = self.original_img_resized.width
        img_height = self.original_img_resized.height
        
        # 计算图像在画布上的位置（居中显示）
        img_x0 = (canvas_width - img_width) // 2
        img_y0 = (canvas_height - img_height) // 2
        
        # 检查鼠标是否在图像范围内
        if (img_x0 <= event.x <= img_x0 + img_width and 
            img_y0 <= event.y <= img_y0 + img_height):
            # 在图像范围内 → 显示光标
            self._ensure_cursor_overlay()
            self._overlay_visible = True
            self.ui_manager.point.config(cursor="none")
            self._hide_system_cursor()
            self._update_cursor_overlay_radius()
        else:
            # 不在图像范围内 → 隐藏光标
            self._overlay_visible = False
            self.ui_manager.point.config(cursor="arrow")
            self._show_system_cursor()
            if hasattr(self, '_overlay_square') and getattr(self, '_overlay_square', None):
                self.ui_manager.point.coords(self._overlay_square, 0, 0, 0, 0)

    def _show_overlay_center(self):
        self._ensure_cursor_overlay()

        img_width = self.original_img_resized.width
        display_ratio = img_width / self.original_img.width
        r = max(3, int(self.brush_size * display_ratio))

        cw = self.ui_manager.point.winfo_width() or self.ui_manager.point.winfo_reqwidth()
        ch = self.ui_manager.point.winfo_height() or self.ui_manager.point.winfo_reqheight()
        cx = cw // 2
        cy = ch // 2

        self.last_x, self.last_y = cx, cy
        self.ui_manager.point.coords(self._overlay_square, cx - r, cy - r, cx + r, cy + r)
        self.ui_manager.point.tag_raise(self._overlay_square)
        self._overlay_visible = True

    # ===== Windows 系统指针隐藏/恢复（避免打包后 cursor='none' 显示黑色小方块） =====
    def _hide_system_cursor(self):
        try:
            import sys
            if sys.platform != 'win32':
                return
            import ctypes
            # 将计数降到 < 0 以确保隐藏
            while ctypes.windll.user32.ShowCursor(False) >= 0:
                pass
            self._cursor_hidden = True
        except Exception:
            # 忽略任何失败，保持 Tk 的 cursor='none' 作为退路
            pass

    def _show_system_cursor(self):
        try:
            import sys
            if sys.platform != 'win32':
                return
            import ctypes
            # 将计数提升到 >= 0 以确保显示
            while ctypes.windll.user32.ShowCursor(True) < 0:
                pass
            self._cursor_hidden = False
        except Exception:
            pass
    
    def compare_two_folders(self):
        """比较两个文件夹的3D数据"""
        try:
            folder1 = filedialog.askdirectory(title="选择第一个数据文件夹")
            if not folder1:
                return
            folder2 = filedialog.askdirectory(title="选择第二个数据文件夹")
            if not folder2:
                return
            
            # 提取组名：父文件夹的父文件夹名
            def get_group_name(folder_path):
                # 获取父文件夹的父文件夹名
                parent_parent = os.path.dirname(os.path.dirname(folder_path))
                group_name = os.path.basename(parent_parent)
                return group_name

            group1_name = get_group_name(folder1)
            group2_name = get_group_name(folder2)

            def prompt_area_scale(group_label: str, initial: float = 1.0):
                """弹窗询问每组 1 pixel 对应的 μm² 面积，返回浮点值或 None."""
                while True:
                    value = simpledialog.askfloat(
                        title="像素面积设置",
                        prompt=f"请输入组“{group_label}”中 1 pixel 对应的面积 (μm²)：",
                        parent=self.ui_manager.main_window,
                        initialvalue=float(initial),
                        minvalue=0.0
                    )
                    if value is None:
                        return None
                    if value <= 0:
                        messagebox.showerror("输入错误", "面积值必须大于 0，请重新输入。")
                        continue
                    return float(value)

            scale1 = prompt_area_scale(group1_name)
            if scale1 is None:
                return
            scale2 = prompt_area_scale(group2_name)
            if scale2 is None:
                return

            def load_folder_data(folder, viewer, prefer: str = 'auto', area_scale: float = 1.0):
                """从文件夹中加载 3D 点数据。
                兼容两类输入：
                - cellinfo__*.csv（细胞）
                - nuinfo__*.csv（细胞核）
                若同时存在，优先使用 nuinfo__。
                prefer: 'auto' | 'cell' | 'nucleus'
                area_scale: 1 pixel 对应的 μm² 值，将用于缩放面积数据
                返回: (xs, ys, zs, areas_scaled, info_kind)
                """
                # 优先使用 nuinfo，如果没有则回退到 cellinfo
                pattern_nu = os.path.join(folder, "nuinfo__*.csv")
                pattern_cell = os.path.join(folder, "cellinfo__*.csv")
                files_nu = sorted(
                    glob.glob(pattern_nu),
                    key=lambda p: int(re.search(r'nuinfo__([0-9]+)', os.path.basename(p)).group(1)))
                files_cell = sorted(
                    glob.glob(pattern_cell),
                    key=lambda p: int(re.search(r'cellinfo__([0-9]+)', os.path.basename(p)).group(1)))

                # 选择使用的数据类型
                use_nu = False
                if prefer == 'nucleus':
                    use_nu = bool(files_nu)
                elif prefer == 'cell':
                    use_nu = False
                else:  # auto
                    use_nu = bool(files_nu)

                if use_nu and files_nu:
                    files = files_nu
                    info_prefix = 'nuinfo__'
                    idx_regex = r'nuinfo__([0-9]+)'
                    info_kind = 'nucleus'
                    print(f"Found {len(files)} nuinfo files in folder {folder}")
                elif files_cell:
                    files = files_cell
                    info_prefix = 'cellinfo__'
                    idx_regex = r'cellinfo__([0-9]+)'
                    info_kind = 'cell'
                    print(f"Found {len(files)} cellinfo files in folder {folder}")
                else:
                    print(f"No nuinfo__*.csv or cellinfo__*.csv files found in folder {folder}")
                    return np.array([]), np.array([]), np.array([]), np.array([]), 'none'

                xs, ys, zs, areas = [], [], [], []
                ref_idx = int(re.search(idx_regex, os.path.basename(files[0])).group(1))
                # 确定掩码目录：nuinfo 在 nucleus_output，掩码在同级 AC；cellinfo 与掩码同目录
                if info_prefix.startswith('nuinfo'):
                    parent_dir = os.path.dirname(folder)
                    mask_dir = os.path.join(parent_dir, 'AC')
                else:
                    mask_dir = folder
                ref_mask_path = os.path.join(mask_dir, f"mask_{ref_idx:05d}.jpg")
                if not os.path.exists(ref_mask_path):
                    print(f"Warning: mask not found: {ref_mask_path}")
                _, ref_edge = viewer.preprocess_mask(ref_mask_path)

                for i, f in enumerate(files):
                    idx = int(re.search(idx_regex, os.path.basename(f)).group(1))
                    mask_path = os.path.join(mask_dir, f"mask_{idx:05d}.jpg")
                    if not os.path.exists(mask_path):
                        print(f"Warning: mask not found: {mask_path}")
                    try:
                        df = pd.read_csv(f, header=0)
                        if df.shape[0] < 2:
                            print(f"File {f} has insufficient rows, skipped")
                            continue
                        x = df.iloc[1:, 7].astype(float).values
                        y = df.iloc[1:, 8].astype(float).values
                        a = df.iloc[1:, 2].astype(float).values
                        print(f"File {f} loaded successfully with {len(x)} data points ({info_prefix[:-2]})")
                    except Exception as e:
                        print(f"Failed to load file {f}: {e}")
                        continue

                    try:
                        _, edge_points = viewer.preprocess_mask(mask_path)
                        R, t = viewer.rigid_transform(edge_points, ref_edge)
                        coords = np.vstack([x, y]).T.astype(np.float32)
                        coords_transformed = (R @ coords.T).T + t
                        z = np.full_like(x, fill_value=i * 70, dtype=float)

                        xs.append(coords_transformed[:, 0])
                        ys.append(coords_transformed[:, 1])
                        zs.append(z)
                        areas.append(a)
                    except Exception as e:
                        print(f"Error processing file {f}: {e}")
                        continue

                result_x = np.concatenate(xs) if xs else np.array([])
                result_y = np.concatenate(ys) if ys else np.array([])
                result_z = np.concatenate(zs) if zs else np.array([])
                result_a = np.concatenate(areas) if areas else np.array([])
                if result_a.size > 0:
                    result_a = result_a.astype(float) * float(area_scale)
                print(f"Final number of data points in folder {folder}: {len(result_x)}")
                return result_x, result_y, result_z, result_a, info_kind

            win = tk.Toplevel(self.ui_manager.main_window)
            win.geometry("1500x800")
            win.title("3D Comparison")

            # 左右主容器
            frame_left = tk.Frame(win, highlightthickness=0, bd=0, takefocus=0, bg="#ffffff")
            frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            frame_right = tk.Frame(win, highlightthickness=0, bd=0, takefocus=0, bg="#ffffff")
            frame_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            frame_legend = tk.Frame(win, highlightthickness=0, bd=0, takefocus=0, bg="#ffffff", width=80)
            frame_legend.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

            # 在左右容器内部分别创建：顶部控制面板 + 下方渲染区域
            # 左侧控制面板（固定在顶部，不会被清理）
            control_left = ttk.Frame(frame_left)
            control_left.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 4))
            
            # 左侧渲染区域（会被清理重建）
            render_left = tk.Frame(frame_left, highlightthickness=0, bd=0, takefocus=0, bg="#ffffff")
            render_left.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # 右侧控制面板（固定在顶部，不会被清理）
            control_right = ttk.Frame(frame_right)
            control_right.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 4))
            
            # 右侧渲染区域（会被清理重建）
            render_right = tk.Frame(frame_right, highlightthickness=0, bd=0, takefocus=0, bg="#ffffff")
            render_right.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # 创建两个3D查看器实例
            viewer1 = ThreeDViewer_VTK(win)
            viewer2 = ThreeDViewer_VTK(win)

            # 状态容器，便于在回调里更新
            state = {
                'prefer1': 'auto',
                'prefer2': 'auto',
                'scale1': scale1,
                'scale2': scale2,
            }

            # 初始加载（auto）
            x1, y1, z1, a1, info1 = load_folder_data(folder1, viewer1, prefer=state['prefer1'], area_scale=state['scale1'])
            x2, y2, z2, a2, info2 = load_folder_data(folder2, viewer2, prefer=state['prefer2'], area_scale=state['scale2'])

            def render_both():
                # 类型和有效性检查
                if not isinstance(a1, np.ndarray) or not isinstance(a2, np.ndarray):
                    print("Warning: Data type error, skipping render")
                    return
                if not isinstance(x1, np.ndarray) or not isinstance(x2, np.ndarray):
                    print("Warning: Coordinate type error, skipping render")
                    return
                
                # 至少需要一侧有数据
                if a1.size == 0 and a2.size == 0:
                    messagebox.showerror("Error", "Both folders have no valid data")
                    return
                
                # 统一色标范围（收集有效数据）
                valid_areas = []
                if a1.size > 0:
                    valid_areas.append(a1)
                if a2.size > 0:
                    valid_areas.append(a2)
                
                all_areas = np.concatenate(valid_areas)
                a_min, a_max = float(np.min(all_areas)), float(np.max(all_areas))
                cmap = cm.get_cmap('jet').resampled(7)
                norm = Normalize(vmin=a_min, vmax=a_max)

                def get_colors(areas):
                    if areas.size == 0:
                        return np.array([]).reshape(0, 3).astype(np.uint8)
                    return (np.array(cmap(norm(areas)))[:, :3] * 255).astype(np.uint8)

                colors1 = get_colors(a1)
                colors2 = get_colors(a2)

                # 清理并重绘两侧（注意：清理的是渲染子frame，不是包含控制面板的父frame）
                try:
                    self._cleanup_vtk_frame(render_left)
                except Exception:
                    pass
                try:
                    self._cleanup_vtk_frame(render_right)
                except Exception:
                    pass

                # 在组名上附加数据类型标签
                lbl1 = f"{group1_name} - {'Nucleus' if info1 == 'nucleus' else 'Cell'}"
                lbl2 = f"{group2_name} - {'Nucleus' if info2 == 'nucleus' else 'Cell'}"

                rw1, iren1 = viewer1.create_renderer(render_left, x1, y1, z1, a1, colors1, group_name=lbl1)
                rw2, iren2 = viewer2.create_renderer(render_right, x2, y2, z2, a2, colors2, group_name=lbl2)

                # 保存引用防止被回收
                win._rw1 = rw1
                win._iren1 = iren1
                win._rw2 = rw2
                win._iren2 = iren2

            # 创建控制面板控件（control_left/control_right frame已在上方创建）
            ttk.Label(control_left, text=f"{group1_name} 数据源:").pack(side=tk.LEFT)
            var1 = tk.StringVar(value='Auto')
            cmb1 = ttk.Combobox(control_left, textvariable=var1, state='readonly', values=['Auto', 'Cell', 'Nucleus'], width=10)
            cmb1.pack(side=tk.LEFT, padx=6)
            scale_label_left = ttk.Label(control_left, text=f"像素面积: {state['scale1']:.3f} μm²/pixel")
            scale_label_left.pack(side=tk.LEFT, padx=(10, 0))

            ttk.Label(control_right, text=f"{group2_name} 数据源:").pack(side=tk.LEFT)
            var2 = tk.StringVar(value='Auto')
            cmb2 = ttk.Combobox(control_right, textvariable=var2, state='readonly', values=['Auto', 'Cell', 'Nucleus'], width=10)
            cmb2.pack(side=tk.LEFT, padx=6)
            scale_label_right = ttk.Label(control_right, text=f"像素面积: {state['scale2']:.3f} μm²/pixel")
            scale_label_right.pack(side=tk.LEFT, padx=(10, 0))

            def pref_to_key(v: str) -> str:
                v = (v or '').strip().lower()
                if v in ('cell', 'nucleus'):
                    return v
                return 'auto'

            def reload_left(_e=None):
                nonlocal x1, y1, z1, a1, info1
                state['prefer1'] = pref_to_key(var1.get())
                new_x1, new_y1, new_z1, new_a1, new_info1 = load_folder_data(
                    folder1, viewer1, prefer=state['prefer1'], area_scale=state['scale1']
                )
                # 确保所有返回值都是 ndarray
                for var in (new_x1, new_y1, new_z1, new_a1):
                    if not isinstance(var, np.ndarray):
                        new_x1 = new_y1 = new_z1 = new_a1 = np.array([])
                        break
                x1, y1, z1, a1, info1 = new_x1, new_y1, new_z1, new_a1, new_info1
                render_both()

            def reload_right(_e=None):
                nonlocal x2, y2, z2, a2, info2
                state['prefer2'] = pref_to_key(var2.get())
                new_x2, new_y2, new_z2, new_a2, new_info2 = load_folder_data(
                    folder2, viewer2, prefer=state['prefer2'], area_scale=state['scale2']
                )
                # 确保所有返回值都是 ndarray
                for var in (new_x2, new_y2, new_z2, new_a2):
                    if not isinstance(var, np.ndarray):
                        new_x2 = new_y2 = new_z2 = new_a2 = np.array([])
                        break
                x2, y2, z2, a2, info2 = new_x2, new_y2, new_z2, new_a2, new_info2
                render_both()

            cmb1.bind('<<ComboboxSelected>>', reload_left)
            cmb2.bind('<<ComboboxSelected>>', reload_right)

            # 首次渲染
            render_both()

            # 在 legend 绘制前，根据当前数据计算色标范围
            try:
                all_areas_legend = np.concatenate([a1, a2])
                a_min = float(np.min(all_areas_legend)) if all_areas_legend.size else 0.0
                a_max = float(np.max(all_areas_legend)) if all_areas_legend.size else 1.0
            except Exception:
                a_min, a_max = 0.0, 1.0
            cmap = cm.get_cmap('jet').resampled(7)
            norm = Normalize(vmin=a_min, vmax=a_max)

            # 创建球体图例
            legend_canvas = tk.Canvas(frame_legend, width=140, height=400, bg="white")
            legend_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 计算图例球体的参数
            legend_height = 320  # 减少高度，为标题留出更多空间
            legend_margin = 40   # 增加上边距，避免与标题重合
            n_legend_spheres = 7
            
            # 预计算所有球体的半径，用于计算合适的间距
            sphere_radii = []
            for i in range(n_legend_spheres):
                area_value = a_min + (a_max - a_min) * i / (n_legend_spheres - 1)
                radius = 10.0 + (area_value - a_min) / (a_max - a_min) * 30.0
                sphere_radii.append(radius)
            
            # 计算最小间距，确保球体之间有足够的空间
            max_radius = max(sphere_radii)
            min_spacing = max_radius * 2.5  # 球体中心之间的最小距离
            
            # 计算总需要的空间
            total_needed_space = (n_legend_spheres - 1) * min_spacing + max_radius * 2
            available_space = legend_height - 2 * legend_margin
            
            # 如果需要的空间超过可用空间，调整间距
            if total_needed_space > available_space:
                actual_spacing = available_space / (n_legend_spheres - 1)
            else:
                actual_spacing = min_spacing
            
            # 创建图例球体
            for i in range(n_legend_spheres):
                # 计算当前球体对应的面积值
                area_value = a_min + (a_max - a_min) * i / (n_legend_spheres - 1)
                
                # 计算球体颜色
                color = cmap(norm(area_value))
                color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                
                # 获取球体半径
                radius = sphere_radii[i]
                
                # 计算球体中心位置（使用均匀间距）
                y_pos = legend_margin + max_radius + i * actual_spacing
                
                # 绘制球体（用圆形代替）
                x1 = 50 - radius/2
                y1 = y_pos - radius/2
                x2 = 50 + radius/2
                y2 = y_pos + radius/2
                
                # 绘制球体
                legend_canvas.create_oval(x1, y1, x2, y2, 
                                        fill=f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}",
                                        outline="black", width=1)
                
                # 计算数字标签位置，确保与球体保持合适距离
                text_x = 50 + radius/2 + 10  # 球体右边缘 + 10像素间距
                
                # 添加数值标签
                legend_canvas.create_text(text_x, y_pos, text=f"{area_value:.1f}", 
                                        anchor="w", font=("Arial", 8), fill="white")
            
            # 添加标题（改为白色）
            legend_canvas.create_text(70, 20, text="Area Value", 
                                    font=("Arial", 10, "bold"), anchor="center", fill="white")

            win._legend_canvas = legend_canvas
            win._frame_left = frame_left
            win._frame_right = frame_right
            win._viewer1 = viewer1
            win._viewer2 = viewer2

            def _on_close():
                self.on_compare_window_close(win)

            win.protocol("WM_DELETE_WINDOW", _on_close)
            
        except Exception as e:
            print(f"3D comparison error: {e}")
            import traceback
            traceback.print_exc()

    def on_compare_window_close(self, win):
        """关闭3D比较窗口时的清理工作"""
        try:
            # 先隐藏窗口，避免销毁过程中的最后一次 Configure/Expose
            try:
                win.withdraw()
            except Exception:
                pass

            # 清理两个渲染区
            if hasattr(win, "_frame_left") and win._frame_left is not None:
                try:
                    win._viewer1._cleanup_vtk_frame(win._frame_left)
                except Exception:
                    pass
                win._frame_left = None

            if hasattr(win, "_frame_right") and win._frame_right is not None:
                try:
                    win._viewer2._cleanup_vtk_frame(win._frame_right)
                except Exception:
                    pass
                win._frame_right = None

            # 不再对 _rw1/_rw2 调 Finalize/ReleaseGraphicsResources，直接断引用
            for attr in ("_rw1", "_rw2", "_iren1", "_iren2", "_viewer1", "_viewer2"):
                if hasattr(win, attr):
                    setattr(win, attr, None)

            # 清理图例画布
            if hasattr(win, "_legend_canvas") and win._legend_canvas is not None:
                try:
                    win._legend_canvas.destroy()
                except Exception:
                    pass
                win._legend_canvas = None
        except Exception as e:
            print(f"Error when closing window: {e}")
        finally:
            try:
                win.destroy()
            except Exception:
                pass

    def update_segmentation_display(self):
        """实时分割显示 - 与原始ACseg_rat.py一致"""
        try:
            # 读取当前帧图像（第一张）
            image_path = os.path.join(self.output_dir, "00000.jpg")
            image_pil = Image.open(image_path).convert("RGB")
            image_np = np.array(image_pil)
            device = torch.device("cuda")
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            sam2_checkpoint = "C:/Users/hantang/anaconda3/envs/pytorch/lib/site-packages/sam2/samgp.pt"
            model_cfg = "C:/Users/hantang/anaconda3/envs/pytorch/lib/site-packages/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
            predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
            inference_state = predictor.init_state(video_path=os.path.join(self.output_dir, "1st"))
            predictor.reset_state(inference_state)
            points2 = np.array(self.points, np.float32)
            labels2 = np.array(self.labels, np.int32)
            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points2,
                labels=labels2,
            ) 
            mask_logits = out_mask_logits[0].cpu().numpy()
            threshold = 0.5
            binary_mask = (mask_logits > threshold).astype(np.uint8)

            self.latest_mask = binary_mask
            current_mask = self.latest_mask
            if len(current_mask.shape) > 2:
                current_mask = current_mask[0]
            darkened_img = (image_np * 0.6).astype(np.uint8)

            fg_img = image_np.copy()
            fg_pil = Image.fromarray(fg_img)
            enhancer = ImageEnhance.Brightness(fg_pil)
            fg_enhanced = enhancer.enhance(1.5)
            fg_enhanced_np = np.array(fg_enhanced)

            combined_img = darkened_img.copy()
            combined_img[current_mask > 0] = fg_enhanced_np[current_mask > 0]

            kernel = np.ones((3,3), np.uint8)
            mask_uint8 = (current_mask * 255).astype(np.uint8)
            dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
            eroded = cv2.erode(mask_uint8, kernel, iterations=1)
            edge = cv2.subtract(dilated, eroded)
            edge = (edge > 0).astype(np.uint8)

            edge_blue = np.zeros_like(combined_img)
            kernel = np.ones((5, 5), np.uint8)
            dilated_edge = cv2.dilate(edge.astype(np.uint8), kernel, iterations=2)
            edge_blue[dilated_edge == 1] = [0, 120, 255]

            edge_img = Image.fromarray(edge_blue).filter(ImageFilter.GaussianBlur(radius=4))
            combined_pil = Image.fromarray(combined_img)
            combined_pil = Image.blend(combined_pil, edge_img, alpha=0.7)
            blue_layer = np.zeros_like(image_np, dtype=np.uint8)
            blue_layer[:, :] = [0, 0, 255]
            blue_mask = (current_mask > 0).astype(np.uint8)
            blue_layer = cv2.bitwise_and(blue_layer, blue_layer, mask=blue_mask)
            alpha = 0.1
            combined_with_blue = cv2.addWeighted(np.array(combined_pil), 1 - alpha, blue_layer, alpha, 0)
            new_width = int(combined_pil.width * self.ratio)
            new_height = int(combined_pil.height * self.ratio)
            resized_img = Image.fromarray(combined_with_blue).resize((new_width, new_height), Image.LANCZOS)
            if not hasattr(self, 'image_history'):
                self.image_history = []
            self.image_history.append(resized_img)
            self.tk_image2 = ImageTk.PhotoImage(resized_img)
            if not hasattr(self, 'count'):
                self.count = 0
            if self.count == 0:
                self.ui_manager.point.delete(self.firstframe)
            else:
                if hasattr(self, 'segview'):
                    self.ui_manager.point.delete(self.segview)
            self.segview = self.ui_manager.point.create_image(500, 400, anchor='center', image=self.tk_image2)
            self.count += 1
        except Exception as e:
            print(f"Real-time segmentation display error: {e}")
            import traceback
            traceback.print_exc()
