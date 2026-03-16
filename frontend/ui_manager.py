"""
UI管理器
负责所有界面相关的逻辑和组件管理
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import ImageTk
import matplotlib.pyplot as plt
from backend.functionality import FunctionalityManager
from backend.functionality import ThreeDViewer_VTK
import importlib
import os

class UIManager:
    """UI管理器"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.main_window = None
        self.mode_window = None
        self.current_window = None
        self.ui_elements = {}
        self.selected_mode = None
        self.selected_animal = None
        self.count = 0
        self.image_history = []
        self.annotations = []
        self.draw_mode = False
        self.roi_drawing = False
        self.roi_points = []
        self.roi_id = None
        self.background_blurred = False
        
        # 初始化样式
        self._init_styles()
        
        # 初始化功能管理器
        self.functionality = FunctionalityManager(self)
    
    def _init_styles(self):
        """初始化UI样式"""
        self.style = ttk.Style(theme='superhero')
        
        # 配置按钮样式
        self.style.configure('Primary.TButton', 
                           font=('Segoe UI', 12, 'bold'), 
                           padding=(8, 8), 
                           background=self.style.colors.get("info"), 
                           foreground='white', 
                           anchor='center')
        
        self.style.configure('Secondary.TButton', 
                           font=('Segoe UI', 18, 'bold'), 
                           padding=(5, 12), 
                           background=self.style.colors.get("primary"), 
                           foreground="#f0f0f0", 
                           anchor='center')
        
        self.style.configure('Third.TButton',
                           background="ivory", 
                           borderwidth=0, 
                           takefocus=False)
        
        self.style.configure('Four.TButton',
                           background="white", 
                           borderwidth=0, 
                           takefocus=False)
        
        self.style.configure('draw.TButton',
                           background="ivory", 
                           borderwidth=0, 
                           takefocus=False)
        
        self.style.configure('blur.TButton',
                           background="#E5DDC5", 
                           takefocus=False, 
                           foreground="black", 
                           font=('Segoe UI', 12, 'bold'), 
                           padding=(5, 5))
        
        self.style.configure('RedTlabel.TLabel',
                           background="ivory", 
                           foreground="red", 
                           font=('Segoe UI', 15, 'bold'), 
                           anchor='center')
        
        self.style.configure('exit.TButton', 
                           background="red", 
                           foreground="white", 
                           font=('Segoe UI', 12, 'bold'), 
                           padding=(5, 5), 
                           anchor='center')
        
        # 配置样式映射
        self.style.map('Third.TButton', 
                      background=[('active', "#F5EEDC"), ('pressed', "#E5DDC5"), ('hover', 'ivory')])
        self.style.map('Four.TButton', 
                      background=[('active', "#F5EEDC"), ('pressed', "#E5DDC5"), ('hover', 'white')])
        self.style.map('blur.TButton', 
                      background=[('active', "grey"), ('pressed', "grey"), ('hover', '#E5DDC5')])
        
        self.style.configure('draw_active.TButton',
                           background='ivory',
                           borderwidth=6,
                           relief='solid')
        
        self.style.map('draw_active.TButton',
                      bordercolor=[('!disabled', 'red')],
                      relief=[('pressed', 'sunken'), ('!pressed', 'solid')])
    
    def create_mode_selection(self):
        """创建模式选择窗口"""
        self.mode_window = self.root
        self.mode_window.deiconify()
        self.mode_window.title("Select Mode")
        self.mode_window.state('zoomed')
        
        for widget in self.mode_window.winfo_children():
            widget.destroy()
        bg_color = self.style.lookup('TFrame', 'background')
        self.mode_window.configure(bg=bg_color)
        # 标题
        title = ttk.Label(self.mode_window, text="Select Analysis Mode", 
                         font=("Segoe UI", 36, "bold"), bootstyle=PRIMARY)
        title.place(relx=0.5, rely=0.15, anchor="center")
        
        # 按钮容器
        button_frame = ttk.Frame(self.mode_window)
        button_frame.place(relx=0.5, rely=0.5, anchor="center")

        strip = self.create_mode_cards(button_frame)
        strip.pack()

        # 退出按钮
        exit_btn = ttk.Button(
            self.mode_window,
            text="EXIT",
            command=self.mode_window.destroy,
            style='exit.TButton',
            width=25
        )
        exit_btn.place(relx=0.5, rely=0.9, anchor="center")
        
        self.mode_window.protocol("WM_DELETE_WINDOW", self.exit_app)
    
    def exit_app(self):
        if self.root:
            self.root.quit()
            self.root.destroy()
    def create_mode_cards(self, parent):
        """创建模式选择卡片"""
        import math
        from PIL import Image, ImageOps, ImageFilter, ImageDraw
        
        bg_color = "white"
        card_w, card_h = 430, 530
        radius = 25
        gap = 10
        btn_h = 70
        img_h = 350
        btn_color = "#4e7ca1"
        btn_outline = "ivory"
        card_fill = "#0e2841"
        card_outline = "ivory"
        scale_factor = 1.03
        lift_dy = 6
        dim_stipple = "gray75"
        dim_fill = "#000000"

        # 模式信息
        mode_infos = [
            {"text": "Growth Plate", "image": "C:/Users/hantang/Pictures/Saved Pictures/GP.png", "command": self.create_growth_plate},
            {"text": "Articular Cartilage", "image": "C:/Users/hantang/Pictures/Saved Pictures/AC.png", "command": self.create_articular_cartilage},
            {"text": "Meniscus", "image": "C:/Users/hantang/Pictures/Saved Pictures/Menis.png", "command": self.create_meniscus},
        ]

        # 卡片容器
        strip = tk.Frame(parent, bg=bg_color)
        strip.grid_rowconfigure(0, weight=0)

        self._mode_cards = []
        self._mode_images = []
        self._mode_card_dim_layers = []

        def load_image(path, size):
            w, h = size
            try:
                im = Image.open(path).convert("RGBA")
                im = ImageOps.contain(im, (w, h), Image.LANCZOS)
            except Exception:
                im = Image.new("RGBA", (w, h), (90, 90, 90, 255))
            return ImageTk.PhotoImage(im)
        
        def make_glow_image(w, h, glow_color="#a0ffff", blur_radius=25, spread=25, radius=25):
            img = Image.new("RGBA", (w + spread*2, h + spread*2), (43, 62, 80, 255))
            draw = ImageDraw.Draw(img)

            left, top = spread, spread
            right, bottom = spread + w, spread + h
            draw.rounded_rectangle(
                [left, top, right, bottom],
                radius=radius,
                fill=glow_color
            )
            
            glow = img.filter(ImageFilter.GaussianBlur(blur_radius))
            return glow

        def round_rect(canvas, x1, y1, x2, y2, r, **kwargs):
            points = []

            steps = 24
            angle_step = 90 / steps

            for i in range(steps + 1):
                angle = math.radians(180 + i * angle_step)
                px = x1 + r + r * math.cos(angle)
                py = y1 + r + r * math.sin(angle)
                points.append((px, py))

            for i in range(steps + 1):
                angle = math.radians(270 + i * angle_step)
                px = x2 - r + r * math.cos(angle)
                py = y1 + r + r * math.sin(angle)
                points.append((px, py))

            for i in range(steps + 1):
                angle = math.radians(0 + i * angle_step)
                px = x2 - r + r * math.cos(angle)
                py = y2 - r + r * math.sin(angle)
                points.append((px, py))

            for i in range(steps + 1):
                angle = math.radians(90 + i * angle_step)
                px = x1 + r + r * math.cos(angle)
                py = y2 - r + r * math.sin(angle)
                points.append((px, py))

            return canvas.create_polygon(points, smooth=True, splinesteps=36, **kwargs)

        def create_card(idx, info):
            canvas = tk.Canvas(
                strip, width=card_w + 40, height=card_h + 40,
                bg="#2b3e50", highlightthickness=0, bd=0, cursor="hand2"
            )
            canvas.grid(row=0, column=idx, padx=(0 if idx == 0 else gap, 0), pady=0)

            group = f"cardgrp{idx}"
            y_offset = lift_dy + 10
            x_offset = 16
            # 卡片背景
            card = round_rect(canvas, 8 + x_offset, 8+y_offset, card_w - 8 + x_offset, card_h - 8 + y_offset, radius,
                            fill=card_fill, outline=card_outline, width=4, tags=(group, "card"))

            # 按钮
            btn_margin = 18
            btn = round_rect(canvas, 8 + btn_margin + x_offset, 8 + btn_margin + y_offset,
                            card_w - 8 - btn_margin + x_offset, 8 + btn_margin + btn_h, 16,
                            fill=btn_color, outline=btn_outline, width=3, tags=(group, "btn-bg"))

            canvas.create_text(
                card_w // 2 + x_offset, 8 + btn_margin + btn_h // 2 + y_offset // 2,
                text=info.get("text", ""), font=("Segoe UI", 20, "bold"),
                fill="white", tags=(group, "btn-text")
            )

            img_w = card_w - 2 * btn_margin - 8
            tk_im = load_image(info.get("image", ""), (img_w, img_h))
            self._mode_images.append(tk_im)
            img_x = card_w // 2 + x_offset
            img_y = 8 + btn_margin + btn_h + 16 + img_h // 2 + y_offset
            img_item = canvas.create_image(img_x, img_y, image=tk_im, tags=(group, "thumb"))

            glow_img = make_glow_image(card_w, card_h, glow_color="ivory", blur_radius=10, spread=20, radius=20)
            glow_tk = ImageTk.PhotoImage(glow_img)
            self._mode_images.append(glow_tk)
            glow_item = canvas.create_image(card_w/2 + x_offset, card_h/2 + y_offset, image=glow_tk, tags=(group, "glow"))
            canvas.itemconfigure(glow_item, state="hidden")
            canvas.tag_lower(glow_item, "card")

            dim = round_rect(canvas, 8 + x_offset, 8 + y_offset, card_w - 8 + x_offset, card_h - 8 + y_offset, radius,
                            fill=dim_fill, outline="", width=0, tags=(group, "dim"))
            canvas.itemconfigure(dim, state="hidden")
            canvas.itemconfig(dim, stipple=dim_stipple)
            self._mode_card_dim_layers.append(dim)

            canvas._hovered = False
            canvas._original_bbox = canvas.bbox(group)
            
            def on_enter(_):
                if canvas._hovered:
                    return
                canvas._hovered = True

                canvas.scale(group, card_w/2, card_h/2, scale_factor, scale_factor)
                canvas.move(group, 0, -lift_dy)

                canvas.itemconfigure(glow_item, state="normal")
                canvas.tag_raise(glow_item)
                canvas.tag_raise("card")
                canvas.tag_raise("btn-bg")
                canvas.tag_raise("btn-text")
                canvas.tag_raise("thumb")

                # 只让其他卡片的 dim 显示（使用它们自己 canvas 上的 dim id）
                for j, card in enumerate(self._mode_cards):
                    if card and j != idx:
                        c = card["canvas"]
                        dim_id = card["dim"]
                        c.itemconfigure(dim_id, state="normal")
                        c.tag_raise(dim_id)  # 用 id，而不是 "dim" 标签
                    elif card and j == idx:
                        c = card["canvas"]
                        dim_id = card["dim"]
                        c.itemconfigure(dim_id, state="hidden")
                        
            def on_leave(_):
                if not canvas._hovered:
                    return
                canvas._hovered = False

                canvas.itemconfigure(glow_item, state="hidden")

                # 反向变换 + 绝对复位（防止累计误差）
                canvas.scale(group, card_w/2, card_h/2, 1/scale_factor, 1/scale_factor)

                # 复位坐标（用 bbox 差值）
                x0, y0, x1, y1 = canvas.bbox(group)
                ox0, oy0, _, _ = canvas._original_bbox
                dx = ox0 - x0
                dy = oy0 - y0
                canvas.move(group, dx, dy)

                # 隐藏所有 dim
                for rec in self._mode_cards:
                    if rec:
                        rec["canvas"].itemconfigure(rec["dim"], state="hidden")

            def on_click(_):
                cmd = info.get("command")
                if callable(cmd):
                    cmd()

            canvas.bind("<Enter>", on_enter)
            canvas.bind("<Leave>", on_leave)
            canvas.bind("<Button-1>", on_click)

            return {
                "canvas": canvas,
                "group": group,
                "glow": glow_item,
                "dim": dim,
                "idx": idx,
            }

        self._mode_cards = [None] * len(mode_infos)  # 预分配卡片列表
        for idx, info in enumerate(mode_infos):
            card_record = create_card(idx, info)     # 获取卡片结构
            self._mode_cards[idx] = card_record      # 按照索引精确放入

        return strip
    
    def create_growth_plate(self):
        """创建生长板模式界面"""
        self.create_main_interface("Growth Plate")
        
    def create_articular_cartilage(self):
        """创建关节软骨模式界面"""
        self.create_main_interface("Articular Cartilage")
        
    def create_meniscus(self):
        """创建半月板模式界面"""
        self.create_main_interface("Meniscus")
    
    def create_main_interface(self, selected_mode):
        """创建主界面"""
        # 关闭模式选择窗口
        if self.mode_window and self.mode_window.winfo_exists():
            for widget in self.mode_window.winfo_children():
                widget.destroy()
        
        # 创建主窗口
        self.main_window = self.root
        self.main_window.title(f"Main Interface - {selected_mode}")
        self.main_window.state('zoomed')  # 窗口自动最大化
        self.main_window.config(bg='white')
        
        # 存储当前模式
        self.selected_mode = selected_mode
        
        # 初始化UI元素字典
        self.ui_elements = {}
        
        # 创建基础UI框架（所有模式共用的部分）
        self.create_common_ui()
        
        # 创建顶部工具栏
        toolbar = ttk.Frame(self.main_window, padding=10)
        toolbar.pack(fill='x', pady=(0, 5))
        
        # 模式切换按钮
        switch_btn = ttk.Button(
            toolbar,
            text=f"Switch Mode ({selected_mode})",
            command=self.switch_mode,
            style='Primary.TButton',
            width=30
        )
        switch_btn.pack(side='left', padx=(0, 20))
        
        # 动物类型选择
        animal_frame = ttk.Frame(toolbar)
        animal_frame.pack(side='left', padx=(0, 20))
        
        ttk.Label(animal_frame, text="Animal Model:", font=("Segoe UI", 11)).pack(side='left', padx=(0, 5))
        
        self.animal_var = ttk.StringVar(value="mouse")
        animal_combo = ttk.Combobox(
            animal_frame, 
            textvariable=self.animal_var,
            values=["mouse", "hamster", "rat"],
            state="readonly",
            width=12
        )
        animal_combo.pack(side='left')
        animal_combo.bind("<<ComboboxSelected>>", self.on_animal_changed)
        
        # 根据选择的模式加载特定UI和功能
        if selected_mode == "Growth Plate":
            self.create_growth_plate_ui()
        elif selected_mode == "Articular Cartilage":
            self.create_articular_cartilage_ui()
        elif selected_mode == "Meniscus":
            self.create_meniscus_ui()

        # 初始化动物类型
        self.on_animal_changed()
        
        # 分析按钮
        analyze_btn = ttk.Button(
            toolbar, 
            text="Cellinfo plot", 
            command=self.select_analysis_mode,
            style='Primary.TButton',
            width=12
        )
        analyze_btn.pack(side='left', padx=(0, 20))
        
        # 帮助按钮
        help_btn = ttk.Button(
            toolbar, 
            text="Help", 
            command=self.show_help,
            style='Primary.TButton',
            width=12
        )
        help_btn.pack(side='left')
        
        # 3D比较按钮
        compare_btn = ttk.Button(
            toolbar, 
            text="3D Compare", 
            command=self.compare_two_folders,
            style='Primary.TButton',
            width=12
        )
        compare_btn.pack(side='left', padx=(10, 0))
        
        self.main_window.protocol("WM_DELETE_WINDOW", self.exit_app)
    
    def create_common_ui(self):
        """创建所有UI元素"""
        # 主画布
        self.ui_elements['rectangle'] = ttk.Canvas(self.main_window, width=1012, height=812, bg="black")
        self.ui_elements['rectangle'].create_rectangle(4, 4, 1008, 808, fill="white")
        self.ui_elements['rectangle'].pack()
        
        # 右侧面板1
        self.rectangle2 = ttk.Canvas(self.main_window, width=408, height=380, bg="black")
        self.rectangle2.create_rectangle(4, 4, 404, 376, fill="white")
        self.rectangle2.pack()
        
        # 右侧面板2
        self.ui_elements['rectangle3'] = ttk.Canvas(self.main_window, width=408, height=328, bg="black")
        self.ui_elements['rectangle3'].create_rectangle(4, 4, 404, 324, fill="white")
        self.ui_elements['rectangle3'].pack()
        
        # 右侧LOGO面板
        self.ui_elements['rectangle4'] = ttk.Canvas(self.main_window, width=408, height=145, bg="white", bd=0, highlightthickness=0)
        # 尝试加载LOGO图片，如果失败则显示占位符
        try:
            from PIL import Image
            logo = Image.open("C:/Users/hantang/Pictures/Saved Pictures/BME_Logo_Final_v2.png")
            logo = logo.resize((406, 142))
            self.logo_photo = ImageTk.PhotoImage(logo)
            self.ui_elements['rectangle4'].create_image(1, 1, anchor="nw", image=self.logo_photo)
        except Exception as e:
            print(f"Failed to load BME Logo: {e}")
            # 如果图片不存在，创建占位符
            self.ui_elements['rectangle4'].create_text(203, 72, text="BME LOGO", font=("Arial", 24, "bold"), fill="gray")
        self.ui_elements['rectangle4'].pack()
        
        # 放置主画布
        self.ui_elements['rectangle'].place(relx=0.5, rely=0.5, anchor="center")
        self.rectangle2.place(relx=0.872, rely=0.715, anchor="center")
        self.ui_elements['rectangle3'].place(relx=0.872, rely=0.362, anchor="center")
        self.ui_elements['rectangle4'].place(relx=0.872, rely=0.122, anchor="center")
        
        # 主交互画布
        self.point = ttk.Canvas(self.main_window, width=1000, height=800)
        self.point.create_rectangle(0, 0, 1000, 800, fill="ivory")
        self.point.place(relx=0.5, rely=0.5, anchor="center")
        
        # 标题标签
        self.labelframe = ttk.Frame(self.main_window, width=340, height=50)
        self.labelframe.place(relx=0.325, rely=0.1, anchor="center")
        self.ui_elements['winlabel'] = ttk.Label(self.main_window, text="Segmentation Workspace", 
                                                font=('Segoe UI', 20, "bold"), bootstyle="inverse-primary")
        self.ui_elements['winlabel'].place(relx=0.325, rely=0.1, anchor="center")
        
        # 绑定鼠标事件
        self.point.bind("<Button-1>", self.on_click)
        self.point.bind("<Button-3>", self.on_right_click)
        self.point.bind("<B1-Motion>", self.on_mouse_move)
        self.point.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # 底部功能框架
        self.functionframe = ttk.Frame(self.main_window, width=1012, height=80)
        self.functionframe.place(relx=0.5, rely=0.94, anchor="center")
        
        # 链接框架
        self.linkframe = ttk.Frame(self.main_window, width=408, height=75)
        self.linkframe.place(relx=0.872, rely=0.942, anchor="center")
        # 创建按钮图片（使用占位符）
        self.create_button_images()
        
        # 撤销按钮
        self.btn_undo = ttk.Button(self.main_window, image=self.undo_img, 
                                                 command=self.undo_action, style='Third.TButton')
        self.btn_undo.place(relx=0.45, rely=0.85, anchor="center")
        
        # 重做按钮
        self.btn_redo = ttk.Button(self.main_window, image=self.redo_img, 
                                                 command=self.redo_action, style='Third.TButton')
        self.btn_redo.place(relx=0.55, rely=0.85, anchor="center")
        
        # 画笔按钮
        self.draw_btn = ttk.Button(self.main_window, image=self.brush_img, 
                                     command=self.toggle_draw_mode, style='draw.TButton')
        self.draw_btn.place(relx=0.69, rely=0.15, anchor="center")
        
        # 背景调整按钮
        self.blur_btn = ttk.Button(self.main_window, text="Adjust BG", 
                                  command=self.apply_brightness_adjustment, style="blur.TButton")
        self.blur_btn.place(relx=0.73, rely=0.15, anchor="center")
        
        # 确认按钮
        self.btn_ok = ttk.Button(self.main_window, image=self.success_img, 
                                command=self.next_mask, style='Four.TButton')
        self.btn_ok.place(relx=0.83, rely=0.868, anchor="center")
        
        # 拒绝按钮
        self.btn_not = ttk.Button(self.main_window, image=self.fail_img, 
                                 command=self.show_on_point, style='Four.TButton')
        self.btn_not.place(relx=0.914, rely=0.868, anchor="center")
        if not hasattr(self, 'viewer') or not self.viewer:
            self.viewer = ThreeDViewer_VTK(self.main_window, width=400, height=320)
            self.viewer.place(relx=0.872, rely=0.362, anchor="center")
        
        # 恢复三维可视化状态（如果存在）
        if hasattr(self, 'saved_viewer_state') and self.saved_viewer_state:
            self.viewer.ac_dir = self.saved_viewer_state.get('ac_dir')
        # GitHub链接按钮
        self.btn_git = ttk.Button(self.main_window, image=self.git_img, 
                                 command=self.open_github, style='Four.TButton')
        self.btn_git.place(relx=0.843, rely=0.942, anchor="center")
        
        self.btn_git2 = ttk.Button(self.main_window, image=self.git2_img, 
                                  command=self.open_github2, style='Four.TButton')
        self.btn_git2.place(relx=0.93, rely=0.942, anchor="center")
        
        # 链接标签
        self.link = ttk.Label(self.linkframe, text="LINK", font=("Segoe UI", 14, "bold"))
        self.link.place(relx=0.08, rely=0.5, anchor="center")
        

        
        # 主要功能按钮
        self.ui_elements['button'] = ttk.Button(
            self.main_window, text="Open Image", command=self.open_image, 
            style='Secondary.TButton', width=15)
        self.ui_elements['button'].place(relx=0.35, rely=0.94, anchor="center")
        
        self.ui_elements['button2'] = ttk.Button(
            self.main_window, text="Start SAM", command=self.start_sam, 
            style='Secondary.TButton', width=15)
        self.ui_elements['button2'].place(relx=0.5, rely=0.94, anchor="center")
        
        self.ui_elements['button3'] = ttk.Button(
            self.main_window, text="Cellpose", command=self.startcellpose, 
            style='Secondary.TButton', width=15)
        self.ui_elements['button3'].place(relx=0.65, rely=0.94, anchor="center")
        
        # 绑定canvas事件
        self.bind_canvas_events()
    
    def create_button_images(self):
        """创建按钮图片"""
        from PIL import Image
        
        # 定义图像路径
        image_paths = {
            'undo': "C:/Users/hantang/Pictures/Saved Pictures/undo2.png",
            'redo': "C:/Users/hantang/Pictures/Saved Pictures/redo2.png",
            'brush': "C:/Users/hantang/Pictures/Saved Pictures/brush.png",
            'success': "C:/Users/hantang/Pictures/Saved Pictures/Success.png",
            'fail': "C:/Users/hantang/Pictures/Saved Pictures/fail.png",
            'git': "C:/Users/hantang/Pictures/Saved Pictures/git.png",
            'git2': "C:/Users/hantang/Pictures/Saved Pictures/git2.png",
            'pen': "C:/Users/hantang/Pictures/Saved Pictures/pen.png",
            'eraser': "C:/Users/hantang/Pictures/Saved Pictures/eraser.png"
        }
        
        # 加载图像
        for name, path in image_paths.items():
            try:
                img = Image.open(path)
                if name in ['undo', 'redo', 'brush', 'success', 'fail', 'pen', 'eraser']:
                    img = img.resize((35, 35))
                elif name in ['git', 'git2']:
                    img = img.resize((134, 55))
                setattr(self, f'{name}_img', ImageTk.PhotoImage(img))
            except Exception as e:
                print(f"Failed to load image {path}: {e}")
                # 创建占位符图像
                if name in ['undo', 'redo', 'brush', 'success', 'fail', 'pen', 'eraser']:
                    placeholder = Image.new('RGBA', (35, 35), (200, 200, 200, 255))
                else:
                    placeholder = Image.new('RGBA', (134, 55), (200, 200, 200, 255))
                setattr(self, f'{name}_img', ImageTk.PhotoImage(placeholder))
    
    def create_growth_plate_ui(self):
        """创建生长板模式的UI"""
        try:
            from PIL import Image
            logo = Image.open("C:/Users/hantang/Pictures/Saved Pictures/LOGO.jpg")
            logo = logo.resize((400, 325))
            self.LOGO_photo = ImageTk.PhotoImage(logo)
            self.ui_elements['rectangle5a'] = ttk.Canvas(self.main_window, width=400, height=325, 
                                                        bg="white", bd=0, highlightthickness=0)
            self.ui_elements['rectangle5a'].create_image(0, 0, anchor="nw", image=self.LOGO_photo)
            self.ui_elements['rectangle5a'].pack()
            self.ui_elements['rectangle5a'].place(relx=0.12, rely=0.5, anchor="center")
        except Exception as e:
            print(f"Failed to load Growth Plate Logo: {e}")
            # 如果图片不存在，创建占位符
            placeholder = ttk.Canvas(self.main_window, width=400, height=325, bg="white", bd=1, relief="solid")
            placeholder.create_text(200, 162, text="Growth Plate\nLOGO", font=("Arial", 24, "bold"), fill="gray")
            placeholder.place(relx=0.12, rely=0.5, anchor="center")
    
    def create_articular_cartilage_ui(self):
        """创建关节软骨模式的UI"""
        try:
            from PIL import Image
            logo = Image.open("C:/Users/hantang/Pictures/Saved Pictures/LOGO2.jpg")
            logo = logo.resize((400, 256))
            self.LOGO_photo2 = ImageTk.PhotoImage(logo)
            self.ui_elements['rectangle5b'] = ttk.Canvas(self.main_window, width=400, height=256, 
                                                        bg="white", bd=0, highlightthickness=0)
            self.ui_elements['rectangle5b'].create_image(0, 0, anchor="nw", image=self.LOGO_photo2)
            self.ui_elements['rectangle5b'].pack()
            self.ui_elements['rectangle5b'].place(relx=0.12, rely=0.5, anchor="center")
        except Exception as e:
            print(f"Failed to load Articular Cartilage Logo: {e}")
            # 如果图片不存在，创建占位符
            placeholder = ttk.Canvas(self.main_window, width=400, height=256, bg="white", bd=1, relief="solid")
            placeholder.create_text(200, 128, text="Articular Cartilage\nLOGO", font=("Arial", 24, "bold"), fill="gray")
            placeholder.place(relx=0.12, rely=0.5, anchor="center")
    
    def create_meniscus_ui(self):
        """创建半月板模式的UI"""
        try:
            from PIL import Image
            logo = Image.open("C:/Users/hantang/Pictures/Saved Pictures/LOGO3.jpg")
            logo = logo.resize((400, 256))
            self.LOGO_photo3 = ImageTk.PhotoImage(logo)
            self.ui_elements['rectangle5c'] = ttk.Canvas(self.main_window, width=400, height=236, 
                                                        bg="white", bd=0, highlightthickness=0)
            self.ui_elements['rectangle5c'].create_image(0, 0, anchor="nw", image=self.LOGO_photo3)
            self.ui_elements['rectangle5c'].pack()
            self.ui_elements['rectangle5c'].place(relx=0.12, rely=0.5, anchor="center")
        except Exception as e:
            print(f"Failed to load Meniscus Logo: {e}")
            # 如果图片不存在，创建占位符
            placeholder = ttk.Canvas(self.main_window, width=400, height=236, bg="white", bd=1, relief="solid")
            placeholder.create_text(200, 118, text="Meniscus\nLOGO", font=("Arial", 24, "bold"), fill="gray")
            placeholder.place(relx=0.12, rely=0.5, anchor="center")
    
    def _canvas_to_image_coords(self, x, y):
        canvas_width = 1000
        canvas_height = 800
        if self.selected_animal == "rat":
            scaled_width = 5500 * self.functionality.ratio
            scaled_height = 2970 * self.functionality.ratio
        elif self.selected_animal == "hamster":
            scaled_width = 4500 * self.functionality.ratio
            scaled_height = 2250 * self.functionality.ratio
        else:
            scaled_width = 2500 * self.functionality.ratio
            scaled_height = 1250 * self.functionality.ratio
        image_x0 = (canvas_width - scaled_width) // 2
        image_y0 = (canvas_height - scaled_height) // 2
        scaled_x = x - image_x0
        scaled_y = y - image_y0
        original_x = int(scaled_x / self.functionality.ratio)
        original_y = int(scaled_y / self.functionality.ratio)
        return original_x, original_y

    def on_click(self, event):
        try:
            # 检查绘制模式状态
            if hasattr(self, 'functionality') and hasattr(self.functionality, 'draw_mode'):
                draw_mode = self.functionality.draw_mode
            else:
                draw_mode = getattr(self, 'draw_mode', False)
            if hasattr(self, 'functionality') and hasattr(self.functionality, 'SAMbanned'): 
                SAMbanned = self.functionality.SAMbanned
            else:
                SAMbanned = getattr(self, 'SAMbanned', False)
            if draw_mode:
                # ROI多边形绘制
                if hasattr(self, 'functionality'):
                    self.functionality.on_click(event)
            elif SAMbanned:
                pass
            else:
                # 前景点
                x, y = self._canvas_to_image_coords(event.x, event.y)
                self.functionality.points.append([x, y])
                self.functionality.labels.append(1)
                self.functionality.update_segmentation_display()
                # 画黑点+十字
                circle = self.point.create_oval(event.x-8, event.y-8, event.x+8, event.y+8, fill="black", outline="white", width=2, tags="circles")
                horizontal_line = self.point.create_line(event.x - 5, event.y, event.x + 5, event.y, fill="white", width=2, tags="circles")
                vertical_line = self.point.create_line(event.x, event.y - 5, event.x, event.y + 5, fill="white", width=2, tags="circles")
                if not hasattr(self, 'annotations'):
                    self.annotations = []
                self.annotations.append((circle, horizontal_line, vertical_line))
                # 与后端的注释列表保持同步（用于撤销时正确删除）
                if hasattr(self, 'functionality') and hasattr(self.functionality, 'annotations'):
                    self.functionality.annotations.append((circle, horizontal_line, vertical_line))
                self.point.tag_raise("circles")
        except Exception as e:
            print(f"Click event error: {e}")

    def on_right_click(self, event):
        try:
            # 检查绘制模式状态
            if hasattr(self, 'functionality') and hasattr(self.functionality, 'draw_mode'):
                draw_mode = self.functionality.draw_mode
            else:
                draw_mode = getattr(self, 'draw_mode', False)
            if hasattr(self, 'functionality') and hasattr(self.functionality, 'SAMbanned'): 
                SAMbanned = self.functionality.SAMbanned
            else:
                SAMbanned = getattr(self, 'SAMbanned', False)
            if draw_mode:
                # 取消ROI
                if hasattr(self, 'functionality'):
                    self.functionality.delete_roi()
            elif SAMbanned:
                pass
            else:
                # 背景点
                x, y = self._canvas_to_image_coords(event.x, event.y)
                self.functionality.points.append([x, y])
                self.functionality.labels.append(0)
                self.functionality.update_segmentation_display()
                # 画红点+十字
                circle = self.point.create_oval(event.x-8, event.y-8, event.x+8, event.y+8, fill="red", outline="white", width=2, tags="circles")
                horizontal_line = self.point.create_line(event.x - 5, event.y, event.x + 5, event.y, fill="white", width=2, tags="circles")
                if not hasattr(self, 'annotations'):
                    self.annotations = []
                self.annotations.append((circle, horizontal_line))
                # 与后端的注释列表保持同步（用于撤销时正确删除）
                if hasattr(self, 'functionality') and hasattr(self.functionality, 'annotations'):
                    self.functionality.annotations.append((circle, horizontal_line))
                self.point.tag_raise("circles")
        except Exception as e:
            print(f"Right-click event error: {e}")

    def on_mouse_move(self, event):
        # 检查绘制模式状态
        if hasattr(self, 'functionality') and hasattr(self.functionality, 'draw_mode'):
            draw_mode = self.functionality.draw_mode
        else:
            draw_mode = getattr(self, 'draw_mode', False)
        
        if draw_mode and hasattr(self, 'functionality'):
            self.functionality.on_mouse_move(event)

    def on_mouse_up(self, event):
        # 检查绘制模式状态
        if hasattr(self, 'functionality') and hasattr(self.functionality, 'draw_mode'):
            draw_mode = self.functionality.draw_mode
        else:
            draw_mode = getattr(self, 'draw_mode', False)
        
        if draw_mode and hasattr(self, 'functionality'):
            self.functionality.on_mouse_up(event)

    def bind_canvas_events(self):
        self.point.bind("<Button-1>", self.on_click)
        self.point.bind("<Button-3>", self.on_right_click)
        self.point.bind("<B1-Motion>", self.on_mouse_move)
        self.point.bind("<ButtonRelease-1>", self.on_mouse_up)

    # 在create_common_ui最后调用self.bind_canvas_events()
    
    def undo_action(self):
        """撤销操作"""
        if hasattr(self, 'functionality'):
            self.functionality.undo_action()
        else:
            self.show_info_message("功能", "撤销操作")
    
    def redo_action(self):
        """重做操作"""
        if hasattr(self, 'functionality'):
            self.functionality.redo_action()
        else:
            self.show_info_message("功能", "重做操作")
    
    def toggle_draw_mode(self):
        """切换绘制模式"""
        if hasattr(self, 'functionality'):
            # 先调用functionality的方法
            self.functionality.toggle_draw_mode()
            # 同步状态
            self.draw_mode = self.functionality.draw_mode
            print(f"Draw mode: {'ON' if self.draw_mode else 'OFF'}")
        else:
            self.show_info_message("功能", "切换绘制模式")
    
    def apply_brightness_adjustment(self):
        """应用亮度调整"""
        if hasattr(self, 'functionality'):
            self.functionality.apply_brightness_adjustment()
        else:
            self.show_info_message("功能", "调整背景亮度")
    
    def next_mask(self):
        """下一个掩码"""
        if hasattr(self, 'functionality'):
            self.functionality.next_mask()
        else:
            self.show_info_message("功能", "下一个掩码")
    
    def show_on_point(self):
        """显示点"""
        if hasattr(self, 'functionality'):
            self.functionality.show_on_point()
        else:
            self.show_info_message("功能", "显示点")
    
    def open_github(self):
        """打开GitHub链接"""
        import webbrowser
        webbrowser.open("https://github.com/Tangtang2001")
    
    def open_github2(self):
        """打开GitHub2链接"""
        import webbrowser
        webbrowser.open("https://github.com/wenchunyi-polyubme")
    
    def open_image(self):
        """打开图像"""
        if hasattr(self, 'functionality'):
            self.functionality.open_image()
        else:
            self.show_info_message("功能", "Open Image 功能")
    
    def start_sam(self):
        """启动SAM"""
        if hasattr(self, 'functionality'):
            self.functionality.start_sam()
        else:
            self.show_info_message("功能", "Start SAM 功能")
    
    def startcellpose(self):
        """启动Cellpose或直接进行Post_analysis"""
        if hasattr(self, 'functionality'):
            # 检查 cellpose/post_analysis 状态
            func = self.functionality
            if self.selected_mode == "Growth Plate":
                input_path = os.path.join(func.output_dir, "GP")
            elif self.selected_mode == "Articular Cartilage":
                input_path = os.path.join(func.output_dir, "AC")
            else:
                input_path = os.path.join(func.output_dir, "Meniscus")
            input_path = input_path.replace("\\", "/")
            if getattr(func, 'cellpose_ready', False) and not getattr(func, 'post_analysis_ready', False):
                # 已分割未分析，弹窗提示
                if messagebox.askokcancel("Cell Orientation Distribution Analysis", "Cellpose segmentation detected. You can now run Cell Orientation Distribution Analysis. Proceed?"):
                    func.run_cellpose_post_analysis(target_dir=input_path, max_wait_per_file=180)
            else:
                func.startcellpose()
        else:
            self.show_info_message("功能", "Start SAM 功能")
    
    def compare_two_folders(self):
        """比较两个文件夹"""
        if hasattr(self, 'functionality'):
            self.functionality.compare_two_folders()
        else:
            self.show_info_message("功能", "3D Compare 功能")
    
    def switch_mode(self):
        """切换模式"""
        # 关闭当前主窗口
        if hasattr(self, 'viewer') and self.viewer:
            try:
                # 保存相机位置等信息（如果需要）
                self.saved_viewer_state = {
                    'ac_dir': self.viewer.ac_dir,
                    # 可以添加其他需要保存的状态
                }
                # 清理但不销毁viewer
                self.viewer.clear_3d_data()
            except Exception as e:
                print(f"Failed to save 3D visualization state: {e}")
        
        # 清除主界面内容
        for widget in self.main_window.winfo_children():
            if widget != self.viewer.frame:  # 不要销毁viewer的frame
                widget.destroy()
        # 重新创建模式选择界面
        self.create_mode_selection()
    
    def on_animal_changed(self, event=None):
        """当动物类型改变时的处理"""
        animal = self.animal_var.get() if event else "mouse"
        self.selected_animal = animal
        self.load_animal_specific_library()
    
    def load_animal_specific_library(self):
        """加载动物特定的工具库"""
        # 根据模式和动物类型确定要导入的库
        if self.selected_mode == "Growth Plate":
            if self.selected_animal == "mouse":
                lib_name = "GP_mouse_utils"
            elif self.selected_animal == "hamster":
                lib_name = "GP_hamster_utils"
            else:  # rat
                lib_name = "GP_rat_utils"
        
        elif self.selected_mode == "Articular Cartilage":
            if self.selected_animal == "mouse":
                lib_name = "AC_mouse_utils"
            elif self.selected_animal == "hamster":
                lib_name = "AC_hamster_utils"
            else:  # rat
                lib_name = "AC_rat_utils"
        
        else:  # Meniscus
            if self.selected_animal == "mouse":
                lib_name = "MN_mouse_utils"
            elif self.selected_animal == "hamster":
                lib_name = "MN_hamster_utils"
            else:  # rat
                lib_name = "MN_rat_utils"
        
        # 动态导入库
        try:
            self.current_utils = importlib.import_module(lib_name)
            print(f"Loading library for {self.selected_animal} {self.selected_mode}")
        except ImportError as e:
            print(f"Error loading library {lib_name}: {e}")
            # 回退到默认库
            self.current_utils = importlib.import_module("default_utils")
        # 检查库是否实现了必要的接口
        required_functions = ["show_mask", "barprogress", "calculate_distance", "run_ijm_macro", "process_single_file", "Drag"]
        for func in required_functions:
            if not hasattr(self.current_utils, func):
                raise AttributeError(f"Library {lib_name} does not implement required function: {func}")
        
        # 将 current_utils 传递给 FunctionalityManager
        if hasattr(self, 'functionality'):
            self.functionality.current_utils = self.current_utils
    
    def select_analysis_mode(self):
        """选择分析模式"""
        if hasattr(self, 'functionality'):
            self.functionality.select_analysis_mode()
        else:
            self.show_info_message("分析模式", "Cellinfo plot 功能")
    
    def show_help(self):
        """显示帮助信息"""
        help_message = (
            "Welcome to the Analysis Tool!\n\n"
            "1. Select a mode (Growth Plate, Articular Cartilage, Meniscus) to start.\n"
            "2. Choose an animal model (mouse, hamster, rat) from the dropdown.\n"
            "3. Click 'Cellinfo plot' to analyze cell parameters.\n"
            "4. Use the buttons to open images, start SAM, or run Cellpose.\n"
            "5. For detailed instructions, refer to the user manual."
        )
        self.show_info_message("Help", help_message)
    
    def round_rect(self, canvas, x1, y1, x2, y2, radius, **kwargs):
        """绘制圆角矩形"""
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return canvas.create_polygon(
            points, smooth=True, splinesteps=72, **kwargs
        )
    
    def create_comparison_window(self, title="3D Comparison"):
        """创建比较窗口"""
        win = ttk.Toplevel(self.main_window)
        win.geometry("1500x800")
        win.title(title)
        
        # 创建左右两个frame用于3D可视化
        frame_left = ttk.Frame(win)
        frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        frame_right = ttk.Frame(win)
        frame_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建图例frame
        frame_legend = ttk.Frame(win, width=80)
        frame_legend.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        return win, frame_left, frame_right, frame_legend
    
    def create_colorbar_legend(self, frame_legend, fig):
        """创建颜色条图例"""
        from PIL import Image
        import numpy as np
        
        # 将matplotlib图形转换为PhotoImage
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        img = Image.fromarray(img, mode='RGBA').convert('RGB')
        legend_img = ImageTk.PhotoImage(img)

        label = tk.Label(frame_legend, image=legend_img)
        label.image = legend_img
        label.pack(fill=tk.BOTH, expand=True)
        
        return label
    
    def show_error_message(self, title, message):
        """显示错误消息"""
        messagebox.showerror(title, message)
    
    def show_info_message(self, title, message):
        """显示信息消息"""
        messagebox.showinfo(title, message)
    
    def ask_directory(self, title):
        """选择目录对话框"""
        return filedialog.askdirectory(title=title)
    
    def ask_file(self, title, filetypes):
        """选择文件对话框"""
        return filedialog.askopenfilename(title=title, filetypes=filetypes)
    
    def bind_window_close(self, window, callback):
        """绑定窗口关闭事件"""
        window.protocol("WM_DELETE_WINDOW", callback)
    
    def cleanup_window(self, window):
        """清理窗口资源"""
        try:
            # 清理matplotlib图形
            if hasattr(window, '_fig') and window._fig is not None:
                plt.close(window._fig)
                window._fig = None
        except Exception as e:
            print(f"Error when cleaning up windows: {e}")
        finally:
            try:
                window.destroy()
            except Exception:
                pass
