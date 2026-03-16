# -*- coding: utf-8 -*-
"""
PyQt 版本的 UI 管理器（ui_manager2）
- 在不修改现有 Tk/ttkbootstrap 版本的前提下，复刻当前界面的布局与视觉效果。
- 尽量 1:1 复刻：模式选择卡片的悬浮发光与缩放、主界面工具栏、右侧面板、主画布与按钮等。
- 仅提供 UI 结构与交互占位，后续可逐步对接 backend 功能。

依赖:
- PyQt5 (推荐) 或 PyQt6（稍作替换即可）
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Optional

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QHBoxLayout, QVBoxLayout, QGridLayout, QComboBox, QFileDialog, QMessageBox,
    QGraphicsDropShadowEffect, QStackedWidget, QToolButton
)


# -------------------------- 工具与常量 --------------------------
# 尽量贴近 ttkbootstrap 的 superhero 主题配色
THEME = {
    "primary": "#df691a",      # 橙色（primary）
    "info": "#5bc0de",         # 青色（info）
    "success": "#5cb85c",
    "danger": "#d9534f",
    "warning": "#f0ad4e",
    "dark": "#2b3e50",
    "light": "#f7f7f7",
    "panel_border": "#e8e8e8",
}

CARD_FILL = QtGui.QColor("#0e2841")
CARD_OUTLINE = QtGui.QColor("ivory")
BTN_COLOR = QtGui.QColor("#4e7ca1")
BTN_OUTLINE = QtGui.QColor("ivory")
BG_CANVAS = QtGui.QColor("ivory")
BG_WHITE = QtGui.QColor("white")
BG_DARK = QtGui.QColor(THEME["dark"])  # 深色背景

TITLE_FONT = QFont("Segoe UI", 36, QFont.Bold)
LABEL_FONT = QFont("Segoe UI", 20, QFont.Bold)
BTN_FONT_PRIMARY = QFont("Segoe UI", 12, QFont.Bold)
BTN_FONT_SECONDARY = QFont("Segoe UI", 18, QFont.Bold)


def load_icon(path: str, size: Optional[QtCore.QSize] = None) -> QIcon:
    try:
        pm = QPixmap(path)
        if not pm or pm.isNull():
            raise FileNotFoundError(path)
        if size is not None:
            pm = pm.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QIcon(pm)
    except Exception:
        # 占位符
        pm = QPixmap(64, 64)
        pm.fill(QtGui.QColor(200, 200, 200))
        return QIcon(pm)


def load_pixmap(path: str, size: Optional[QtCore.QSize] = None) -> QPixmap:
    try:
        pm = QPixmap(path)
        if not pm or pm.isNull():
            raise FileNotFoundError(path)
        if size is not None:
            pm = pm.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pm
    except Exception:
        pm = QPixmap(size or QtCore.QSize(200, 120))
        pm.fill(QtGui.QColor(90, 90, 90))
        return pm


# -------------------------- 模式卡片组件 --------------------------
class GlowCard(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal()

    def __init__(self, title: str, image_path: str, w: int = 430, h: int = 530, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("GlowCard")
        self.setFixedSize(w + 40, h + 40)
        self.setStyleSheet("background: #2b3e50; border: 0px;")

        # 内部卡片容器
        card = QFrame(self)
        card.setObjectName("card")
        card.setGeometry(16, 16, w, h)
        card.setStyleSheet(f"#card {{ background: {CARD_FILL.name()}; border: 4px solid {CARD_OUTLINE.name()}; border-radius: 20px; }}")

        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(18, 18, 18, 18)
        vbox.setSpacing(12)

        # 按钮头条
        btn = QFrame(card)
        btn.setObjectName("btn")
        btn.setFixedHeight(70)
        btn.setStyleSheet(f"#btn {{ background: {BTN_COLOR.name()}; border: 3px solid {BTN_OUTLINE.name()}; border-radius: 16px; }}")
        btn_layout = QHBoxLayout(btn)
        btn_layout.setContentsMargins(8, 8, 8, 8)
        btn_label = QLabel(title)
        btn_label.setAlignment(Qt.AlignCenter)
        btn_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        btn_label.setStyleSheet("color: white;")
        btn_layout.addWidget(btn_label)

        # 图片
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        pm = load_pixmap(image_path, QtCore.QSize(w - 2 * 26, 350))
        img_label.setPixmap(pm)

        vbox.addWidget(btn)
        vbox.addWidget(img_label, 1)

        # Glow + Shadow
        glow = QGraphicsDropShadowEffect(self)
        glow.setBlurRadius(32)
        glow.setColor(QtGui.QColor("ivory"))
        glow.setOffset(0, 0)
        card.setGraphicsEffect(glow)
        card.setProperty("_glow", glow)
        glow.setEnabled(False)

        # 动画：悬停时轻微缩放
        self._scale_anim = QtCore.QPropertyAnimation(card, b"geometry", self)
        self._scale_anim.setDuration(120)
        self._card_rect = card.geometry()
        self._hover_rect = QtCore.QRect(self._card_rect)
        self._hover_rect.adjust(-8, -8, 8, 8)

    def enterEvent(self, e: QtCore.QEvent) -> None:
        card = self.findChild(QFrame, "card")
        glow: QGraphicsDropShadowEffect = card.property("_glow")
        if glow:
            glow.setEnabled(True)
        self._scale_anim.stop()
        self._scale_anim.setStartValue(card.geometry())
        self._scale_anim.setEndValue(self._hover_rect)
        self._scale_anim.start()
        return super().enterEvent(e)

    def leaveEvent(self, e: QtCore.QEvent) -> None:
        card = self.findChild(QFrame, "card")
        glow: QGraphicsDropShadowEffect = card.property("_glow")
        if glow:
            glow.setEnabled(False)
        self._scale_anim.stop()
        self._scale_anim.setStartValue(card.geometry())
        self._scale_anim.setEndValue(self._card_rect)
        self._scale_anim.start()
        return super().leaveEvent(e)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        return super().mousePressEvent(e)


# -------------------------- 模式选择窗口 --------------------------
class ModeSelectionWindow(QWidget):
    def __init__(self, on_select: Callable[[str], None], parent: QWidget | None = None):
        super().__init__(parent)
        self.on_select = on_select
        self.setWindowTitle("Select Mode")
        self.setMinimumSize(1300, 800)
        self.setStyleSheet("background: white;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 标题
        title = QLabel("Select Analysis Mode")
        title.setFont(TITLE_FONT)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #0e6aba; margin-top: 36px; margin-bottom: 24px;")
        root.addWidget(title)

        # 三张卡片
        cards = QHBoxLayout()
        cards.setContentsMargins(32, 16, 32, 16)
        cards.setSpacing(10)
        root.addLayout(cards)

        card_infos = [
            ("Growth Plate", "C:/Users/hantang/Pictures/Saved Pictures/GP.png", "Growth Plate"),
            ("Articular Cartilage", "C:/Users/hantang/Pictures/Saved Pictures/AC.png", "Articular Cartilage"),
            ("Meniscus", "C:/Users/hantang/Pictures/Saved Pictures/Menis.png", "Meniscus"),
        ]

        for title_text, img_path, mode_value in card_infos:
            card = GlowCard(title_text, img_path, w=430, h=530)
            card.clicked.connect(lambda _=False, m=mode_value: self.on_select(m))
            cards.addWidget(card)

        # 退出按钮
        btn_exit = QPushButton("EXIT")
        btn_exit.setFont(BTN_FONT_PRIMARY)
        btn_exit.setStyleSheet("background: red; color: white; padding: 8px 24px; border: none; border-radius: 6px;")
        btn_exit.clicked.connect(self.close)
        box_exit = QHBoxLayout()
        box_exit.addStretch(1)
        box_exit.addWidget(btn_exit)
        box_exit.addStretch(1)
        root.addLayout(box_exit)
        root.addSpacing(12)


# -------------------------- 主窗口 --------------------------
class MainWindow(QMainWindow):
    def __init__(self, selected_mode: str):
        super().__init__()
        self.selected_mode = selected_mode
        self.setWindowTitle(f"Main Interface - {selected_mode}")
        self.showMaximized()

        # 中央根容器：顶部工具栏 + 舞台区域（承载所有绝对定位部件）
        central = QWidget(self)
        self.setCentralWidget(central)
        self._outer = QVBoxLayout(central)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)

        # 顶部工具栏
        self._toolbar_wrap = QWidget()
        toolbar = QHBoxLayout(self._toolbar_wrap)
        toolbar.setContentsMargins(12, 10, 12, 10)
        toolbar.setSpacing(10)

        btn_switch = QPushButton(f"Switch Mode ({selected_mode})")
        btn_switch.setFont(BTN_FONT_PRIMARY)
        btn_switch.setStyleSheet(
            f"background: {THEME['info']}; color: white; padding: 8px 16px; border-radius: 6px;"
        )
        btn_switch.clicked.connect(self.on_switch_mode)
        toolbar.addWidget(btn_switch)

        # Animal Model
        box_animal = QHBoxLayout()
        lab_animal = QLabel("Animal Model:")
        lab_animal.setFont(QFont("Segoe UI", 11))
        cmb_animal = QComboBox()
        cmb_animal.addItems(["mouse", "hamster", "rat"])
        cmb_animal.setCurrentText("mouse")
        box_animal.addWidget(lab_animal)
        box_animal.addWidget(cmb_animal)
        animal_wrap = QWidget()
        animal_wrap.setLayout(box_animal)
        toolbar.addWidget(animal_wrap)

        # 分析按钮
        btn_analyze = QPushButton("Cellinfo plot")
        btn_analyze.setFont(BTN_FONT_PRIMARY)
        btn_analyze.setStyleSheet(
            f"background: {THEME['info']}; color: white; padding: 8px 16px; border-radius: 6px;"
        )
        toolbar.addWidget(btn_analyze)

        btn_help = QPushButton("Help")
        btn_help.setFont(BTN_FONT_PRIMARY)
        btn_help.setStyleSheet(
            f"background: {THEME['info']}; color: white; padding: 8px 16px; border-radius: 6px;"
        )
        btn_help.clicked.connect(self.on_help)
        toolbar.addWidget(btn_help)

        btn_compare = QPushButton("3D Compare")
        btn_compare.setFont(BTN_FONT_PRIMARY)
        btn_compare.setStyleSheet(
            f"background: {THEME['info']}; color: white; padding: 8px 16px; border-radius: 6px;"
        )
        toolbar.addWidget(btn_compare)

        toolbar.addStretch(1)
        self._outer.addWidget(self._toolbar_wrap)

        # 舞台容器：使用可滚动区域以避免内容固定尺寸强制窗口最小尺寸
        self._stage = QWidget()
        self._stage.setStyleSheet("background: white;")
        # 不设置过高的最小尺寸，避免强制窗口扩大到内容大小
        # self._stage.setMinimumHeight(700)
        # self._stage.setMinimumWidth(1200)
        # 将舞台放入 QScrollArea，使窗口可以比舞台小并显示滚动条
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self._stage)
        self._outer.addWidget(scroll, 1)

        # 维护相对定位的对象列表
        self._rel_items = []  # list of dict(widget, size, relx, rely)

        def place_rel(widget: QWidget, size: QtCore.QSize, relx: float, rely: float):
            widget.setParent(self._stage)
            widget.setFixedSize(size)
            self._rel_items.append({
                "w": widget, "size": size, "relx": relx, "rely": rely
            })
            widget.show()
            self._reflow_rel()

        def load_btn_icon(name, size=(35, 35)):
            paths = {
                'undo': "C:/Users/hantang/Pictures/Saved Pictures/undo2.png",
                'redo': "C:/Users/hantang/Pictures/Saved Pictures/redo2.png",
                'brush': "C:/Users/hantang/Pictures/Saved Pictures/brush.png",
                'success': "C:/Users/hantang/Pictures/Saved Pictures/Success.png",
                'fail': "C:/Users/hantang/Pictures/Saved Pictures/fail.png",
                'git': "C:/Users/hantang/Pictures/Saved Pictures/git.png",
                'git2': "C:/Users/hantang/Pictures/Saved Pictures/git2.png",
                'pen': "C:/Users/hantang/Pictures/Saved Pictures/pen.png",
                'eraser': "C:/Users/hantang/Pictures/Saved Pictures/eraser.png",
            }
        # ========== 舞台内控件：严格按 Tk 坐标放置 ==========
        # 标题标签（inverse-primary） 340x50 @ (0.325, 0.1)
        self._title_lab = QLabel("Segmentation Workspace")
        self._title_lab.setAlignment(Qt.AlignCenter)
        self._title_lab.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self._title_lab.setStyleSheet(
            f"background:{THEME['primary']}; color:white; border-radius:4px;"
        )
        place_rel(self._title_lab, QtCore.QSize(340, 50), 0.325, 0.10)

        # 主交互画布（1000x800，ivory 背景） @ (0.5, 0.5)
        self._canvas = QLabel()
        self._canvas.setStyleSheet("background: ivory; border: 1px solid #ccc;")
        self._canvas.setAlignment(Qt.AlignCenter)
        place_rel(self._canvas, QtCore.QSize(1000, 800), 0.5, 0.5)

        # 右侧三块白面板：
        def make_panel(size: QtCore.QSize) -> QFrame:
            panel = QFrame()
            panel.setStyleSheet(
                f"background: white; border: 1px solid {THEME['panel_border']};"
            )
            panel.setFixedSize(size)
            return panel

        # LOGO 面板 408x145 @ (0.872, 0.122)
        self._panel_logo = make_panel(QtCore.QSize(408, 145))
        # 在面板内添加 logo
        _logo = QLabel(self._panel_logo)
        _logo.setGeometry(1, 1, 406, 142)
        _logo_pm = load_pixmap("C:/Users/hantang/Pictures/Saved Pictures/BME_Logo_Final_v2.png", QtCore.QSize(406, 142))
        _logo.setPixmap(_logo_pm)
        _logo.setAlignment(Qt.AlignCenter)
        place_rel(self._panel_logo, QtCore.QSize(408, 145), 0.872, 0.122)

        # 中间面板 408x328 @ (0.872, 0.362)
        self._panel_mid = make_panel(QtCore.QSize(408, 328))
        place_rel(self._panel_mid, QtCore.QSize(408, 328), 0.872, 0.362)

        # 底部面板 408x380 @ (0.872, 0.715)
        self._panel_bottom = make_panel(QtCore.QSize(408, 380))
        place_rel(self._panel_bottom, QtCore.QSize(408, 380), 0.872, 0.715)

        # 图标按钮与功能按钮（位置与 Tk 对齐）
        btn_undo = QToolButton()
        btn_undo.setIcon(load_icon("C:/Users/hantang/Pictures/Saved Pictures/undo2.png", QtCore.QSize(35, 35)))
        btn_undo.setIconSize(QtCore.QSize(35, 35))
        btn_undo.setAutoRaise(True)
        place_rel(btn_undo, QtCore.QSize(35, 35), 0.45, 0.85)

        btn_redo = QToolButton()
        btn_redo.setIcon(load_icon("C:/Users/hantang/Pictures/Saved Pictures/redo2.png", QtCore.QSize(35, 35)))
        btn_redo.setIconSize(QtCore.QSize(35, 35))
        btn_redo.setAutoRaise(True)
        place_rel(btn_redo, QtCore.QSize(35, 35), 0.55, 0.85)

        btn_brush = QToolButton()
        btn_brush.setIcon(load_icon("C:/Users/hantang/Pictures/Saved Pictures/brush.png", QtCore.QSize(35, 35)))
        btn_brush.setIconSize(QtCore.QSize(35, 35))
        btn_brush.setAutoRaise(True)
        place_rel(btn_brush, QtCore.QSize(35, 35), 0.69, 0.15)

        btn_adjust = QPushButton("Adjust BG")
        btn_adjust.setFont(QFont("Segoe UI", 12, QFont.Bold))
        btn_adjust.setStyleSheet(
            "background:#E5DDC5; color:black; padding:6px 10px; border:none; border-radius:4px;"
        )
        place_rel(btn_adjust, QtCore.QSize(120, 36), 0.73, 0.15)

        btn_ok = QToolButton()
        btn_ok.setIcon(load_icon("C:/Users/hantang/Pictures/Saved Pictures/Success.png", QtCore.QSize(35, 35)))
        btn_ok.setIconSize(QtCore.QSize(35, 35))
        btn_ok.setAutoRaise(True)
        place_rel(btn_ok, QtCore.QSize(35, 35), 0.83, 0.868)

        btn_not = QToolButton()
        btn_not.setIcon(load_icon("C:/Users/hantang/Pictures/Saved Pictures/fail.png", QtCore.QSize(35, 35)))
        btn_not.setIconSize(QtCore.QSize(35, 35))
        btn_not.setAutoRaise(True)
        place_rel(btn_not, QtCore.QSize(35, 35), 0.914, 0.868)

        # 右下 Git 链接按钮（作为图片按钮）
        btn_git = QToolButton()
        btn_git.setIcon(load_icon("C:/Users/hantang/Pictures/Saved Pictures/git.png", QtCore.QSize(134, 55)))
        btn_git.setIconSize(QtCore.QSize(134, 55))
        btn_git.setAutoRaise(True)
        place_rel(btn_git, QtCore.QSize(134, 55), 0.843, 0.942)

        btn_git2 = QToolButton()
        btn_git2.setIcon(load_icon("C:/Users/hantang/Pictures/Saved Pictures/git2.png", QtCore.QSize(134, 55)))
        btn_git2.setIconSize(QtCore.QSize(134, 55))
        btn_git2.setAutoRaise(True)
        place_rel(btn_git2, QtCore.QSize(134, 55), 0.93, 0.942)

        # 底部主功能按钮（Open Image / Start SAM / Cellpose）
        def make_primary_button(text):
            btn = QPushButton(text)
            btn.setFont(BTN_FONT_SECONDARY)
            btn.setStyleSheet(
                f"background: {THEME['primary']}; color: #f0f0f0; padding: 8px 24px; border-radius: 6px;"
            )
            btn.setFixedWidth(180)
            return btn

        btn_open = make_primary_button("Open Image")
        btn_start_sam = make_primary_button("Start SAM")
        btn_cellpose = make_primary_button("Cellpose")

        place_rel(btn_open, QtCore.QSize(180, 46), 0.35, 0.94)
        place_rel(btn_start_sam, QtCore.QSize(180, 46), 0.50, 0.94)
        place_rel(btn_cellpose, QtCore.QSize(180, 46), 0.65, 0.94)

        # 事件占位（后续可与 backend 对接）
        btn_open.clicked.connect(self.on_open_image)
        btn_start_sam.clicked.connect(self.on_start_sam)
        btn_cellpose.clicked.connect(self.on_cellpose)

        # 保存对象以便 on_resize 重新布局
        self._toolbar = self._toolbar_wrap
        # 防止主窗口强制非常大的最小尺寸，设置合理的下限
        try:
            self.setMinimumSize(900, 700)
        except Exception:
            pass

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        # 尝试在窗口尺寸变化时重排舞台元素
        self._reflow_rel()

    def _reflow_rel(self):
        """根据 relx/rely 重新放置 _stage 内的元素，尽量模拟 Tk 的 place(relx,rely)。
        Tk 的 rel 坐标是基于整个窗口的，我们需要扣除顶部工具栏高度。
        """
        if not hasattr(self, "_stage"):
            return
        try:
            total_w = self.centralWidget().width()
            total_h = self.centralWidget().height()
            toolbar_h = self._toolbar.height() if hasattr(self, "_toolbar") and self._toolbar else 0
            stage_w = self._stage.width()
            stage_h = self._stage.height()
            # central 总高度 = toolbar + stage
            # 目标像素坐标（基于 Tk 语义）：
            # x_abs = relx * total_w; y_abs = rely * total_h
            # 放入 stage 内部坐标：y_stage = y_abs - toolbar_h
            for it in self._rel_items:
                w: QWidget = it["w"]
                size: QtCore.QSize = it["size"]
                rx = it["relx"]; ry = it["rely"]
                x_abs = int(rx * total_w)
                y_abs = int(ry * total_h)
                x_stage = x_abs - size.width() // 2
                y_stage = (y_abs - toolbar_h) - size.height() // 2
                # 边界保护
                x_stage = max(0, min(stage_w - size.width(), x_stage))
                y_stage = max(0, min(stage_h - size.height(), y_stage))
                w.move(x_stage, y_stage)
        except Exception:
            pass

    # ------ 事件处理占位 ------
    def on_switch_mode(self):
        # 关闭当前窗口，返回模式选择
        self.close()
        self._open_mode_select()

    def on_open_image(self):
        pass

    def on_start_sam(self):
        pass

    def on_cellpose(self):
        pass

    def on_help(self):
        QMessageBox.information(self, "Help", (
            "Welcome to the Analysis Tool!\n\n"
            "1. Select a mode (Growth Plate, Articular Cartilage, Meniscus) to start.\n"
            "2. Choose an animal model (mouse, hamster, rat) from the dropdown.\n"
            "3. Click 'Cellinfo plot' to analyze cell parameters.\n"
            "4. Use the buttons to open images, start SAM, or run Cellpose.\n"
            "5. For detailed instructions, refer to the user manual."
        ))

    # 外部注入：由 UIManager2 控制，当切换模式时重新弹模式选择窗口
    def _set_mode_select_opener(self, opener: Callable[[], None]):
        self._open_mode_select = opener


# -------------------------- 对外入口：UIManager2 --------------------------
class UIManager2:
    """对齐 ui_manager.UIManager 的 PyQt 版本外观管理器（仅 UI）。"""
    def __init__(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self._mode_win: Optional[ModeSelectionWindow] = None
        self._main_win: Optional[MainWindow] = None

    def run(self):
        self.show_mode_selection()
        return self.app.exec_()

    # ---- 模式选择 ----
    def show_mode_selection(self):
        if self._main_win:
            try:
                self._main_win.close()
            except Exception:
                pass
            self._main_win = None
        self._mode_win = ModeSelectionWindow(on_select=self._on_mode_chosen)
        self._mode_win.show()

    def _on_mode_chosen(self, mode: str):
        # 进入主界面
        if self._mode_win:
            self._mode_win.close()
            self._mode_win = None
        self._main_win = MainWindow(mode)
        # 提供返回模式选择的回调
        self._main_win._set_mode_select_opener(self.show_mode_selection)
        self._main_win.show()


# 允许独立运行以快速预览 UI（不增加额外业务逻辑）
if __name__ == "__main__":
    ui = UIManager2()
    sys.exit(ui.run())
