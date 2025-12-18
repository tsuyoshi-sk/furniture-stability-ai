#!/usr/bin/env python3
"""
Furniture Drawing Generator
家具図面生成システム - 技術図面・三面図・組立図

Features:
- Technical Drawings: 正確な寸法付き図面
- Three-View: 三面図（正面・側面・上面）
- Assembly Diagrams: 組立図・分解図
- Parts List: パーツリスト・カットリスト生成
- OBJ Auto Analysis: OBJファイルから自動で図面生成
- DXF Export: CAD互換出力
"""
import sys
import math
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import svgwrite
from svgwrite import cm, mm
import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "inference"))
OUTPUT_DIR = PROJECT_ROOT / "output/drawings"
OUTPUT_DIR.mkdir(exist_ok=True)


class ViewType(Enum):
    FRONT = "front"
    SIDE = "side"
    TOP = "top"
    ISOMETRIC = "isometric"
    EXPLODED = "exploded"


@dataclass
class Dimension:
    """寸法線データ"""
    start: Tuple[float, float]
    end: Tuple[float, float]
    value: float
    unit: str = "mm"
    offset: float = 20  # 寸法線オフセット
    text_offset: float = 5


@dataclass
class Part:
    """パーツデータ"""
    name: str
    name_ja: str
    width: float
    depth: float
    thickness: float
    quantity: int = 1
    material: str = "plywood"
    notes: str = ""


@dataclass
class FurnitureSpec:
    """家具仕様"""
    name: str
    name_ja: str
    type: str
    width: float      # W (mm)
    depth: float      # D (mm)
    height: float     # H (mm)
    thickness: float  # 板厚 (mm)
    parts: List[Part] = field(default_factory=list)
    shelves: int = 0
    doors: int = 0
    drawers: int = 0
    material: str = "plywood"
    notes: str = ""


# =============================================================================
# OBJ ANALYZER
# =============================================================================

class OBJAnalyzer:
    """
    OBJファイルを解析して家具仕様を自動生成

    - 寸法の自動計算
    - 家具タイプの推定
    - 内部構造（棚板等）の検出
    """

    # 家具タイプの日本語名
    TYPE_NAMES = {
        'chair': '椅子',
        'table': 'テーブル',
        'shelf': '棚',
        'cabinet': 'キャビネット',
        'desk': 'デスク',
        'sofa': 'ソファ',
        'stool': 'スツール',
        'bench': 'ベンチ',
        'bookshelf': '本棚',
    }

    def __init__(self):
        self.vertices = None
        self.faces = None
        self.bounds = None

    def load_obj(self, obj_path: Path) -> np.ndarray:
        """OBJファイルから頂点を読み込み"""
        vertices = []
        faces = []

        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.strip().split()[1:]
                    face = [int(p.split('/')[0]) - 1 for p in parts]
                    faces.append(face)

        self.vertices = np.array(vertices)
        self.faces = faces
        return self.vertices

    def calculate_bounds(self) -> Dict[str, float]:
        """バウンディングボックスを計算"""
        if self.vertices is None:
            raise ValueError("OBJファイルが読み込まれていません")

        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)

        # X=幅, Y=高さ, Z=奥行き と仮定
        self.bounds = {
            'min_x': float(min_coords[0]),
            'max_x': float(max_coords[0]),
            'min_y': float(min_coords[1]),
            'max_y': float(max_coords[1]),
            'min_z': float(min_coords[2]),
            'max_z': float(max_coords[2]),
            'width': float(max_coords[0] - min_coords[0]),
            'height': float(max_coords[1] - min_coords[1]),
            'depth': float(max_coords[2] - min_coords[2]),
        }
        return self.bounds

    def detect_horizontal_planes(self, tolerance: float = 0.02) -> List[float]:
        """
        水平面（棚板など）を検出

        Args:
            tolerance: Y座標の許容誤差（高さに対する割合）

        Returns:
            検出された水平面のY座標リスト
        """
        if self.vertices is None or self.bounds is None:
            raise ValueError("先にload_objとcalculate_boundsを実行してください")

        height = self.bounds['height']
        min_y = self.bounds['min_y']
        max_y = self.bounds['max_y']

        # Y座標のヒストグラムを作成
        y_coords = self.vertices[:, 1]
        num_bins = max(20, int(height / 10))  # 10mm単位
        hist, bin_edges = np.histogram(y_coords, bins=num_bins)

        # ピークを検出（頂点が集中している高さ = 水平面）
        threshold = len(self.vertices) * 0.01  # 全頂点の1%以上
        planes = []

        for i, count in enumerate(hist):
            if count > threshold:
                plane_y = (bin_edges[i] + bin_edges[i + 1]) / 2
                # 上端・下端付近は除外
                relative_y = (plane_y - min_y) / height
                if 0.05 < relative_y < 0.95:
                    planes.append(float(plane_y))

        # 近い平面をマージ
        if planes:
            planes = sorted(planes)
            merged = [planes[0]]
            for p in planes[1:]:
                if p - merged[-1] > height * tolerance:
                    merged.append(p)
            planes = merged

        return planes

    def estimate_thickness(self) -> float:
        """板厚を推定"""
        if self.bounds is None:
            raise ValueError("先にcalculate_boundsを実行してください")

        # 最小寸法から推定
        dims = [self.bounds['width'], self.bounds['height'], self.bounds['depth']]
        min_dim = min(dims)

        # 一般的な板厚
        standard_thicknesses = [6, 9, 12, 15, 18, 21, 24, 25, 30]

        # 高さが最大の場合（棚など）、幅と奥行きの小さい方を参考に
        if self.bounds['height'] == max(dims):
            ref_dim = min(self.bounds['width'], self.bounds['depth'])
            estimated = ref_dim * 0.02  # 幅の2%程度
        else:
            estimated = min_dim * 0.05

        # 最も近い標準厚さを選択
        closest = min(standard_thicknesses, key=lambda x: abs(x - estimated))
        return float(closest)

    def estimate_furniture_type(self) -> str:
        """家具タイプを推定"""
        if self.bounds is None:
            raise ValueError("先にcalculate_boundsを実行してください")

        w = self.bounds['width']
        h = self.bounds['height']
        d = self.bounds['depth']

        aspect_ratio = h / max(w, d) if max(w, d) > 0 else 1

        # ヒューリスティックによる推定
        if aspect_ratio > 2.5:
            return 'bookshelf'
        elif aspect_ratio > 1.5:
            return 'cabinet'
        elif h < 100:  # 低い
            return 'table'
        elif w > d * 1.5 and h < 100:
            return 'desk'
        elif d > w and h < 80:
            return 'bench'
        else:
            return 'shelf'

    def analyze(self, obj_path: Path, furniture_type: Optional[str] = None) -> FurnitureSpec:
        """
        OBJファイルを解析してFurnitureSpecを生成

        Args:
            obj_path: OBJファイルパス
            furniture_type: 家具タイプ（指定しない場合は自動推定）

        Returns:
            FurnitureSpec
        """
        obj_path = Path(obj_path)

        # OBJ読み込み
        self.load_obj(obj_path)
        self.calculate_bounds()

        # 寸法をmmに変換（OBJがm単位の場合）
        scale = 1.0
        max_dim = max(self.bounds['width'], self.bounds['height'], self.bounds['depth'])
        if max_dim < 10:  # おそらくメートル単位
            scale = 1000.0

        width = self.bounds['width'] * scale
        height = self.bounds['height'] * scale
        depth = self.bounds['depth'] * scale

        # 家具タイプ
        if furniture_type is None:
            # 安定性AIから推定を試みる
            try:
                from inference import FurnitureStabilityPredictor
                predictor = FurnitureStabilityPredictor(verbose=False)
                result = predictor.predict(obj_path, use_tta=False)
                furniture_type = result.get('furniture_type', self.estimate_furniture_type())
            except:
                furniture_type = self.estimate_furniture_type()

        # 棚板検出
        planes = self.detect_horizontal_planes()
        num_shelves = len(planes)

        # 板厚推定（mm単位で推定）
        thickness = self._estimate_thickness_mm(width, height, depth)

        # 名前
        name = obj_path.stem
        name_ja = self.TYPE_NAMES.get(furniture_type, furniture_type)

        return FurnitureSpec(
            name=name,
            name_ja=name_ja,
            type=furniture_type,
            width=width,
            depth=depth,
            height=height,
            thickness=thickness,
            shelves=num_shelves,
            material='plywood',
            notes=f'Auto-generated from {obj_path.name}'
        )

    def _estimate_thickness_mm(self, width: float, height: float, depth: float) -> float:
        """mm単位で板厚を推定"""
        # 一般的な板厚
        standard_thicknesses = [6, 9, 12, 15, 18, 21, 24, 25, 30]

        # 寸法に基づいて推定
        min_dim = min(width, depth)

        # 家具サイズに基づいたヒューリスティック
        if height > 1500:  # 大型家具
            estimated = 18
        elif height > 800:  # 中型家具
            estimated = 15
        elif min_dim < 300:  # 小型または薄型
            estimated = 12
        else:
            estimated = 18

        # 最も近い標準厚さを選択
        closest = min(standard_thicknesses, key=lambda x: abs(x - estimated))
        return float(closest)


# =============================================================================
# DRAWING STYLES
# =============================================================================

class DrawingStyle:
    """図面スタイル設定"""
    # Line styles
    OUTLINE = {'stroke': '#000000', 'stroke-width': 0.5, 'fill': 'none'}
    HIDDEN = {'stroke': '#000000', 'stroke-width': 0.3, 'fill': 'none',
              'stroke-dasharray': '5,3'}
    CENTER = {'stroke': '#FF0000', 'stroke-width': 0.2, 'fill': 'none',
              'stroke-dasharray': '10,3,2,3'}
    DIMENSION = {'stroke': '#0000FF', 'stroke-width': 0.2, 'fill': 'none'}
    HATCH = {'stroke': '#808080', 'stroke-width': 0.1, 'fill': 'none'}

    # Text styles
    TITLE = {'font-size': '14px', 'font-family': 'Arial', 'font-weight': 'bold'}
    LABEL = {'font-size': '10px', 'font-family': 'Arial'}
    DIM_TEXT = {'font-size': '8px', 'font-family': 'Arial'}


# =============================================================================
# DRAWING GENERATOR
# =============================================================================

class FurnitureDrawingGenerator:
    """
    家具図面生成器

    Usage:
        gen = FurnitureDrawingGenerator()

        # Generate three-view drawing
        gen.generate_three_view(
            FurnitureSpec(
                name="Bookshelf",
                name_ja="本棚",
                type="bookshelf",
                width=800,
                depth=300,
                height=1800,
                thickness=18,
                shelves=5
            ),
            output="bookshelf_drawing.svg"
        )
    """

    def __init__(self, scale: float = 0.1, margin: float = 50):
        """
        Initialize generator

        Args:
            scale: Drawing scale (0.1 = 1:10)
            margin: Page margin in mm
        """
        self.scale = scale
        self.margin = margin

    def _to_drawing_coords(self, x: float, y: float) -> Tuple[float, float]:
        """Convert real dimensions to drawing coordinates"""
        return (x * self.scale + self.margin, y * self.scale + self.margin)

    def _scaled(self, value: float) -> float:
        """Scale a value"""
        return value * self.scale

    # =========================================================================
    # DIMENSION LINES
    # =========================================================================

    def _draw_dimension_horizontal(
        self,
        dwg: svgwrite.Drawing,
        x1: float, x2: float, y: float,
        value: float,
        offset: float = 20,
        above: bool = True
    ):
        """Draw horizontal dimension line"""
        y_dim = y - offset if above else y + offset

        # Extension lines
        dwg.add(dwg.line(
            (x1, y), (x1, y_dim - 3 if above else y_dim + 3),
            **DrawingStyle.DIMENSION
        ))
        dwg.add(dwg.line(
            (x2, y), (x2, y_dim - 3 if above else y_dim + 3),
            **DrawingStyle.DIMENSION
        ))

        # Dimension line
        dwg.add(dwg.line((x1, y_dim), (x2, y_dim), **DrawingStyle.DIMENSION))

        # Arrows
        arrow_size = 3
        # Left arrow
        dwg.add(dwg.line((x1, y_dim), (x1 + arrow_size, y_dim - arrow_size/2), **DrawingStyle.DIMENSION))
        dwg.add(dwg.line((x1, y_dim), (x1 + arrow_size, y_dim + arrow_size/2), **DrawingStyle.DIMENSION))
        # Right arrow
        dwg.add(dwg.line((x2, y_dim), (x2 - arrow_size, y_dim - arrow_size/2), **DrawingStyle.DIMENSION))
        dwg.add(dwg.line((x2, y_dim), (x2 - arrow_size, y_dim + arrow_size/2), **DrawingStyle.DIMENSION))

        # Dimension text
        text_x = (x1 + x2) / 2
        text_y = y_dim - 2 if above else y_dim + 8
        dwg.add(dwg.text(
            f"{value:.0f}",
            insert=(text_x, text_y),
            text_anchor="middle",
            **DrawingStyle.DIM_TEXT
        ))

    def _draw_dimension_vertical(
        self,
        dwg: svgwrite.Drawing,
        x: float, y1: float, y2: float,
        value: float,
        offset: float = 20,
        left: bool = True
    ):
        """Draw vertical dimension line"""
        x_dim = x - offset if left else x + offset

        # Extension lines
        dwg.add(dwg.line(
            (x, y1), (x_dim - 3 if left else x_dim + 3, y1),
            **DrawingStyle.DIMENSION
        ))
        dwg.add(dwg.line(
            (x, y2), (x_dim - 3 if left else x_dim + 3, y2),
            **DrawingStyle.DIMENSION
        ))

        # Dimension line
        dwg.add(dwg.line((x_dim, y1), (x_dim, y2), **DrawingStyle.DIMENSION))

        # Arrows
        arrow_size = 3
        dwg.add(dwg.line((x_dim, y1), (x_dim - arrow_size/2, y1 + arrow_size), **DrawingStyle.DIMENSION))
        dwg.add(dwg.line((x_dim, y1), (x_dim + arrow_size/2, y1 + arrow_size), **DrawingStyle.DIMENSION))
        dwg.add(dwg.line((x_dim, y2), (x_dim - arrow_size/2, y2 - arrow_size), **DrawingStyle.DIMENSION))
        dwg.add(dwg.line((x_dim, y2), (x_dim + arrow_size/2, y2 - arrow_size), **DrawingStyle.DIMENSION))

        # Dimension text (rotated)
        text_y = (y1 + y2) / 2
        text_x = x_dim - 3 if left else x_dim + 10
        g = dwg.g(transform=f"rotate(-90, {text_x}, {text_y})")
        g.add(dwg.text(
            f"{value:.0f}",
            insert=(text_x, text_y + 3),
            text_anchor="middle",
            **DrawingStyle.DIM_TEXT
        ))
        dwg.add(g)

    # =========================================================================
    # BOOKSHELF DRAWING
    # =========================================================================

    def _draw_bookshelf_front(
        self,
        dwg: svgwrite.Drawing,
        spec: FurnitureSpec,
        origin: Tuple[float, float]
    ):
        """Draw bookshelf front view"""
        ox, oy = origin
        w = self._scaled(spec.width)
        h = self._scaled(spec.height)
        t = self._scaled(spec.thickness)

        # Outer frame
        dwg.add(dwg.rect((ox, oy), (w, h), **DrawingStyle.OUTLINE))

        # Inner space
        inner_x = ox + t
        inner_y = oy + t
        inner_w = w - 2 * t
        inner_h = h - 2 * t
        dwg.add(dwg.rect((inner_x, inner_y), (inner_w, inner_h), **DrawingStyle.OUTLINE))

        # Shelves
        if spec.shelves > 0:
            shelf_spacing = inner_h / (spec.shelves + 1)
            for i in range(1, spec.shelves + 1):
                shelf_y = inner_y + i * shelf_spacing
                dwg.add(dwg.line(
                    (inner_x, shelf_y),
                    (inner_x + inner_w, shelf_y),
                    **DrawingStyle.OUTLINE
                ))
                # Shelf thickness
                dwg.add(dwg.line(
                    (inner_x, shelf_y + t),
                    (inner_x + inner_w, shelf_y + t),
                    **DrawingStyle.OUTLINE
                ))

        # Dimensions
        self._draw_dimension_horizontal(dwg, ox, ox + w, oy, spec.width, offset=25)
        self._draw_dimension_vertical(dwg, ox, oy, oy + h, spec.height, offset=25)
        self._draw_dimension_horizontal(dwg, inner_x, inner_x + inner_w, oy + h, spec.width - 2*spec.thickness, offset=-20, above=False)

        # Label
        dwg.add(dwg.text(
            "正面図 (Front View)",
            insert=(ox + w/2, oy - 35),
            text_anchor="middle",
            **DrawingStyle.LABEL
        ))

    def _draw_bookshelf_side(
        self,
        dwg: svgwrite.Drawing,
        spec: FurnitureSpec,
        origin: Tuple[float, float]
    ):
        """Draw bookshelf side view"""
        ox, oy = origin
        d = self._scaled(spec.depth)
        h = self._scaled(spec.height)
        t = self._scaled(spec.thickness)

        # Outer frame
        dwg.add(dwg.rect((ox, oy), (d, h), **DrawingStyle.OUTLINE))

        # Inner lines (thickness)
        dwg.add(dwg.line((ox + t, oy), (ox + t, oy + h), **DrawingStyle.HIDDEN))

        # Shelves (side view)
        if spec.shelves > 0:
            inner_h = h - 2 * t
            shelf_spacing = inner_h / (spec.shelves + 1)
            for i in range(1, spec.shelves + 1):
                shelf_y = oy + t + i * shelf_spacing
                dwg.add(dwg.line((ox, shelf_y), (ox + d, shelf_y), **DrawingStyle.HIDDEN))

        # Dimensions
        self._draw_dimension_horizontal(dwg, ox, ox + d, oy, spec.depth, offset=25)

        # Label
        dwg.add(dwg.text(
            "側面図 (Side View)",
            insert=(ox + d/2, oy - 35),
            text_anchor="middle",
            **DrawingStyle.LABEL
        ))

    def _draw_bookshelf_top(
        self,
        dwg: svgwrite.Drawing,
        spec: FurnitureSpec,
        origin: Tuple[float, float]
    ):
        """Draw bookshelf top view"""
        ox, oy = origin
        w = self._scaled(spec.width)
        d = self._scaled(spec.depth)
        t = self._scaled(spec.thickness)

        # Outer frame
        dwg.add(dwg.rect((ox, oy), (w, d), **DrawingStyle.OUTLINE))

        # Inner frame (showing sides)
        dwg.add(dwg.rect((ox + t, oy + t), (w - 2*t, d - 2*t), **DrawingStyle.OUTLINE))

        # Dimensions
        self._draw_dimension_horizontal(dwg, ox, ox + w, oy, spec.width, offset=25)
        self._draw_dimension_vertical(dwg, ox + w, oy, oy + d, spec.depth, offset=-25, left=False)
        self._draw_dimension_horizontal(dwg, ox, ox + t, oy + d, spec.thickness, offset=-15, above=False)

        # Label
        dwg.add(dwg.text(
            "上面図 (Top View)",
            insert=(ox + w/2, oy - 35),
            text_anchor="middle",
            **DrawingStyle.LABEL
        ))

    # =========================================================================
    # DESK DRAWING
    # =========================================================================

    def _draw_desk_front(
        self,
        dwg: svgwrite.Drawing,
        spec: FurnitureSpec,
        origin: Tuple[float, float]
    ):
        """Draw desk front view"""
        ox, oy = origin
        w = self._scaled(spec.width)
        h = self._scaled(spec.height)
        t = self._scaled(spec.thickness)
        leg_w = self._scaled(50)  # Leg width

        # Desktop
        dwg.add(dwg.rect((ox, oy), (w, t), **DrawingStyle.OUTLINE))

        # Legs
        leg_h = h - t
        # Left leg
        dwg.add(dwg.rect((ox + 20 * self.scale, oy + t), (leg_w, leg_h), **DrawingStyle.OUTLINE))
        # Right leg
        dwg.add(dwg.rect((ox + w - 20 * self.scale - leg_w, oy + t), (leg_w, leg_h), **DrawingStyle.OUTLINE))

        # Dimensions
        self._draw_dimension_horizontal(dwg, ox, ox + w, oy, spec.width, offset=25)
        self._draw_dimension_vertical(dwg, ox, oy, oy + h, spec.height, offset=25)
        self._draw_dimension_vertical(dwg, ox + 20 * self.scale + leg_w/2, oy + t, oy + h, spec.height - spec.thickness, offset=15, left=False)

        # Label
        dwg.add(dwg.text(
            "正面図 (Front View)",
            insert=(ox + w/2, oy - 35),
            text_anchor="middle",
            **DrawingStyle.LABEL
        ))

    def _draw_desk_side(
        self,
        dwg: svgwrite.Drawing,
        spec: FurnitureSpec,
        origin: Tuple[float, float]
    ):
        """Draw desk side view"""
        ox, oy = origin
        d = self._scaled(spec.depth)
        h = self._scaled(spec.height)
        t = self._scaled(spec.thickness)
        leg_d = self._scaled(50)

        # Desktop
        dwg.add(dwg.rect((ox, oy), (d, t), **DrawingStyle.OUTLINE))

        # Legs (side view)
        leg_h = h - t
        dwg.add(dwg.rect((ox + 20 * self.scale, oy + t), (leg_d, leg_h), **DrawingStyle.OUTLINE))
        dwg.add(dwg.rect((ox + d - 20 * self.scale - leg_d, oy + t), (leg_d, leg_h), **DrawingStyle.OUTLINE))

        # Dimensions
        self._draw_dimension_horizontal(dwg, ox, ox + d, oy, spec.depth, offset=25)

        # Label
        dwg.add(dwg.text(
            "側面図 (Side View)",
            insert=(ox + d/2, oy - 35),
            text_anchor="middle",
            **DrawingStyle.LABEL
        ))

    def _draw_desk_top(
        self,
        dwg: svgwrite.Drawing,
        spec: FurnitureSpec,
        origin: Tuple[float, float]
    ):
        """Draw desk top view"""
        ox, oy = origin
        w = self._scaled(spec.width)
        d = self._scaled(spec.depth)

        # Desktop surface
        dwg.add(dwg.rect((ox, oy), (w, d), **DrawingStyle.OUTLINE))

        # Leg positions (hidden)
        leg_size = self._scaled(50)
        leg_offset = 20 * self.scale
        dwg.add(dwg.rect((ox + leg_offset, oy + leg_offset), (leg_size, leg_size), **DrawingStyle.HIDDEN))
        dwg.add(dwg.rect((ox + w - leg_offset - leg_size, oy + leg_offset), (leg_size, leg_size), **DrawingStyle.HIDDEN))
        dwg.add(dwg.rect((ox + leg_offset, oy + d - leg_offset - leg_size), (leg_size, leg_size), **DrawingStyle.HIDDEN))
        dwg.add(dwg.rect((ox + w - leg_offset - leg_size, oy + d - leg_offset - leg_size), (leg_size, leg_size), **DrawingStyle.HIDDEN))

        # Dimensions
        self._draw_dimension_horizontal(dwg, ox, ox + w, oy, spec.width, offset=25)
        self._draw_dimension_vertical(dwg, ox + w, oy, oy + d, spec.depth, offset=-25, left=False)

        # Label
        dwg.add(dwg.text(
            "上面図 (Top View)",
            insert=(ox + w/2, oy - 35),
            text_anchor="middle",
            **DrawingStyle.LABEL
        ))

    # =========================================================================
    # MAIN GENERATORS
    # =========================================================================

    def generate_three_view(
        self,
        spec: FurnitureSpec,
        output: Optional[str] = None,
        include_parts_list: bool = True
    ) -> Path:
        """
        Generate three-view technical drawing

        Args:
            spec: Furniture specification
            output: Output filename
            include_parts_list: Include parts list table

        Returns:
            Path to generated SVG file
        """
        # Calculate drawing size
        page_width = self._scaled(spec.width) + self._scaled(spec.depth) + 150
        page_height = self._scaled(spec.height) + self._scaled(spec.depth) + 200

        if include_parts_list:
            page_height += 150

        dwg = svgwrite.Drawing(
            filename=str(OUTPUT_DIR / (output or f"{spec.name}_drawing.svg")),
            size=(f"{page_width}mm", f"{page_height}mm"),
            viewBox=f"0 0 {page_width} {page_height}"
        )

        # Title block
        dwg.add(dwg.text(
            f"{spec.name_ja} ({spec.name})",
            insert=(page_width / 2, 20),
            text_anchor="middle",
            **DrawingStyle.TITLE
        ))
        dwg.add(dwg.text(
            f"Scale: 1:{int(1/self.scale)} | Material: {spec.material} | Thickness: {spec.thickness}mm",
            insert=(page_width / 2, 35),
            text_anchor="middle",
            **DrawingStyle.LABEL
        ))

        # Choose drawing method based on furniture type
        if spec.type in ('bookshelf', 'shelf', 'cabinet'):
            front_origin = (self.margin, 60)
            side_origin = (self.margin + self._scaled(spec.width) + 50, 60)
            top_origin = (self.margin, 60 + self._scaled(spec.height) + 60)

            self._draw_bookshelf_front(dwg, spec, front_origin)
            self._draw_bookshelf_side(dwg, spec, side_origin)
            self._draw_bookshelf_top(dwg, spec, top_origin)

        elif spec.type == 'desk':
            front_origin = (self.margin, 60)
            side_origin = (self.margin + self._scaled(spec.width) + 50, 60)
            top_origin = (self.margin, 60 + self._scaled(spec.height) + 60)

            self._draw_desk_front(dwg, spec, front_origin)
            self._draw_desk_side(dwg, spec, side_origin)
            self._draw_desk_top(dwg, spec, top_origin)

        else:
            # Generic box drawing
            self._draw_bookshelf_front(dwg, spec, (self.margin, 60))

        # Parts list
        if include_parts_list:
            self._draw_parts_list(dwg, spec, (self.margin, page_height - 130))

        # Save
        output_path = OUTPUT_DIR / (output or f"{spec.name}_drawing.svg")
        dwg.saveas(str(output_path))
        print(f"Drawing saved: {output_path}")

        return output_path

    def _draw_parts_list(
        self,
        dwg: svgwrite.Drawing,
        spec: FurnitureSpec,
        origin: Tuple[float, float]
    ):
        """Draw parts list table"""
        ox, oy = origin

        # Generate parts list
        parts = self._calculate_parts(spec)

        # Table header
        dwg.add(dwg.text("パーツリスト (Parts List)", insert=(ox, oy - 10), **DrawingStyle.LABEL))

        # Table
        row_height = 15
        col_widths = [30, 80, 50, 50, 50, 30, 50]
        headers = ["No.", "Name", "W", "D", "T", "Qty", "Material"]

        # Header row
        x = ox
        for i, (header, col_w) in enumerate(zip(headers, col_widths)):
            dwg.add(dwg.rect((x, oy), (col_w, row_height), fill='#E0E0E0', stroke='#000', stroke_width=0.3))
            dwg.add(dwg.text(header, insert=(x + 3, oy + 11), **DrawingStyle.DIM_TEXT))
            x += col_w

        # Data rows
        for i, part in enumerate(parts):
            y = oy + (i + 1) * row_height
            x = ox
            data = [
                str(i + 1),
                part.name_ja,
                f"{part.width:.0f}",
                f"{part.depth:.0f}",
                f"{part.thickness:.0f}",
                str(part.quantity),
                part.material
            ]
            for j, (val, col_w) in enumerate(zip(data, col_widths)):
                dwg.add(dwg.rect((x, y), (col_w, row_height), fill='none', stroke='#000', stroke_width=0.3))
                dwg.add(dwg.text(val, insert=(x + 3, y + 11), **DrawingStyle.DIM_TEXT))
                x += col_w

    def _calculate_parts(self, spec: FurnitureSpec) -> List[Part]:
        """Calculate parts list from specification"""
        parts = []

        if spec.type in ('bookshelf', 'shelf', 'cabinet'):
            # Side panels
            parts.append(Part(
                name="Side Panel",
                name_ja="側板",
                width=spec.depth,
                depth=spec.height,
                thickness=spec.thickness,
                quantity=2,
                material=spec.material
            ))
            # Top and bottom
            parts.append(Part(
                name="Top/Bottom",
                name_ja="天板/地板",
                width=spec.width - 2 * spec.thickness,
                depth=spec.depth,
                thickness=spec.thickness,
                quantity=2,
                material=spec.material
            ))
            # Back panel
            parts.append(Part(
                name="Back Panel",
                name_ja="背板",
                width=spec.width - 2 * spec.thickness,
                depth=spec.height - 2 * spec.thickness,
                thickness=spec.thickness / 3,  # Thinner back
                quantity=1,
                material=spec.material
            ))
            # Shelves
            if spec.shelves > 0:
                parts.append(Part(
                    name="Shelf",
                    name_ja="棚板",
                    width=spec.width - 2 * spec.thickness,
                    depth=spec.depth - spec.thickness,
                    thickness=spec.thickness,
                    quantity=spec.shelves,
                    material=spec.material
                ))

        elif spec.type == 'desk':
            # Desktop
            parts.append(Part(
                name="Desktop",
                name_ja="天板",
                width=spec.width,
                depth=spec.depth,
                thickness=spec.thickness,
                quantity=1,
                material=spec.material
            ))
            # Legs
            parts.append(Part(
                name="Leg",
                name_ja="脚",
                width=50,
                depth=50,
                thickness=spec.height - spec.thickness,
                quantity=4,
                material="steel"
            ))

        return parts

    def generate_cut_list(
        self,
        spec: FurnitureSpec,
        output: Optional[str] = None
    ) -> Path:
        """
        Generate cut list for manufacturing

        Args:
            spec: Furniture specification
            output: Output filename

        Returns:
            Path to JSON file
        """
        parts = self._calculate_parts(spec)

        cut_list = {
            'furniture': {
                'name': spec.name,
                'name_ja': spec.name_ja,
                'dimensions': {
                    'width': spec.width,
                    'depth': spec.depth,
                    'height': spec.height
                }
            },
            'parts': [
                {
                    'no': i + 1,
                    'name': p.name,
                    'name_ja': p.name_ja,
                    'dimensions': {
                        'width': p.width,
                        'depth': p.depth,
                        'thickness': p.thickness
                    },
                    'quantity': p.quantity,
                    'material': p.material,
                    'area_mm2': p.width * p.depth,
                    'volume_mm3': p.width * p.depth * p.thickness
                }
                for i, p in enumerate(parts)
            ],
            'totals': {
                'total_parts': sum(p.quantity for p in parts),
                'total_area_m2': sum(p.width * p.depth * p.quantity for p in parts) / 1_000_000,
                'total_volume_m3': sum(p.width * p.depth * p.thickness * p.quantity for p in parts) / 1_000_000_000
            }
        }

        output_path = OUTPUT_DIR / (output or f"{spec.name}_cutlist.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cut_list, f, indent=2, ensure_ascii=False)

        print(f"Cut list saved: {output_path}")
        return output_path


# =============================================================================
# DXF CAD DRAWING GENERATOR
# =============================================================================

class CADDrawingGenerator:
    """
    CAD用DXF図面生成

    Features:
    - AutoCAD/Fusion360互換のDXF出力
    - レイヤー分け（外形線、寸法線、中心線等）
    - 正確な寸法注記
    - 三面図レイアウト
    """

    # CADレイヤー定義
    LAYERS = {
        'OUTLINE': {'color': 7, 'linetype': 'CONTINUOUS', 'desc': '外形線'},
        'DIMENSION': {'color': 1, 'linetype': 'CONTINUOUS', 'desc': '寸法線'},
        'CENTERLINE': {'color': 2, 'linetype': 'CENTER', 'desc': '中心線'},
        'HIDDEN': {'color': 8, 'linetype': 'HIDDEN', 'desc': '隠れ線'},
        'TEXT': {'color': 7, 'linetype': 'CONTINUOUS', 'desc': 'テキスト'},
        'HATCH': {'color': 8, 'linetype': 'CONTINUOUS', 'desc': 'ハッチング'},
        'PARTS': {'color': 3, 'linetype': 'CONTINUOUS', 'desc': 'パーツ'},
    }

    def __init__(self, scale: float = 1.0):
        """
        Args:
            scale: 図面スケール（1.0 = 1:1、0.1 = 1:10）
        """
        self.scale = scale
        self.margin = 50  # mm

    def _scaled(self, value: float) -> float:
        """スケール適用"""
        return value * self.scale

    def generate_dxf(
        self,
        spec: FurnitureSpec,
        output: Optional[str] = None,
        include_dimensions: bool = True,
        include_parts_layout: bool = True
    ) -> Path:
        """
        DXF図面を生成

        Args:
            spec: 家具仕様
            output: 出力ファイル名
            include_dimensions: 寸法線を含める
            include_parts_layout: パーツ配置図を含める

        Returns:
            出力ファイルパス
        """
        # DXFドキュメント作成
        doc = ezdxf.new('R2010')  # AutoCAD 2010形式
        doc.units = units.MM

        # レイヤー設定
        self._setup_layers(doc)

        # 寸法スタイル設定
        self._setup_dimension_style(doc)

        msp = doc.modelspace()

        # 三面図の配置位置
        front_origin = (self.margin, self.margin + self._scaled(spec.height) + 100)
        side_origin = (self.margin + self._scaled(spec.width) + 80, self.margin + self._scaled(spec.height) + 100)
        top_origin = (self.margin, self.margin)

        # 三面図描画
        if spec.type in ('bookshelf', 'shelf', 'cabinet'):
            self._draw_bookshelf_front_dxf(msp, spec, front_origin, include_dimensions)
            self._draw_bookshelf_side_dxf(msp, spec, side_origin, include_dimensions)
            self._draw_bookshelf_top_dxf(msp, spec, top_origin, include_dimensions)
        elif spec.type == 'desk':
            self._draw_desk_front_dxf(msp, spec, front_origin, include_dimensions)
            self._draw_desk_side_dxf(msp, spec, side_origin, include_dimensions)
            self._draw_desk_top_dxf(msp, spec, top_origin, include_dimensions)
        else:
            # 汎用の箱型
            self._draw_generic_front_dxf(msp, spec, front_origin, include_dimensions)
            self._draw_generic_side_dxf(msp, spec, side_origin, include_dimensions)
            self._draw_generic_top_dxf(msp, spec, top_origin, include_dimensions)

        # ビューラベル
        self._add_view_labels(msp, spec, front_origin, side_origin, top_origin)

        # タイトルブロック
        self._draw_title_block(msp, spec)

        # パーツ配置図（オプション）
        if include_parts_layout:
            self._draw_parts_layout_dxf(msp, spec)

        # 保存
        output_path = OUTPUT_DIR / (output or f"{spec.name}_cad.dxf")
        doc.saveas(str(output_path))
        print(f"DXF saved: {output_path}")
        return output_path

    def _setup_layers(self, doc):
        """レイヤー設定"""
        for name, props in self.LAYERS.items():
            doc.layers.add(
                name,
                color=props['color'],
                linetype=props['linetype']
            )

    def _setup_dimension_style(self, doc):
        """寸法スタイル設定"""
        dim_style = doc.dimstyles.new('FURNITURE')
        dim_style.dxf.dimtxt = 3.5  # テキストサイズ
        dim_style.dxf.dimasz = 2.5  # 矢印サイズ
        dim_style.dxf.dimexe = 1.5  # 補助線延長
        dim_style.dxf.dimexo = 1.0  # 補助線オフセット
        dim_style.dxf.dimgap = 1.0  # テキストギャップ
        dim_style.dxf.dimdec = 0    # 小数点以下桁数

    def _draw_bookshelf_front_dxf(self, msp, spec: FurnitureSpec, origin: Tuple[float, float], dims: bool):
        """本棚正面図（DXF）"""
        ox, oy = origin
        w = self._scaled(spec.width)
        h = self._scaled(spec.height)
        t = self._scaled(spec.thickness)

        # 外枠
        self._draw_rect(msp, ox, oy, w, h, 'OUTLINE')

        # 内枠（板厚分内側）
        inner_w = w - 2 * t
        inner_h = h - 2 * t
        self._draw_rect(msp, ox + t, oy + t, inner_w, inner_h, 'OUTLINE')

        # 棚板
        if spec.shelves > 0:
            shelf_spacing = inner_h / (spec.shelves + 1)
            for i in range(1, spec.shelves + 1):
                y = oy + t + shelf_spacing * i
                # 棚板上下の線
                msp.add_line((ox + t, y - t/2), (ox + w - t, y - t/2), dxfattribs={'layer': 'OUTLINE'})
                msp.add_line((ox + t, y + t/2), (ox + w - t, y + t/2), dxfattribs={'layer': 'OUTLINE'})

        # 寸法線
        if dims:
            # 全幅（上部）
            self._add_dimension(msp, (ox, oy + h + 15), (ox + w, oy + h + 15), spec.width, vertical=False)
            # 全高（左側）
            self._add_dimension(msp, (ox - 15, oy), (ox - 15, oy + h), spec.height, vertical=True)
            # 内寸幅
            self._add_dimension(msp, (ox + t, oy - 10), (ox + w - t, oy - 10), spec.width - 2 * spec.thickness, vertical=False)

    def _draw_bookshelf_side_dxf(self, msp, spec: FurnitureSpec, origin: Tuple[float, float], dims: bool):
        """本棚側面図（DXF）"""
        ox, oy = origin
        d = self._scaled(spec.depth)
        h = self._scaled(spec.height)
        t = self._scaled(spec.thickness)

        # 外枠
        self._draw_rect(msp, ox, oy, d, h, 'OUTLINE')

        # 内枠
        self._draw_rect(msp, ox + t, oy + t, d - 2*t, h - 2*t, 'OUTLINE')

        # 棚板断面
        if spec.shelves > 0:
            inner_h = h - 2 * t
            shelf_spacing = inner_h / (spec.shelves + 1)
            for i in range(1, spec.shelves + 1):
                y = oy + t + shelf_spacing * i
                self._draw_rect(msp, ox + t, y - t/2, d - 2*t, t, 'OUTLINE')

        # 背板（破線）
        back_t = self._scaled(spec.thickness / 3)
        msp.add_line((ox + d - back_t, oy + t), (ox + d - back_t, oy + h - t), dxfattribs={'layer': 'HIDDEN'})

        # 寸法線
        if dims:
            # 奥行き
            self._add_dimension(msp, (ox, oy + h + 15), (ox + d, oy + h + 15), spec.depth, vertical=False)

    def _draw_bookshelf_top_dxf(self, msp, spec: FurnitureSpec, origin: Tuple[float, float], dims: bool):
        """本棚上面図（DXF）"""
        ox, oy = origin
        w = self._scaled(spec.width)
        d = self._scaled(spec.depth)
        t = self._scaled(spec.thickness)

        # 外枠
        self._draw_rect(msp, ox, oy, w, d, 'OUTLINE')

        # 側板（破線で示す）
        msp.add_line((ox + t, oy), (ox + t, oy + d), dxfattribs={'layer': 'HIDDEN'})
        msp.add_line((ox + w - t, oy), (ox + w - t, oy + d), dxfattribs={'layer': 'HIDDEN'})

        # 背板（破線）
        back_t = self._scaled(spec.thickness / 3)
        msp.add_line((ox, oy + d - back_t), (ox + w, oy + d - back_t), dxfattribs={'layer': 'HIDDEN'})

        # 寸法線
        if dims:
            self._add_dimension(msp, (ox, oy - 15), (ox + w, oy - 15), spec.width, vertical=False)
            self._add_dimension(msp, (ox + w + 15, oy), (ox + w + 15, oy + d), spec.depth, vertical=True)

    def _draw_desk_front_dxf(self, msp, spec: FurnitureSpec, origin: Tuple[float, float], dims: bool):
        """デスク正面図（DXF）"""
        ox, oy = origin
        w = self._scaled(spec.width)
        h = self._scaled(spec.height)
        t = self._scaled(spec.thickness)
        leg_w = self._scaled(50)  # 脚の幅

        # 天板
        self._draw_rect(msp, ox, oy + h - t, w, t, 'OUTLINE')

        # 脚
        leg_h = h - t
        self._draw_rect(msp, ox + t, oy, leg_w, leg_h, 'OUTLINE')
        self._draw_rect(msp, ox + w - t - leg_w, oy, leg_w, leg_h, 'OUTLINE')

        # 寸法線
        if dims:
            self._add_dimension(msp, (ox, oy + h + 15), (ox + w, oy + h + 15), spec.width, vertical=False)
            self._add_dimension(msp, (ox - 15, oy), (ox - 15, oy + h), spec.height, vertical=True)
            # 脚の高さ
            self._add_dimension(msp, (ox + t + leg_w + 10, oy), (ox + t + leg_w + 10, oy + leg_h), spec.height - spec.thickness, vertical=True)

    def _draw_desk_side_dxf(self, msp, spec: FurnitureSpec, origin: Tuple[float, float], dims: bool):
        """デスク側面図（DXF）"""
        ox, oy = origin
        d = self._scaled(spec.depth)
        h = self._scaled(spec.height)
        t = self._scaled(spec.thickness)
        leg_w = self._scaled(50)

        # 天板
        self._draw_rect(msp, ox, oy + h - t, d, t, 'OUTLINE')

        # 脚
        self._draw_rect(msp, ox + t, oy, leg_w, h - t, 'OUTLINE')
        self._draw_rect(msp, ox + d - t - leg_w, oy, leg_w, h - t, 'OUTLINE')

        # 寸法線
        if dims:
            self._add_dimension(msp, (ox, oy + h + 15), (ox + d, oy + h + 15), spec.depth, vertical=False)

    def _draw_desk_top_dxf(self, msp, spec: FurnitureSpec, origin: Tuple[float, float], dims: bool):
        """デスク上面図（DXF）"""
        ox, oy = origin
        w = self._scaled(spec.width)
        d = self._scaled(spec.depth)
        t = self._scaled(spec.thickness)
        leg_w = self._scaled(50)

        # 天板
        self._draw_rect(msp, ox, oy, w, d, 'OUTLINE')

        # 脚の位置（破線）
        for lx in [ox + t, ox + w - t - leg_w]:
            for ly in [oy + t, oy + d - t - leg_w]:
                self._draw_rect(msp, lx, ly, leg_w, leg_w, 'HIDDEN')

        # 寸法線
        if dims:
            self._add_dimension(msp, (ox, oy - 15), (ox + w, oy - 15), spec.width, vertical=False)
            self._add_dimension(msp, (ox + w + 15, oy), (ox + w + 15, oy + d), spec.depth, vertical=True)

    def _draw_generic_front_dxf(self, msp, spec: FurnitureSpec, origin: Tuple[float, float], dims: bool):
        """汎用正面図（DXF）"""
        ox, oy = origin
        w = self._scaled(spec.width)
        h = self._scaled(spec.height)

        self._draw_rect(msp, ox, oy, w, h, 'OUTLINE')

        if dims:
            self._add_dimension(msp, (ox, oy + h + 15), (ox + w, oy + h + 15), spec.width, vertical=False)
            self._add_dimension(msp, (ox - 15, oy), (ox - 15, oy + h), spec.height, vertical=True)

    def _draw_generic_side_dxf(self, msp, spec: FurnitureSpec, origin: Tuple[float, float], dims: bool):
        """汎用側面図（DXF）"""
        ox, oy = origin
        d = self._scaled(spec.depth)
        h = self._scaled(spec.height)

        self._draw_rect(msp, ox, oy, d, h, 'OUTLINE')

        if dims:
            self._add_dimension(msp, (ox, oy + h + 15), (ox + d, oy + h + 15), spec.depth, vertical=False)

    def _draw_generic_top_dxf(self, msp, spec: FurnitureSpec, origin: Tuple[float, float], dims: bool):
        """汎用上面図（DXF）"""
        ox, oy = origin
        w = self._scaled(spec.width)
        d = self._scaled(spec.depth)

        self._draw_rect(msp, ox, oy, w, d, 'OUTLINE')

        if dims:
            self._add_dimension(msp, (ox, oy - 15), (ox + w, oy - 15), spec.width, vertical=False)
            self._add_dimension(msp, (ox + w + 15, oy), (ox + w + 15, oy + d), spec.depth, vertical=True)

    def _draw_rect(self, msp, x: float, y: float, w: float, h: float, layer: str = 'OUTLINE'):
        """矩形を描画"""
        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        msp.add_lwpolyline(points, close=True, dxfattribs={'layer': layer})

    def _add_dimension(self, msp, p1: Tuple[float, float], p2: Tuple[float, float], value: float, vertical: bool = False):
        """寸法線を追加"""
        if vertical:
            # 垂直寸法
            msp.add_linear_dim(
                base=(p1[0], (p1[1] + p2[1]) / 2),
                p1=p1,
                p2=p2,
                angle=90,
                dimstyle='FURNITURE',
                override={'dimtxt': 3.5},
                dxfattribs={'layer': 'DIMENSION'}
            ).render()
        else:
            # 水平寸法
            msp.add_linear_dim(
                base=((p1[0] + p2[0]) / 2, p1[1]),
                p1=p1,
                p2=p2,
                dimstyle='FURNITURE',
                override={'dimtxt': 3.5},
                dxfattribs={'layer': 'DIMENSION'}
            ).render()

    def _add_view_labels(self, msp, spec: FurnitureSpec, front_origin, side_origin, top_origin):
        """ビューラベルを追加"""
        labels = [
            (front_origin, "正面図 (Front View)"),
            (side_origin, "側面図 (Side View)"),
            (top_origin, "上面図 (Top View)")
        ]

        for origin, label in labels:
            msp.add_text(
                label,
                dxfattribs={
                    'layer': 'TEXT',
                    'height': 5,
                    'style': 'Standard'
                }
            ).set_placement((origin[0], origin[1] - 10))

    def _draw_title_block(self, msp, spec: FurnitureSpec):
        """タイトルブロック"""
        # 図面右下にタイトルブロック
        x, y = 300, -50

        # 枠
        self._draw_rect(msp, x, y, 150, 40, 'OUTLINE')

        # テキスト
        msp.add_text(
            f"{spec.name_ja} ({spec.name})",
            dxfattribs={'layer': 'TEXT', 'height': 4}
        ).set_placement((x + 5, y + 30))

        msp.add_text(
            f"W{spec.width:.0f} x D{spec.depth:.0f} x H{spec.height:.0f} mm",
            dxfattribs={'layer': 'TEXT', 'height': 3}
        ).set_placement((x + 5, y + 22))

        msp.add_text(
            f"板厚: {spec.thickness:.0f}mm | 材質: {spec.material}",
            dxfattribs={'layer': 'TEXT', 'height': 2.5}
        ).set_placement((x + 5, y + 15))

        msp.add_text(
            f"Scale: 1:{int(1/self.scale) if self.scale < 1 else 1}",
            dxfattribs={'layer': 'TEXT', 'height': 2.5}
        ).set_placement((x + 5, y + 8))

    def _draw_parts_layout_dxf(self, msp, spec: FurnitureSpec):
        """パーツレイアウト図（板取り図）"""
        # FurnitureDrawingGeneratorからパーツ計算を借用
        gen = FurnitureDrawingGenerator()
        parts = gen._calculate_parts(spec)

        if not parts:
            return

        # パーツを配置（簡易板取り）
        x_start = 500
        y_start = 0
        x = x_start
        y = y_start
        row_height = 0
        max_width = 400

        msp.add_text(
            "パーツレイアウト (Parts Layout)",
            dxfattribs={'layer': 'TEXT', 'height': 5}
        ).set_placement((x_start, y_start + 20))

        for part in parts:
            for _ in range(part.quantity):
                pw = self._scaled(part.width)
                ph = self._scaled(part.depth)

                # 次の行へ
                if x + pw > x_start + max_width:
                    x = x_start
                    y -= row_height + 10
                    row_height = 0

                # パーツ描画
                self._draw_rect(msp, x, y - ph, pw, ph, 'PARTS')

                # ラベル
                msp.add_text(
                    part.name_ja,
                    dxfattribs={'layer': 'TEXT', 'height': 2}
                ).set_placement((x + 2, y - ph/2))

                # 寸法テキスト
                msp.add_text(
                    f"{part.width:.0f}x{part.depth:.0f}",
                    dxfattribs={'layer': 'TEXT', 'height': 1.5}
                ).set_placement((x + 2, y - ph/2 - 3))

                row_height = max(row_height, ph)
                x += pw + 10


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def generate_drawing_from_obj(
    obj_path: Path,
    output: Optional[str] = None,
    furniture_type: Optional[str] = None,
    material: str = "plywood",
    scale: float = 0.1,
    cutlist: bool = True,
    dxf: bool = False,
    dxf_scale: float = 1.0
) -> Tuple[Path, Optional[Path], Optional[Path], FurnitureSpec]:
    """
    OBJファイルから図面を自動生成

    Args:
        obj_path: OBJファイルパス
        output: 出力ファイル名
        furniture_type: 家具タイプ（省略時は自動推定）
        material: 材質
        scale: SVG図面スケール
        cutlist: カットリストも生成するか
        dxf: DXF(CAD)図面も生成するか
        dxf_scale: DXF図面のスケール（1.0 = 1:1実寸）

    Returns:
        (drawing_path, cutlist_path, dxf_path, spec)
    """
    obj_path = Path(obj_path)

    # OBJ解析
    analyzer = OBJAnalyzer()
    spec = analyzer.analyze(obj_path, furniture_type=furniture_type)
    spec.material = material

    # SVG図面生成
    gen = FurnitureDrawingGenerator(scale=scale)
    output_name = output or f"{obj_path.stem}_drawing.svg"
    drawing_path = gen.generate_three_view(spec, output=output_name)

    # カットリスト
    cutlist_path = None
    if cutlist:
        cutlist_name = output_name.replace('.svg', '_cutlist.json') if output else f"{obj_path.stem}_cutlist.json"
        cutlist_path = gen.generate_cut_list(spec, output=cutlist_name)

    # DXF図面生成
    dxf_path = None
    if dxf:
        cad_gen = CADDrawingGenerator(scale=dxf_scale)
        dxf_name = f"{obj_path.stem}_cad.dxf"
        dxf_path = cad_gen.generate_dxf(spec, output=dxf_name)

    return drawing_path, cutlist_path, dxf_path, spec


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Furniture Drawing Generator - OBJから自動図面生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # OBJファイルから自動生成
  python3 furniture_drawing_generator.py shelf.obj

  # CAD用DXF図面を生成
  python3 furniture_drawing_generator.py shelf.obj --dxf

  # 家具タイプを指定
  python3 furniture_drawing_generator.py table.obj --type desk

  # 手動で寸法を指定
  python3 furniture_drawing_generator.py --manual --type bookshelf --width 900 --height 1800

  # バッチ処理
  python3 furniture_drawing_generator.py *.obj --batch --dxf
        """
    )

    # OBJファイル入力
    parser.add_argument('input', nargs='*', help='OBJファイル（複数可）')

    # 共通オプション
    parser.add_argument('--type', choices=['bookshelf', 'desk', 'table', 'cabinet', 'shelf', 'chair'],
                        help='Furniture type (auto-detected if not specified)')
    parser.add_argument('--material', default='plywood', help='Material')
    parser.add_argument('--scale', type=float, default=0.1, help='Drawing scale (default: 0.1 = 1:10)')
    parser.add_argument('--output', '-o', help='Output filename')
    parser.add_argument('--cutlist', action='store_true', default=True, help='Generate cut list (default: True)')
    parser.add_argument('--no-cutlist', action='store_true', help='Skip cut list generation')

    # CAD (DXF) オプション
    parser.add_argument('--dxf', action='store_true', help='Generate CAD DXF file (AutoCAD/Fusion360 compatible)')
    parser.add_argument('--dxf-scale', type=float, default=1.0, help='DXF scale (default: 1.0 = 1:1 full scale)')
    parser.add_argument('--dxf-only', action='store_true', help='Generate DXF only (skip SVG)')

    # 手動入力モード
    parser.add_argument('--manual', action='store_true', help='Manual mode (specify dimensions)')
    parser.add_argument('--width', type=float, default=800, help='Width (mm) for manual mode')
    parser.add_argument('--depth', type=float, default=300, help='Depth (mm) for manual mode')
    parser.add_argument('--height', type=float, default=1800, help='Height (mm) for manual mode')
    parser.add_argument('--thickness', type=float, default=18, help='Board thickness (mm)')
    parser.add_argument('--shelves', type=int, default=5, help='Number of shelves')

    # バッチモード
    parser.add_argument('--batch', '-b', action='store_true', help='Batch mode (summary only)')

    args = parser.parse_args()

    cutlist_enabled = args.cutlist and not args.no_cutlist
    dxf_enabled = args.dxf or args.dxf_only

    # 手動モード
    if args.manual or not args.input:
        if not args.type and not args.manual:
            parser.print_help()
            print("\n使用例: python3 furniture_drawing_generator.py shelf.obj")
            return

        furniture_type = args.type or 'bookshelf'
        spec = FurnitureSpec(
            name=furniture_type.capitalize(),
            name_ja={'bookshelf': '本棚', 'desk': 'デスク', 'table': 'テーブル',
                     'cabinet': 'キャビネット', 'shelf': '棚', 'chair': '椅子'}.get(furniture_type, furniture_type),
            type=furniture_type,
            width=args.width,
            depth=args.depth,
            height=args.height,
            thickness=args.thickness,
            shelves=args.shelves,
            material=args.material
        )

        # SVG生成
        if not args.dxf_only:
            gen = FurnitureDrawingGenerator(scale=args.scale)
            drawing_path = gen.generate_three_view(spec, output=args.output)
            print(f"\nSVG Drawing: {drawing_path}")

            if cutlist_enabled:
                cutlist_path = gen.generate_cut_list(spec)
                print(f"Cut list: {cutlist_path}")

        # DXF生成
        if dxf_enabled:
            cad_gen = CADDrawingGenerator(scale=args.dxf_scale)
            dxf_path = cad_gen.generate_dxf(spec)
            print(f"DXF (CAD): {dxf_path}")

        return

    # OBJファイルモード
    results = []
    for input_path in args.input:
        obj_path = Path(input_path)
        if not obj_path.exists():
            print(f"⚠ ファイルが見つかりません: {obj_path}")
            continue

        try:
            print(f"\n解析中: {obj_path.name}...")

            drawing_path, cutlist_path, dxf_path, spec = generate_drawing_from_obj(
                obj_path,
                output=args.output if len(args.input) == 1 else None,
                furniture_type=args.type,
                material=args.material,
                scale=args.scale,
                cutlist=cutlist_enabled,
                dxf=dxf_enabled,
                dxf_scale=args.dxf_scale
            )

            if args.batch:
                dxf_mark = " [DXF]" if dxf_path else ""
                print(f"  ✓ {spec.name_ja} ({spec.type}) - {spec.width:.0f}x{spec.depth:.0f}x{spec.height:.0f}mm, 棚板: {spec.shelves}{dxf_mark}")
            else:
                print(f"\n{'='*60}")
                print(f"  ファイル: {obj_path.name}")
                print(f"  家具タイプ: {spec.name_ja} ({spec.type})")
                print(f"  寸法: W{spec.width:.0f} x D{spec.depth:.0f} x H{spec.height:.0f} mm")
                print(f"  板厚: {spec.thickness:.0f} mm")
                print(f"  棚板数: {spec.shelves}")
                print(f"  材質: {spec.material}")
                print(f"{'='*60}")
                print(f"  SVG図面: {drawing_path}")
                if cutlist_path:
                    print(f"  カットリスト: {cutlist_path}")
                if dxf_path:
                    print(f"  DXF (CAD): {dxf_path}")

            results.append((obj_path.name, spec, drawing_path, dxf_path))

        except Exception as e:
            print(f"⚠ エラー ({obj_path.name}): {e}")
            import traceback
            traceback.print_exc()

    # バッチサマリー
    if args.batch and len(results) > 1:
        dxf_count = sum(1 for r in results if r[3] is not None)
        dxf_info = f" ({dxf_count}件 DXF)" if dxf_enabled else ""
        print(f"\n合計: {len(results)}件の図面を生成しました{dxf_info}")


if __name__ == "__main__":
    main()
