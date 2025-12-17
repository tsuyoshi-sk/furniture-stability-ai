#!/usr/bin/env python3
"""
Furniture Drawing Generator
家具図面生成システム - 技術図面・三面図・組立図

Features:
- Technical Drawings: 正確な寸法付き図面
- Three-View: 三面図（正面・側面・上面）
- Assembly Diagrams: 組立図・分解図
- Parts List: パーツリスト・カットリスト生成
- DXF Export: CAD互換出力
"""
import math
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import svgwrite
from svgwrite import cm, mm

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
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
        if spec.type == 'bookshelf':
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

        if spec.type == 'bookshelf':
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
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Furniture Drawing Generator')
    parser.add_argument('--type', choices=['bookshelf', 'desk', 'table', 'cabinet'],
                        default='bookshelf', help='Furniture type')
    parser.add_argument('--width', type=float, default=800, help='Width (mm)')
    parser.add_argument('--depth', type=float, default=300, help='Depth (mm)')
    parser.add_argument('--height', type=float, default=1800, help='Height (mm)')
    parser.add_argument('--thickness', type=float, default=18, help='Board thickness (mm)')
    parser.add_argument('--shelves', type=int, default=5, help='Number of shelves')
    parser.add_argument('--material', default='plywood', help='Material')
    parser.add_argument('--scale', type=float, default=0.1, help='Drawing scale')
    parser.add_argument('--output', '-o', help='Output filename')
    parser.add_argument('--cutlist', action='store_true', help='Also generate cut list')
    args = parser.parse_args()

    spec = FurnitureSpec(
        name=args.type.capitalize(),
        name_ja={'bookshelf': '本棚', 'desk': 'デスク', 'table': 'テーブル', 'cabinet': 'キャビネット'}.get(args.type, args.type),
        type=args.type,
        width=args.width,
        depth=args.depth,
        height=args.height,
        thickness=args.thickness,
        shelves=args.shelves,
        material=args.material
    )

    gen = FurnitureDrawingGenerator(scale=args.scale)

    # Generate three-view drawing
    drawing_path = gen.generate_three_view(spec, output=args.output)
    print(f"\nDrawing: {drawing_path}")

    # Generate cut list if requested
    if args.cutlist:
        cutlist_path = gen.generate_cut_list(spec)
        print(f"Cut list: {cutlist_path}")


if __name__ == "__main__":
    main()
