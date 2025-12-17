#!/usr/bin/env python3
"""
Bracket Database for Load Capacity Calculation
Based on:
- AA-SYSTEM 2026 catalog data
- SUGATSUNE No.390 catalog (2021-2024)
"""
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


@dataclass
class Bracket:
    """Bracket specification"""
    code: str                    # Product code (e.g., "A-32")
    name: str                    # Japanese name
    name_en: str                 # English name
    category: str                # wood, glass, hanger, etc.
    type: str                    # horizontal, inclined, etc.
    sizes: List[int]             # Available sizes (mm)
    load_capacity_per_pair: Dict[str, float]  # {support_type: kg}
    material: str                # Steel, Stainless, etc.
    thickness_mm: float          # Material thickness
    shelf_type: str              # wood, glass, both
    notes: str = ""


# =============================================================================
# WOOD BRACKET DATABASE
# 木棚板専用ブラケット
# =============================================================================

WOOD_BRACKETS = [
    # B-032/B-033 Series - Basic fold bracket
    Bracket(
        code="B-032",
        name="フォールドブラケット",
        name_en="Fold Bracket",
        category="wood",
        type="horizontal",
        sizes=[150, 200, 250, 300, 350],
        load_capacity_per_pair={
            "ASF-1": 19,      # Channel support 11mm
            "S1B-50/50": 17,  # Slit pipe 50/50
        },
        material="Steel",
        thickness_mm=2.6,
        shelf_type="wood",
        notes="Basic entry-level bracket"
    ),
    Bracket(
        code="B-132",
        name="フォールドブラケット",
        name_en="Fold Bracket Medium",
        category="wood",
        type="horizontal",
        sizes=[150, 200, 250, 300, 350],
        load_capacity_per_pair={
            "ASF-1": 35,
            "S1B-50/50": 34,
            "ASF-1_alt": 30,
            "S1B-50/50_alt": 29,
        },
        material="Steel",
        thickness_mm=2.6,
        shelf_type="wood",
        notes="Medium duty bracket"
    ),
    Bracket(
        code="B-232",
        name="フォールドブラケット",
        name_en="Fold Bracket Heavy",
        category="wood",
        type="horizontal",
        sizes=[400, 450],
        load_capacity_per_pair={
            "ASF-1": 33,
            "S1B-50/50": 34,
            "ASF-1_long": 28,
            "S1B-50/50_long": 31,
        },
        material="Steel",
        thickness_mm=2.6,
        shelf_type="wood",
        notes="Heavy duty for longer shelves"
    ),
    # A-32/A-33 Series - Standard wood bracket
    Bracket(
        code="A-32",
        name="ウッドブラケット",
        name_en="Wood Bracket",
        category="wood",
        type="horizontal",
        sizes=[150, 200, 250, 300, 350, 400, 450, 550, 600],
        load_capacity_per_pair={
            "ASF-1": 23,
            "S1B-50/50": 34,
            "ASF-1_short": 17,
            "S1B-50/50_short": 31,
            "ASF-1_medium": 32,
            "S1B-50/50_medium": 31,
        },
        material="Steel",
        thickness_mm=2.6,
        shelf_type="wood",
        notes="Standard wood shelf bracket, 0.75° UP angle"
    ),
    # DA-32/DA-33 - With fixing dowel
    Bracket(
        code="DA-32",
        name="かしめダボ付きウッドブラケット",
        name_en="Wood Bracket with Dowel",
        category="wood",
        type="horizontal",
        sizes=[250, 300, 350, 400],
        load_capacity_per_pair={
            "ASF-1": 30,
            "S1B-50/50": 31,
            "ASF-1_alt": 23,
            "S1B-50/50_alt": 24,
        },
        material="Steel",
        thickness_mm=2.6,
        shelf_type="wood",
        notes="Pre-installed fixing dowel"
    ),
    # A-15S - Dowel-only bracket
    Bracket(
        code="A-15S",
        name="ダボ付きウッドブラケット",
        name_en="Dowel Wood Bracket",
        category="wood",
        type="horizontal",
        sizes=[100, 150, 200, 250, 300, 350],
        load_capacity_per_pair={
            "default": 15,  # Estimated based on smaller design
        },
        material="Steel",
        thickness_mm=2.5,
        shelf_type="wood",
        notes="Automatic lock mechanism, dowel only fixing"
    ),
    # R-032W/R-033W - R-type bracket
    Bracket(
        code="R-032W",
        name="Rタイプウッドブラケット",
        name_en="R-Type Wood Bracket",
        category="wood",
        type="horizontal",
        sizes=[250, 300],
        load_capacity_per_pair={
            "default": 25,  # Estimated
        },
        material="Steel",
        thickness_mm=2.6,
        shelf_type="wood",
        notes="R-type with automatic lock"
    ),
    # AL-55S/AL-56S - Inclined bracket
    Bracket(
        code="AL-55S",
        name="木棚用傾斜ブラケット",
        name_en="Inclined Wood Bracket",
        category="wood",
        type="inclined",
        sizes=[100, 150, 200, 250, 300, 350, 400],
        load_capacity_per_pair={
            "ASF-1": 17,
            "S1B-50/50": 16,
            "ASF-1_long": 22,
            "S1B-50/50_long": 18,
        },
        material="Steel",
        thickness_mm=2.6,
        shelf_type="wood",
        notes="Adjustable angle 15° increments up to 90°"
    ),
]


# =============================================================================
# GLASS BRACKET DATABASE
# ガラス棚板専用ブラケット
# =============================================================================

GLASS_BRACKETS = [
    # G-32 Series
    Bracket(
        code="G-32",
        name="ガラスブラケット",
        name_en="Glass Bracket",
        category="glass",
        type="horizontal",
        sizes=[150, 200, 250, 300, 350, 400],
        load_capacity_per_pair={
            "ASF-1": 15,
            "S1B-50/50": 14,
        },
        material="Steel",
        thickness_mm=2.0,
        shelf_type="glass",
        notes="With transparent cushion"
    ),
    # Multi-function holder bracket
    Bracket(
        code="HF-32",
        name="ホルダー用多機能ブラケット",
        name_en="Multi-function Holder Bracket",
        category="glass",
        type="horizontal",
        sizes=[150, 200, 250, 300, 350],
        load_capacity_per_pair={
            "default": 12,
        },
        material="Steel",
        thickness_mm=2.0,
        shelf_type="both",
        notes="For glass and wood shelves"
    ),
]


# =============================================================================
# SPECIALTY BRACKETS (Stainless, Designer Series)
# =============================================================================

SPECIALTY_BRACKETS = [
    # DE-G522 Glass Rod
    Bracket(
        code="DE-G522",
        name="グラスロッド",
        name_en="Glass Rod",
        category="glass",
        type="horizontal",
        sizes=[100, 150, 200, 250, 300, 350, 400, 450],
        load_capacity_per_pair={
            "DE-L": 32,  # At size 300
        },
        material="Stainless SUS-304",
        thickness_mm=5.0,
        shelf_type="glass",
        notes="22x5t flat bar, designer series"
    ),
    # QU-SA Sail Arm
    Bracket(
        code="QU-SA",
        name="セイルアーム",
        name_en="Sail Arm",
        category="glass",
        type="horizontal",
        sizes=[300, 350, 400, 450],
        load_capacity_per_pair={
            "QU-system": 25,
        },
        material="Stainless SUS-304",
        thickness_mm=1.5,
        shelf_type="glass",
        notes="19x19 square pipe, Quattrocca system"
    ),
    # QU-SH1200 Sail Hanger
    Bracket(
        code="QU-SH1200",
        name="セイルハンガー",
        name_en="Sail Hanger",
        category="hanger",
        type="horizontal",
        sizes=[300, 350, 400, 450],
        load_capacity_per_pair={
            "QU-system": 16,
        },
        material="Stainless SUS-304",
        thickness_mm=1.5,
        shelf_type="hanger",
        notes="For clothing hangers"
    ),
]


# =============================================================================
# SUGATSUNE BRACKETS (スガツネ No.390 カタログ)
# 耐荷重データは公式カタログ値に基づく（kgf/本 → kg/pair に換算）
# =============================================================================

SUGATSUNE_BRACKETS = [
    # =========================================================================
    # BT型 - ステンレス鋼製棚受 (SUS304, サテン仕上)
    # 深絞り成形 + 補強板で高強度
    # =========================================================================
    Bracket(
        code="BT-85",
        name="ステンレス鋼製棚受",
        name_en="Stainless Steel Bracket BT",
        category="wood",
        type="horizontal",
        sizes=[69],  # L寸法
        load_capacity_per_pair={"wall": 30},  # 15kgf/本 × 2
        material="Stainless SUS-304",
        thickness_mm=4.0,
        shelf_type="wood",
        notes="SUGATSUNE BT型 H=86mm 深絞り成形+補強板"
    ),
    Bracket(
        code="BT-120",
        name="ステンレス鋼製棚受",
        name_en="Stainless Steel Bracket BT",
        category="wood",
        type="horizontal",
        sizes=[100],  # L寸法
        load_capacity_per_pair={"wall": 40},  # 20kgf/本 × 2
        material="Stainless SUS-304",
        thickness_mm=4.0,
        shelf_type="wood",
        notes="SUGATSUNE BT型 H=120mm"
    ),
    Bracket(
        code="BT-180",
        name="ステンレス鋼製棚受",
        name_en="Stainless Steel Bracket BT",
        category="wood",
        type="horizontal",
        sizes=[149],  # L=148.5mm
        load_capacity_per_pair={"wall": 50},  # 25kgf/本 × 2
        material="Stainless SUS-304",
        thickness_mm=4.0,
        shelf_type="wood",
        notes="SUGATSUNE BT型 H=180.5mm"
    ),
    Bracket(
        code="BT-240",
        name="ステンレス鋼製棚受",
        name_en="Stainless Steel Bracket BT",
        category="wood",
        type="horizontal",
        sizes=[199],  # L寸法
        load_capacity_per_pair={"wall": 56},  # 28kgf/本 × 2
        material="Stainless SUS-304",
        thickness_mm=4.0,
        shelf_type="wood",
        notes="SUGATSUNE BT型 H=240mm"
    ),
    Bracket(
        code="BT-300",
        name="ステンレス鋼製棚受",
        name_en="Stainless Steel Bracket BT",
        category="wood",
        type="horizontal",
        sizes=[239],  # L=238.5mm
        load_capacity_per_pair={"wall": 70},  # 35kgf/本 × 2
        material="Stainless SUS-304",
        thickness_mm=4.0,
        shelf_type="wood",
        notes="SUGATSUNE BT型 H=301mm"
    ),
    Bracket(
        code="BT-380",
        name="ステンレス鋼製棚受",
        name_en="Stainless Steel Bracket BT",
        category="wood",
        type="horizontal",
        sizes=[318],  # L寸法
        load_capacity_per_pair={"wall": 130},  # 65kgf/本 × 2
        material="Stainless SUS-304",
        thickness_mm=5.0,
        shelf_type="wood",
        notes="SUGATSUNE BT型 H=378mm 重量用"
    ),
    Bracket(
        code="BT-480",
        name="ステンレス鋼製棚受",
        name_en="Stainless Steel Bracket BT",
        category="wood",
        type="horizontal",
        sizes=[400],  # L寸法
        load_capacity_per_pair={"wall": 165},  # 82.5kgf/本 × 2
        material="Stainless SUS-304",
        thickness_mm=5.5,
        shelf_type="wood",
        notes="SUGATSUNE BT型 H=480mm 超重量用"
    ),
    # =========================================================================
    # BTK型 - 棚受 (鋼SECC, 粉体焼付塗装)
    # BT型と同寸法・同耐荷重、カラーバリエーション有
    # =========================================================================
    Bracket(
        code="BTK-85",
        name="棚受",
        name_en="Bracket BTK",
        category="wood",
        type="horizontal",
        sizes=[69],
        load_capacity_per_pair={"wall": 30},  # 15kgf/本 × 2
        material="Steel SECC",
        thickness_mm=4.0,
        shelf_type="wood",
        notes="SUGATSUNE BTK型 H=86mm ホワイト/ブラック/アンバー"
    ),
    Bracket(
        code="BTK-120",
        name="棚受",
        name_en="Bracket BTK",
        category="wood",
        type="horizontal",
        sizes=[100],
        load_capacity_per_pair={"wall": 40},  # 20kgf/本 × 2
        material="Steel SECC",
        thickness_mm=4.0,
        shelf_type="wood",
        notes="SUGATSUNE BTK型 H=120mm"
    ),
    Bracket(
        code="BTK-180",
        name="棚受",
        name_en="Bracket BTK",
        category="wood",
        type="horizontal",
        sizes=[149],
        load_capacity_per_pair={"wall": 50},  # 25kgf/本 × 2
        material="Steel SECC",
        thickness_mm=4.0,
        shelf_type="wood",
        notes="SUGATSUNE BTK型 H=180.5mm"
    ),
    Bracket(
        code="BTK-240",
        name="棚受",
        name_en="Bracket BTK",
        category="wood",
        type="horizontal",
        sizes=[199],
        load_capacity_per_pair={"wall": 56},  # 28kgf/本 × 2
        material="Steel SECC",
        thickness_mm=4.0,
        shelf_type="wood",
        notes="SUGATSUNE BTK型 H=240mm"
    ),
    Bracket(
        code="BTK-300",
        name="棚受",
        name_en="Bracket BTK",
        category="wood",
        type="horizontal",
        sizes=[239],
        load_capacity_per_pair={"wall": 70},  # 35kgf/本 × 2
        material="Steel SECC",
        thickness_mm=4.0,
        shelf_type="wood",
        notes="SUGATSUNE BTK型 H=301mm"
    ),
    Bracket(
        code="BTK-380",
        name="棚受",
        name_en="Bracket BTK",
        category="wood",
        type="horizontal",
        sizes=[318],
        load_capacity_per_pair={"wall": 130},  # 65kgf/本 × 2
        material="Steel SECC",
        thickness_mm=5.0,
        shelf_type="wood",
        notes="SUGATSUNE BTK型 H=378mm 重量用"
    ),
    Bracket(
        code="BTK-480",
        name="棚受",
        name_en="Bracket BTK",
        category="wood",
        type="horizontal",
        sizes=[400],
        load_capacity_per_pair={"wall": 165},  # 82.5kgf/本 × 2
        material="Steel SECC",
        thickness_mm=5.5,
        shelf_type="wood",
        notes="SUGATSUNE BTK型 H=480mm 超重量用"
    ),
    # BTK-UB型 - コンパクトタイプ
    Bracket(
        code="BTK-UB100",
        name="棚受コンパクトタイプ",
        name_en="Compact Bracket BTK-UB",
        category="wood",
        type="horizontal",
        sizes=[100],
        load_capacity_per_pair={"wall": 24},  # 12kgf × 2
        material="Steel",
        thickness_mm=2.0,
        shelf_type="wood",
        notes="SUGATSUNE BTK-UB型 高さ半分以下"
    ),
    Bracket(
        code="BTK-UB120",
        name="棚受コンパクトタイプ",
        name_en="Compact Bracket BTK-UB",
        category="wood",
        type="horizontal",
        sizes=[120],
        load_capacity_per_pair={"wall": 40},  # 20kgf × 2
        material="Steel",
        thickness_mm=2.0,
        shelf_type="wood",
        notes="SUGATSUNE BTK-UB型"
    ),
    Bracket(
        code="BTK-UB160",
        name="棚受コンパクトタイプ",
        name_en="Compact Bracket BTK-UB",
        category="wood",
        type="horizontal",
        sizes=[160],
        load_capacity_per_pair={"wall": 50},  # 25kgf × 2
        material="Steel",
        thickness_mm=2.0,
        shelf_type="wood",
        notes="SUGATSUNE BTK-UB型"
    ),
    # SU-A型 - ステンレス鋼製アングル
    Bracket(
        code="SU-A70",
        name="ステンレス鋼製アングル",
        name_en="Stainless Angle SU-A",
        category="wood",
        type="horizontal",
        sizes=[70],
        load_capacity_per_pair={"wall": 48},  # 24kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="both",
        notes="SUGATSUNE SU-A型"
    ),
    Bracket(
        code="SU-A90",
        name="ステンレス鋼製アングル",
        name_en="Stainless Angle SU-A",
        category="wood",
        type="horizontal",
        sizes=[90],
        load_capacity_per_pair={"wall": 34},  # 17kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="both",
        notes="SUGATSUNE SU-A型"
    ),
    Bracket(
        code="SU-A120",
        name="ステンレス鋼製アングル",
        name_en="Stainless Angle SU-A",
        category="wood",
        type="horizontal",
        sizes=[120],
        load_capacity_per_pair={"wall": 26},  # 13kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="both",
        notes="SUGATSUNE SU-A型"
    ),
    Bracket(
        code="SU-A150",
        name="ステンレス鋼製アングル",
        name_en="Stainless Angle SU-A",
        category="wood",
        type="horizontal",
        sizes=[150],
        load_capacity_per_pair={"wall": 20},  # 10kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="both",
        notes="SUGATSUNE SU-A型"
    ),
    Bracket(
        code="SU-A180",
        name="ステンレス鋼製アングル",
        name_en="Stainless Angle SU-A",
        category="wood",
        type="horizontal",
        sizes=[180],
        load_capacity_per_pair={"wall": 18},  # 9kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="both",
        notes="SUGATSUNE SU-A型"
    ),
    # XL-SA01型 - 面取り形状アングル
    Bracket(
        code="XL-SA01-120",
        name="面取り形状アングル",
        name_en="Chamfered Angle XL-SA01",
        category="wood",
        type="horizontal",
        sizes=[120],
        load_capacity_per_pair={"wall": 46},  # 23kgf × 2
        material="Stainless SUS-304",
        thickness_mm=8.0,
        shelf_type="both",
        notes="SUGATSUNE XL-SA01型"
    ),
    Bracket(
        code="XL-SA01-150",
        name="面取り形状アングル",
        name_en="Chamfered Angle XL-SA01",
        category="wood",
        type="horizontal",
        sizes=[150],
        load_capacity_per_pair={"wall": 36},  # 18kgf × 2
        material="Stainless SUS-304",
        thickness_mm=8.0,
        shelf_type="both",
        notes="SUGATSUNE XL-SA01型"
    ),
    Bracket(
        code="XL-SA01-180",
        name="面取り形状アングル",
        name_en="Chamfered Angle XL-SA01",
        category="wood",
        type="horizontal",
        sizes=[180],
        load_capacity_per_pair={"wall": 28},  # 14kgf × 2
        material="Stainless SUS-304",
        thickness_mm=8.0,
        shelf_type="both",
        notes="SUGATSUNE XL-SA01型"
    ),
    Bracket(
        code="XL-SA01-240",
        name="面取り形状アングル",
        name_en="Chamfered Angle XL-SA01",
        category="wood",
        type="horizontal",
        sizes=[240],
        load_capacity_per_pair={"wall": 24},  # 12kgf × 2
        material="Stainless SUS-304",
        thickness_mm=8.0,
        shelf_type="both",
        notes="SUGATSUNE XL-SA01型"
    ),
    # 10913型 - 重量用棚受
    Bracket(
        code="10913-295",
        name="重量用棚受",
        name_en="Heavy Duty Bracket 10913",
        category="wood",
        type="horizontal",
        sizes=[295],
        load_capacity_per_pair={"wall": 140},  # 70kgf × 2
        material="Steel",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE 10913型 エレメントシステム"
    ),
    Bracket(
        code="10913-395",
        name="重量用棚受",
        name_en="Heavy Duty Bracket 10913",
        category="wood",
        type="horizontal",
        sizes=[395],
        load_capacity_per_pair={"wall": 140},  # 70kgf × 2
        material="Steel",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE 10913型 エレメントシステム"
    ),
    Bracket(
        code="10913-495",
        name="重量用棚受",
        name_en="Heavy Duty Bracket 10913",
        category="wood",
        type="horizontal",
        sizes=[495],
        load_capacity_per_pair={"wall": 140},  # 70kgf × 2
        material="Steel",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE 10913型 エレメントシステム"
    ),
    # 10910型 - スマート棚受
    Bracket(
        code="10910-120",
        name="スマート棚受",
        name_en="Smart Bracket 10910",
        category="wood",
        type="horizontal",
        sizes=[120],
        load_capacity_per_pair={"wall": 14},  # 7kgf × 2
        material="Steel",
        thickness_mm=2.0,
        shelf_type="wood",
        notes="SUGATSUNE 10910型"
    ),
    Bracket(
        code="10910-170",
        name="スマート棚受",
        name_en="Smart Bracket 10910",
        category="wood",
        type="horizontal",
        sizes=[170],
        load_capacity_per_pair={"wall": 14},  # 7kgf × 2
        material="Steel",
        thickness_mm=2.0,
        shelf_type="wood",
        notes="SUGATSUNE 10910型"
    ),
    Bracket(
        code="10910-220",
        name="スマート棚受",
        name_en="Smart Bracket 10910",
        category="wood",
        type="horizontal",
        sizes=[220],
        load_capacity_per_pair={"wall": 20},  # 10kgf × 2
        material="Steel",
        thickness_mm=2.0,
        shelf_type="wood",
        notes="SUGATSUNE 10910型"
    ),
    Bracket(
        code="10910-267",
        name="スマート棚受",
        name_en="Smart Bracket 10910",
        category="wood",
        type="horizontal",
        sizes=[267],
        load_capacity_per_pair={"wall": 30},  # 15kgf × 2
        material="Steel",
        thickness_mm=2.0,
        shelf_type="wood",
        notes="SUGATSUNE 10910型"
    ),
    Bracket(
        code="10910-317",
        name="スマート棚受",
        name_en="Smart Bracket 10910",
        category="wood",
        type="horizontal",
        sizes=[317],
        load_capacity_per_pair={"wall": 30},  # 15kgf × 2
        material="Steel",
        thickness_mm=2.0,
        shelf_type="wood",
        notes="SUGATSUNE 10910型"
    ),
    # BY型 - ステンレス鋼製ブラケット
    Bracket(
        code="BY-300",
        name="ステンレス鋼製ブラケット",
        name_en="Stainless Bracket BY",
        category="wood",
        type="horizontal",
        sizes=[300],
        load_capacity_per_pair={"wall": 100},  # 50kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE BY型 91°タイプ"
    ),
    Bracket(
        code="BY-400",
        name="ステンレス鋼製ブラケット",
        name_en="Stainless Bracket BY",
        category="wood",
        type="horizontal",
        sizes=[400],
        load_capacity_per_pair={"wall": 100},  # 50kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE BY型"
    ),
    Bracket(
        code="BY-500",
        name="ステンレス鋼製ブラケット",
        name_en="Stainless Bracket BY",
        category="wood",
        type="horizontal",
        sizes=[500],
        load_capacity_per_pair={"wall": 100},  # 50kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE BY型"
    ),
    # EB型 - 折りたたみ棚受
    Bracket(
        code="EB-200",
        name="折りたたみ棚受",
        name_en="Folding Bracket EB",
        category="wood",
        type="folding",
        sizes=[200],
        load_capacity_per_pair={"wall": 112},  # 56kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE EB型 ヒンジ不要"
    ),
    Bracket(
        code="EB-303",
        name="折りたたみ棚受",
        name_en="Folding Bracket EB",
        category="wood",
        type="folding",
        sizes=[303],
        load_capacity_per_pair={"wall": 100},  # 50kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE EB型 ヒンジ不要"
    ),
    Bracket(
        code="EB-317",
        name="折りたたみ棚受",
        name_en="Folding Bracket EB",
        category="wood",
        type="folding",
        sizes=[317],
        load_capacity_per_pair={"wall": 220},  # 110kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE EB型 重量用"
    ),
    Bracket(
        code="EB-303-DA",
        name="折りたたみ棚受ダンパー内蔵",
        name_en="Folding Bracket EB-DA with Damper",
        category="wood",
        type="folding",
        sizes=[303],
        load_capacity_per_pair={"wall": 100},  # 50kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE EB-DA型 ソフトクローズ"
    ),
    Bracket(
        code="EB-317-DA",
        name="折りたたみ棚受ダンパー内蔵",
        name_en="Folding Bracket EB-DA with Damper",
        category="wood",
        type="folding",
        sizes=[317],
        load_capacity_per_pair={"wall": 220},  # 110kgf × 2
        material="Stainless SUS-304",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE EB-DA型 ソフトクローズ 重量用"
    ),
    # 388型 - 伸縮棚受
    Bracket(
        code="388-25-252",
        name="伸縮棚受",
        name_en="Telescopic Bracket 388",
        category="wood",
        type="telescopic",
        sizes=[252],
        load_capacity_per_pair={"wall": 62},  # 31kgf × 2
        material="Aluminum",
        thickness_mm=2.0,
        shelf_type="wood",
        notes="SUGATSUNE 388型 アルミ軽量"
    ),
    Bracket(
        code="388-25-402",
        name="伸縮棚受",
        name_en="Telescopic Bracket 388",
        category="wood",
        type="telescopic",
        sizes=[402],
        load_capacity_per_pair={"wall": 66},  # 33kgf × 2
        material="Aluminum",
        thickness_mm=2.0,
        shelf_type="wood",
        notes="SUGATSUNE 388型"
    ),
    Bracket(
        code="388-25-549",
        name="伸縮棚受",
        name_en="Telescopic Bracket 388",
        category="wood",
        type="telescopic",
        sizes=[549],
        load_capacity_per_pair={"wall": 72},  # 36kgf × 2
        material="Aluminum",
        thickness_mm=2.0,
        shelf_type="wood",
        notes="SUGATSUNE 388型"
    ),
    # SSA型 - ステンレス鋼製棚受アングル
    Bracket(
        code="SSA-100",
        name="ステンレス鋼製棚受アングル",
        name_en="Stainless Shelf Angle SSA",
        category="wood",
        type="horizontal",
        sizes=[100],
        load_capacity_per_pair={"wall": 150},  # 150kgf/2pcs
        material="Stainless SUS-304",
        thickness_mm=2.0,
        shelf_type="both",
        notes="SUGATSUNE SSA型 L型棚受"
    ),
    Bracket(
        code="SSA-150",
        name="ステンレス鋼製棚受アングル",
        name_en="Stainless Shelf Angle SSA",
        category="wood",
        type="horizontal",
        sizes=[150],
        load_capacity_per_pair={"wall": 150},  # 150kgf/2pcs
        material="Stainless SUS-304",
        thickness_mm=2.0,
        shelf_type="both",
        notes="SUGATSUNE SSA型"
    ),
    Bracket(
        code="SSA-200",
        name="ステンレス鋼製棚受アングル",
        name_en="Stainless Shelf Angle SSA",
        category="wood",
        type="horizontal",
        sizes=[200],
        load_capacity_per_pair={"wall": 150},  # 150kgf/2pcs
        material="Stainless SUS-304",
        thickness_mm=2.0,
        shelf_type="both",
        notes="SUGATSUNE SSA型"
    ),
    # 隠し棚受 IT7020型
    Bracket(
        code="IT7020-N200",
        name="隠し棚受",
        name_en="Hidden Bracket IT7020",
        category="wood",
        type="hidden",
        sizes=[200],
        load_capacity_per_pair={"wall": 14.2},  # 7.1kgf × 2 at 100mm
        material="Zinc Alloy/Steel",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE IT7020型 金物が見えない 適応板厚25mm以上"
    ),
    Bracket(
        code="IT7020-N300",
        name="隠し棚受",
        name_en="Hidden Bracket IT7020",
        category="wood",
        type="hidden",
        sizes=[300],
        load_capacity_per_pair={"wall": 20.4},  # 10.2kgf × 2
        material="Zinc Alloy/Steel",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE IT7020型 適応板厚30mm以上"
    ),
    Bracket(
        code="IT7020-N400",
        name="隠し棚受",
        name_en="Hidden Bracket IT7020",
        category="wood",
        type="hidden",
        sizes=[400],
        load_capacity_per_pair={"wall": 42.8},  # 21.4kgf × 2
        material="Zinc Alloy/Steel",
        thickness_mm=3.0,
        shelf_type="wood",
        notes="SUGATSUNE IT7020型 適応板厚40mm以上"
    ),
    # =========================================================================
    # 棚柱システム用棚受 (SPHL/SPH/SP/SM型棚柱用)
    # 耐荷重は棚板1枚当り（棚受4ヶ使い）の値
    # =========================================================================
    Bracket(
        code="SPHL-25",
        name="ステンレス鋼製棚受 SPHL用",
        name_en="Stainless Bracket for SPHL",
        category="shelf_post",
        type="shelf_post",
        sizes=[25],
        load_capacity_per_pair={"SPHL-1820": 150},  # 150kgf/4ヶ = 75kg/2ヶ
        material="Stainless SUS-304",
        thickness_mm=1.5,
        shelf_type="both",
        notes="SUGATSUNE SPHL-1820棚柱用 コンパクト"
    ),
    Bracket(
        code="SPHL-30",
        name="ステンレス鋼製棚受 SPHL用",
        name_en="Stainless Bracket for SPHL",
        category="shelf_post",
        type="shelf_post",
        sizes=[30],
        load_capacity_per_pair={"SPHL-1820": 150},  # 150kgf/4ヶ
        material="Stainless SUS-304",
        thickness_mm=1.5,
        shelf_type="both",
        notes="SUGATSUNE SPHL-1820棚柱用 カバー付 ガラス板対応"
    ),
    Bracket(
        code="SPHL-FB200",
        name="棚柱用ブラケット SPHL用",
        name_en="Bracket for SPHL",
        category="shelf_post",
        type="shelf_post_bracket",
        sizes=[200],
        load_capacity_per_pair={
            "depth_200-250": 20,   # 棚板奥行200-250mmで20kgf/2本
            "depth_250-300": 14,   # 棚板奥行250-300mmで14kgf/2本
        },
        material="Stainless SUS-430",
        thickness_mm=1.5,
        shelf_type="wood",
        notes="SUGATSUNE SPHL型用ブラケット 抜け止め付"
    ),
    Bracket(
        code="SPH-15",
        name="ステンレス鋼製棚受 SPH用",
        name_en="Stainless Bracket for SPH",
        category="shelf_post",
        type="shelf_post",
        sizes=[15],
        load_capacity_per_pair={"SPH": 130},  # 130kgf/4ヶ
        material="Stainless SUS-304",
        thickness_mm=1.0,
        shelf_type="wood",
        notes="SUGATSUNE SPH棚柱用 コンパクト"
    ),
    Bracket(
        code="SPH-20",
        name="ステンレス鋼製棚受 SPH用",
        name_en="Stainless Bracket for SPH",
        category="shelf_post",
        type="shelf_post",
        sizes=[20],
        load_capacity_per_pair={"SPH": 130},  # 130kgf/4ヶ
        material="Stainless SUS-304",
        thickness_mm=1.5,
        shelf_type="both",
        notes="SUGATSUNE SPH棚柱用 NBRカバー付 ガラス板対応"
    ),
    Bracket(
        code="SPH-5",
        name="ステンレス鋼製棚受 SPH用",
        name_en="Stainless Bracket for SPH",
        category="shelf_post",
        type="shelf_post",
        sizes=[21],  # 実寸法
        load_capacity_per_pair={"SPH": 130},  # 130kgf/4ヶ
        material="Stainless SUS-304",
        thickness_mm=2.0,
        shelf_type="glass",
        notes="SUGATSUNE SPH棚柱用 カバー+板押さえ付 ガラス棚専用 適応ガラス厚4-8mm"
    ),
    Bracket(
        code="SPB-15R",
        name="ステンレス鋼製棚受 SP/SPS用",
        name_en="Stainless Bracket for SP/SPS",
        category="shelf_post",
        type="shelf_post",
        sizes=[15],
        load_capacity_per_pair={"SP": 100},  # 100kgf/4ヶ
        material="Stainless SUS-304",
        thickness_mm=1.5,
        shelf_type="both",
        notes="SUGATSUNE SP/SPS棚柱用 エラストマーカバー付"
    ),
    Bracket(
        code="SPB-20",
        name="ステンレス鋼製棚受 SP/SPS用",
        name_en="Stainless Bracket for SP/SPS",
        category="shelf_post",
        type="shelf_post",
        sizes=[20],
        load_capacity_per_pair={"SP": 80},  # 80kgf/4ヶ
        material="Stainless SUS-304",
        thickness_mm=1.5,
        shelf_type="both",
        notes="SUGATSUNE SP/SPS棚柱用"
    ),
    Bracket(
        code="SPF-20LC",
        name="レベル調整棚受 SP/SPS用",
        name_en="Level Adjusting Bracket for SP/SPS",
        category="shelf_post",
        type="shelf_post",
        sizes=[20],
        load_capacity_per_pair={"SP": 80},  # 80kgf/4ヶ
        material="Stainless SUS-304",
        thickness_mm=1.0,
        shelf_type="both",
        notes="SUGATSUNE SP/SPS棚柱用 高さ調整+2mm シリコンカバー"
    ),
    Bracket(
        code="SPB-200",
        name="棚柱用ブラケット SP/SPS用",
        name_en="Bracket for SP/SPS",
        category="shelf_post",
        type="shelf_post_bracket",
        sizes=[200],
        load_capacity_per_pair={
            "2pcs": 10,   # 10kgf/2本
            "3pcs": 15,   # 15kgf/3本
        },
        material="Stainless SUS-304",
        thickness_mm=1.5,
        shelf_type="both",
        notes="SUGATSUNE SP/SPS型用ブラケット 棚奥行200mm以下 ずれ止め+抜け止め付"
    ),
    Bracket(
        code="SMB-15R",
        name="ステンレス鋼製棚受 SM用",
        name_en="Stainless Bracket for SM",
        category="shelf_post",
        type="shelf_post",
        sizes=[15],
        load_capacity_per_pair={"SM": 90},  # 90kgf/4ヶ
        material="Stainless SUS-304",
        thickness_mm=1.5,
        shelf_type="both",
        notes="SUGATSUNE SM棚柱用 エラストマーカバー付"
    ),
    Bracket(
        code="SPM-20B",
        name="ステンレス鋼製棚受 SM用",
        name_en="Stainless Bracket for SM",
        category="shelf_post",
        type="shelf_post",
        sizes=[20],
        load_capacity_per_pair={"SM": 60},  # 60kgf/4ヶ
        material="Stainless SUS-304",
        thickness_mm=1.0,
        shelf_type="wood",
        notes="SUGATSUNE SM棚柱用 薄型 本棚に最適"
    ),
]


# =============================================================================
# 棚柱 (SHELF POSTS)
# =============================================================================

@dataclass
class ShelfPost:
    """Shelf post specification (棚柱)"""
    code: str
    name: str
    name_en: str
    lengths: List[int]          # Available lengths (mm)
    pitch_mm: int               # Slot pitch
    max_load_4pcs: float        # Maximum load with 4 shelf supports (kgf)
    material: str
    thickness_mm: float
    compatible_brackets: List[str]
    notes: str = ""


SHELF_POSTS = [
    ShelfPost(
        code="SPHL-1820",
        name="ステンレス棚柱 重量用",
        name_en="Stainless Shelf Post Heavy Duty",
        lengths=[1820],
        pitch_mm=20,
        max_load_4pcs=150,
        material="Stainless SUS-304",
        thickness_mm=1.2,
        compatible_brackets=["SPHL-25", "SPHL-30", "SPHL-FB200"],
        notes="SUGATSUNE SPHL型 板厚1.2mm 重量用"
    ),
    ShelfPost(
        code="SPH-1820",
        name="ステンレス棚柱",
        name_en="Stainless Shelf Post SPH",
        lengths=[1820],
        pitch_mm=22,
        max_load_4pcs=130,
        material="Stainless SUS-304",
        thickness_mm=0.8,
        compatible_brackets=["SPH-15", "SPH-20", "SPH-5"],
        notes="SUGATSUNE SPH型"
    ),
    ShelfPost(
        code="SPH-2520",
        name="ステンレス棚柱",
        name_en="Stainless Shelf Post SPH",
        lengths=[2524],
        pitch_mm=22,
        max_load_4pcs=130,
        material="Stainless SUS-304",
        thickness_mm=0.8,
        compatible_brackets=["SPH-15", "SPH-20", "SPH-5"],
        notes="SUGATSUNE SPH型 ロングタイプ"
    ),
    ShelfPost(
        code="SP-1820",
        name="ステンレス棚柱",
        name_en="Stainless Shelf Post SP",
        lengths=[455, 650, 845, 1040, 1820, 2600],
        pitch_mm=15,
        max_load_4pcs=100,
        material="Stainless SUS-304",
        thickness_mm=0.8,
        compatible_brackets=["SPB-15R", "SPB-20", "SPF-20LC", "SPB-200"],
        notes="SUGATSUNE SP型 対応棚受最多"
    ),
    ShelfPost(
        code="SPS-1820",
        name="ステンレス棚柱",
        name_en="Stainless Shelf Post SPS",
        lengths=[1600, 1820, 2600],
        pitch_mm=15,
        max_load_4pcs=100,
        material="Stainless SUS-430",
        thickness_mm=0.8,
        compatible_brackets=["SPB-15R", "SPB-20", "SPF-20LC", "SPB-200"],
        notes="SUGATSUNE SPS型"
    ),
    ShelfPost(
        code="SM-1820",
        name="ステンレス棚柱 目隠し付",
        name_en="Stainless Shelf Post with Cover",
        lengths=[1820, 2600],
        pitch_mm=15,
        max_load_4pcs=90,
        material="Stainless SUS-304",
        thickness_mm=0.8,
        compatible_brackets=["SMB-15R", "SPM-20B"],
        notes="SUGATSUNE SM型 取付穴目隠し付"
    ),
]


# =============================================================================
# SUPPORT TYPES
# 基礎支柱
# =============================================================================

@dataclass
class Support:
    """Support/Column specification"""
    code: str
    name: str
    name_en: str
    type: str           # channel, pipe, pecker
    rise_mm: float      # 立ち上がり寸法
    slot_pitch_mm: int  # Slot pitch
    material: str
    thickness_mm: float
    compatible_brackets: List[str]


SUPPORTS = [
    Support(
        code="ASF-1",
        name="チャンネルサポート 11mm",
        name_en="Channel Support 11mm",
        type="channel",
        rise_mm=11,
        slot_pitch_mm=25,
        material="Steel",
        thickness_mm=2.0,
        compatible_brackets=["B-032", "B-132", "B-232", "A-32", "DA-32", "AL-55S", "G-32"]
    ),
    Support(
        code="ASF-10",
        name="チャンネルサポート 14mm",
        name_en="Channel Support 14mm",
        type="channel",
        rise_mm=14,
        slot_pitch_mm=25,
        material="Steel",
        thickness_mm=2.0,
        compatible_brackets=["B-032", "B-132", "B-232", "A-32", "DA-32", "AL-55S", "G-32"]
    ),
    Support(
        code="S1B-50/50",
        name="スリットパイプ 50/50",
        name_en="Slit Pipe 50/50",
        type="pipe",
        rise_mm=50,
        slot_pitch_mm=50,
        material="Steel",
        thickness_mm=1.6,
        compatible_brackets=["B-032", "B-132", "B-232", "A-32", "DA-32", "AL-55S"]
    ),
    Support(
        code="PSF-17",
        name="チャンネルペッカーサポート 11mm",
        name_en="Channel Pecker Support 11mm",
        type="pecker",
        rise_mm=11,
        slot_pitch_mm=25,
        material="Steel",
        thickness_mm=2.0,
        compatible_brackets=["B-032", "A-32", "G-32"]
    ),
]


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_all_brackets() -> List[Bracket]:
    """Get all brackets"""
    return WOOD_BRACKETS + GLASS_BRACKETS + SPECIALTY_BRACKETS + SUGATSUNE_BRACKETS


def get_bracket_by_code(code: str) -> Optional[Bracket]:
    """Find bracket by product code"""
    for bracket in get_all_brackets():
        if bracket.code == code or bracket.code.lower() == code.lower():
            return bracket
    return None


def get_brackets_by_category(category: str) -> List[Bracket]:
    """Get brackets by category (wood, glass, hanger)"""
    return [b for b in get_all_brackets() if b.category == category]


def get_support_by_code(code: str) -> Optional[Support]:
    """Find support by code"""
    for support in SUPPORTS:
        if support.code == code:
            return support
    return None


def calculate_bracket_load_capacity(
    bracket_code: str,
    support_code: str = "ASF-1",
    num_brackets: int = 2
) -> float:
    """
    Calculate load capacity for a bracket-support combination

    Args:
        bracket_code: Bracket product code
        support_code: Support product code
        num_brackets: Number of brackets (default 2 for one shelf)

    Returns:
        Load capacity in kg
    """
    bracket = get_bracket_by_code(bracket_code)
    if not bracket:
        raise ValueError(f"Unknown bracket code: {bracket_code}")

    # Find load capacity for this support type
    load_per_pair = None

    # Try exact match
    if support_code in bracket.load_capacity_per_pair:
        load_per_pair = bracket.load_capacity_per_pair[support_code]
    # Try default
    elif "default" in bracket.load_capacity_per_pair:
        load_per_pair = bracket.load_capacity_per_pair["default"]
    # Use first available
    else:
        load_per_pair = list(bracket.load_capacity_per_pair.values())[0]

    # Scale by number of brackets (2 brackets = 1 pair = base load)
    return load_per_pair * (num_brackets / 2)


def export_database_to_json(filepath: Optional[Path] = None) -> dict:
    """Export database to JSON format"""
    if filepath is None:
        filepath = PROJECT_ROOT / "data/json/bracket_database.json"

    data = {
        "version": "2026",
        "source": "AA-SYSTEM 2026 Catalog + SUGATSUNE No.390",
        "brackets": {
            "wood": [asdict(b) for b in WOOD_BRACKETS],
            "glass": [asdict(b) for b in GLASS_BRACKETS],
            "specialty": [asdict(b) for b in SPECIALTY_BRACKETS],
            "sugatsune": [asdict(b) for b in SUGATSUNE_BRACKETS],
        },
        "supports": [asdict(s) for s in SUPPORTS],
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Database exported to: {filepath}")
    return data


def print_bracket_summary():
    """Print summary of all brackets"""
    print("=" * 70)
    print("BRACKET DATABASE SUMMARY")
    print("=" * 70)

    # AA-SYSTEM brackets
    print("\n[AA-SYSTEM 2026]")
    categories = ["wood", "glass"]
    for cat in categories:
        brackets = [b for b in WOOD_BRACKETS + GLASS_BRACKETS + SPECIALTY_BRACKETS
                   if b.category == cat]
        if not brackets:
            continue

        print(f"\n{cat.upper()} BRACKETS ({len(brackets)} items)")
        print("-" * 70)
        for b in brackets:
            load_values = list(b.load_capacity_per_pair.values())
            avg_load = sum(load_values) / len(load_values)
            print(f"  {b.code:12s} | {b.name:20s} | ~{avg_load:.0f}kg/pair | sizes: {b.sizes[:3]}...")

    # SUGATSUNE brackets
    print("\n" + "=" * 70)
    print("[SUGATSUNE No.390]")
    print("-" * 70)
    for b in SUGATSUNE_BRACKETS:
        load_values = list(b.load_capacity_per_pair.values())
        avg_load = sum(load_values) / len(load_values)
        sizes_str = str(b.sizes[:2]) if len(b.sizes) > 2 else str(b.sizes)
        print(f"  {b.code:15s} | {b.name:20s} | ~{avg_load:.0f}kg/pair | {b.type}")

    print("\n" + "=" * 70)
    print(f"Total brackets: {len(get_all_brackets())}")
    print(f"  AA-SYSTEM: {len(WOOD_BRACKETS) + len(GLASS_BRACKETS) + len(SPECIALTY_BRACKETS)}")
    print(f"  SUGATSUNE: {len(SUGATSUNE_BRACKETS)}")
    print(f"Total supports: {len(SUPPORTS)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Bracket Database')
    parser.add_argument('--export', action='store_true', help='Export to JSON')
    parser.add_argument('--summary', action='store_true', help='Print summary')
    parser.add_argument('--lookup', type=str, help='Look up bracket by code')
    args = parser.parse_args()

    if args.export:
        export_database_to_json()

    if args.summary:
        print_bracket_summary()

    if args.lookup:
        bracket = get_bracket_by_code(args.lookup)
        if bracket:
            print(f"\nBracket: {bracket.code}")
            print(f"  Name: {bracket.name}")
            print(f"  Category: {bracket.category}")
            print(f"  Type: {bracket.type}")
            print(f"  Sizes: {bracket.sizes}")
            print(f"  Load capacity: {bracket.load_capacity_per_pair}")
        else:
            print(f"Bracket not found: {args.lookup}")

    if not any([args.export, args.summary, args.lookup]):
        print_bracket_summary()
        export_database_to_json()
