#!/usr/bin/env python3
"""
Okamura Office Furniture Database (Extended)
Auto-generated from Okamura Office Comprehensive Catalog 2026 (YOZ005-25D)
313 PDF files processed - 577 unique dimensions extracted

Used for:
- Furniture stability analysis training data
- Desk/table dimension reference
- Standard office furniture specifications
"""
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


@dataclass
class OfficeFurniture:
    """Office furniture specification"""
    code: str                    # Product code
    name: str                    # Japanese name
    name_en: str                 # English name
    category: str                # desk, table, storage, etc.
    width_mm: int                # Width (W)
    depth_mm: int                # Depth (D)
    height_mm: int               # Height (H)
    weight_kg: Optional[float]   # Weight if available
    material: str                # Primary material
    load_capacity_kg: Optional[float] = None  # Load capacity if specified
    notes: str = ""


# =============================================================================
# EXTRACTED DIMENSIONS DATABASE
# From Okamura Catalog 2026 (313 PDFs)
# =============================================================================


# Desk/Table dimensions (from catalog extraction)
OKAMURA_TABLE_DIMENSIONS: List[Tuple[int, int, int]] = [
    (450, 495, 775),
    (1200, 600, 720),
    (900, 600, 720),
    (1500, 600, 720),
    (555, 520, 740),
    (1916, 600, 720),
    (1800, 600, 720),
    (1800, 450, 720),
    (1200, 400, 720),
    (700, 450, 720),
    (518, 577, 788),
    (518, 532, 793),
    (400, 320, 750),
    (537, 558, 733),
    (537, 558, 771),
    (537, 563, 736),
    (537, 563, 772),
    (450, 495, 750),
    (450, 495, 710),
    (1260, 600, 720),
    (1685, 600, 720),
    (1370, 600, 780),
    (650, 600, 780),
    (414, 418, 760),
    (470, 460, 750),
    (440, 470, 750),
    (380, 450, 780),
    (600, 600, 740),
    (1500, 450, 720),
    (475, 475, 750),
    (460, 545, 740),
    (420, 535, 755),
    (600, 600, 730),
    (1500, 600, 730),
    (2100, 650, 770),
    (1200, 650, 770),
    (600, 650, 770),
    (1800, 600, 730),
    (521, 547, 760),
    (1200, 450, 720),
    (410, 410, 752),
    (532, 532, 754),
    (1000, 450, 720),
    (700, 450, 790),
    (700, 450, 724),
    (1200, 600, 800),
    (800, 600, 700),
    (800, 600, 740),
    (750, 500, 720),
    (920, 448, 700),
]

# Standing table dimensions
OKAMURA_STANDING_TABLE_DIMENSIONS: List[Tuple[int, int, int]] = [
    (900, 300, 1200),
    (864, 299, 1146),
    (857, 268, 1146),
    (900, 300, 1050),
    (900, 450, 1100),
    (1920, 570, 1195),
    (1920, 720, 1195),
    (900, 500, 1100),
    (900, 500, 1050),
    (900, 500, 1250),
    (900, 500, 1200),
    (864, 445, 1137),
    (864, 499, 1146),
    (857, 468, 1146),
    (1442, 948, 1308),
    (1000, 404, 1500),
    (1800, 600, 1004),
    (2400, 1200, 1004),
    (810, 200, 1395),
    (500, 200, 1395),
    (1246, 600, 1500),
    (900, 420, 1460),
    (1800, 900, 1004),
    (1200, 1200, 1004),
    (1600, 1200, 1004),
    (1600, 900, 1004),
    (1200, 695, 1320),
    (880, 650, 1013),
    (904, 773, 1360),
]

# Storage/Shelf dimensions
OKAMURA_STORAGE_DIMENSIONS: List[Tuple[int, int, int]] = [
    (900, 300, 2100),
    (864, 299, 2011),
    (857, 268, 2011),
    (900, 500, 2100),
    (864, 499, 2011),
    (857, 468, 2011),
    (589, 1122, 1660),
    (700, 700, 1560),
    (505, 495, 1615),
    (3270, 3270, 2200),
    (3670, 3670, 2200),
    (4870, 4870, 2200),
    (2470, 2470, 2200),
    (2770, 2770, 2200),
    (2640, 2640, 2240),
    (5160, 2640, 2240),
    (3840, 3840, 2240),
    (7560, 3840, 2240),
    (5040, 5040, 2240),
    (9960, 5040, 2240),
    (1000, 404, 1800),
    (1800, 404, 1529),
    (1246, 600, 1800),
    (2100, 1500, 2310),
    (1954, 1380, 2125),
    (2400, 2400, 2310),
    (2254, 2280, 2125),
    (2400, 1100, 2319),
    (2217, 943, 2205),
    (2100, 1100, 2310),
]

# Furniture items with product codes
OKAMURA_PRODUCTS: List[OfficeFurniture] = [
    OfficeFurniture(
        code="81F6NF-MAU5",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1200,
        depth_mm=600,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="81F6NJ-MAU3",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=900,
        depth_mm=600,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="81F6MG-MAP5",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1500,
        depth_mm=600,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="81F6MF-MAR5",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1200,
        depth_mm=600,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="4LMXCM-WJ57",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=2100,
        depth_mm=1050,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="4LMZAP-MGA7",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=2400,
        depth_mm=1200,
        height_mm=1000,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="4LMWAK-MJQ1",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1800,
        depth_mm=900,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="4B08AD-ZA75",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=860,
        depth_mm=367,
        height_mm=337,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="4H08AD-ZA75",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=860,
        depth_mm=355,
        height_mm=337,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="9315DR-Z32",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=880,
        depth_mm=550,
        height_mm=515,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="8158DD-Z927",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=589,
        depth_mm=1122,
        height_mm=1660,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="8151CS-Z927",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1654,
        depth_mm=580,
        height_mm=900,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="H146DS-Z721",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=718,
        depth_mm=850,
        height_mm=800,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="H146DR-Z721",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=718,
        depth_mm=850,
        height_mm=800,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="H146DS-Z721",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=718,
        depth_mm=850,
        height_mm=800,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="H146DS-Z721",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=718,
        depth_mm=850,
        height_mm=800,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="H146DR-Z721",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=718,
        depth_mm=850,
        height_mm=800,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="L685CA-MX62",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1916,
        depth_mm=600,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="L685GB-MEP2",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1916,
        depth_mm=800,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="L667HS-M725",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1800,
        depth_mm=900,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="L668SZ-WA10",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1500,
        depth_mm=450,
        height_mm=950,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="L683AD-MG99",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=2400,
        depth_mm=1200,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="L683BD-MK37",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=2400,
        depth_mm=1200,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="4LMECL-MME5",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1050,
        depth_mm=1050,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="4LMCAB-MGK6",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=750,
        depth_mm=750,
        height_mm=620,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="4LMDAA-MHQ8",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=600,
        depth_mm=750,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="4LSAWS-WE54",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=2400,
        depth_mm=1200,
        height_mm=736,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="81F1HB-MX63",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1800,
        depth_mm=600,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="81F1AX-MK37",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1800,
        depth_mm=450,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="8186KQ-MX62",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1800,
        depth_mm=600,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="8185MX-MX61",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1800,
        depth_mm=450,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="8185CS-Z712",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=700,
        depth_mm=700,
        height_mm=1560,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="93HREA-MAT5",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1800,
        depth_mm=600,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="93HRAF-MMD2",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1200,
        depth_mm=400,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="93HRES-MMD3",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=700,
        depth_mm=450,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="4LRYDG-WJ95",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=4800,
        depth_mm=1200,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="L525BD-MP53",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=2400,
        depth_mm=1200,
        height_mm=740,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="81F2CB-MG99",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1800,
        depth_mm=600,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="81F2AX-MK37",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1800,
        depth_mm=450,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="MZSA5E-MRW2",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=3600,
        depth_mm=1200,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="MZSB4C-MQA2",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=4800,
        depth_mm=1200,
        height_mm=1000,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="MZSC3F-MST3",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=2400,
        depth_mm=1200,
        height_mm=620,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="MZSE1J-MRX1",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1200,
        depth_mm=1200,
        height_mm=1000,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="3CA7DE-MBD3",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=4000,
        depth_mm=1400,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="3CA1BB-MBB3",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=2100,
        depth_mm=1100,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="3CA9CC-MBA1",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1200,
        depth_mm=1200,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="MC76BJ-MX62",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1800,
        depth_mm=900,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="81E7SB-MBX3",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1800,
        depth_mm=600,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="81E6HD-MH47",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1500,
        depth_mm=600,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
    OfficeFurniture(
        code="81E7TR-MBX3",
        name="オカムラ製品",
        name_en="Okamura Product",
        category="table",
        width_mm=1260,
        depth_mm=600,
        height_mm=720,
        weight_kg=None,
        material="Melamine/Steel",
        notes="Catalog 2026"
    ),
]

# Weight data from catalog (kg)
OKAMURA_WEIGHTS: List[float] = [1.0, 1.8, 2.0, 2.1, 2.3, 2.8, 2.9, 3.2, 3.5, 3.8, 3.9, 4.2, 4.4, 4.7, 5.0, 5.3, 5.4, 5.8, 5.9, 6.0, 6.7, 6.9, 7.3, 7.5, 7.8, 8.0, 8.1, 8.2, 8.6, 8.8, 9.0, 10.0, 10.7, 11.0, 11.1, 12.1, 12.5, 14.5, 15.0, 16.0, 16.2, 17.2, 18.0, 20.0, 21.5, 22.7, 30.2, 31.0, 32.0, 35.5]


# =============================================================================
# STATISTICS
# =============================================================================

EXTRACTION_STATS = {
    'total_pdfs': 313,
    'total_dimensions': 1577,
    'unique_dimensions': 577,
    'furniture_items_with_codes': 113,
    'dimension_ranges': {
        'width': (100, 9960),
        'depth': (111, 5040),
        'height': (100, 2503),
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_dimensions() -> List[Tuple[int, int, int]]:
    """Get all unique dimensions"""
    return (
        OKAMURA_TABLE_DIMENSIONS + 
        OKAMURA_STANDING_TABLE_DIMENSIONS + 
        OKAMURA_STORAGE_DIMENSIONS
    )


def get_all_products() -> List[OfficeFurniture]:
    """Get all products with codes"""
    return OKAMURA_PRODUCTS


def get_dimension_stats() -> Dict:
    """Get dimension statistics"""
    return EXTRACTION_STATS


def get_dimensions_by_category(category: str) -> List[Tuple[int, int, int]]:
    """Get dimensions by category"""
    if category == 'table':
        return OKAMURA_TABLE_DIMENSIONS
    elif category == 'standing':
        return OKAMURA_STANDING_TABLE_DIMENSIONS
    elif category == 'storage':
        return OKAMURA_STORAGE_DIMENSIONS
    return []


def get_typical_table_dimensions() -> List[Tuple[int, int, int]]:
    """Get common table dimensions for 3D model generation"""
    return [
        (1200, 600, 720),   # Standard desk
        (1400, 700, 720),   # Medium desk
        (1600, 800, 720),   # Large desk
        (1800, 900, 720),   # Conference table
        (900, 600, 720),    # Compact desk
        (1200, 750, 1000),  # Standing desk
    ]


def export_to_json(filepath: Optional[Path] = None) -> Path:
    """Export database to JSON"""
    if filepath is None:
        filepath = PROJECT_ROOT / "data/json/okamura_furniture_extended.json"
    
    export_data = {
        'source': 'Okamura Office Comprehensive Catalog 2026 (YOZ005-25D)',
        'statistics': EXTRACTION_STATS,
        'table_dimensions': OKAMURA_TABLE_DIMENSIONS,
        'standing_dimensions': OKAMURA_STANDING_TABLE_DIMENSIONS,
        'storage_dimensions': OKAMURA_STORAGE_DIMENSIONS,
        'products': [asdict(p) for p in OKAMURA_PRODUCTS],
        'weights': OKAMURA_WEIGHTS
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return filepath


def print_summary():
    """Print database summary"""
    print("=" * 60)
    print("OKAMURA FURNITURE DATABASE (EXTENDED)")
    print("=" * 60)
    
    stats = EXTRACTION_STATS
    print(f"\nSource: 313 PDFs from Okamura Catalog 2026")
    print(f"Total dimensions extracted: {stats['total_dimensions']}")
    print(f"Unique dimensions: {stats['unique_dimensions']}")
    print(f"Products with codes: {stats['furniture_items_with_codes']}")
    
    print(f"\nDimension Ranges:")
    print(f"  Width: {stats['dimension_ranges']['width'][0]}-{stats['dimension_ranges']['width'][1]} mm")
    print(f"  Depth: {stats['dimension_ranges']['depth'][0]}-{stats['dimension_ranges']['depth'][1]} mm")
    print(f"  Height: {stats['dimension_ranges']['height'][0]}-{stats['dimension_ranges']['height'][1]} mm")
    
    print(f"\nCategories:")
    print(f"  Tables: {len(OKAMURA_TABLE_DIMENSIONS)} dimensions")
    print(f"  Standing tables: {len(OKAMURA_STANDING_TABLE_DIMENSIONS)} dimensions")
    print(f"  Storage: {len(OKAMURA_STORAGE_DIMENSIONS)} dimensions")
    print(f"  Products with codes: {len(OKAMURA_PRODUCTS)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Okamura Furniture Database (Extended)')
    parser.add_argument('--summary', action='store_true', help='Print summary')
    parser.add_argument('--export', action='store_true', help='Export to JSON')
    args = parser.parse_args()
    
    if args.summary or (not args.summary and not args.export):
        print_summary()
    
    if args.export:
        path = export_to_json()
        print(f"\nExported to: {path}")
