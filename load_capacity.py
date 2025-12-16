#!/usr/bin/env python3
"""
Furniture Load Capacity Prediction System
- Physics-based structural analysis
- Dynamic load consideration
- Material property estimation
- ML refinement (future)
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

SCRIPT_DIR = Path(__file__).parent


# =============================================================================
# Material Properties Database
# =============================================================================

@dataclass
class Material:
    """Material properties"""
    name: str
    density: float          # kg/m³
    elastic_modulus: float  # MPa (N/mm²)
    yield_strength: float   # MPa
    allowable_stress: float # MPa (with safety factor)

MATERIALS = {
    'pine': Material('Pine Wood', 500, 10000, 40, 8),
    'oak': Material('Oak Wood', 750, 12000, 50, 10),
    'plywood': Material('Plywood', 600, 8000, 30, 6),
    'mdf': Material('MDF', 750, 3500, 20, 4),
    'particle_board': Material('Particle Board', 650, 2500, 12, 2.5),
    'steel': Material('Steel', 7850, 200000, 250, 160),
    'aluminum': Material('Aluminum', 2700, 70000, 270, 90),
    'plastic_abs': Material('ABS Plastic', 1050, 2300, 40, 13),
}

DEFAULT_MATERIAL = 'plywood'  # Conservative default


# =============================================================================
# Geometry Analysis
# =============================================================================

def load_obj_vertices(filepath: Path) -> np.ndarray:
    """Load vertices from OBJ file"""
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices, dtype=np.float32)


def load_obj_faces(filepath: Path) -> List[List[int]]:
    """Load faces from OBJ file"""
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('f '):
                parts = line.strip().split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]
                faces.append(face)
    return faces


@dataclass
class ShelfGeometry:
    """Extracted shelf geometry"""
    span_length: float      # mm - distance between supports
    width: float            # mm - depth of shelf
    thickness: float        # mm - board thickness
    support_type: str       # 'fixed', 'simply_supported', 'cantilever'
    support_positions: List[float]  # positions along span
    cross_section_area: float      # mm²
    moment_of_inertia: float       # mm⁴
    section_modulus: float         # mm³
    num_supports: int
    is_shelf_board: bool


def analyze_shelf_geometry(vertices: np.ndarray, faces: List[List[int]]) -> ShelfGeometry:
    """
    Analyze 3D mesh to extract shelf board geometry
    Assumes Y is up, X is width, Z is depth
    """
    if len(vertices) == 0:
        raise ValueError("Empty mesh")

    # Bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    dims = max_coords - min_coords

    # Convert to mm (assuming model is in meters)
    dims_mm = dims * 1000

    # Identify shelf orientation (longest dimension is span)
    sorted_dims = np.argsort(dims_mm)[::-1]

    span_length = dims_mm[sorted_dims[0]]
    width = dims_mm[sorted_dims[1]]
    thickness = dims_mm[sorted_dims[2]]

    # Ensure minimum thickness
    if thickness < 1:
        thickness = max(1, dims_mm[1])  # Use Y dimension if very thin

    # Cross-section properties (rectangular)
    cross_section_area = width * thickness
    moment_of_inertia = (width * thickness**3) / 12  # I = bh³/12
    section_modulus = (width * thickness**2) / 6      # S = bh²/6

    # Detect support type from geometry
    support_type, support_positions, num_supports = detect_supports(vertices, dims)

    # Check if this looks like a shelf board
    is_shelf_board = (
        span_length > thickness * 5 and  # Long and thin
        width > thickness * 2 and         # Wide plate shape
        thickness < 100                    # Not too thick
    )

    return ShelfGeometry(
        span_length=span_length,
        width=width,
        thickness=thickness,
        support_type=support_type,
        support_positions=support_positions,
        cross_section_area=cross_section_area,
        moment_of_inertia=moment_of_inertia,
        section_modulus=section_modulus,
        num_supports=num_supports,
        is_shelf_board=is_shelf_board
    )


def detect_supports(vertices: np.ndarray, dims: np.ndarray) -> Tuple[str, List[float], int]:
    """
    Detect support positions from mesh geometry
    Returns: (support_type, support_positions, num_supports)
    """
    # Analyze vertex distribution at bottom
    y_min = np.min(vertices[:, 1])
    y_threshold = y_min + dims[1] * 0.1

    bottom_vertices = vertices[vertices[:, 1] <= y_threshold]

    if len(bottom_vertices) < 4:
        return 'simply_supported', [0, 1], 2

    # Find support clusters along span (X axis)
    x_positions = bottom_vertices[:, 0]
    x_min, x_max = np.min(x_positions), np.max(x_positions)
    span = x_max - x_min

    if span < 0.01:
        return 'cantilever', [0], 1

    # Bin positions to find support clusters
    num_bins = 10
    hist, bin_edges = np.histogram(x_positions, bins=num_bins)

    # Find peaks (support locations)
    threshold = np.max(hist) * 0.3
    support_bins = np.where(hist > threshold)[0]

    if len(support_bins) == 0:
        return 'simply_supported', [0, 1], 2

    # Convert to normalized positions (0-1)
    support_positions = []
    for bin_idx in support_bins:
        pos = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
        normalized_pos = (pos - x_min) / span if span > 0 else 0
        support_positions.append(normalized_pos)

    # Determine support type
    num_supports = len(set([round(p, 1) for p in support_positions]))

    if num_supports == 1:
        if support_positions[0] < 0.2:
            return 'cantilever', [0], 1
        else:
            return 'simply_supported', [0, 1], 2
    elif num_supports == 2:
        return 'simply_supported', sorted(support_positions)[:2], 2
    else:
        return 'multi_supported', sorted(support_positions), num_supports


# =============================================================================
# Load Capacity Calculation
# =============================================================================

@dataclass
class LoadCapacityResult:
    """Load capacity prediction result"""
    max_static_load: float      # kg
    max_dynamic_load: float     # kg (with impact factor)
    safety_factor: float
    max_deflection: float       # mm at max load
    critical_stress: float      # MPa at max load
    allowable_stress: float     # MPa
    weak_point: str
    confidence: float           # 0-1
    warnings: List[str]
    geometry: ShelfGeometry
    material: Material


def calculate_load_capacity(
    geometry: ShelfGeometry,
    material: Material,
    dynamic_factor: float = 2.0,
    max_deflection_ratio: float = 1/200,  # L/200 typical for shelves
    safety_factor: float = 2.5
) -> LoadCapacityResult:
    """
    Calculate load capacity using beam theory

    For uniformly distributed load on simply supported beam:
    - Max moment: M = wL²/8
    - Max stress: σ = M/S = M*c/I
    - Max deflection: δ = 5wL⁴/(384EI)

    For cantilever:
    - Max moment: M = wL²/2
    - Max deflection: δ = wL⁴/(8EI)
    """
    warnings = []

    L = geometry.span_length  # mm
    I = geometry.moment_of_inertia  # mm⁴
    S = geometry.section_modulus  # mm³
    E = material.elastic_modulus  # MPa = N/mm²
    sigma_allow = material.allowable_stress  # MPa

    # Adjust for support type
    if geometry.support_type == 'cantilever':
        moment_coef = 0.5      # wL²/2
        deflection_coef = 1/8  # wL⁴/(8EI)
        warnings.append("Cantilever support - higher stress concentration")
    elif geometry.support_type == 'fixed':
        moment_coef = 1/12     # wL²/12 for fixed-fixed
        deflection_coef = 1/384
    else:  # simply_supported or multi_supported
        moment_coef = 1/8      # wL²/8
        deflection_coef = 5/384

    # Calculate max allowable distributed load (N/mm) from stress limit
    # σ = M/S = (w*L²*moment_coef)/S
    # w = σ*S / (L²*moment_coef)
    if L > 0 and moment_coef > 0:
        w_stress = sigma_allow * S / (L**2 * moment_coef)  # N/mm
    else:
        w_stress = 0

    # Calculate max allowable load from deflection limit
    # δ = deflection_coef * w * L⁴ / (E * I)
    # w = δ * E * I / (deflection_coef * L⁴)
    max_deflection = L * max_deflection_ratio  # mm
    if L > 0 and deflection_coef > 0:
        w_deflection = max_deflection * E * I / (deflection_coef * L**4)  # N/mm
    else:
        w_deflection = 0

    # Use the smaller (more conservative) value
    if w_stress <= 0 and w_deflection <= 0:
        w_max = 0
        weak_point = "Invalid geometry"
    elif w_stress < w_deflection:
        w_max = w_stress
        weak_point = "Stress limit"
    else:
        w_max = w_deflection
        weak_point = "Deflection limit"

    # Convert to total load (kg)
    # w is N/mm, total load = w * L (N), convert to kg
    total_load_N = w_max * L  # N
    total_load_kg = total_load_N / 9.81  # kg

    # Apply safety factor for static load
    max_static_load = total_load_kg / safety_factor

    # Apply dynamic factor
    max_dynamic_load = max_static_load / dynamic_factor

    # Calculate stress at max static load
    w_actual = (max_static_load * 9.81) / L if L > 0 else 0  # N/mm
    M_actual = w_actual * L**2 * moment_coef  # N*mm
    critical_stress = M_actual / S if S > 0 else 0  # MPa

    # Calculate deflection at max static load
    if E > 0 and I > 0 and L > 0:
        actual_deflection = deflection_coef * w_actual * L**4 / (E * I)
    else:
        actual_deflection = 0

    # Confidence based on geometry validity
    confidence = 1.0
    if not geometry.is_shelf_board:
        confidence *= 0.7
        warnings.append("Geometry may not be a typical shelf board")
    if geometry.thickness < 10:
        confidence *= 0.8
        warnings.append("Very thin board - results may be less accurate")
    if geometry.span_length > 2000:
        confidence *= 0.9
        warnings.append("Long span - consider intermediate supports")
    if max_dynamic_load < 1:
        confidence *= 0.5
        warnings.append("Very low capacity - check geometry/material")

    return LoadCapacityResult(
        max_static_load=max_static_load,
        max_dynamic_load=max_dynamic_load,
        safety_factor=safety_factor,
        max_deflection=actual_deflection,
        critical_stress=critical_stress,
        allowable_stress=sigma_allow,
        weak_point=weak_point,
        confidence=confidence,
        warnings=warnings,
        geometry=geometry,
        material=material
    )


# =============================================================================
# Material Estimation
# =============================================================================

def estimate_material(geometry: ShelfGeometry) -> str:
    """
    Estimate material based on geometry
    This is a heuristic - actual material should be specified when possible
    """
    thickness = geometry.thickness
    span = geometry.span_length

    # Thin and long -> likely plywood or metal
    if thickness < 15 and span > 500:
        return 'plywood'

    # Very thin -> metal or plastic
    if thickness < 5:
        return 'steel'

    # Medium thickness -> solid wood or MDF
    if 15 <= thickness <= 25:
        return 'mdf'

    # Thick -> solid wood
    if thickness > 25:
        return 'pine'

    return DEFAULT_MATERIAL


# =============================================================================
# Main Predictor Class
# =============================================================================

class LoadCapacityPredictor:
    """Shelf load capacity predictor"""

    def __init__(self,
                 default_material: str = DEFAULT_MATERIAL,
                 dynamic_factor: float = 2.0,
                 safety_factor: float = 2.5):
        self.default_material = default_material
        self.dynamic_factor = dynamic_factor
        self.safety_factor = safety_factor

    def predict(self,
                obj_path: Path,
                material: Optional[str] = None,
                auto_estimate_material: bool = True) -> LoadCapacityResult:
        """
        Predict load capacity for a shelf OBJ file

        Args:
            obj_path: Path to OBJ file
            material: Material name (from MATERIALS dict)
            auto_estimate_material: Estimate material from geometry if not specified

        Returns:
            LoadCapacityResult with predictions
        """
        obj_path = Path(obj_path)

        # Load mesh
        vertices = load_obj_vertices(obj_path)
        faces = load_obj_faces(obj_path)

        # Analyze geometry
        geometry = analyze_shelf_geometry(vertices, faces)

        # Determine material
        if material and material in MATERIALS:
            mat = MATERIALS[material]
        elif auto_estimate_material:
            estimated = estimate_material(geometry)
            mat = MATERIALS[estimated]
        else:
            mat = MATERIALS[self.default_material]

        # Calculate load capacity
        result = calculate_load_capacity(
            geometry=geometry,
            material=mat,
            dynamic_factor=self.dynamic_factor,
            safety_factor=self.safety_factor
        )

        return result

    def predict_batch(self,
                      obj_paths: List[Path],
                      material: Optional[str] = None) -> List[LoadCapacityResult]:
        """Predict load capacity for multiple files"""
        results = []
        for path in obj_paths:
            try:
                result = self.predict(path, material)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        return results


def format_result(result: LoadCapacityResult) -> str:
    """Format result for display"""
    lines = [
        "=" * 60,
        "Load Capacity Analysis",
        "=" * 60,
        "",
        "Geometry:",
        f"  Span: {result.geometry.span_length:.1f} mm",
        f"  Width: {result.geometry.width:.1f} mm",
        f"  Thickness: {result.geometry.thickness:.1f} mm",
        f"  Support: {result.geometry.support_type}",
        f"  Is shelf board: {'Yes' if result.geometry.is_shelf_board else 'No'}",
        "",
        f"Material: {result.material.name}",
        f"  Elastic modulus: {result.material.elastic_modulus} MPa",
        f"  Allowable stress: {result.material.allowable_stress} MPa",
        "",
        "-" * 60,
        "Load Capacity Results",
        "-" * 60,
        f"  Max Static Load:  {result.max_static_load:.1f} kg",
        f"  Max Dynamic Load: {result.max_dynamic_load:.1f} kg",
        f"  Safety Factor: {result.safety_factor}",
        f"  Dynamic Factor: {result.max_static_load/result.max_dynamic_load:.1f}x",
        "",
        f"  Critical Stress: {result.critical_stress:.2f} MPa",
        f"  Max Deflection: {result.max_deflection:.2f} mm",
        f"  Limiting Factor: {result.weak_point}",
        "",
        f"  Confidence: {result.confidence*100:.0f}%",
    ]

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Shelf Load Capacity Prediction')
    parser.add_argument('input', nargs='?', help='OBJ file')
    parser.add_argument('--material', '-m', choices=list(MATERIALS.keys()),
                        help='Material type')
    parser.add_argument('--dynamic-factor', '-d', type=float, default=2.0,
                        help='Dynamic load factor (default: 2.0)')
    parser.add_argument('--safety-factor', '-s', type=float, default=2.5,
                        help='Safety factor (default: 2.5)')
    parser.add_argument('--list-materials', action='store_true',
                        help='List available materials')
    parser.add_argument('--output', '-o', help='Save result to JSON')
    args = parser.parse_args()

    if args.list_materials:
        print("\nAvailable Materials:")
        print("-" * 60)
        for key, mat in MATERIALS.items():
            print(f"  {key:15s}: E={mat.elastic_modulus:6d} MPa, "
                  f"σ_allow={mat.allowable_stress:3.0f} MPa")
        return

    if not args.input:
        parser.print_help()
        print("\n\nExample:")
        print("  python3 load_capacity.py shelf.obj")
        print("  python3 load_capacity.py shelf.obj --material oak")
        print("  python3 load_capacity.py --list-materials")
        return

    predictor = LoadCapacityPredictor(
        dynamic_factor=args.dynamic_factor,
        safety_factor=args.safety_factor
    )

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    result = predictor.predict(input_path, material=args.material)

    print(f"\nFile: {input_path.name}")
    print(format_result(result))

    if args.output:
        output_data = {
            'file': str(input_path),
            'max_static_load_kg': result.max_static_load,
            'max_dynamic_load_kg': result.max_dynamic_load,
            'safety_factor': result.safety_factor,
            'max_deflection_mm': result.max_deflection,
            'critical_stress_mpa': result.critical_stress,
            'weak_point': result.weak_point,
            'confidence': result.confidence,
            'material': result.material.name,
            'geometry': {
                'span_mm': result.geometry.span_length,
                'width_mm': result.geometry.width,
                'thickness_mm': result.geometry.thickness,
                'support_type': result.geometry.support_type,
            },
            'warnings': result.warnings
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
