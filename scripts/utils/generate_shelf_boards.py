#!/usr/bin/env python3
"""
Generate shelf board test cases with known load capacities
For training and validating the load capacity model
"""
import numpy as np
from pathlib import Path
import random
import json

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data/datasets/dataset_load"


def save_obj(vertices, faces, filepath):
    """Save as OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {' '.join(str(i+1) for i in face)}\n")


def create_box(center, size):
    """Create a box mesh"""
    cx, cy, cz = center
    sx, sy, sz = size

    vertices = [
        [cx - sx/2, cy - sy/2, cz - sz/2],
        [cx + sx/2, cy - sy/2, cz - sz/2],
        [cx + sx/2, cy + sy/2, cz - sz/2],
        [cx - sx/2, cy + sy/2, cz - sz/2],
        [cx - sx/2, cy - sy/2, cz + sz/2],
        [cx + sx/2, cy - sy/2, cz + sz/2],
        [cx + sx/2, cy + sy/2, cz + sz/2],
        [cx - sx/2, cy + sy/2, cz + sz/2],
    ]

    faces = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [2, 6, 7, 3],
        [0, 3, 7, 4],
        [1, 5, 6, 2],
    ]

    return np.array(vertices), faces


def calculate_theoretical_capacity(span_m, width_m, thickness_m, E_mpa, sigma_allow_mpa,
                                   support_type='simply_supported', safety_factor=2.5,
                                   max_deflection_ratio=1/200):
    """
    Calculate theoretical load capacity using beam theory
    Returns: max load in kg (minimum of stress and deflection limits)
    """
    # Convert to mm for calculation
    L = span_m * 1000  # mm
    b = width_m * 1000  # mm
    h = thickness_m * 1000  # mm

    # Cross-section properties
    I = (b * h**3) / 12  # mm⁴
    S = (b * h**2) / 6   # mm³

    # Support factor
    if support_type == 'cantilever':
        moment_coef = 0.5
        deflection_coef = 1/8
    elif support_type == 'fixed':
        moment_coef = 1/12
        deflection_coef = 1/384
    else:  # simply_supported
        moment_coef = 1/8
        deflection_coef = 5/384

    # Max load from stress limit (uniformly distributed)
    # σ = M/S = (w*L²*coef)/S
    # w = σ*S / (L²*coef)
    w_stress = sigma_allow_mpa * S / (L**2 * moment_coef)  # N/mm
    max_load_stress = (w_stress * L) / 9.81 / safety_factor

    # Max load from deflection limit
    # δ = deflection_coef * w * L⁴ / (E * I)
    # w = δ * E * I / (deflection_coef * L⁴)
    max_deflection = L * max_deflection_ratio
    w_deflection = max_deflection * E_mpa * I / (deflection_coef * L**4)
    max_load_deflection = (w_deflection * L) / 9.81 / safety_factor

    # Use the more conservative (smaller) value
    max_load_kg = min(max_load_stress, max_load_deflection)

    return max_load_kg


# Material properties (E in MPa, sigma_allow in MPa)
MATERIALS = {
    'plywood_12mm': {'E': 8000, 'sigma': 6, 'thickness': 0.012},
    'plywood_18mm': {'E': 8000, 'sigma': 6, 'thickness': 0.018},
    'plywood_24mm': {'E': 8000, 'sigma': 6, 'thickness': 0.024},
    'mdf_16mm': {'E': 3500, 'sigma': 4, 'thickness': 0.016},
    'mdf_19mm': {'E': 3500, 'sigma': 4, 'thickness': 0.019},
    'particle_15mm': {'E': 2500, 'sigma': 2.5, 'thickness': 0.015},
    'particle_18mm': {'E': 2500, 'sigma': 2.5, 'thickness': 0.018},
    'pine_20mm': {'E': 10000, 'sigma': 8, 'thickness': 0.020},
    'pine_25mm': {'E': 10000, 'sigma': 8, 'thickness': 0.025},
    'oak_20mm': {'E': 12000, 'sigma': 10, 'thickness': 0.020},
}

SPANS = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]  # meters
WIDTHS = [0.2, 0.25, 0.3, 0.35, 0.4]    # meters


def generate_dataset(num_samples=500):
    """Generate shelf board dataset with known load capacities"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = []

    for i in range(num_samples):
        # Random parameters
        material_key = random.choice(list(MATERIALS.keys()))
        material = MATERIALS[material_key]

        span = random.choice(SPANS) + random.uniform(-0.05, 0.05)
        width = random.choice(WIDTHS) + random.uniform(-0.02, 0.02)
        thickness = material['thickness'] * random.uniform(0.9, 1.1)

        # Create shelf board (simple box)
        vertices, faces = create_box(
            center=[0, thickness/2, 0],
            size=[span, thickness, width]
        )

        # Calculate theoretical capacity
        capacity = calculate_theoretical_capacity(
            span_m=span,
            width_m=width,
            thickness_m=thickness,
            E_mpa=material['E'],
            sigma_allow_mpa=material['sigma']
        )

        # Add some noise to simulate real-world variation
        capacity_with_noise = capacity * random.uniform(0.85, 1.15)

        # Save OBJ
        filename = f"shelf_board_{i:05d}.obj"
        filepath = OUTPUT_DIR / filename
        save_obj(vertices, faces, filepath)

        # Save metadata
        metadata.append({
            'filename': filename,
            'material': material_key,
            'span_m': span,
            'width_m': width,
            'thickness_m': thickness,
            'E_mpa': material['E'],
            'sigma_allow_mpa': material['sigma'],
            'theoretical_capacity_kg': capacity,
            'target_capacity_kg': capacity_with_noise
        })

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples}")

    # Save metadata
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated {num_samples} shelf boards")
    print(f"Data saved to: {OUTPUT_DIR}")
    print(f"Metadata: {metadata_path}")

    # Statistics
    capacities = [m['target_capacity_kg'] for m in metadata]
    print(f"\nLoad Capacity Statistics:")
    print(f"  Min: {min(capacities):.1f} kg")
    print(f"  Max: {max(capacities):.1f} kg")
    print(f"  Mean: {np.mean(capacities):.1f} kg")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', '-n', type=int, default=500)
    args = parser.parse_args()

    generate_dataset(num_samples=args.num)
