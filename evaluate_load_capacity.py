#!/usr/bin/env python3
"""
Evaluate load capacity prediction accuracy
"""
import json
import numpy as np
from pathlib import Path
from load_capacity import LoadCapacityPredictor, MATERIALS

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "dataset_load"


def evaluate():
    """Evaluate predictions against known theoretical values"""
    # Load metadata
    with open(DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)

    predictor = LoadCapacityPredictor(safety_factor=2.5, dynamic_factor=2.0)

    results = []
    errors_with_material = []
    errors_without_material = []

    for item in metadata:
        filepath = DATA_DIR / item['filename']

        # Map material names
        material_map = {
            'plywood_12mm': 'plywood',
            'plywood_18mm': 'plywood',
            'plywood_24mm': 'plywood',
            'mdf_16mm': 'mdf',
            'mdf_19mm': 'mdf',
            'particle_15mm': 'particle_board',
            'particle_18mm': 'particle_board',
            'pine_20mm': 'pine',
            'pine_25mm': 'pine',
            'oak_20mm': 'oak',
        }
        material_key = material_map.get(item['material'], 'plywood')

        try:
            # Predict with correct material
            result_with_mat = predictor.predict(filepath, material=material_key)
            pred_with = result_with_mat.max_static_load

            # Predict without material (auto-estimate)
            result_without_mat = predictor.predict(filepath, material=None)
            pred_without = result_without_mat.max_static_load

            actual = item['theoretical_capacity_kg']

            error_with = abs(pred_with - actual) / actual * 100
            error_without = abs(pred_without - actual) / actual * 100

            errors_with_material.append(error_with)
            errors_without_material.append(error_without)

            results.append({
                'file': item['filename'],
                'actual': actual,
                'pred_with_mat': pred_with,
                'pred_without_mat': pred_without,
                'error_with': error_with,
                'error_without': error_without
            })

        except Exception as e:
            print(f"Error: {item['filename']}: {e}")

    # Statistics
    print("=" * 60)
    print("Load Capacity Prediction Evaluation")
    print("=" * 60)
    print(f"\nSamples evaluated: {len(results)}")

    print("\n--- With Correct Material ---")
    print(f"Mean Error:   {np.mean(errors_with_material):.2f}%")
    print(f"Median Error: {np.median(errors_with_material):.2f}%")
    print(f"Max Error:    {np.max(errors_with_material):.2f}%")
    print(f"< 5% error:   {sum(1 for e in errors_with_material if e < 5) / len(errors_with_material) * 100:.1f}%")
    print(f"< 10% error:  {sum(1 for e in errors_with_material if e < 10) / len(errors_with_material) * 100:.1f}%")

    print("\n--- Without Material (Auto-estimate) ---")
    print(f"Mean Error:   {np.mean(errors_without_material):.2f}%")
    print(f"Median Error: {np.median(errors_without_material):.2f}%")
    print(f"Max Error:    {np.max(errors_without_material):.2f}%")
    print(f"< 10% error:  {sum(1 for e in errors_without_material if e < 10) / len(errors_without_material) * 100:.1f}%")
    print(f"< 30% error:  {sum(1 for e in errors_without_material if e < 30) / len(errors_without_material) * 100:.1f}%")

    # Error by material
    print("\n--- Error by Material (with correct material) ---")
    material_errors = {}
    for item, r in zip(metadata, results):
        mat = item['material'].split('_')[0]
        if mat not in material_errors:
            material_errors[mat] = []
        material_errors[mat].append(r['error_with'])

    for mat, errs in sorted(material_errors.items()):
        print(f"  {mat:15s}: Mean={np.mean(errs):.2f}%, Max={np.max(errs):.2f}%")

    return results


if __name__ == "__main__":
    evaluate()
