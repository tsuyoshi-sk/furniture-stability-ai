#!/usr/bin/env python3
"""
Hybrid Load Capacity Predictor
Combines:
1. Rule-based thickness-to-material estimation
2. Physics-based beam theory calculation
3. ML correction factor for fine-tuning

This approach leverages domain knowledge that certain thicknesses
are associated with certain materials in real-world furniture.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Tuple
import argparse

SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR / "models_load"

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# =============================================================================
# Thickness-based Material Estimation Rules
# Based on real-world furniture manufacturing standards
# =============================================================================

THICKNESS_MATERIAL_RULES = {
    # (min_mm, max_mm): [(material, probability, E, sigma), ...]
    (8, 12): [
        ('plywood_thin', 0.70, 8000, 6),
        ('mdf_thin', 0.20, 3500, 4),
        ('particle', 0.10, 2500, 2.5),
    ],
    (12, 16): [
        ('plywood_medium', 0.40, 8000, 6),
        ('mdf', 0.35, 3500, 4),
        ('particle', 0.25, 2500, 2.5),
    ],
    (16, 20): [
        ('mdf', 0.35, 3500, 4),
        ('plywood', 0.30, 8000, 6),
        ('particle', 0.20, 2500, 2.5),
        ('pine', 0.15, 10000, 8),
    ],
    (20, 26): [
        ('pine', 0.30, 10000, 8),
        ('plywood_thick', 0.25, 8000, 6),
        ('oak', 0.25, 12000, 10),
        ('mdf_thick', 0.20, 3500, 4),
    ],
    (26, 35): [
        ('oak', 0.40, 12000, 10),
        ('pine', 0.35, 10000, 8),
        ('plywood_thick', 0.25, 8000, 6),
    ],
}


def estimate_material_from_thickness(thickness_mm: float) -> list:
    """
    Estimate likely materials based on thickness

    Returns list of (material, probability, E, sigma)
    """
    for (min_t, max_t), materials in THICKNESS_MATERIAL_RULES.items():
        if min_t <= thickness_mm < max_t:
            return materials

    # Default for extreme values
    if thickness_mm < 8:
        return [('plywood_thin', 1.0, 8000, 6)]
    else:
        return [('oak', 0.5, 12000, 10), ('pine', 0.5, 10000, 8)]


def physics_load_capacity(span_mm, width_mm, thickness_mm, E, sigma, safety_factor=2.5):
    """
    Calculate load capacity using Euler-Bernoulli beam theory
    """
    L, b, h = span_mm, width_mm, thickness_mm

    # Cross-section properties
    I = (b * h**3) / 12  # Moment of inertia
    S = (b * h**2) / 6   # Section modulus

    # Simply supported beam with uniform load
    moment_coef = 1/8
    deflection_coef = 5/384

    # Stress limit: σ = M/S = wL²/(8S)
    w_stress = sigma * S / (L**2 * moment_coef)  # N/mm
    max_load_stress = (w_stress * L) / 9.81 / safety_factor  # kg

    # Deflection limit: δ = 5wL⁴/(384EI) ≤ L/200
    max_deflection = L / 200
    w_deflection = max_deflection * E * I / (deflection_coef * L**4)
    max_load_deflection = (w_deflection * L) / 9.81 / safety_factor

    return min(max_load_stress, max_load_deflection)


def predict_load_capacity_rule_based(
    span_mm: float,
    width_mm: float,
    thickness_mm: float,
    safety_factor: float = 2.5
) -> Dict:
    """
    Predict load capacity using rule-based material estimation

    Returns dict with:
    - predicted_capacity: weighted average by material probability
    - min_capacity: conservative (lowest) estimate
    - max_capacity: optimistic estimate
    - material_estimates: list of (material, prob, capacity)
    """
    materials = estimate_material_from_thickness(thickness_mm)

    capacities = []
    for material, prob, E, sigma in materials:
        cap = physics_load_capacity(span_mm, width_mm, thickness_mm, E, sigma, safety_factor)
        capacities.append((material, prob, cap))

    # Weighted average
    total_prob = sum(prob for _, prob, _ in capacities)
    predicted = sum(prob * cap for _, prob, cap in capacities) / total_prob

    # Range
    min_cap = min(cap for _, _, cap in capacities)
    max_cap = max(cap for _, _, cap in capacities)

    return {
        'predicted_capacity': predicted,
        'min_capacity': min_cap,
        'max_capacity': max_cap,
        'material_estimates': capacities,
    }


# =============================================================================
# ML Correction Model
# =============================================================================

class CorrectionModel(nn.Module):
    """
    Small ML model that learns a correction factor
    Input: physics features + rule-based prediction
    Output: correction multiplier (centered at 1.0)
    """

    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # Initialize to output ~0 (correction = 1.0)
        self.net[-1].weight.data.fill_(0)
        self.net[-1].bias.data.fill_(0)

    def forward(self, x):
        # Output correction as multiplier: exp(output) ≈ 1 + output for small values
        correction = self.net(x)
        return torch.exp(correction.clamp(-0.5, 0.5))  # Limit to 0.6 - 1.6x


def create_features(span_mm, width_mm, thickness_mm, rule_pred, rule_min, rule_max):
    """Create features for correction model"""
    return np.array([
        np.log(span_mm + 1) / 10,
        np.log(width_mm + 1) / 8,
        np.log(thickness_mm + 1) / 5,
        thickness_mm / 30,
        span_mm / width_mm / 5,
        span_mm / thickness_mm / 100,
        np.log(rule_pred + 1) / 5,
        np.log(rule_min + 1) / 5,
        np.log(rule_max + 1) / 5,
        (rule_max - rule_min) / (rule_pred + 1),
    ], dtype=np.float32)


# =============================================================================
# Hybrid Predictor
# =============================================================================

class HybridLoadCapacityPredictor:
    """
    Hybrid predictor combining rules + physics + ML correction
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.correction_model = None
        if model_path and model_path.exists():
            self.correction_model = CorrectionModel()
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.correction_model.load_state_dict(checkpoint['model_state_dict'])
            self.correction_model.eval()

    def predict(
        self,
        span_mm: float,
        width_mm: float,
        thickness_mm: float,
        safety_factor: float = 2.5,
        use_ml_correction: bool = True
    ) -> Dict:
        """
        Predict load capacity

        Returns dict with all estimates
        """
        # Rule-based prediction
        rule_result = predict_load_capacity_rule_based(
            span_mm, width_mm, thickness_mm, safety_factor
        )

        result = {
            'rule_based': rule_result['predicted_capacity'],
            'min_estimate': rule_result['min_capacity'],
            'max_estimate': rule_result['max_capacity'],
            'material_estimates': rule_result['material_estimates'],
        }

        # ML correction
        if use_ml_correction and self.correction_model is not None:
            features = create_features(
                span_mm, width_mm, thickness_mm,
                rule_result['predicted_capacity'],
                rule_result['min_capacity'],
                rule_result['max_capacity']
            )
            with torch.no_grad():
                correction = self.correction_model(
                    torch.from_numpy(features).unsqueeze(0)
                ).item()
            result['ml_corrected'] = rule_result['predicted_capacity'] * correction
            result['correction_factor'] = correction
        else:
            result['ml_corrected'] = rule_result['predicted_capacity']
            result['correction_factor'] = 1.0

        # Final recommendation: conservative estimate
        result['recommended'] = result['min_estimate'] * 1.1  # 10% above minimum

        return result


# =============================================================================
# Training
# =============================================================================

def train_correction_model(data_dir: Path, epochs: int = 200):
    """Train the ML correction model"""
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    print("Training ML correction model...")

    # Load data
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Prepare training data
    X = []
    y = []

    for item in metadata:
        span = item['span_m'] * 1000
        width = item['width_m'] * 1000
        thickness = item['thickness_m'] * 1000
        target = item['target_capacity_kg']

        rule_result = predict_load_capacity_rule_based(span, width, thickness)
        features = create_features(
            span, width, thickness,
            rule_result['predicted_capacity'],
            rule_result['min_capacity'],
            rule_result['max_capacity']
        )

        X.append(features)
        # Target: log correction factor
        correction = target / rule_result['predicted_capacity']
        y.append(np.log(correction))

    X = np.array(X)
    y = np.array(y, dtype=np.float32)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model
    model = CorrectionModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            pred = model.net(batch_x).squeeze()
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                pred = model.net(batch_x).squeeze()
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * len(batch_y)

        val_loss /= len(val_dataset)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, MODEL_DIR / "correction_model.pth")

    print(f"Best Val Loss: {best_loss:.4f}")
    return MODEL_DIR / "correction_model.pth"


def evaluate_hybrid(data_dir: Path, model_path: Optional[Path] = None):
    """Evaluate hybrid predictor"""
    print("\n" + "=" * 60)
    print("Evaluating Hybrid Predictor")
    print("=" * 60)

    predictor = HybridLoadCapacityPredictor(model_path)

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    errors_rule = []
    errors_ml = []
    errors_conservative = []
    range_contains = []

    for item in metadata:
        span = item['span_m'] * 1000
        width = item['width_m'] * 1000
        thickness = item['thickness_m'] * 1000
        target = item['target_capacity_kg']

        result = predictor.predict(span, width, thickness)

        # Rule-based error
        err_rule = abs(result['rule_based'] - target) / target * 100
        errors_rule.append(err_rule)

        # ML-corrected error
        err_ml = abs(result['ml_corrected'] - target) / target * 100
        errors_ml.append(err_ml)

        # Conservative (safe) check
        if result['min_estimate'] <= target:
            errors_conservative.append(0)  # Safe
        else:
            errors_conservative.append(
                (result['min_estimate'] - target) / target * 100
            )

        # Range coverage
        in_range = result['min_estimate'] * 0.9 <= target <= result['max_estimate'] * 1.1
        range_contains.append(in_range)

    # Results
    print(f"\n1. Rule-Based (thickness → material → physics):")
    print(f"   MAPE: {np.mean(errors_rule):.1f}%")
    print(f"   Within 10%: {np.mean(np.array(errors_rule) < 10) * 100:.1f}%")
    print(f"   Within 20%: {np.mean(np.array(errors_rule) < 20) * 100:.1f}%")
    print(f"   Within 30%: {np.mean(np.array(errors_rule) < 30) * 100:.1f}%")

    if model_path and model_path.exists():
        print(f"\n2. ML-Corrected:")
        print(f"   MAPE: {np.mean(errors_ml):.1f}%")
        print(f"   Within 10%: {np.mean(np.array(errors_ml) < 10) * 100:.1f}%")
        print(f"   Within 20%: {np.mean(np.array(errors_ml) < 20) * 100:.1f}%")
        print(f"   Within 30%: {np.mean(np.array(errors_ml) < 30) * 100:.1f}%")

    print(f"\n3. Safety Metrics:")
    print(f"   Safe Rate (min ≤ target): {np.mean(np.array(errors_conservative) == 0) * 100:.1f}%")
    print(f"   Range Coverage: {np.mean(range_contains) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Hybrid Load Capacity Predictor')
    parser.add_argument('--train', action='store_true', help='Train correction model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate predictor')
    parser.add_argument('--data-dir', type=Path, default=SCRIPT_DIR / "dataset_load_v2")
    parser.add_argument('--predict', nargs=3, type=float, metavar=('SPAN', 'WIDTH', 'THICKNESS'),
                        help='Predict capacity for given dimensions (mm)')
    args = parser.parse_args()

    if args.train:
        model_path = train_correction_model(args.data_dir)
        evaluate_hybrid(args.data_dir, model_path)

    elif args.evaluate:
        model_path = MODEL_DIR / "correction_model.pth"
        if not model_path.exists():
            model_path = None
        evaluate_hybrid(args.data_dir, model_path)

    elif args.predict:
        span, width, thickness = args.predict
        model_path = MODEL_DIR / "correction_model.pth"
        predictor = HybridLoadCapacityPredictor(
            model_path if model_path.exists() else None
        )
        result = predictor.predict(span, width, thickness)

        print(f"\n寸法: {span:.0f} x {width:.0f} x {thickness:.0f} mm")
        print(f"\n耐荷重予測:")
        print(f"  ルールベース: {result['rule_based']:.1f} kg")
        print(f"  ML補正後: {result['ml_corrected']:.1f} kg")
        print(f"  最小推定: {result['min_estimate']:.1f} kg")
        print(f"  最大推定: {result['max_estimate']:.1f} kg")
        print(f"  推奨値: {result['recommended']:.1f} kg")
        print(f"\n材質推定:")
        for mat, prob, cap in result['material_estimates']:
            print(f"  {mat}: {prob*100:.0f}% → {cap:.1f} kg")

    else:
        # Default: train and evaluate
        model_path = train_correction_model(args.data_dir)
        evaluate_hybrid(args.data_dir, model_path)


if __name__ == "__main__":
    main()
