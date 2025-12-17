#!/usr/bin/env python3
"""
Load Capacity Prediction v3 - Physics-Informed Neural Network
Direct end-to-end load prediction with physics constraints
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data/datasets/dataset_load_v2"  # Use v2 dataset
MODEL_DIR = PROJECT_ROOT / "models/models_load"

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

NUM_POINTS = 512


def load_obj_vertices(filepath):
    """Load vertices from OBJ"""
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices, dtype=np.float32)


def extract_physics_features(vertices):
    """
    Extract physics-relevant features for beam theory
    Focus on features that directly impact load capacity
    """
    if len(vertices) == 0:
        return np.zeros(25, dtype=np.float32)

    # Bounding box dimensions
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    dims = max_coords - min_coords

    # Sort dimensions (span > width > thickness for shelf boards)
    sorted_idx = np.argsort(dims)[::-1]
    span = dims[sorted_idx[0]]
    width = dims[sorted_idx[1]]
    thickness = dims[sorted_idx[2]]

    # Convert to mm
    L = span * 1000
    b = width * 1000
    h = thickness * 1000

    # Cross-section properties (critical for load capacity)
    I = (b * h**3) / 12  # Moment of inertia
    S = (b * h**2) / 6   # Section modulus

    # Slenderness ratio (key for deflection)
    slenderness = L / (h + 1e-6)

    # Physics-based dimensionless parameters
    # These relate directly to beam theory formulas
    aspect_Lh = L / (h + 1e-6)  # Most important for deflection
    aspect_Lb = L / (b + 1e-6)
    aspect_bh = b / (h + 1e-6)

    # Deflection sensitivity: δ ∝ L⁴ / (E * I) = L⁴ / (E * b * h³)
    # Normalized: L⁴ / (b * h³)
    deflection_factor = L**4 / (b * h**3 + 1e-6)

    # Stress sensitivity: σ ∝ L² / S = L² / (b * h²)
    stress_factor = L**2 / (b * h**2 + 1e-6)

    # Load capacity scales inversely with these factors
    # P_max ∝ b * h² / L² (from stress) or b * h³ / L⁴ (from deflection)

    # Features (normalized to reasonable ranges)
    features = np.array([
        # Direct dimensions (log scale)
        np.log(L + 1),
        np.log(b + 1),
        np.log(h + 1),

        # Key ratios
        L / 1000,  # normalized span
        b / 400,   # normalized width
        h / 30,    # normalized thickness

        # Aspect ratios (scale invariant)
        aspect_Lh / 100,
        aspect_Lb / 10,
        aspect_bh / 20,

        # Cross-section properties (log scale)
        np.log(I + 1) / 20,
        np.log(S + 1) / 15,

        # Physics factors (log scale, normalized)
        np.log(deflection_factor + 1) / 30,
        np.log(stress_factor + 1) / 15,

        # Derived quantities (normalized)
        slenderness / 100,
        np.log(L * b + 1) / 15,  # load area proxy
        h**2 / 1000,  # thickness squared (important for stress)
        h**3 / 30000,  # thickness cubed (important for deflection)

        # Thickness categories (for material estimation)
        float(h < 15),  # thin
        float(15 <= h < 20),  # medium
        float(h >= 20),  # thick

        # Combined physics terms
        (b * h**2) / 10000,  # section modulus proxy
        (b * h**3) / 300000,  # moment of inertia proxy
        L**2 / (b * h**2 + 1e-6) / 1000,  # stress factor normalized
        L**4 / (b * h**3 + 1e-6) / 1e8,  # deflection factor normalized
        (b * h**2) / (L**2 + 1e-6) * 1000,  # inverse stress factor (load capacity proxy)
    ], dtype=np.float32)

    # Clip to prevent extreme values
    features = np.clip(features, -10, 10)

    return features


def normalize_point_cloud(points):
    """Normalize point cloud"""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
    return points


def sample_points(points, num_points):
    """Sample fixed number of points"""
    n = len(points)
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if n >= num_points:
        indices = np.random.choice(n, num_points, replace=False)
    else:
        indices = np.random.choice(n, num_points, replace=True)
    return points[indices]


class LoadCapacityDataset(Dataset):
    """Dataset for direct load capacity prediction"""

    def __init__(self, metadata, data_dir, num_points=NUM_POINTS, augment=False):
        self.metadata = metadata
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.augment = augment

        # Compute dataset statistics for normalization
        capacities = [m['target_capacity_kg'] for m in metadata]
        self.cap_mean = np.mean(capacities)
        self.cap_std = np.std(capacities)
        self.cap_log_mean = np.mean(np.log(np.array(capacities) + 1))
        self.cap_log_std = np.std(np.log(np.array(capacities) + 1))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        filepath = self.data_dir / item['filename']

        vertices = load_obj_vertices(filepath)
        physics_features = extract_physics_features(vertices)

        # Get dimensions in mm
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        dims = (max_coords - min_coords) * 1000
        sorted_idx = np.argsort(dims)[::-1]
        span_mm = dims[sorted_idx[0]]
        width_mm = dims[sorted_idx[1]]
        thickness_mm = dims[sorted_idx[2]]

        # Normalize and sample
        vertices = normalize_point_cloud(vertices)
        if self.augment:
            vertices = self._augment(vertices)
        points = sample_points(vertices, self.num_points)

        # Target: log-transformed capacity (better for regression)
        target_capacity = item['target_capacity_kg']
        target_log = np.log(target_capacity + 1)

        return {
            'points': torch.from_numpy(points).float(),
            'physics_features': torch.from_numpy(physics_features).float(),
            'target_capacity': torch.tensor(target_capacity).float(),
            'target_log': torch.tensor(target_log).float(),
            'span_mm': torch.tensor(span_mm).float(),
            'width_mm': torch.tensor(width_mm).float(),
            'thickness_mm': torch.tensor(thickness_mm).float(),
        }

    def _augment(self, points):
        """Data augmentation - minimal to preserve physics"""
        # Small rotation around Y axis
        angle = random.uniform(-0.1, 0.1)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        points = points @ R.T

        # Very small noise
        noise = np.random.normal(0, 0.001, points.shape).astype(np.float32)
        points = points + noise

        return points


class PointNetEncoder(nn.Module):
    """PointNet encoder"""

    def __init__(self, output_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2)[0]
        return x


class PhysicsInformedLoadPredictor(nn.Module):
    """
    Physics-informed neural network for load capacity prediction

    Architecture:
    1. PointNet encodes shape
    2. Physics feature MLP encodes beam properties
    3. Combined features predict material "strength factor"
    4. Output: strength factor applied to physics-based calculation
    """

    def __init__(self, point_dim=128, physics_dim=25):
        super().__init__()

        # Point cloud encoder (learns shape features)
        self.point_encoder = PointNetEncoder(point_dim)

        # Physics feature encoder
        self.physics_encoder = nn.Sequential(
            nn.Linear(physics_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        combined_dim = point_dim + 128

        # Main predictor - outputs log capacity directly
        self.capacity_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Auxiliary: predict material strength factor (E * sigma proxy)
        self.strength_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Positive output
        )

    def forward(self, points, physics_features, dims=None):
        """
        Args:
            points: Point cloud [B, N, 3]
            physics_features: Physics features [B, 25]
            dims: (span, width, thickness) in mm, optional for physics constraint
        """
        point_feat = self.point_encoder(points)
        physics_feat = self.physics_encoder(physics_features)
        combined = torch.cat([point_feat, physics_feat], dim=1)

        # Direct capacity prediction (log scale)
        log_capacity = self.capacity_head(combined)

        # Strength factor prediction (for interpretability)
        strength_factor = self.strength_head(combined)

        return log_capacity.squeeze(-1), strength_factor.squeeze(-1)


def physics_loss(pred_log, target_log, dims, strength_pred):
    """
    Physics-informed loss that penalizes unrealistic predictions

    Real load capacity must satisfy:
    - P ∝ b * h² / L²  (stress limit)
    - P ∝ b * h³ / L⁴  (deflection limit)
    """
    span, width, thickness = dims

    # Reconstruction loss
    mse_loss = nn.functional.mse_loss(pred_log, target_log)

    # Huber loss for robustness
    huber_loss = nn.functional.smooth_l1_loss(pred_log, target_log)

    # Physics consistency: predictions should correlate with physics features
    # log(P) should decrease with log(L) and increase with log(h)
    # This is a soft constraint, not enforced strictly
    pred_cap = torch.exp(pred_log)

    # Stress-based capacity: P_stress ∝ b * h² / L²
    physics_stress = width * thickness**2 / (span**2 + 1e-6)

    # Deflection-based capacity: P_defl ∝ b * h³ / L⁴
    physics_defl = width * thickness**3 / (span**4 + 1e-6)

    # Minimum of two (conservative physics estimate)
    physics_min = torch.minimum(physics_stress, physics_defl)

    # Soft physics constraint: prediction should correlate with physics
    # Use Pearson correlation as loss
    pred_centered = pred_cap - pred_cap.mean()
    phys_centered = physics_min - physics_min.mean()
    correlation = (pred_centered * phys_centered).sum() / (
        pred_centered.norm() * phys_centered.norm() + 1e-6
    )
    physics_corr_loss = 1 - correlation

    # Combined loss
    total_loss = mse_loss + 0.5 * huber_loss + 0.1 * physics_corr_loss

    return total_loss, {
        'mse': mse_loss.item(),
        'huber': huber_loss.item(),
        'physics_corr': physics_corr_loss.item(),
    }


def train_epoch(model, loader, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total = 0

    for batch in loader:
        points = batch['points'].to(device)
        physics_features = batch['physics_features'].to(device)
        target_log = batch['target_log'].to(device)
        span = batch['span_mm'].to(device)
        width = batch['width_mm'].to(device)
        thickness = batch['thickness_mm'].to(device)

        optimizer.zero_grad()

        pred_log, strength = model(points, physics_features)

        loss, loss_dict = physics_loss(
            pred_log, target_log,
            (span, width, thickness),
            strength
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(target_log)
        total += len(target_log)

    return total_loss / total


def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            points = batch['points'].to(device)
            physics_features = batch['physics_features'].to(device)
            target_capacity = batch['target_capacity'].numpy()

            pred_log, strength = model(points, physics_features)
            pred_capacity = torch.exp(pred_log).cpu().numpy()

            predictions.extend(pred_capacity)
            targets.extend(target_capacity)

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Metrics
    errors = np.abs(predictions - targets) / targets * 100
    mape = np.mean(errors)
    within_5 = np.mean(errors < 5) * 100
    within_10 = np.mean(errors < 10) * 100
    within_20 = np.mean(errors < 20) * 100
    within_30 = np.mean(errors < 30) * 100

    # Correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]

    return {
        'mape': mape,
        'within_5': within_5,
        'within_10': within_10,
        'within_20': within_20,
        'within_30': within_30,
        'correlation': correlation,
    }


def main():
    print("=" * 60)
    print("Load Capacity Prediction v3")
    print("Physics-Informed Neural Network")
    print("=" * 60)

    # Load dataset
    with open(DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)

    print(f"\nDataset: {len(metadata)} samples")

    # Split data
    train_meta, val_meta = train_test_split(metadata, test_size=0.2, random_state=42)
    print(f"Train: {len(train_meta)}, Val: {len(val_meta)}")

    # Create datasets
    train_dataset = LoadCapacityDataset(train_meta, DATA_DIR, augment=True)
    val_dataset = LoadCapacityDataset(val_meta, DATA_DIR, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Create model
    model = PhysicsInformedLoadPredictor().to(DEVICE)
    print(f"\nDevice: {DEVICE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    # Training loop
    best_mape = float('inf')
    patience = 50
    patience_counter = 0

    print("\n" + "-" * 60)
    print("Training")
    print("-" * 60)

    for epoch in range(500):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, DEVICE)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                  f"MAPE={val_metrics['mape']:.1f}%, "
                  f"<5%={val_metrics['within_5']:.1f}%, "
                  f"<10%={val_metrics['within_10']:.1f}%, "
                  f"<20%={val_metrics['within_20']:.1f}%, "
                  f"Corr={val_metrics['correlation']:.3f}")

        # Save best model
        if val_metrics['mape'] < best_mape:
            best_mape = val_metrics['mape']
            patience_counter = 0

            MODEL_DIR.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
            }, MODEL_DIR / "load_capacity_v3_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Final evaluation
    print("\n" + "-" * 60)
    print("Final Evaluation")
    print("-" * 60)

    checkpoint = torch.load(MODEL_DIR / "load_capacity_v3_best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    val_metrics = evaluate(model, val_loader, DEVICE)

    print(f"\nLoad Capacity Prediction:")
    print(f"  MAPE: {val_metrics['mape']:.1f}%")
    print(f"  Within 5%: {val_metrics['within_5']:.1f}%")
    print(f"  Within 10%: {val_metrics['within_10']:.1f}%")
    print(f"  Within 20%: {val_metrics['within_20']:.1f}%")
    print(f"  Within 30%: {val_metrics['within_30']:.1f}%")
    print(f"  Correlation: {val_metrics['correlation']:.3f}")

    print(f"\nModel saved: {MODEL_DIR / 'load_capacity_v3_best.pth'}")


if __name__ == "__main__":
    main()
