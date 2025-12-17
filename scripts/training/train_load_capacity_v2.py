#!/usr/bin/env python3
"""
Train ML model for load capacity prediction - Version 2
Strategy:
1. Realistic dataset with dimension-material correlation
2. Ensemble model with uncertainty estimation
3. Conservative estimation for high reliability
4. Output range (min-max) instead of single value
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
DATA_DIR = PROJECT_ROOT / "data/datasets/dataset_load_v2"
MODEL_DIR = PROJECT_ROOT / "models/models_load"

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

NUM_POINTS = 512

# Material definitions with typical thickness ranges
MATERIALS = {
    'plywood_thin': {'E': 8000, 'sigma': 6, 'thickness_range': (0.009, 0.015), 'class': 0},
    'plywood_medium': {'E': 8000, 'sigma': 6, 'thickness_range': (0.015, 0.021), 'class': 0},
    'plywood_thick': {'E': 8000, 'sigma': 6, 'thickness_range': (0.021, 0.030), 'class': 0},
    'mdf_thin': {'E': 3500, 'sigma': 4, 'thickness_range': (0.012, 0.018), 'class': 1},
    'mdf_thick': {'E': 3500, 'sigma': 4, 'thickness_range': (0.018, 0.025), 'class': 1},
    'particle_thin': {'E': 2500, 'sigma': 2.5, 'thickness_range': (0.012, 0.018), 'class': 2},
    'particle_thick': {'E': 2500, 'sigma': 2.5, 'thickness_range': (0.018, 0.025), 'class': 2},
    'pine': {'E': 10000, 'sigma': 8, 'thickness_range': (0.018, 0.030), 'class': 3},
    'oak': {'E': 12000, 'sigma': 10, 'thickness_range': (0.018, 0.030), 'class': 4},
}

# Material properties by class (for physics calculations)
CLASS_PROPERTIES = {
    0: {'E': 8000, 'sigma': 6, 'name': 'plywood'},
    1: {'E': 3500, 'sigma': 4, 'name': 'mdf'},
    2: {'E': 2500, 'sigma': 2.5, 'name': 'particle'},
    3: {'E': 10000, 'sigma': 8, 'name': 'pine'},
    4: {'E': 12000, 'sigma': 10, 'name': 'oak'},
}

NUM_CLASSES = 5


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
        [0, 1, 2, 3], [4, 7, 6, 5],
        [0, 4, 5, 1], [2, 6, 7, 3],
        [0, 3, 7, 4], [1, 5, 6, 2],
    ]

    return np.array(vertices), faces


def physics_load_capacity(span_mm, width_mm, thickness_mm, E, sigma, safety_factor=2.5):
    """Calculate load capacity using beam theory"""
    L, b, h = span_mm, width_mm, thickness_mm

    I = (b * h**3) / 12
    S = (b * h**2) / 6
    moment_coef = 1/8
    deflection_coef = 5/384

    # Stress limit
    w_stress = sigma * S / (L**2 * moment_coef)
    max_stress = (w_stress * L) / 9.81 / safety_factor

    # Deflection limit
    max_defl = L / 200
    w_defl = max_defl * E * I / (deflection_coef * L**4)
    max_defl_load = (w_defl * L) / 9.81 / safety_factor

    return min(max_stress, max_defl_load)


def generate_realistic_dataset(num_samples=3000):
    """Generate dataset with realistic dimension-material correlations"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    SPANS = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
    WIDTHS = [0.2, 0.25, 0.3, 0.35, 0.4]

    metadata = []

    for i in range(num_samples):
        # Choose material (some materials more common)
        material_weights = {
            'plywood_thin': 0.08, 'plywood_medium': 0.15, 'plywood_thick': 0.07,
            'mdf_thin': 0.12, 'mdf_thick': 0.10,
            'particle_thin': 0.12, 'particle_thick': 0.10,
            'pine': 0.13, 'oak': 0.13,
        }
        material_key = random.choices(
            list(material_weights.keys()),
            weights=list(material_weights.values())
        )[0]
        material = MATERIALS[material_key]

        # Generate dimensions with material-appropriate thickness
        span = random.choice(SPANS) + random.uniform(-0.05, 0.05)
        width = random.choice(WIDTHS) + random.uniform(-0.02, 0.02)
        t_min, t_max = material['thickness_range']
        thickness = random.uniform(t_min, t_max)

        # Create shelf board
        vertices, faces = create_box(
            center=[0, thickness/2, 0],
            size=[span, thickness, width]
        )

        # Calculate theoretical capacity
        capacity = physics_load_capacity(
            span * 1000, width * 1000, thickness * 1000,
            material['E'], material['sigma']
        )

        # Small noise for real-world variation
        capacity_with_noise = capacity * random.uniform(0.95, 1.05)

        # Save
        filename = f"shelf_board_{i:05d}.obj"
        save_obj(vertices, faces, DATA_DIR / filename)

        metadata.append({
            'filename': filename,
            'material': material_key,
            'material_class': material['class'],
            'span_m': span,
            'width_m': width,
            'thickness_m': thickness,
            'E_mpa': material['E'],
            'sigma_allow_mpa': material['sigma'],
            'theoretical_capacity_kg': capacity,
            'target_capacity_kg': capacity_with_noise,
        })

        if (i + 1) % 500 == 0:
            print(f"Generated {i+1}/{num_samples}")

    with open(DATA_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated {num_samples} samples to {DATA_DIR}")
    return metadata


def load_obj_vertices(filepath):
    """Load vertices from OBJ"""
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices, dtype=np.float32)


def extract_geometric_features(vertices):
    """Extract geometric features from point cloud"""
    if len(vertices) == 0:
        return np.zeros(20, dtype=np.float32)

    # Bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    dims = max_coords - min_coords

    # Sort dimensions
    sorted_idx = np.argsort(dims)[::-1]
    span = dims[sorted_idx[0]]
    width = dims[sorted_idx[1]]
    thickness = dims[sorted_idx[2]]

    # Convert to mm
    span_mm = span * 1000
    width_mm = width * 1000
    thickness_mm = thickness * 1000

    # Basic geometry
    volume = span * width * thickness

    # Aspect ratios (important for identifying shelf boards)
    aspect_sw = span / (width + 1e-6)
    aspect_st = span / (thickness + 1e-6)
    aspect_wt = width / (thickness + 1e-6)

    # Beam properties
    I_mm4 = width_mm * (thickness_mm ** 3) / 12
    S_mm3 = width_mm * (thickness_mm ** 2) / 6
    slenderness = span_mm / (thickness_mm + 1e-6)

    # Thickness-based features (correlate with material type)
    thickness_category = 0  # thin
    if thickness_mm > 18:
        thickness_category = 2  # thick
    elif thickness_mm > 14:
        thickness_category = 1  # medium

    # Features with careful normalization
    features = np.array([
        # Raw dimensions (log scale for better learning)
        np.log1p(span_mm),
        np.log1p(width_mm),
        np.log1p(thickness_mm),

        # Ratios (scale invariant)
        aspect_sw,
        aspect_st,
        aspect_wt,

        # Beam properties (log scale)
        np.log1p(I_mm4),
        np.log1p(S_mm3),
        np.log1p(slenderness),

        # Thickness indicators (important for material prediction)
        thickness_mm / 30.0,  # normalized thickness
        (thickness_mm - 12) / 18.0,  # centered thickness
        float(thickness_category) / 2.0,

        # Span indicators
        span_mm / 1200.0,  # normalized span
        (span_mm - 400) / 800.0,  # centered span

        # Width indicators
        width_mm / 400.0,  # normalized width

        # Combined features
        np.log1p(span_mm * width_mm / thickness_mm),  # load area / thickness
        np.log1p(volume * 1e9),
        thickness_mm / span_mm,  # relative thickness
        width_mm / span_mm,  # relative width
        np.log1p(span_mm**4 / (thickness_mm**3 + 1e-6)),  # deflection sensitivity
    ], dtype=np.float32)

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
    """Dataset for load capacity prediction"""

    def __init__(self, metadata, data_dir, num_points=NUM_POINTS, augment=False):
        self.metadata = metadata
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        filepath = self.data_dir / item['filename']

        # Load vertices
        vertices = load_obj_vertices(filepath)

        # Extract features before normalization
        geo_features = extract_geometric_features(vertices)

        # Get dimensions
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

        return {
            'points': torch.from_numpy(points).float(),
            'geo_features': torch.from_numpy(geo_features).float(),
            'material_class': torch.tensor(item['material_class']).long(),
            'target_capacity': torch.tensor(item['target_capacity_kg']).float(),
            'E': torch.tensor(item['E_mpa']).float(),
            'sigma': torch.tensor(item['sigma_allow_mpa']).float(),
            'span_mm': span_mm,
            'width_mm': width_mm,
            'thickness_mm': thickness_mm,
        }

    def _augment(self, points):
        """Data augmentation"""
        # Random rotation around Y axis
        angle = random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        points = points @ R.T

        # Small scale variation
        scale = random.uniform(0.98, 1.02)
        points = points * scale

        # Small noise
        noise = np.random.normal(0, 0.002, points.shape).astype(np.float32)
        points = points + noise

        return points


class PointNetEncoder(nn.Module):
    """Enhanced PointNet encoder"""

    def __init__(self, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, dim=2)[0]
        return x


class EnsembleLoadPredictor(nn.Module):
    """
    Ensemble model for load capacity prediction
    Outputs:
    - Material class probabilities
    - Load capacity estimate
    - Uncertainty estimate
    """

    def __init__(self, point_dim=256, geo_dim=20, num_classes=NUM_CLASSES, num_heads=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads

        # Shared encoder
        self.point_encoder = PointNetEncoder(point_dim)

        # Geometry encoder
        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        combined_dim = point_dim + 128

        # Multiple prediction heads for ensemble
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes + 2)  # classes + (E, sigma)
            ) for _ in range(num_heads)
        ])

        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, points, geo_features):
        point_feat = self.point_encoder(points)
        geo_feat = self.geo_encoder(geo_features)
        combined = torch.cat([point_feat, geo_feat], dim=1)

        # Get predictions from all heads
        outputs = [head(combined) for head in self.heads]

        # Average class logits
        class_logits = torch.stack([o[:, :self.num_classes] for o in outputs]).mean(dim=0)

        # Average property predictions
        properties = torch.stack([o[:, self.num_classes:] for o in outputs]).mean(dim=0)

        # Calculate prediction variance (uncertainty)
        prop_std = torch.stack([o[:, self.num_classes:] for o in outputs]).std(dim=0).mean(dim=1, keepdim=True)

        # Learned uncertainty
        uncertainty = self.uncertainty_head(combined)
        total_uncertainty = uncertainty + prop_std * 0.5

        return class_logits, properties, total_uncertainty


def train_epoch(model, loader, criterion_class, criterion_prop, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total = 0

    for batch in loader:
        points = batch['points'].to(device)
        geo_features = batch['geo_features'].to(device)
        material_class = batch['material_class'].to(device)
        E = batch['E'].to(device)
        sigma = batch['sigma'].to(device)

        # Normalize targets
        E_norm = torch.log1p(E) / 12
        sigma_norm = sigma / 15

        optimizer.zero_grad()

        class_logits, properties, uncertainty = model(points, geo_features)

        # Classification loss with label smoothing
        loss_class = criterion_class(class_logits, material_class)

        # Property regression loss
        targets = torch.stack([E_norm, sigma_norm], dim=1)
        loss_prop = criterion_prop(properties, targets)

        # Total loss
        loss = loss_class + loss_prop * 3

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(material_class)
        total += len(material_class)

    return total_loss / total


def evaluate(model, loader, device, conservative=True):
    """
    Evaluate model with optional conservative estimation

    Conservative mode: uses the lower bound of possible load capacities
    """
    model.eval()

    correct = 0
    total = 0
    load_errors = []
    conservative_errors = []
    range_contains_target = []

    with torch.no_grad():
        for batch in loader:
            points = batch['points'].to(device)
            geo_features = batch['geo_features'].to(device)
            material_class = batch['material_class']
            target_capacity = batch['target_capacity'].numpy()
            span_mm = batch['span_mm'].numpy()
            width_mm = batch['width_mm'].numpy()
            thickness_mm = batch['thickness_mm'].numpy()

            class_logits, properties, uncertainty = model(points, geo_features)

            # Classification accuracy
            _, predicted = torch.max(class_logits, 1)
            correct += (predicted.cpu() == material_class).sum().item()
            total += len(material_class)

            # Get probabilities
            probs = torch.softmax(class_logits, dim=1).cpu().numpy()

            # Predicted properties
            properties = properties.cpu().numpy()
            E_pred = np.expm1(properties[:, 0] * 12)
            sigma_pred = properties[:, 1] * 15

            for i in range(len(target_capacity)):
                # Point estimate using predicted properties
                pred_capacity = physics_load_capacity(
                    span_mm[i], width_mm[i], thickness_mm[i],
                    E_pred[i], sigma_pred[i]
                )
                error = abs(pred_capacity - target_capacity[i]) / target_capacity[i] * 100
                load_errors.append(error)

                # Conservative estimate: use weakest likely material
                # Calculate capacity for all materials
                capacities = []
                for cls_id, props in CLASS_PROPERTIES.items():
                    cap = physics_load_capacity(
                        span_mm[i], width_mm[i], thickness_mm[i],
                        props['E'], props['sigma']
                    )
                    capacities.append((cls_id, cap, probs[i, cls_id]))

                # Sort by capacity
                capacities.sort(key=lambda x: x[1])

                # Conservative: weighted toward lower capacities
                # Use probability-weighted minimum of likely materials
                likely_materials = [c for c in capacities if c[2] > 0.1]
                if not likely_materials:
                    likely_materials = capacities[:2]

                # Conservative estimate: minimum of likely materials
                conservative_cap = min(c[1] for c in likely_materials)

                # Optimistic estimate: maximum of likely materials
                optimistic_cap = max(c[1] for c in likely_materials)

                # Check if range contains target
                range_contains = conservative_cap <= target_capacity[i] <= optimistic_cap * 1.2
                range_contains_target.append(range_contains)

                # Conservative error (should be safe, i.e., underestimate)
                if conservative_cap <= target_capacity[i]:
                    cons_error = 0  # Safe underestimate
                else:
                    cons_error = (conservative_cap - target_capacity[i]) / target_capacity[i] * 100
                conservative_errors.append(cons_error)

    class_acc = correct / total * 100
    mape = np.mean(load_errors)
    within_10 = np.mean(np.array(load_errors) < 10) * 100
    within_20 = np.mean(np.array(load_errors) < 20) * 100
    within_30 = np.mean(np.array(load_errors) < 30) * 100

    # Conservative metrics
    safe_rate = np.mean(np.array(conservative_errors) == 0) * 100
    range_coverage = np.mean(range_contains_target) * 100

    return {
        'class_acc': class_acc,
        'mape': mape,
        'within_10': within_10,
        'within_20': within_20,
        'within_30': within_30,
        'safe_rate': safe_rate,
        'range_coverage': range_coverage,
    }


def main():
    print("=" * 60)
    print("Load Capacity ML Model v2")
    print("Ensemble + Uncertainty + Conservative Estimation")
    print("=" * 60)

    # Generate or load dataset
    metadata_path = DATA_DIR / "metadata.json"
    if not metadata_path.exists():
        print("\nGenerating realistic dataset...")
        metadata = generate_realistic_dataset(3000)
    else:
        with open(metadata_path) as f:
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
    model = EnsembleLoadPredictor(num_heads=5).to(DEVICE)
    print(f"\nDevice: {DEVICE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_prop = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30)

    # Training loop
    best_metric = 0  # range_coverage
    patience = 40
    patience_counter = 0

    print("\n" + "-" * 60)
    print("Training")
    print("-" * 60)

    for epoch in range(400):
        train_loss = train_epoch(
            model, train_loader, criterion_class, criterion_prop, optimizer, DEVICE
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, DEVICE)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                  f"ClassAcc={val_metrics['class_acc']:.1f}%, "
                  f"MAPE={val_metrics['mape']:.1f}%, "
                  f"Safe={val_metrics['safe_rate']:.1f}%, "
                  f"Range={val_metrics['range_coverage']:.1f}%")

        # Save best model (optimize for range coverage)
        metric = val_metrics['range_coverage']
        if metric > best_metric:
            best_metric = metric
            patience_counter = 0

            MODEL_DIR.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
            }, MODEL_DIR / "load_capacity_v2_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Final evaluation
    print("\n" + "-" * 60)
    print("Final Evaluation")
    print("-" * 60)

    checkpoint = torch.load(MODEL_DIR / "load_capacity_v2_best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    val_metrics = evaluate(model, val_loader, DEVICE)

    print(f"\nMaterial Classification: {val_metrics['class_acc']:.1f}%")
    print(f"Point Estimate MAPE: {val_metrics['mape']:.1f}%")
    print(f"  Within 10%: {val_metrics['within_10']:.1f}%")
    print(f"  Within 20%: {val_metrics['within_20']:.1f}%")
    print(f"  Within 30%: {val_metrics['within_30']:.1f}%")
    print(f"\nConservative Estimation:")
    print(f"  Safe Rate (underestimate): {val_metrics['safe_rate']:.1f}%")
    print(f"  Range Contains Target: {val_metrics['range_coverage']:.1f}%")

    print(f"\nModel saved: {MODEL_DIR / 'load_capacity_v2_best.pth'}")


if __name__ == "__main__":
    main()
