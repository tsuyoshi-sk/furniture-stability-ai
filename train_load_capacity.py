#!/usr/bin/env python3
"""
Train ML model for load capacity prediction
Strategy: Predict material properties (E, sigma) from geometry, then use physics model
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
DATA_DIR = SCRIPT_DIR / "dataset_load"
MODEL_DIR = SCRIPT_DIR / "models_load"

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

NUM_POINTS = 512

# Material classes
MATERIAL_CLASSES = {
    'plywood': 0,
    'mdf': 1,
    'particle': 2,
    'pine': 3,
    'oak': 4,
}

# Material properties for physics calculation
MATERIAL_PROPS = {
    0: {'E': 8000, 'sigma': 6},    # plywood
    1: {'E': 3500, 'sigma': 4},    # mdf
    2: {'E': 2500, 'sigma': 2.5},  # particle
    3: {'E': 10000, 'sigma': 8},   # pine
    4: {'E': 12000, 'sigma': 10},  # oak
}


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
        return np.zeros(15, dtype=np.float32)

    # Bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    dims = max_coords - min_coords

    # Sort dimensions
    sorted_idx = np.argsort(dims)[::-1]
    span = dims[sorted_idx[0]]
    width = dims[sorted_idx[1]]
    thickness = dims[sorted_idx[2]]

    # Basic geometry
    volume = span * width * thickness
    surface = 2 * (span * width + width * thickness + span * thickness)

    # Aspect ratios
    aspect_sw = span / (width + 1e-6)
    aspect_st = span / (thickness + 1e-6)
    aspect_wt = width / (thickness + 1e-6)

    # Beam properties (scaled)
    I_proxy = width * (thickness ** 3) / 12
    S_proxy = width * (thickness ** 2) / 6
    slenderness = span / (thickness + 1e-6)

    # Normalize to reasonable ranges
    features = np.array([
        np.log1p(span * 1000),      # log scale mm
        np.log1p(width * 1000),
        np.log1p(thickness * 1000),
        np.log1p(volume * 1e9),     # mm³
        np.log1p(surface * 1e6),    # mm²
        aspect_sw,
        aspect_st,
        aspect_wt,
        np.log1p(I_proxy * 1e12),
        np.log1p(S_proxy * 1e9),
        np.log1p(slenderness),
        span / width if width > 0 else 0,
        thickness / span if span > 0 else 0,
        width / span if span > 0 else 0,
        volume / (span * width * thickness + 1e-9) if span * width * thickness > 0 else 1,
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

        # Extract geometric features (before normalization)
        geo_features = extract_geometric_features(vertices)

        # Dimensions in mm
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        dims = (max_coords - min_coords) * 1000
        sorted_idx = np.argsort(dims)[::-1]
        span_mm = dims[sorted_idx[0]]
        width_mm = dims[sorted_idx[1]]
        thickness_mm = dims[sorted_idx[2]]

        # Normalize and sample point cloud
        vertices = normalize_point_cloud(vertices)

        if self.augment:
            vertices = self._augment(vertices)

        points = sample_points(vertices, self.num_points)

        # Material class
        material_name = item['material'].split('_')[0]
        material_class = MATERIAL_CLASSES.get(material_name, 0)

        # Target capacity
        target_capacity = item['target_capacity_kg']

        # Material properties
        E = item['E_mpa']
        sigma = item['sigma_allow_mpa']

        return {
            'points': torch.from_numpy(points).float(),
            'geo_features': torch.from_numpy(geo_features).float(),
            'material_class': torch.tensor(material_class).long(),
            'target_capacity': torch.tensor(target_capacity).float(),
            'E': torch.tensor(E).float(),
            'sigma': torch.tensor(sigma).float(),
            'span_mm': span_mm,
            'width_mm': width_mm,
            'thickness_mm': thickness_mm,
        }

    def _augment(self, points):
        """Data augmentation"""
        angle = random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        points = points @ R.T

        scale = random.uniform(0.95, 1.05)
        points = points * scale

        noise = np.random.normal(0, 0.005, points.shape).astype(np.float32)
        points = points + noise

        return points


class PointNetEncoder(nn.Module):
    """PointNet encoder"""

    def __init__(self, output_dim=256):
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


class MaterialPredictor(nn.Module):
    """Predict material class and properties from geometry"""

    def __init__(self, point_dim=256, geo_dim=15, num_classes=5):
        super().__init__()

        self.point_encoder = PointNetEncoder(point_dim)

        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        combined_dim = point_dim + 64

        # Material classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # Property regressor (E, sigma)
        self.property_regressor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # E, sigma
        )

    def forward(self, points, geo_features):
        point_feat = self.point_encoder(points)
        geo_feat = self.geo_encoder(geo_features)
        combined = torch.cat([point_feat, geo_feat], dim=1)

        class_logits = self.classifier(combined)
        properties = self.property_regressor(combined)

        return class_logits, properties


def train_epoch(model, loader, criterion_class, criterion_prop, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        points = batch['points'].to(device)
        geo_features = batch['geo_features'].to(device)
        material_class = batch['material_class'].to(device)
        E = batch['E'].to(device)
        sigma = batch['sigma'].to(device)

        # Normalize targets
        E_norm = torch.log1p(E) / 12  # log scale normalized
        sigma_norm = sigma / 15

        optimizer.zero_grad()

        class_logits, properties = model(points, geo_features)

        # Classification loss
        loss_class = criterion_class(class_logits, material_class)

        # Property regression loss
        targets = torch.stack([E_norm, sigma_norm], dim=1)
        loss_prop = criterion_prop(properties, targets)

        loss = loss_class + loss_prop * 2  # Weight property loss more

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(material_class)
        _, predicted = torch.max(class_logits, 1)
        correct += (predicted == material_class).sum().item()
        total += len(material_class)

    return total_loss / total, correct / total * 100


def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()

    correct = 0
    total = 0
    load_errors = []

    with torch.no_grad():
        for batch in loader:
            points = batch['points'].to(device)
            geo_features = batch['geo_features'].to(device)
            material_class = batch['material_class']
            target_capacity = batch['target_capacity'].numpy()
            span_mm = batch['span_mm'].numpy()
            width_mm = batch['width_mm'].numpy()
            thickness_mm = batch['thickness_mm'].numpy()

            class_logits, properties = model(points, geo_features)

            # Classification accuracy
            _, predicted = torch.max(class_logits, 1)
            correct += (predicted.cpu() == material_class).sum().item()
            total += len(material_class)

            # Predicted properties
            properties = properties.cpu().numpy()
            E_pred = np.expm1(properties[:, 0] * 12)
            sigma_pred = properties[:, 1] * 15

            # Calculate load capacity with predicted properties
            for i in range(len(target_capacity)):
                pred_capacity = physics_load_capacity(
                    span_mm[i], width_mm[i], thickness_mm[i],
                    E_pred[i], sigma_pred[i]
                )
                error = abs(pred_capacity - target_capacity[i]) / target_capacity[i] * 100
                load_errors.append(error)

    class_acc = correct / total * 100
    mape = np.mean(load_errors)
    within_10 = np.mean(np.array(load_errors) < 10) * 100
    within_20 = np.mean(np.array(load_errors) < 20) * 100
    within_30 = np.mean(np.array(load_errors) < 30) * 100

    return {
        'class_acc': class_acc,
        'mape': mape,
        'within_10': within_10,
        'within_20': within_20,
        'within_30': within_30,
    }


def main():
    print("=" * 60)
    print("Load Capacity ML Model Training")
    print("(Material Property Prediction + Physics)")
    print("=" * 60)

    # Load metadata
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
    model = MaterialPredictor().to(DEVICE)
    print(f"\nDevice: {DEVICE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion_class = nn.CrossEntropyLoss()
    criterion_prop = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)

    # Training loop
    best_mape = float('inf')
    patience = 30
    patience_counter = 0

    print("\n" + "-" * 60)
    print("Training")
    print("-" * 60)

    for epoch in range(300):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion_class, criterion_prop, optimizer, DEVICE
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, DEVICE)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                  f"ClassAcc={val_metrics['class_acc']:.1f}%, "
                  f"MAPE={val_metrics['mape']:.1f}%, "
                  f"<10%={val_metrics['within_10']:.1f}%, "
                  f"<20%={val_metrics['within_20']:.1f}%")

        # Save best model
        if val_metrics['mape'] < best_mape:
            best_mape = val_metrics['mape']
            patience_counter = 0

            MODEL_DIR.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_mape': best_mape,
                'best_class_acc': val_metrics['class_acc'],
            }, MODEL_DIR / "load_capacity_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Final evaluation
    print("\n" + "-" * 60)
    print("Final Evaluation")
    print("-" * 60)

    checkpoint = torch.load(MODEL_DIR / "load_capacity_best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    val_metrics = evaluate(model, val_loader, DEVICE)

    print(f"\nMaterial Classification Accuracy: {val_metrics['class_acc']:.1f}%")
    print(f"Load Capacity MAPE: {val_metrics['mape']:.1f}%")
    print(f"Within 10% error: {val_metrics['within_10']:.1f}%")
    print(f"Within 20% error: {val_metrics['within_20']:.1f}%")
    print(f"Within 30% error: {val_metrics['within_30']:.1f}%")

    print(f"\nModel saved: {MODEL_DIR / 'load_capacity_best.pth'}")


if __name__ == "__main__":
    main()
