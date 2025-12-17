#!/usr/bin/env python3
"""
Furniture Image Generator
家具画像生成AI - Stable Diffusionベース

Features:
- Text-to-Image: 説明文から家具画像を生成
- Spec-to-Image: 仕様から家具画像を生成
- Style Transfer: 既存画像のスタイル変換
- Batch Generation: 複数バリエーション一括生成
"""
import os
import json
import torch
from pathlib import Path
from typing import Optional, List, Dict, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output/images"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class GeneratedImage:
    """Generated image result"""
    image: Image.Image
    prompt: str
    negative_prompt: str
    seed: int
    path: Optional[Path] = None
    metadata: Dict = None


# =============================================================================
# PROMPT ENGINEERING FOR FURNITURE
# =============================================================================

FURNITURE_STYLES = {
    'modern': 'modern minimalist design, clean lines, contemporary',
    'scandinavian': 'Scandinavian design, light wood, minimalist Nordic style',
    'industrial': 'industrial style, metal and wood, exposed hardware',
    'japanese': 'Japanese design, zen aesthetic, natural materials, wabi-sabi',
    'mid_century': 'mid-century modern, retro 1950s-60s style, organic curves',
    'traditional': 'traditional classic design, ornate details, rich wood tones',
    'rustic': 'rustic farmhouse style, reclaimed wood, weathered finish'
}

FURNITURE_MATERIALS = {
    'oak': 'solid oak wood, natural grain pattern',
    'walnut': 'dark walnut wood, rich brown tones',
    'pine': 'light pine wood, natural knots',
    'plywood': 'birch plywood, layered edges visible',
    'mdf': 'smooth MDF surface, painted finish',
    'steel': 'brushed steel, metallic finish',
    'glass': 'tempered glass, transparent',
    'marble': 'marble surface, natural veining'
}

QUALITY_BOOSTERS = [
    "professional product photography",
    "studio lighting",
    "8k resolution",
    "highly detailed",
    "photorealistic",
    "soft shadows",
    "clean white background"
]

NEGATIVE_PROMPTS = [
    "blurry", "low quality", "distorted", "deformed",
    "bad anatomy", "watermark", "text", "logo",
    "oversaturated", "underexposed", "noise", "grain",
    "cropped", "out of frame", "duplicate"
]


def build_furniture_prompt(
    furniture_type: str,
    material: str = 'oak',
    style: str = 'modern',
    dimensions: Optional[Dict] = None,
    features: Optional[List[str]] = None,
    quality: str = 'high'
) -> tuple[str, str]:
    """
    Build optimized prompt for furniture image generation

    Args:
        furniture_type: e.g., 'bookshelf', 'desk', 'chair', 'table'
        material: Material key from FURNITURE_MATERIALS
        style: Style key from FURNITURE_STYLES
        dimensions: {'width': 800, 'depth': 400, 'height': 1800}
        features: Additional features ['5 shelves', 'drawer', 'glass door']
        quality: 'high', 'medium', 'draft'

    Returns:
        (prompt, negative_prompt)
    """
    parts = []

    # Core description
    material_desc = FURNITURE_MATERIALS.get(material, material)
    style_desc = FURNITURE_STYLES.get(style, style)

    parts.append(f"{furniture_type}")
    parts.append(material_desc)
    parts.append(style_desc)

    # Dimensions if provided
    if dimensions:
        w = dimensions.get('width', 0)
        d = dimensions.get('depth', 0)
        h = dimensions.get('height', 0)
        if w and h:
            # Describe proportions
            aspect = h / w if w > 0 else 1
            if aspect > 2:
                parts.append("tall vertical design")
            elif aspect < 0.5:
                parts.append("wide horizontal design")
            else:
                parts.append("balanced proportions")

    # Features
    if features:
        parts.extend(features)

    # Quality boosters
    if quality == 'high':
        parts.extend(QUALITY_BOOSTERS)
    elif quality == 'medium':
        parts.extend(QUALITY_BOOSTERS[:4])

    prompt = ", ".join(parts)
    negative_prompt = ", ".join(NEGATIVE_PROMPTS)

    return prompt, negative_prompt


# =============================================================================
# IMAGE GENERATOR CLASS
# =============================================================================

class FurnitureImageGenerator:
    """
    Furniture Image Generator using Stable Diffusion

    Usage:
        generator = FurnitureImageGenerator()

        # Generate from text
        image = generator.generate("modern oak bookshelf with 5 shelves")

        # Generate from specs
        image = generator.generate_from_specs({
            'type': 'bookshelf',
            'material': 'oak',
            'width': 800,
            'height': 1800,
            'shelves': 5
        })
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize generator

        Args:
            model_id: HuggingFace model ID
            device: 'cuda', 'mps', or 'cpu'
            dtype: torch.float16 or torch.float32
        """
        self.model_id = model_id
        self.device = device or self._detect_device()
        self.dtype = dtype if self.device != 'cpu' else torch.float32
        self.pipe = None
        self._loaded = False

    def _detect_device(self) -> str:
        """Detect best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self):
        """Load the diffusion model"""
        if self._loaded:
            return

        print(f"Loading model {self.model_id} on {self.device}...")

        try:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            )

            # Use faster scheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            self.pipe = self.pipe.to(self.device)

            # Memory optimization
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()

            self._loaded = True
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        save: bool = True,
        filename: Optional[str] = None
    ) -> GeneratedImage:
        """
        Generate furniture image from text prompt

        Args:
            prompt: Text description
            negative_prompt: What to avoid
            width: Image width (multiple of 8)
            height: Image height (multiple of 8)
            num_inference_steps: Quality steps (20-50)
            guidance_scale: Prompt adherence (5-15)
            seed: Random seed for reproducibility
            save: Whether to save the image
            filename: Output filename

        Returns:
            GeneratedImage with PIL Image
        """
        self.load_model()

        # Ensure dimensions are multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Set seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Default negative prompt
        if negative_prompt is None:
            negative_prompt = ", ".join(NEGATIVE_PROMPTS)

        # Enhance prompt for furniture
        enhanced_prompt = f"furniture product photo, {prompt}, professional studio lighting, white background"

        print(f"Generating image...")
        print(f"  Prompt: {enhanced_prompt[:80]}...")
        print(f"  Size: {width}x{height}, Steps: {num_inference_steps}")

        # Generate
        with torch.inference_mode():
            result = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )

        image = result.images[0]

        # Save if requested
        output_path = None
        if save:
            if filename is None:
                filename = f"furniture_{seed}.png"
            output_path = OUTPUT_DIR / filename
            image.save(output_path)
            print(f"  Saved: {output_path}")

        return GeneratedImage(
            image=image,
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            path=output_path,
            metadata={
                'model': self.model_id,
                'steps': num_inference_steps,
                'guidance': guidance_scale,
                'size': f"{width}x{height}"
            }
        )

    def generate_from_specs(
        self,
        specs: Dict,
        style: str = 'modern',
        **kwargs
    ) -> GeneratedImage:
        """
        Generate furniture image from specifications

        Args:
            specs: {
                'type': 'bookshelf',
                'material': 'oak',
                'width': 800,
                'depth': 300,
                'height': 1800,
                'shelves': 5,
                'features': ['adjustable shelves']
            }
            style: Design style
            **kwargs: Additional generate() arguments

        Returns:
            GeneratedImage
        """
        furniture_type = specs.get('type', 'furniture')
        material = specs.get('material', 'wood')
        dimensions = {
            'width': specs.get('width', 0),
            'depth': specs.get('depth', 0),
            'height': specs.get('height', 0)
        }

        features = specs.get('features', [])
        if specs.get('shelves'):
            features.append(f"{specs['shelves']} shelves")
        if specs.get('doors'):
            features.append(f"{specs['doors']} doors")
        if specs.get('drawers'):
            features.append(f"{specs['drawers']} drawers")

        prompt, negative_prompt = build_furniture_prompt(
            furniture_type=furniture_type,
            material=material,
            style=style,
            dimensions=dimensions,
            features=features
        )

        return self.generate(prompt, negative_prompt, **kwargs)

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int = 4,
        vary_style: bool = True,
        vary_material: bool = False,
        **kwargs
    ) -> List[GeneratedImage]:
        """
        Generate multiple variations of furniture

        Args:
            base_prompt: Base furniture description
            num_variations: Number of variations
            vary_style: Vary design styles
            vary_material: Vary materials

        Returns:
            List of GeneratedImage
        """
        results = []

        styles = list(FURNITURE_STYLES.keys()) if vary_style else ['modern']
        materials = list(FURNITURE_MATERIALS.keys())[:4] if vary_material else ['oak']

        for i in range(num_variations):
            style = styles[i % len(styles)]
            material = materials[i % len(materials)]

            style_desc = FURNITURE_STYLES[style]
            material_desc = FURNITURE_MATERIALS[material]

            prompt = f"{base_prompt}, {style_desc}, {material_desc}"

            result = self.generate(
                prompt,
                filename=f"variation_{i}_{style}_{material}.png",
                seed=None,  # Random seed for each
                **kwargs
            )
            results.append(result)

        return results

    def generate_catalog_page(
        self,
        furniture_list: List[Dict],
        output_path: Optional[Path] = None
    ) -> Image.Image:
        """
        Generate a catalog page with multiple furniture items

        Args:
            furniture_list: List of furniture specs
            output_path: Where to save the combined image

        Returns:
            Combined PIL Image
        """
        images = []
        for specs in furniture_list:
            result = self.generate_from_specs(
                specs,
                width=256,
                height=256,
                num_inference_steps=20,
                save=False
            )
            images.append(result.image)

        # Create grid
        n = len(images)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        cell_size = 256
        grid = Image.new('RGB', (cols * cell_size, rows * cell_size), 'white')

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            grid.paste(img, (col * cell_size, row * cell_size))

        if output_path:
            grid.save(output_path)
            print(f"Catalog page saved: {output_path}")

        return grid


# =============================================================================
# LIGHTWEIGHT ALTERNATIVE (No GPU Required)
# =============================================================================

class SimpleFurnitureGenerator:
    """
    Simple furniture image generator using basic shapes
    For systems without GPU or when quick previews are needed
    """

    def __init__(self):
        self.colors = {
            'oak': '#C4A35A',
            'walnut': '#5C4033',
            'pine': '#E8D4A8',
            'white': '#FFFFFF',
            'black': '#2C2C2C',
            'steel': '#8C8C8C'
        }

    def generate_bookshelf(
        self,
        width: int = 400,
        height: int = 600,
        shelves: int = 5,
        material: str = 'oak',
        thickness: int = 20
    ) -> Image.Image:
        """Generate simple bookshelf illustration"""
        from PIL import ImageDraw

        img = Image.new('RGB', (width + 100, height + 100), 'white')
        draw = ImageDraw.Draw(img)

        color = self.colors.get(material, '#C4A35A')
        darker = self._darken_color(color)

        # Draw frame
        x, y = 50, 50

        # Sides
        draw.rectangle([x, y, x + thickness, y + height], fill=color, outline=darker)
        draw.rectangle([x + width - thickness, y, x + width, y + height], fill=color, outline=darker)

        # Top and bottom
        draw.rectangle([x, y, x + width, y + thickness], fill=color, outline=darker)
        draw.rectangle([x, y + height - thickness, x + width, y + height], fill=color, outline=darker)

        # Shelves
        shelf_spacing = (height - 2 * thickness) / shelves
        for i in range(1, shelves):
            shelf_y = y + thickness + i * shelf_spacing
            draw.rectangle(
                [x + thickness, shelf_y - thickness/2, x + width - thickness, shelf_y + thickness/2],
                fill=color, outline=darker
            )

        # Add shadow
        shadow_offset = 5
        draw.rectangle(
            [x + width, y + shadow_offset, x + width + shadow_offset, y + height + shadow_offset],
            fill='#E0E0E0'
        )
        draw.rectangle(
            [x + shadow_offset, y + height, x + width + shadow_offset, y + height + shadow_offset],
            fill='#E0E0E0'
        )

        return img

    def generate_desk(
        self,
        width: int = 500,
        height: int = 300,
        depth: int = 250,
        material: str = 'oak'
    ) -> Image.Image:
        """Generate simple desk illustration (isometric view)"""
        from PIL import ImageDraw

        img = Image.new('RGB', (width + 150, height + 150), 'white')
        draw = ImageDraw.Draw(img)

        color = self.colors.get(material, '#C4A35A')
        darker = self._darken_color(color)
        lighter = self._lighten_color(color)

        # Isometric offsets
        iso_x = depth // 3
        iso_y = depth // 4

        x, y = 50, 50
        top_thick = 25
        leg_width = 30

        # Desktop top surface
        points = [
            (x, y + iso_y),
            (x + width, y + iso_y),
            (x + width + iso_x, y),
            (x + iso_x, y)
        ]
        draw.polygon(points, fill=color, outline=darker)

        # Desktop front
        points = [
            (x, y + iso_y),
            (x + width, y + iso_y),
            (x + width, y + iso_y + top_thick),
            (x, y + iso_y + top_thick)
        ]
        draw.polygon(points, fill=darker, outline=darker)

        # Desktop side
        points = [
            (x + width, y + iso_y),
            (x + width + iso_x, y),
            (x + width + iso_x, y + top_thick),
            (x + width, y + iso_y + top_thick)
        ]
        draw.polygon(points, fill=lighter, outline=darker)

        # Legs
        leg_height = height - top_thick
        leg_positions = [
            (x + 20, y + iso_y + top_thick),
            (x + width - leg_width - 20, y + iso_y + top_thick)
        ]

        for lx, ly in leg_positions:
            # Front face of leg
            draw.rectangle(
                [lx, ly, lx + leg_width, ly + leg_height],
                fill=darker, outline=darker
            )
            # Side face of leg
            points = [
                (lx + leg_width, ly),
                (lx + leg_width + iso_x//2, ly - iso_y//2),
                (lx + leg_width + iso_x//2, ly + leg_height - iso_y//2),
                (lx + leg_width, ly + leg_height)
            ]
            draw.polygon(points, fill=color, outline=darker)

        return img

    def generate_table(
        self,
        width: int = 400,
        height: int = 250,
        material: str = 'oak',
        legs: int = 4
    ) -> Image.Image:
        """Generate simple table illustration"""
        from PIL import ImageDraw

        img = Image.new('RGB', (width + 100, height + 100), 'white')
        draw = ImageDraw.Draw(img)

        color = self.colors.get(material, '#C4A35A')
        darker = self._darken_color(color)

        x, y = 50, 50
        top_thick = 30
        leg_width = 25
        leg_height = height - top_thick

        # Table top
        draw.rectangle([x, y, x + width, y + top_thick], fill=color, outline=darker)

        # Legs
        if legs == 4:
            leg_positions = [
                (x + 10, y + top_thick),
                (x + width - leg_width - 10, y + top_thick),
                (x + 10 + leg_width // 2, y + top_thick),  # Back legs (partially visible)
                (x + width - leg_width - 10 + leg_width // 2, y + top_thick)
            ]
            for i, (lx, ly) in enumerate(leg_positions[:2]):  # Only front legs fully visible
                draw.rectangle(
                    [lx, ly, lx + leg_width, ly + leg_height],
                    fill=darker, outline=darker
                )

        return img

    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color"""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        factor = 0.7
        return f'#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}'

    def _lighten_color(self, hex_color: str) -> str:
        """Lighten a hex color"""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        factor = 1.2
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f'#{r:02x}{g:02x}{b:02x}'


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Furniture Image Generator')
    parser.add_argument('--type', choices=['bookshelf', 'desk', 'table', 'cabinet'],
                        default='bookshelf', help='Furniture type')
    parser.add_argument('--material', default='oak', help='Material')
    parser.add_argument('--style', default='modern', help='Design style')
    parser.add_argument('--width', type=int, default=512, help='Image width')
    parser.add_argument('--height', type=int, default=512, help='Image height')
    parser.add_argument('--simple', action='store_true', help='Use simple generator (no GPU)')
    parser.add_argument('--prompt', '-p', help='Custom prompt')
    parser.add_argument('--output', '-o', help='Output filename')
    args = parser.parse_args()

    if args.simple:
        # Use simple generator
        gen = SimpleFurnitureGenerator()

        if args.type == 'bookshelf':
            img = gen.generate_bookshelf(material=args.material)
        elif args.type == 'desk':
            img = gen.generate_desk(material=args.material)
        elif args.type == 'table':
            img = gen.generate_table(material=args.material)
        else:
            img = gen.generate_bookshelf(material=args.material)

        output = args.output or f"simple_{args.type}_{args.material}.png"
        output_path = OUTPUT_DIR / output
        img.save(output_path)
        print(f"Saved: {output_path}")
    else:
        # Use Stable Diffusion
        gen = FurnitureImageGenerator()

        if args.prompt:
            result = gen.generate(
                args.prompt,
                width=args.width,
                height=args.height,
                filename=args.output
            )
        else:
            result = gen.generate_from_specs(
                {
                    'type': args.type,
                    'material': args.material,
                    'shelves': 5 if args.type == 'bookshelf' else None
                },
                style=args.style,
                width=args.width,
                height=args.height,
                filename=args.output
            )

        print(f"\nGenerated: {result.path}")
        print(f"Seed: {result.seed}")


if __name__ == "__main__":
    main()
