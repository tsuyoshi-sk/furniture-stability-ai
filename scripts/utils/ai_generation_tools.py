#!/usr/bin/env python3
"""
AI Furniture Generation Tools Integration
家具画像・図面・3Dモデル生成AIの統合モジュール

Supported APIs:
- Image Generation: Midjourney (via API), Stability AI, DALL-E 3
- 3D Generation: Meshy AI, Tripo AI
- CAD Generation: Zoo Text-to-CAD
"""
import os
import json
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
from abc import ABC, abstractmethod

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


@dataclass
class GenerationResult:
    """Result of AI generation"""
    success: bool
    output_path: Optional[Path]
    output_url: Optional[str]
    metadata: Dict
    error: Optional[str] = None


class FurnitureGenerator(ABC):
    """Base class for furniture generation"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        pass


# =============================================================================
# IMAGE GENERATION
# =============================================================================

class StabilityAIGenerator(FurnitureGenerator):
    """Stability AI (Stable Diffusion) for furniture images"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('STABILITY_API_KEY')
        self.api_url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"

    def generate(self, prompt: str, style: str = "photorealistic",
                 width: int = 1024, height: int = 1024,
                 output_path: Optional[Path] = None) -> GenerationResult:
        """
        Generate furniture image

        Args:
            prompt: Description of furniture (e.g., "modern wooden bookshelf with 5 shelves")
            style: Style preset (photorealistic, 3d-model, line-art)
            width: Image width
            height: Image height
            output_path: Where to save the image

        Returns:
            GenerationResult with image path/URL
        """
        if not self.api_key:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error="STABILITY_API_KEY not set"
            )

        # Enhance prompt for furniture
        enhanced_prompt = f"professional furniture design, {prompt}, studio lighting, white background, product photography"

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "image/*"
                },
                files={"none": ''},
                data={
                    "prompt": enhanced_prompt,
                    "output_format": "png",
                    "aspect_ratio": "1:1" if width == height else "16:9"
                }
            )

            if response.status_code == 200:
                if output_path is None:
                    output_path = PROJECT_ROOT / "output/images" / f"furniture_{hash(prompt)}.png"
                output_path.parent.mkdir(exist_ok=True)

                with open(output_path, 'wb') as f:
                    f.write(response.content)

                return GenerationResult(
                    success=True,
                    output_path=output_path,
                    output_url=None,
                    metadata={"prompt": enhanced_prompt, "model": "sd3"}
                )
            else:
                return GenerationResult(
                    success=False, output_path=None, output_url=None,
                    metadata={}, error=f"API error: {response.status_code}"
                )
        except Exception as e:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error=str(e)
            )


class OpenAIImageGenerator(FurnitureGenerator):
    """OpenAI DALL-E 3 for furniture images"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')

    def generate(self, prompt: str, size: str = "1024x1024",
                 quality: str = "standard", style: str = "natural") -> GenerationResult:
        """
        Generate furniture image with DALL-E 3

        Args:
            prompt: Furniture description
            size: "1024x1024", "1024x1792", or "1792x1024"
            quality: "standard" or "hd"
            style: "natural" or "vivid"
        """
        if not self.api_key:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error="OPENAI_API_KEY not set"
            )

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            enhanced_prompt = f"Professional furniture product photo: {prompt}. Clean white background, studio lighting, high detail."

            response = client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size=size,
                quality=quality,
                style=style,
                n=1
            )

            image_url = response.data[0].url

            return GenerationResult(
                success=True,
                output_path=None,
                output_url=image_url,
                metadata={
                    "prompt": enhanced_prompt,
                    "revised_prompt": response.data[0].revised_prompt,
                    "model": "dall-e-3"
                }
            )
        except Exception as e:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error=str(e)
            )


# =============================================================================
# 3D MODEL GENERATION
# =============================================================================

class MeshyAIGenerator(FurnitureGenerator):
    """Meshy AI for 3D model generation"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('MESHY_API_KEY')
        self.api_url = "https://api.meshy.ai/v2/text-to-3d"

    def generate(self, prompt: str, art_style: str = "realistic",
                 output_format: str = "glb") -> GenerationResult:
        """
        Generate 3D furniture model

        Args:
            prompt: Furniture description
            art_style: "realistic", "cartoon", "low-poly", "sculpture"
            output_format: "glb", "fbx", "obj", "stl"
        """
        if not self.api_key:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error="MESHY_API_KEY not set"
            )

        try:
            # Create task
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "mode": "preview",
                    "prompt": f"furniture: {prompt}",
                    "art_style": art_style,
                    "negative_prompt": "low quality, blurry"
                }
            )

            if response.status_code == 202:
                task_id = response.json().get("result")
                return GenerationResult(
                    success=True,
                    output_path=None,
                    output_url=f"https://api.meshy.ai/v2/text-to-3d/{task_id}",
                    metadata={
                        "task_id": task_id,
                        "status": "processing",
                        "note": "Poll the URL to check completion"
                    }
                )
            else:
                return GenerationResult(
                    success=False, output_path=None, output_url=None,
                    metadata={}, error=f"API error: {response.text}"
                )
        except Exception as e:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error=str(e)
            )


class TripoAIGenerator(FurnitureGenerator):
    """Tripo AI for fast 3D model generation"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('TRIPO_API_KEY')
        self.api_url = "https://api.tripo3d.ai/v2/openapi/task"

    def generate(self, prompt: str, model_version: str = "v2.0-20240919") -> GenerationResult:
        """
        Generate 3D model with Tripo AI (10 second generation)

        Args:
            prompt: Furniture description
            model_version: Model version to use
        """
        if not self.api_key:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error="TRIPO_API_KEY not set"
            )

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "type": "text_to_model",
                    "prompt": f"3D furniture model: {prompt}",
                    "model_version": model_version
                }
            )

            if response.status_code == 200:
                data = response.json()
                return GenerationResult(
                    success=True,
                    output_path=None,
                    output_url=None,
                    metadata={
                        "task_id": data.get("data", {}).get("task_id"),
                        "status": "processing"
                    }
                )
            else:
                return GenerationResult(
                    success=False, output_path=None, output_url=None,
                    metadata={}, error=f"API error: {response.text}"
                )
        except Exception as e:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error=str(e)
            )


# =============================================================================
# CAD/DRAWING GENERATION
# =============================================================================

class ZooTextToCAD(FurnitureGenerator):
    """Zoo.dev Text-to-CAD for manufacturing-ready models"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('ZOO_API_KEY')
        self.api_url = "https://api.zoo.dev/ai/text-to-cad"

    def generate(self, prompt: str, output_format: str = "step") -> GenerationResult:
        """
        Generate CAD model from text

        Args:
            prompt: Furniture specification (e.g., "bookshelf 800mm wide, 300mm deep, 1800mm tall with 5 shelves")
            output_format: "step", "stl", "obj", "gltf"
        """
        if not self.api_key:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error="ZOO_API_KEY not set"
            )

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "prompt": prompt,
                    "output_format": output_format
                }
            )

            if response.status_code == 201:
                data = response.json()
                return GenerationResult(
                    success=True,
                    output_path=None,
                    output_url=data.get("outputs", {}).get(output_format),
                    metadata={
                        "id": data.get("id"),
                        "format": output_format
                    }
                )
            else:
                return GenerationResult(
                    success=False, output_path=None, output_url=None,
                    metadata={}, error=f"API error: {response.text}"
                )
        except Exception as e:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error=str(e)
            )


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

class FurnitureAIStudio:
    """
    Unified interface for all furniture generation AI tools

    Usage:
        studio = FurnitureAIStudio()

        # Generate image
        result = studio.generate_image("modern oak bookshelf with 5 shelves")

        # Generate 3D model
        result = studio.generate_3d("wooden dining table 180cm x 90cm")

        # Generate CAD
        result = studio.generate_cad("shelf board 600mm x 300mm x 18mm thick")
    """

    def __init__(self):
        self.image_generators = {
            'stability': StabilityAIGenerator(),
            'openai': OpenAIImageGenerator(),
        }
        self.model_generators = {
            'meshy': MeshyAIGenerator(),
            'tripo': TripoAIGenerator(),
        }
        self.cad_generators = {
            'zoo': ZooTextToCAD(),
        }

    def generate_image(self, prompt: str, provider: str = "stability", **kwargs) -> GenerationResult:
        """Generate furniture image"""
        generator = self.image_generators.get(provider)
        if not generator:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error=f"Unknown provider: {provider}"
            )
        return generator.generate(prompt, **kwargs)

    def generate_3d(self, prompt: str, provider: str = "meshy", **kwargs) -> GenerationResult:
        """Generate 3D furniture model"""
        generator = self.model_generators.get(provider)
        if not generator:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error=f"Unknown provider: {provider}"
            )
        return generator.generate(prompt, **kwargs)

    def generate_cad(self, prompt: str, provider: str = "zoo", **kwargs) -> GenerationResult:
        """Generate CAD model"""
        generator = self.cad_generators.get(provider)
        if not generator:
            return GenerationResult(
                success=False, output_path=None, output_url=None,
                metadata={}, error=f"Unknown provider: {provider}"
            )
        return generator.generate(prompt, **kwargs)

    def generate_from_specs(self, specs: Dict) -> Dict[str, GenerationResult]:
        """
        Generate all assets from furniture specifications

        Args:
            specs: {
                'name': 'Bookshelf',
                'width_mm': 800,
                'depth_mm': 300,
                'height_mm': 1800,
                'material': 'oak',
                'shelves': 5
            }

        Returns:
            Dict with image, 3d, and cad results
        """
        # Build prompts from specs
        base_desc = f"{specs.get('material', 'wooden')} {specs.get('name', 'furniture')}"
        dims = f"{specs.get('width_mm', 600)}mm wide, {specs.get('depth_mm', 300)}mm deep, {specs.get('height_mm', 1000)}mm tall"

        image_prompt = f"modern {base_desc}, {dims}, professional product photo"
        model_prompt = f"{base_desc}, {dims}"
        cad_prompt = f"{specs.get('name', 'furniture')} {dims}"
        if specs.get('shelves'):
            cad_prompt += f" with {specs['shelves']} shelves"

        return {
            'image': self.generate_image(image_prompt),
            '3d': self.generate_3d(model_prompt),
            'cad': self.generate_cad(cad_prompt)
        }


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

FURNITURE_PROMPTS = {
    'bookshelf': {
        'image': "modern {material} bookshelf with {shelves} shelves, {width}mm wide, clean lines, Scandinavian design",
        '3d': "bookshelf, {material}, {shelves} horizontal shelves, {width}x{depth}x{height}mm",
        'cad': "bookshelf {width}mm wide {depth}mm deep {height}mm tall with {shelves} evenly spaced shelves, {thickness}mm thick boards"
    },
    'desk': {
        'image': "minimalist {material} desk, {width}mm wide, {height}mm tall, clean workspace design",
        '3d': "desk, {material}, rectangular top {width}x{depth}mm, four legs",
        'cad': "desk {width}mm wide {depth}mm deep {height}mm tall, top thickness {thickness}mm"
    },
    'cabinet': {
        'image': "storage cabinet, {material}, {doors} doors, {width}mm wide, modern design",
        '3d': "cabinet with {doors} doors, {material}, {width}x{depth}x{height}mm",
        'cad': "cabinet {width}mm wide {depth}mm deep {height}mm tall with {doors} doors"
    },
    'table': {
        'image': "dining table, {material}, {width}mm x {depth}mm top, {legs} legs, elegant design",
        '3d': "table, {material} top, {legs} legs, {width}x{depth}x{height}mm",
        'cad': "table top {width}mm x {depth}mm, height {height}mm, top thickness {thickness}mm"
    }
}


def build_prompt(furniture_type: str, prompt_type: str, **kwargs) -> str:
    """Build prompt from template"""
    template = FURNITURE_PROMPTS.get(furniture_type, {}).get(prompt_type)
    if not template:
        return f"{furniture_type} furniture"

    # Set defaults
    defaults = {
        'material': 'wood',
        'width': 800,
        'depth': 400,
        'height': 1000,
        'thickness': 18,
        'shelves': 4,
        'doors': 2,
        'legs': 4
    }
    defaults.update(kwargs)

    return template.format(**defaults)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Furniture AI Generation Tools')
    parser.add_argument('--type', choices=['image', '3d', 'cad'], default='image',
                        help='Generation type')
    parser.add_argument('--prompt', '-p', help='Generation prompt')
    parser.add_argument('--provider', help='API provider')
    parser.add_argument('--list-providers', action='store_true', help='List available providers')
    args = parser.parse_args()

    if args.list_providers:
        print("Image Providers: stability, openai")
        print("3D Providers: meshy, tripo")
        print("CAD Providers: zoo")
        print("\nRequired Environment Variables:")
        print("  STABILITY_API_KEY - Stability AI")
        print("  OPENAI_API_KEY - OpenAI DALL-E")
        print("  MESHY_API_KEY - Meshy AI")
        print("  TRIPO_API_KEY - Tripo AI")
        print("  ZOO_API_KEY - Zoo.dev")
        return

    studio = FurnitureAIStudio()

    if args.type == 'image':
        result = studio.generate_image(args.prompt, provider=args.provider or 'stability')
    elif args.type == '3d':
        result = studio.generate_3d(args.prompt, provider=args.provider or 'meshy')
    elif args.type == 'cad':
        result = studio.generate_cad(args.prompt, provider=args.provider or 'zoo')

    print(f"Success: {result.success}")
    if result.success:
        if result.output_path:
            print(f"Output: {result.output_path}")
        if result.output_url:
            print(f"URL: {result.output_url}")
        print(f"Metadata: {json.dumps(result.metadata, indent=2)}")
    else:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    main()
