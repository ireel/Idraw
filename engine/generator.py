from pathlib import Path
from random import Random


class LayeredGenerator:
    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = Random(seed)

    def generate_layers(self, prompts, output_dir, dry_run=False):
        output_dir = Path(output_dir)
        outputs = {
            "lineart": str(output_dir / "01_lineart.png"),
            "flat_color": str(output_dir / "02_flat_color.png"),
            "shading": str(output_dir / "03_shading_light.png"),
            "final": str(output_dir / "04_final_composite.png"),
        }

        if dry_run:
            return outputs

        self._render_placeholder(outputs["lineart"], prompts["lineart"], (240, 240, 240, 0))
        self._render_placeholder(outputs["flat_color"], prompts["flat_color"], (120, 160, 200, 255))
        self._render_placeholder(outputs["shading"], prompts["shading"], (40, 40, 40, 180))
        return outputs

    def _render_placeholder(self, path, text, color):
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:
            raise RuntimeError("pillow 依赖未安装") from exc

        image = Image.new("RGBA", (self.width, self.height), color)
        draw = ImageDraw.Draw(image)
        draw.text((32, 32), text, fill=(255, 255, 255, 255))
        image.save(path)
