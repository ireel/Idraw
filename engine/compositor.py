from PIL import Image, ImageOps

def process_lineart(image):
    """
    Convert white background of lineart to transparent.
    Assumes lineart is black lines on white background.
    """
    image = image.convert("RGBA")
    # Get the alpha channel based on grayscale value (white -> transparent, black -> opaque)
    grayscale = image.convert("L")
    # Invert grayscale: White(255) becomes 0 (transparent), Black(0) becomes 255 (opaque)
    alpha = ImageOps.invert(grayscale)
    
    # Create a new image with black lines and the calculated alpha
    black_lines = Image.new("RGBA", image.size, (0, 0, 0, 255))
    black_lines.putalpha(alpha)
    
    return black_lines

def composite_layers(lineart_path, flat_color_path, shading_path, output_path):
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("pillow 依赖未安装") from exc

    # Load images
    flat = Image.open(flat_color_path).convert("RGBA")
    shading = Image.open(shading_path).convert("RGBA")
    lineart = Image.open(lineart_path).convert("RGBA")

    # Since shading is generated via Img2Img from flat color, it is a full opaque image.
    # We will use shading as the base.
    base = shading

    # Process lineart to make it transparent
    lineart_transparent = process_lineart(lineart)

    # Composite: Base + Lineart
    # Using alpha_composite (Source Over)
    combined = Image.alpha_composite(base, lineart_transparent)

    combined.save(output_path)
