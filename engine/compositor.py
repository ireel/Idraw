def composite_layers(lineart_path, flat_color_path, shading_path, output_path):
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("pillow 依赖未安装") from exc

    flat = Image.open(flat_color_path).convert("RGBA")
    shading = Image.open(shading_path).convert("RGBA")
    lineart = Image.open(lineart_path).convert("RGBA")

    combined = Image.alpha_composite(flat, shading)
    combined = Image.alpha_composite(combined, lineart)
    combined.save(output_path)
