def build_lineart_prompt(subject):
    return f"monochrome lineart, clean lines, transparent background, {subject}"


def build_flat_color_prompt(subject):
    return f"flat color, flat shading, base colors only, no lighting, {subject}"


def build_shading_prompt(subject):
    return f"dramatic lighting, strong rim light, deep shadows, grayscale, {subject}"
