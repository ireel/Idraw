DEFAULT_NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
    "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
    "username, blurry, artist name, bad feet, distorted, ugly"
)

def build_lineart_prompt(subject):
    return (
        f"masterpiece, best quality, monochrome, lineart, clean lines, white background, "
        f"simple background, {subject}"
    )

def build_flat_color_prompt(subject):
    return (
        f"masterpiece, best quality, flat color, flat shading, base colors only, no lighting, "
        f"no shadow, simple background, {subject}"
    )

def build_shading_prompt(subject):
    return (
        f"masterpiece, best quality, dramatic lighting, strong rim light, deep shadows, "
        f"cinematic lighting, volumatic lighting, {subject}"
    )
