import gc
import torch
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline
)
from PIL import Image

class LayeredGenerator:
    def __init__(self, width, height, seed=None):
        # 针对 SD1.5/AnyLoRA 的分辨率优化
        # SD1.5 默认训练分辨率为 512x512，1024x1024 容易产生伪影或双头
        # 推荐使用 512x768 (Portrait) 或 768x512 (Landscape)
        # if width == 1024 and height == 1024:
        #     print("Warning: 1024x1024 resolution on SD1.5 may cause artifacts. Reducing to 512x768.")
        #     self.width = 512
        #     self.height = 768
        # else:
        self.width = width
        self.height = height
            
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Model paths (relative to project root)
        self.models_dir = Path("models")
        self.ckpt_path = self.models_dir / "checkpoints/anyloraCheckpoint_bakedvaeBlessedFp16.safetensors"
        self.lineart_lora_path = self.models_dir / "loras/animeoutlineV4_16.safetensors"
        # Revert to the SD1.5 compatible textual inversion embedding or a compatible LoRA
        # self.flat_lora_path = self.models_dir / "loras/MinimalistFlatColorXL_byKonan.safetensors" # Incompatible SDXL LoRA
        self.flat_lora_path = self.models_dir / "loras/flat_color.pt" 
        self.controlnet_lineart_path = self.models_dir / "controlnet/control_v11p_sd15s2_lineart_anime.pth"
        self.controlnet_depth_path = self.models_dir / "controlnet/control_v11f1p_sd15_depth.pth"

    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_layers(self, prompts, output_dir, dry_run=False):
        output_dir = Path(output_dir)
        paths = {
            "lineart": str(output_dir / "01_lineart.png"),
            "flat_color": str(output_dir / "02_flat_color.png"),
            "shading": str(output_dir / "03_shading_light.png"),
            "final": str(output_dir / "04_final_composite.png"),
        }

        if dry_run:
            return paths

        generator = torch.Generator(device="cpu").manual_seed(self.seed) if self.seed else None
        
        # --- Step 1: Lineart (Txt2Img + LoRA) ---
        print(f"Generating Lineart using {self.ckpt_path}...")
        try:
            pipe = StableDiffusionPipeline.from_single_file(
                str(self.ckpt_path), 
                config="./models/sd15_config",
                torch_dtype=self.dtype, 
                use_safetensors=True
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            
            # Load Lineart LoRA
            print(f"Loading LoRA: {self.lineart_lora_path}")
            pipe.load_lora_weights(str(self.lineart_lora_path))
            
            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)
            
            image_lineart = pipe(
                prompts["lineart"], 
                negative_prompt=prompts.get("negative"),
                width=self.width, 
                height=self.height, 
                num_inference_steps=25,
                generator=generator
            ).images[0]
            
            image_lineart.save(paths["lineart"])
            
        finally:
            if 'pipe' in locals():
                del pipe
            self._clear_memory()

        # --- Step 2: Flat Color (Txt2Img + ControlNet Lineart + LoRA) ---
        print(f"Generating Flat Color using ControlNet: {self.controlnet_lineart_path}...")
        try:
            controlnet = ControlNetModel.from_single_file(
                str(self.controlnet_lineart_path), 
                torch_dtype=self.dtype
            )
            
            pipe = StableDiffusionControlNetPipeline.from_single_file(
                str(self.ckpt_path), 
                config="./models/sd15_config",
                controlnet=controlnet, 
                torch_dtype=self.dtype, 
                use_safetensors=True
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            
            # Load Flat Color LoRA
            print(f"Loading LoRA: {self.flat_lora_path}")
            try:
                pipe.load_lora_weights(str(self.flat_lora_path))
            except Exception as e:
                print(f"Failed to load as LoRA ({e}), trying Textual Inversion...")
                pipe.load_textual_inversion(str(self.flat_lora_path), token="flat_color")
            
            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)
            
            image_flat = pipe(
                prompts["flat_color"],
                negative_prompt=prompts.get("negative"),
                image=image_lineart,
                width=self.width,
                height=self.height,
                num_inference_steps=25,
                controlnet_conditioning_scale=1.0,
                generator=generator
            ).images[0]
            
            image_flat.save(paths["flat_color"])
            
        finally:
            if 'pipe' in locals():
                del pipe
            if 'controlnet' in locals():
                del controlnet
            self._clear_memory()

        # --- Step 3: Shading (Img2Img + ControlNet Depth? + Prompt) ---
        print(f"Generating Shading...")
        try:
            # Attempt to use Depth ControlNet if we can estimate depth
            # For simplicity and robustness, we will use Lineart ControlNet again to maintain structure,
            # but use Img2Img on the Flat Color to add shading.
            # If we had a depth map, we would use Depth ControlNet.
            # Let's stick to Lineart ControlNet for structural consistency, 
            # as generating a depth map requires another model (Depth Anything/MiDaS).
            
            controlnet = ControlNetModel.from_single_file(
                str(self.controlnet_lineart_path), 
                torch_dtype=self.dtype
            )
            
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                str(self.ckpt_path),
                config="./models/sd15_config",
                controlnet=controlnet,
                torch_dtype=self.dtype,
                use_safetensors=True
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            
            # No LoRA for shading (or maybe a detailed LoRA? default to None/Base)
            # Or unload LoRA if we reused pipe (but we are reloading).
            
            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)
            
            # Use Flat Color as starting point (init_image)
            # Use Lineart as control (control_image)
            
            # Ensure flat color is RGB for img2img
            image_flat_rgb = image_flat.convert("RGB")
            
            image_shading = pipe(
                prompts["shading"],
                negative_prompt=prompts.get("negative"),
                image=image_flat_rgb, # init_image
                control_image=image_lineart, # control_image
                strength=0.6, # Reduced strength to preserve more of the flat color structure while adding shading
                num_inference_steps=30,
                controlnet_conditioning_scale=1.0,
                generator=generator
            ).images[0]
            
            image_shading.save(paths["shading"])
            
        finally:
            if 'pipe' in locals():
                del pipe
            if 'controlnet' in locals():
                del controlnet
            self._clear_memory()

        return paths
