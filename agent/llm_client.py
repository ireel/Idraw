class LLMClient:
    def __init__(self, api_key, model, base_url="https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def expand_prompt(self, user_prompt):
        try:
            from openai import OpenAI
            import json
        except Exception as exc:
            raise RuntimeError("openai dependency not installed") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        system_prompt = (
            "You are an expert AI art prompt generator for Anime style illustrations. "
            "Your task is to convert the user's request into 3 distinct sets of English tags (Danbooru style) "
            "for a layered generation process: Lineart, Flat Color, and Shading.\n"
            "Return ONLY a valid JSON object with the following keys:\n"
            "- 'lineart_tags': Tags focusing on character details, pose, costume, line quality, monochrome, sketch.\n"
            "- 'flat_color_tags': Tags focusing on specific colors (e.g., blue hair, red dress), patterns, simple background.\n"
            "- 'shading_tags': Tags focusing on lighting direction, atmosphere, cinematic lighting, rim light, shadows.\n"
            "Do not include generic quality tags like 'best quality' as they are added automatically. "
            "Ensure the tags are comma-separated and descriptive."
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            # Fallback if JSON parsing fails or model doesn't support JSON mode
            print(f"Warning: LLM prompt expansion failed or format error: {e}. Using raw prompt.")
            # Return a fallback dict with the user prompt as base
            return {
                "lineart_tags": user_prompt,
                "flat_color_tags": user_prompt,
                "shading_tags": user_prompt
            }
