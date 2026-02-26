class LLMClient:
    def __init__(self, api_key, model, base_url="https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def expand_prompt(self, user_prompt):
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai 依赖未安装") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "请将用户输入扩写为更细致的插画描述，避免添加画质关键词"},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
