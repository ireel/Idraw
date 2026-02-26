import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from agent.llm_client import LLMClient
from agent.prompts import build_flat_color_prompt, build_lineart_prompt, build_shading_prompt
from engine.compositor import composite_layers
from engine.generator import LayeredGenerator
from engine.tools import create_session_dir, write_json


def build_prompts(user_prompt, llm_client):
    refined_prompt = user_prompt
    if llm_client is not None:
        refined_prompt = llm_client.expand_prompt(user_prompt)
    return {
        "lineart": build_lineart_prompt(refined_prompt),
        "flat_color": build_flat_color_prompt(refined_prompt),
        "shading": build_shading_prompt(refined_prompt),
        "refined": refined_prompt,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="描述要生成的插画内容")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--model", default="anthropic/claude-3.5-sonnet")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    session_dir = create_session_dir(base_output_dir, datetime.utcnow())
    session_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENROUTER_API_KEY")
    llm_client = None
    if api_key:
        llm_client = LLMClient(api_key=api_key, model=args.model)

    prompts = build_prompts(args.prompt, llm_client)
    write_json(session_dir / "session.json", prompts)

    generator = LayeredGenerator(
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    outputs = generator.generate_layers(
        prompts=prompts,
        output_dir=session_dir,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        composite_layers(
            lineart_path=outputs["lineart"],
            flat_color_path=outputs["flat_color"],
            shading_path=outputs["shading"],
            output_path=outputs["final"],
        )

    write_json(session_dir / "manifest.json", outputs)
    print(json.dumps({"session_dir": str(session_dir), "outputs": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
