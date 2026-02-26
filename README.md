# AI-Agent驱动的分层插画生成系统 (Layered-AI-Illustrator)

## 1. 项目概述
本项目旨在开发一个基于 LLM Agent 调度的自动化 AI 绘画工作流。系统将用户的自然语言输入转化为包含四个独立图层的专业插画生成流程。通过自动化提示词工程和动态参数控制，实现“端到端”的高质量赛璐璐风格（平涂）插画生成。

**核心输出物 (每轮任务 4 张图像):**
1. `01_lineart.png`: 纯净线稿（透明或纯白背景）。
2. `02_flat_color.png`: 纯底色填充（无光影，基于线稿约束）。
3. `03_shading_light.png`: 独立的阴影与高光通道（或灰度光影图）。
4. `04_final_composite.png`: 最终合成图（结合前三者并加入后期特效与背景）。

## 2. 系统架构设计

### 2.1 大脑层 (LLM Agent 控制器)
* **平台对接:** 通过 API Key 调用 OpenRouter 平台上的高性能大语言模型（如 Claude-3.5-Sonnet 或 GPT-4o）。
* **工具调用 (Tool Calling / MCP):** LLM 被赋予调用底层生图引擎的权限。LLM 负责：
    * 分析用户意图，扩写美术细节。
    * 为线稿、底色、光影阶段分别生成精确的 Stable Diffusion (SD) 提示词。
    * 动态决策各个生成阶段的参数（如 `CFG Scale`, `Denoising Strength`, `ControlNet Weight` 等）。

### 2.2 执行层 (图像生成引擎)
* **核心库:** 推荐使用 `diffusers` (Python) 编写底层生图脚本，或通过 API 调用本地无头运行的 `ComfyUI`。
* **模型管线 (Pipeline):**
    * **线稿:** SDXL + Lineart LoRA 或特定的动漫大模型。
    * **底色:** 接收上一步线稿作为 ControlNet (Lineart) 输入，配合 Flat Color LoRA 生成。
    * **光影:** 接收线稿和底色作为条件输入（如 ControlNet Depth/Normal），生成光影层。
* **合成模块:** 使用 `OpenCV` 或 `Pillow` (PIL) 将前三个图层按特定混合模式（正片叠底、滤色）进行代码级合成。

### 2.3 硬件优化策略 (针对 8GB VRAM / 16GB RAM)
* **显存管理:** 绝对禁止同时加载多个模型。必须实现**显存分时复用 (Sequential Offloading)**。
    * 阶段一：加载线稿模型 -> 生成线稿 -> 从 VRAM 卸载到 RAM 或清空。
    * 阶段二：加载底色模型 + ControlNet -> 生成底色 -> 卸载。
    * 阶段三：加载光影模型 -> 生成光影 -> 卸载。
* **技术栈配置:** 强制开启 `xformers` 内存高效注意力机制，并考虑使用 `FP8` 或 `FP16` 精度加载模型权重。

## 3. 核心工作流 (Agent 交互时序)

1.  **用户输入:** "画一个赛博朋克风格的武士少女，站在霓虹灯下"
2.  **Agent 拆解任务:** * Agent 生成全局 JSON 配置：包含全局分辨率、随机种子等。
3.  **Agent 执行阶段 1 (线稿):** * 生成 Prompt 1: `monochrome, lineart, clean lines, cyberpunk samurai girl, ...`
    * 调用生图函数 -> 生成并保存 `01_lineart.png`。
4.  **Agent 执行阶段 2 (底色):** * 生成 Prompt 2: `flat color, flat shading, base colors only, ...`
    * 自动配置 ControlNet 权重 (例如 0.8) 并输入 `01_lineart.png`。
    * 调用生图函数 -> 生成并保存 `02_flat_color.png`。
5.  **Agent 执行阶段 3 (光影):** * 生成 Prompt 3: `dramatic lighting, neon rim light, deep shadows, cinematic...`
    * 输入前两张图作为控制条件。
    * 调用生图函数 -> 生成并保存 `03_shading_light.png`。
6.  **代码级合成 (阶段 4):**
    * 图像处理脚本自动将三层对齐叠加，输出 `04_final_composite.png`。

## 4. 推荐目录结构
```text
Layered-AI-Illustrator/
│
├── agent/                 # LLM Agent 逻辑与提示词模板
│   ├── llm_client.py      # OpenRouter API 请求封装
│   └── prompts.py         # 针对线稿、底色、光影的 System Prompt
│
├── engine/                # 图像生成与处理引擎
│   ├── generator.py       # 基于 diffusers/ComfyUI 的生图封装 (含显存优化逻辑)
│   ├── tools.py           # 暴露给 Agent 调用的工具函数 (MCP 规范)
│   └── compositor.py      # 图像合成脚本 (OpenCV/PIL)
│
├── output/                # 生成结果输出目录
│   └── session_timestamp/ # 每次运行生成一个文件夹存放 4 张图
│
├── app.py                 # 主程序入口
├── requirements.txt       # Python 依赖清单
└── PROJECT_PLAN.md        # 本开发文档
```
## 5. 第一阶段开发里程碑
搭建 Python 虚拟环境，安装 openai (用于调用 OpenRouter)、diffusers、torch 等基础库。
编写 llm_client.py，跑通 OpenRouter 的基础请求和 Function Calling (结构化 JSON 输出)。
编写一段测试脚本，验证本地 GPU 环境下的显存动态加载和卸载 (Offloading) 是否正常工作，避免 OOM。