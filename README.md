# Patch-as-Decodable-Token: Towards Unified Multi-Modal Vision Tasks in MLLMs

<font size=4><div align='center' > [[🤗 RIC Demo](https://huggingface.co)] [[🤗 RIC Data](https://huggingface.co)] [[🤗 Checkpoints](https://huggingface.co/collections/)] </div></font>

<font size=4><div align='center'>[[📄 Tech Report](https://arxiv.org)] [[📝 Blog]()]</div></font>

<div align="center">
<img src="./assets/TaskIntroduction.webp" width="900"/>
</div>


Multimodal large language models (MLLMs) have advanced rapidly in recent years. However, existing approaches for vision tasks often rely on indirect representations, such as generating coordinates as text for detection, which limits performance and prevents dense prediction tasks like segmentation. To overcome these challenges, we introduce Patch-as-Decodable Token (PaDT), a unified paradigm that enables MLLMs to directly generate both textual and diverse visual outputs. Central to PaDT are Visual Reference Tokens (VRTs), derived from visual patch embeddings of query images and interleaved seamlessly with LLM's output textual tokens. A lightweight decoder then transforms LLM's outputs into detection, segmentation, and grounding predictions. Unlike prior methods, PaDT processes VRTs independently at each forward pass and dynamically expands the embedding table, thus improving localization and differentiation among similar objects. We further tailor a training strategy for PaDT by randomly selecting VRTs for supervised fine-tuning and introducing a robust per-token cross-entropy loss. Our empirical studies across four visual perception and understanding tasks suggest PaDT consistently achieving state-of-the-art performance, even compared with significantly larger MLLM models. 

