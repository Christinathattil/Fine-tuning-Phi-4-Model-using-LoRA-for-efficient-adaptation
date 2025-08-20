# Fine-tuning-Phi-4-Model-using-LoRA-for-efficient-adaptation
Efficiently fine-tune the Phi-4 language model using LoRA adapters. Train lightweight modules on your dataset to customize the model with minimal compute and storage. Includes loading, training, saving, and pushing adapters to Hugging Face Hub for easy sharing and deployment.

# Fine-tuned Microsoft Phi-4 Model with LoRA Adapters

This repository provides a **domain-adapted version of Microsoft Phi-4**, fine-tuned efficiently using Low-Rank Adaptation (LoRA) adapters. Fine-tuning was performed on a specialized dataset in a resource-efficient manner, leveraging only the adapted parameters and tokenizer files rather than the full model weights.

---

## Model Overview

- **Base Model:** [Microsoft Phi-4](https://huggingface.co/microsoft/phi-4), a state-of-the-art large language model (LLM) for natural language understanding and generation.
- **Fine-tuning Technique:** Low-Rank Adaptation (LoRA) for parameter-efficient training.
- **Dataset:** [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k), formatted in ShareGPT multi-turn chat style.
- **Training Environment:** Kaggle Notebooks (local GPUs), utilizing [Unsloth](https://github.com/unslothai/unsloth), Hugging Face Transformers & Datasets.
- **Artifacts Provided:** Only LoRA adapter weights and tokenizer (full base model weights are excluded to save storage).

---

## Why Choose LoRA?

- **Efficiency:** Updates only a tiny, targeted subset of model parameters, reducing memory and storage requirements.
- **Faster Fine-tuning:** Enables rapid iteration and training of LLMs on consumer-grade hardware.
- **Modularity:** LoRA adapters can be layered onto the base Phi-4 model as needed, avoiding redundant downloads of the full model.

---

## Usage Instructions

### Requirements

- Python 3.8 or higher
- PyTorch (with CUDA for GPU acceleration recommended)
- [Unsloth library](https://github.com/unslothai/unsloth)
- Hugging Face `transformers` & `datasets` packages

### Loading and Running the Fine-tuned Adapter

```python
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Define the path or Hugging Face repo for LoRA adapters + tokenizer
model_name = "christina444/phi-4-finetuned"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load base model with LoRA adapters (4-bit quantized for efficiency)
model = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True  # For memory-efficient inference
)
model = FastLanguageModel.get_peft_model(model)  # Apply the LoRA adapters
model.eval()

# Example: Prepare input and generate a response
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="phi-4")
messages = [
    {"role": "user", "content": "Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,"}
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=64,
    temperature=1.5,
    min_p=0.1
)
print(tokenizer.batch_decode(outputs))
```

## **Model Card & Applications**

This fine-tuned Phi-4 LoRA model inherits general-purpose abilities from the Phi-4 base while gaining specialized knowledge from the FineTome-100k dataset. As such, it is ideal for:

- **Conversational AI and chatbots**
- **Customer support automation**
- **Technical content generation**
- **Task-specific dialogue systems**

---

## **Dataset Details**

- **Source:** [mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)
- **Format:** Multi-turn conversations (ShareGPT style)
- **Preprocessing:** Standardized and harmonized using Unsloth tooling for optimal Phi-4 compatibility

---

## **Citations & Acknowledgments**

- **Base Model:** [Microsoft Phi-4](https://huggingface.co/microsoft/phi-4)
- **Parameter-efficient Fine-tuning:** [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- **Dataset:** [mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)
- **Fine-tuning Tutorial:** [Analytics Vidhya Guide](https://www.analyticsvidhya.com/blog/2025/01/fine-tune-phi-4-locally/)

---

## **Further Reading & Key References**

To deepen your understanding of the underlying methods and context for this work, consider reviewing the following influential publications:

- **LoRA: Low-Rank Adaptation of Large Language Models**  
  *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, et al.*  
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)  
  *This foundational paper introduces the LoRA technique for efficient fine-tuning of large neural networks.*

- **Phi-2 & Phi-3: Language Models by Microsoft**  
  *Microsoft Research*  
  [Phi-2 Model Card](https://huggingface.co/microsoft/phi-2)  
  [Phi-3 Model Card](https://huggingface.co/microsoft/phi-3-mini-4k-instruct)  
  *Background on the Phi series language models that paved the way for Phi-4.*

- **Parameter-Efficient Transfer Learning with Adaptors**  
  *Houlsby et al., 2019*  
  [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)  
  *Examines various approaches to parameter-efficient transfer learning, including the use of adapter modules.*

- **The ShareGPT Dataset**  
  [ShareGPT Paper](https://arxiv.org/abs/2303.17580)  
  *Describes the dataset format used for multi-turn conversation, inspiring FineTome-100k.*

---

## **Notes**

- This repository only includes the **LoRA adapter weights** and **tokenizer** due to storage constraints (as on Kaggle Notebooks).
- To use the fine-tuned model, first load or download the **base Microsoft Phi-4 model**.
- For step-by-step fine-tuning instructions, see the [Analytics Vidhya tutorial](https://www.analyticsvidhya.com/blog/2025/01/fine-tune-phi-4-locally/).
