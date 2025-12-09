# LLM Self-Refine Eval

This project provides a comprehensive framework to evaluate the self-refinement capabilities of Large Language Models (LLMs). The core experiment tests whether an LLM can identify its own reasoning errors and generate effective, corrective hints to improve its performance on a second attempt.

## 🧪 The Evaluation Workflow

The pipeline is executed in three automated stages for each model and dataset pair:

1. **Initial Inference:**  
   The model is prompted with a set of questions from a given dataset and attempts to solve them.

2. **Hint Generation:**  
   For each incorrectly answered question, the model is shown its own flawed reasoning and the correct answer. It is then prompted to generate a "hint sentence" that would have guided it to the correct solution without explicitly revealing the answer.

3. **Post-Hint Inference:**  
   The model is presented with the same questions it initially failed, but this time, the self-generated hint is prepended to the prompt to measure if the hint leads to a correct answer.


## 🚀 How to Use It?

### 1. Setup

First, clone the repository and set up the Python environment:

```bash
git clone https://github.com/your-username/llm-self-refine-eval.git
cd llm-self-refine-eval
pip install -r requirements.txt
python load_models.py
```

### 2. Running the Included Experiments

The project comes pre-packaged with several datasets (in `data/`) and their corresponding prompt templates (in `prompts/`).

You can run the full evaluation pipeline using the main script `src/run.py`. For example:

```bash
python src/run.py \
  --model_path "Qwen/Qwen2.5-Math-1.5B" \
  --dataset "gsm8k" \
  --input_path "data/gsm8k.jsonl" \
  --output_dir "results/Qwen-Math-1.5B/gsm8k/max512" \
  --max_tokens 1024
```

To simplify running multiple experiments, you can use the provided shell scripts located in the `scripts/` directory
