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
  --max_tokens 512
```

To simplify running multiple experiments, you can use the provided shell scripts located in the `scripts/` directory:

- `scripts/run_sample_models.sh`: A quick script to test the pipeline on a small subset of models and datasets.  
- `scripts/run_all_models.sh`: A comprehensive script to run the full evaluation across all predefined models and datasets.


### 3. Analyzing the Results

After running the pipeline, all raw outputs will be saved as `.jsonl` files in the `results/` directory.  

The `src/analysis.py` script aggregates these results into a clean, human-readable summary.

#### How to Run the Analysis

Execute the script from the project's root directory. The `--results_root` argument specifies which folder of results to analyze.

**Basic Usage:**  
This command analyzes everything inside the `results/` directory and saves the output to default file locations (`results/summary.txt` and `results/summary.csv`):

```bash
python src/analysis.py --results_root results/
```

**Advanced Usage:**
You can also specify custom output paths and analyze a specific subset of results.

```bash
python src/analysis.py \
  --results_root results/sample_results \
  --out_txt results/sample_results/statistics/summary.txt \
  --out_csv results/sample_results/statistics/summary.csv
```

This will generate two summary files in the specified locations:

- `summary.txt`: A text file with easy-to-read accuracy metrics.  
- `summary.csv`: A CSV file with detailed metrics for further data analysis and plotting.


## 🛠️ Project Structure
```
.
├── data/                  # Input datasets
├── prompts/               # Prompt templates
├── results/               # Output directory for all experiment results
├── scripts/               # Shell scripts to automate running experiments
├── src/
│   ├── datasets/          # Python modules for processing each dataset
│   ├── __init__.py
│   ├── analysis.py        # Script to aggregate and summarize results
│   ├── inference.py       # Core logic for model inference and hint generation
│   ├── run.py             # Main pipeline execution script
│   └── utils.py           # Helper functions for I/O, parsing, etc.
└── requirements.txt       # Project dependencies
```
