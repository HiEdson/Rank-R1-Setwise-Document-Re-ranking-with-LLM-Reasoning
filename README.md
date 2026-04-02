# Rank-R1: Setwise Document Re-ranking with LLM Reasoning

This repository provides a modular, high-performance implementation of **Setwise Document Re-ranking** using Large Language Models. It specifically supports **Rank-R1**, a reasoning-based re-ranker that leverages model internal thoughts (like DeepSeek-R1) to achieve state-of-the-art results.

## 🚀 Features

- **Setwise Ranking:** Efficiently re-rank documents using sorting algorithms (Heapsort/Bubblesort) where the LLM acts as the judge.
- **Rank-R1 Reasoning:** Support for reasoning models that use `<think>` and `<answer>` tags for better relevance decisions.
- **Local Data Support:** Run experiments on your own JSON datasets without needing large IR-benchmark downloads.
- **Multi-Backend:** Support for Hugging Face (T5, Llama), vLLM (high-throughput), and OpenAI APIs.

---

## 🛠️ Installation

### 1. Set up Environment
We recommend using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 🧪 Quick Start: Simulated Re-ranking
You can try the re-ranker immediately using a small simulated dataset. This verifies the environment and the core logic without downloading 100GB of data.

1. **Generate Mock Data:**
   ```bash
   python generate_simulated_data.py
   ```
2. **Run Re-ranker (Example with Flan-T5-Small):**
   ```bash
   python run.py run \
     --model_name_or_path google/flan-t5-small \
     --run_path sample_data/bm25_simulated.run \
     --save_path sample_data/reranked.txt \
     --queries_path sample_data/queries.json \
     --docs_path sample_data/docs.json \
     --hits 5 \
     --device cpu \
     setwise --num_child 2 --k 5
   ```

---

## 📖 Advanced Usage (Full Datasets)

For large-scale experiments (e.g., TREC DL 2019 or BEIR), use `pyserini` or `ir-datasets`:

```bash
python run.py run \
  --model_name_or_path google/flan-t5-large \
  --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
  --save_path run.setwise.results.txt \
  --ir_dataset_name msmarco-passage/trec-dl-2019 \
  --hits 100 \
  --device cuda \
  setwise --num_child 2 --method heapsort --k 10
```

### Key Arguments:
- `--run_path`: Path to your first-stage (BM25) run file.
- `--model_name_or_path`: Hugging Face model ID (e.g., `meta-llama/Llama-3-8B-Instruct`).
- `--prompt_file`: Required for Rank-R1 models (e.g., `prompt_setwise-R1.toml`).
- `setwise --num_child`: Number of documents to compare in one prompt (affects efficiency/VRAM).

---

## 🏗️ Project Structure

- `rank_r1_core/`: The core logic for Setwise, Pairwise, and R1 re-ranking.
- `run.py`: Main entry point for running experiments.
- `generate_simulated_data.py`: Utility to create small-scale test data.
- `requirements.txt`: Project dependencies.

---

## 🫡 References
This project is based on the research from:
- [1] Zhuang et al. **A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models**, SIGIR 2024.
- [2] Zhuang et al. **Rank-R1: Document Ranking with Large Language Models Reasoning**, 2025.

### Citation
```bibtex
@inproceedings{zhuang2024setwise,
    author={Zhuang, Shengyao and Zhuang, Honglei and Koopman, Bevan and Zuccon, Guido},
    title={A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models},
    booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    year = {2024},
    series = {SIGIR '24}
}
```

---
*Developed as part of the Master in AI project at QMUL.*
