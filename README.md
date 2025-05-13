# Empirical Asset Pricing with Large Language Models

This repository contains the code and data for our paper:  
**"Empirical Asset Pricing and Return Prediction Based on Large Language Models"**

We propose a novel approach to asset pricing by leveraging Large Language Models (LLMs), transforming structured tabular financial data into textual inputs, and using enhanced prompt engineering to achieve superior performance in return prediction tasks.

## 🔍 Overview

Asset pricing is a challenging domain due to its noisy, high-dimensional data. In this work, we are the first to apply LLMs to this field by:
- Converting tabular asset pricing datasets into textual form.
- Designing prompt templates to guide LLMs effectively.
- Evaluating the method across multiple benchmarks.

Our method significantly outperforms traditional machine learning baselines and shows strong **zero-shot** generalization capabilities.

## 📊 Key Features

- 💡 **Textual Transformation of Financial Data**: Converts structured asset pricing inputs into LLM-friendly text.
- 🧠 **Prompt-based Learning**: Uses carefully designed prompts for supervised and zero-shot inference.
- 📈 **Superior Performance**: Outperforms classic ML models (e.g., XGBoost, RF) in return prediction.
- 🧪 **Zero-shot Capability**: LLMs generalize well even without fine-tuning.

## 🧬 Requirements

- Python 3.11.9
- Transformers (e.g., HuggingFace)
- scikit-learn
- pandas, numpy
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

To execute the main program, simply run the following command:

```bash
python Bert.py
```

This will launch the core pipeline for asset return prediction using large language models.

Scripts with the suffix `ceshi` (e.g., `bert_ceshi.py`, `xgboost_ceshi.py`) are used for testing and evaluation purposes. These can be run individually to validate performance or conduct ablation experiments.

## 📂 Dataset

The dataset used in this project can be downloaded from the following websites:

- [China Stock Market & Accounting Research (CSMAR)](https://www.gtarsc.com/)
- [Wind Financial Terminal](https://www.wind.com.cn/)
- [RESSET Financial Database](https://www.resset.cn/)

Please ensure that the dataset is placed in the appropriate `./data` directory as required by the scripts. Make sure file names and structures match those expected in the code for successful execution.
