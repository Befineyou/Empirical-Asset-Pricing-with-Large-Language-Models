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

- Python 3.8+
- Transformers (e.g., HuggingFace)
- scikit-learn
- pandas, numpy
- tqdm

Install dependencies:
```bash
pip install -r requirements.txt
