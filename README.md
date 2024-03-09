# Evaluating Task-Specific Optimization Methods in Large Language Models: A Comparison

This repository contains all the datasets and experimental details for the thesis titled "Evaluating Task Specific Optimization Methods in Large Language Models: A Comparison." This work focuses on exploring and comparing various optimization methods applied to large language models.

## Folders Overview

### Datasets

#### 1. Medical Abstracts Dataset [1]
Located in the `datasets` folder, this dataset contains medical abstracts detailing patient health conditions, focusing on five conditions: digestive system diseases, cardiovascular diseases, neoplasms, nervous system diseases, and general pathological conditions. Each record includes clinical descriptions and respective labels. The dataset is split into:
- **Training Data**: 14,438 records with labeled classes.
- **Test Data**: 14,442 records for class prediction.

#### 2. ArXiv Papers Dataset [2]
This dataset comprises over 42,000 research paper abstracts from fields like Machine Learning, Computational Linguistics, Name Entity Recognition, AI, and Computer Vision, published between 1992 and Feb 2018. The data is in JSON format with fields such as date, download link, summary, title, authors, and tags.

### Experiments [3]

In the `experiments` folder, I describe the methodologies and results of experiments conducted for the thesis. Key aspects include:

- **Optimization Methods**: Application of various optimization methods to large language models, specifically:
  - Zero-Shot Learning
  - Few-Shot Learning
  - Chain of Thoughts
  - Retrieval Augmented Generation
  - Parameter Fine-Tuning with Adapter Tuning (LoRA)
- **Models Used**: [4]
  - llama-2-7b-chat
  - mistral-7b-instruct
  Both are fine-tuned models for chat, each with 7 billion parameters.
- **Experiments Objective**: Testing the optimization methods on text generation and classification tasks using both models.
- **Tools Used**: Experiments were conducted using the Hugging Face Hub and its libraries.

## Thesis Overview

The thesis aims to evaluate and compare state-of-the-art task-specific optimization methods in large language models, such as llama-2-7b-chat and mistral-7b-instruct. It focuses on their effectiveness in resource-constrained environments and their ability to provide insightful results. The primary goal is to determine which optimization methods are more effective and to contribute to the advancement of language model development. A key aspect of the research is exploring how to minimally transform the workflow with generative models to accurately perform a variety of NLP tasks.

---

*Note: This README provides a brief overview of the repository contents. For more detailed information, please refer to the individual files within each folder.*


## References:

[1] T. Schopf, D. Braun, and F. Matthes, "Evaluating Unsupervised Text Classification: Zero-Shot and Similarity-Based Approaches," in Proc. of the 2022 6th International Conference on Natural Language Processing and Information Retrieval (NLPIR '22), Bangkok, Thailand, 2023, pp. 6-15, doi: [10.1145/3582768.3582795](https://doi.org/10.1145/3582768.3582795).

You can find the associated code and data used in this project in the following GitHub repository: [GitHub Repository](https://github.com/sebischair/medical-abstracts-tc-corpus)

[2] The second dataset is sourced from arXiv, an open-source library for research papers. Gratitude to arXiv for facilitating access to valuable knowledge.

Additionally, the dataset can be found on Kaggle at the following link: [arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/neelshah18/arxivdataset).

[3] The following works were used as references in conducting PEFT and RAG:

1. Google Colab Notebook: [Fine-tuning Llama 2.7b on a custom dataset for Material recommendation and optimization for road construction](https://colab.research.google.com/github/AdityaShirke8005/Fine_tuning_Llama_2_7b-Material_recommendation_for_road_construction_on_custom_dataset/blob/main/Fine_tuning_Llama_2_7b_on_a_custom_dataset_for_Material_recommendation_and_optimization_for_road_construction.ipynb).

2. GitHub Repository: [RAG Therapist Notebook](https://github.com/cAPRIcaT3/RAG_therapist/blob/main/RAGModel_therapist.ipynb).

3. Medium Article: [Customized Evaluation Metrics with Hugging Face Trainer](https://medium.com/@rakeshrajpurohit/customized-evaluation-metrics-with-hugging-face-trainer-3ff00d936f99) by Rakesh Rajpurohit.

[4] 
This study will make use of the fine-tuned chat version of the LLaMA-7B model, as provided by Facebook (Meta) Research. For more details on the models family and implementation, visit the [LLaMA GitHub repository](https://github.com/facebookresearch/llama).

Specifically, the Llama-2-7b-chat-hf model, hosted on Hugging Face, is utilized in this study. For additional details, visit:[Llama-2-7b-chat-hf on Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

The Mistral-7B-Instruct-v0.2 model from Hugging Face was also used in this study. For more details on the model, visit: [Mistral-7B-Instruct-v0.2 on Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).

---
