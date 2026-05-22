# Fantasy Football Llama 3 Fine Tuned Model

Transforms base Llama 3 model into a highly specialized Fantasy Football Analyst. 

This repository contains the complete pipeline for generating domain-specific synthetic training data, fine-tuning the model using Low-Rank Adaptation (LoRA), and deploying the finalized weights for local, rapid inference via Ollama. 

### 1. Synthetic Data Generation
The foundation of the model's domain expertise is a custom dataset of synthetic Q&A pairs. 
* **Script:** `generate_data.py`
* **Process:** This script programmatically generates thousands of complex fantasy scenarios (e.g. evaluating a trade offering a future 1st round pick and Joshua Palmer for a tier-1 running back). The data is formatted into instruction-response pairs optimized for LLM fine-tuning.

### 2. LoRA Fine-Tuning
Trains model to adopt highly specialized jargon without requiring massive, expensive compute clusters, the model is fine-tuned using the LoRA (Low-Rank Adaptation) method.
* **Notebook:** `ffb_llama_trainer.py`
* **Process:** Instead of updating all billions of parameters in the base Llama model, LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into the Transformer architecture. This drastically reduces the number of trainable parameters, allowing the model to quickly adapt to the fantasy football instruction dataset while running on a single consumer-grade GPU or Google Colab instance (which was used in this case).

### 3. Deployment via Ollama
The final stage of the lifecycle is containerized local deployment. By utilizing Ollama, the fine-tuned model can be served locally with minimal latency.
