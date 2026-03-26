

## The Problem

Existing attention visualization tools are primarily designed for smaller models like BERT, leaving developers working with massive models like `nvidia/Nemotron-Cascade-2-30B-A3B` without a way to inspect attention heads effectively. This gap makes it difficult to debug, interpret, or optimize the behavior of frontier-scale models, especially in tasks requiring fine-grained token-level analysis.

## Who it's for

This tool is for machine learning engineers and researchers working with large-scale transformer models who need to understand attention patterns for debugging or model interpretability. For example, a developer fine-tuning `Nemotron-Cascade-2-30B-A3B` for a specific NLP task can use this tool to identify which tokens the model focuses on during inference.