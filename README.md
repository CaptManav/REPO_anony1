# Graph Neural Network Recommendation System

## Overview

This project implements a scalable, production-oriented Graph Neural Network (GNN) based recommendation engine using PyTorch Geometric.

The system models user–item interactions as a heterogeneous graph and performs link prediction to generate personalized top-K recommendations.

Unlike traditional collaborative filtering, this approach leverages message passing over graph structure to learn expressive node embeddings that capture higher-order relationships.

---

## Problem Statement

Given a graph containing:

- Users  
- Items  
- Interaction edges (e.g., ratings, clicks, purchases)  

The objective is to predict the probability of a future interaction between a user and an item.

Formally:

Given graph \( G = (V, E) \), learn embeddings \( h_u \) and \( h_i \) such that:

\[
P(u, i) = \sigma(h_u^\top h_i)
\]

The task is framed as a **link prediction problem** and evaluated using ranking metrics.

---

## Why Graph Neural Networks?

Traditional recommendation systems (e.g., matrix factorization) only model direct interactions.

GNNs enable:

- Multi-hop relational reasoning  
- Inductive learning (new users/items)  
- Feature-aware embedding learning  
- Better representation of structured data  

This project explores GraphSAGE and Graph Attention Networks (GAT) to compare performance against classical baselines.

---

## System Architecture

### 1. Data Layer
- Dataset: MovieLens 25M (initial)
- Construction of heterogeneous graph:
  - User nodes
  - Item nodes
  - Interaction edges
- Node features:
  - User activity statistics
  - Item metadata
  - Optional text embeddings

### 2. Graph Processing
- PyTorch Geometric `HeteroData`
- Train/validation/test edge split
- Negative sampling
- Neighbor sampling for scalability

### 3. Model Layer

**Baseline:**
- Matrix Factorization

**GNN Models:**
- GraphSAGE
- Graph Attention Network (GAT)

**Objective:**
- Binary Cross Entropy loss for link prediction

### 4. Evaluation

Ranking-based metrics:
- Precision@K  
- Recall@K  
- NDCG@K  
- AUC  
- Hit Rate  

### 5. Inference Layer
- Export learned node embeddings  
- FAISS-based nearest neighbor retrieval  
- FastAPI endpoint for real-time recommendation  

---

## Project Structure

```
gnn-recommender/
│
├── data/
├── preprocessing/
├── models/
├── training/
├── evaluation/
├── inference/
├── api/
├── experiments/
├── docs/
└── main.py
```


## Scalability Considerations

- Neighbor sampling (PyG NeighborLoader)
- Mini-batch training
- Sparse adjacency handling
- Inductive capability for unseen nodes
- Dockerized deployment pipeline

---

## Roadmap

### Phase 1
- Data preprocessing and graph construction  
- Baseline matrix factorization implementation  

### Phase 2
- GraphSAGE implementation  
- Link prediction training pipeline  
- Ranking metric evaluation  

### Phase 3
- GAT implementation  
- Performance benchmarking and ablation studies  

### Phase 4
- Embedding export  
- FAISS indexing  
- FastAPI deployment  
- Docker containerization  

### Phase 5
- Embedding visualization (UMAP/t-SNE)  
- Attention weight analysis  
- Performance analysis and documentation  

---

## Expected Outcomes

- Demonstrate superiority (or trade-offs) of GNN over classical collaborative filtering  
- Understand message passing dynamics and over-smoothing  
- Implement scalable graph training pipeline  
- Deliver deployable recommendation system  
