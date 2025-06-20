# Marvel Network Link-Prediction

A network-analysis and link-prediction study on the Marvel Universe bipartite graph. We combine classic collaborative-filtering approaches with the **LPFormer** model to predict hero–comic co-appearances, with special focus on the Infinity Saga subgraph.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Data](#data)  
3. [Repository Structure](#repository-structure)  
4. [Installation & Requirements](#installation--requirements)  
5. [Usage](#usage)  
6. [Contributors & Contact](#contributors--contact)  

---

## Project Overview

- **Goal:** Implement and evaluate link-prediction in a Marvel bipartite network (heroes ⇄ comics).  
- **Models:**  
  - **Collaborative Filtering** (CF) baselines  
  - **LPFormer** (a Transformer-based link predictor)  
- **Workflow:**  
  1. **Network Exploration** (full graph + Infinity Saga subgraph)  
  2. **Model Implementation**  
     - CF analysis (in `marvel_project.ipynb`)  
     - LPFormer (in `LPFormer/LPFormer_Subgraph.ipynb`)  
  3. **Evaluation & Comparison** of predictive performance  

---

## Data

All raw and cleaned data live in the `Data/` directory:

- `nodes_corr.csv` – corrected node list (heroes + comics)  
- `edges_corr.csv` – corrected edge list (hero appearances)  
- `nodes.csv` and `edges.csv` - Original Kaggle exports (if you’d like to inspect the raw source)

_No external downloads are required._

---

## Repository Structure

    .
    ├── Data/
    │   ├── nodes.csv           (original Kaggle data)
    │   ├── edges.csv           (original Kaggle data)
    │   ├── nodes_corr.csv      (preprocessed data)
    │   └── edges_corr.csv      (preprocessed data)
    ├── LPFormer/
    │   └── LPFormer_Subgraph.ipynb  # LPFormer implementation & analysis
    ├── CF_Evaluation/
    │   └── results_1-4.csv     # CF output dataframes
    ├── Plots/                  # Figures generated by notebooks
    ├── marvel_project.ipynb    # Main network exploration & CF work
    ├── requirements.txt
    └── README.md


---

## Installation & Requirements

1. Clone this repository:  
   ```bash
   git clone https://github.com/Konpyuuta/social-media-analytics.git
   cd marvel-network
   
2. (Recommended) Create a virtual environment:
   ```bash
    python3 -m venv venv
    source venv/bin/activate

4. Install dependencies:
   ```bash
   pip install -r requirements.txt

---

## Usage

All analyses run directly in Jupyter:

Open `marvel_project.ipynb` (main network + CF).
Open `LPFormer/LPFormer_Subgraph.ipynb` (LPFormer modeling).
No special command-line steps are required—just load the notebooks in your IDE of choice.

---

## Contributors and Contacts

| Name            | Role                             | Contact              |
|-----------------|----------------------------------|----------------------|
| Alessia Bussard | Preprocessing,Analysis of the Network & CF Evaluation | alessia.bussard@gmail.com    |
| Maurice Amon | Graph Persistence & Implementation of CF Recommender Algorithms | amon@zhaw.ch    |
| Filipe Silva | LPFormer implementation and Evaluation | filipe.nunessilva@students.unibe.ch    |

