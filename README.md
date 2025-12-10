# URL Security Classification using Q-Learning

This project implements a reinforcement learning solution for URL-based attack detection using Q-learning on the CSIC 2010 HTTP dataset.

## Project Structure

```
.
├── data_preparation.py    # Data loading, feature extraction, state discretization
├── environment.py         # RL environment (MDP) implementation
├── q_learning.py          # Q-learning algorithm
├── baseline.py            # Rule-based baseline classifier
├── training.py            # Training pipeline
├── evaluation.py          # Evaluation metrics and visualizations
├── main.py                # Main script to run entire pipeline
├── requirements.txt       # Python dependencies
├── dataset/
│   └── csic_database.csv  # CSIC 2010 HTTP dataset
└── README.md              # This file
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the complete pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess the CSIC 2010 dataset
2. Extract features from URLs
3. Train a Q-learning agent
4. Evaluate both RL agent and baseline
5. Generate visualizations
6. Create a comprehensive report

### Output Files

After running, you'll get:
- `q_table.npy`: Saved Q-table
- `q_learning_agent.pkl`: Saved trained agent
- `evaluation_results.json`: Evaluation metrics
- `plots/`: Directory with visualizations
  - `learning_curve.png`: Training progress
  - `confusion_matrix_rl.png`: RL agent confusion matrix
  - `confusion_matrix_baseline.png`: Baseline confusion matrix
  - `q_table_heatmap.png`: Q-table visualization
- `report.md`: Comprehensive report

## Features

### Data Preparation
- Feature extraction: URL length, special characters, digits, parameters, suspicious keywords
- State discretization: Binning and quantization methods
- Train/validation/test split

### RL Environment
- Discrete state space
- Two actions: ALLOW (0) or BLOCK (1)
- Reward structure optimized for security:
  - TP: +2.0, TN: +0.5, FP: -5.0, FN: -8.0

### Q-Learning
- Tabular Q-learning with Bellman update
- ε-greedy exploration strategy
- Hyperparameters: α=0.1, γ=0.90, ε-decay=0.98

### Baseline
- Rule-based classifier with threshold tuning

## Requirements

- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0

## License

This project is for educational purposes.

