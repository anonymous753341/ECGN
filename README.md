# Enhanced Cluster-aware Graph Network (ECGN)

Node classification in graphs often faces challenges due to class imbalance and complex clustering structures. Traditional Graph Neural Networks (GNNs) do not address both issues simultaneously. We introduce **ECGN**, a novel method that integrates cluster-specific training with synthetic node generation. ECGN learns unique node aggregation strategies for different clusters and generates new minority-class nodes to improve decision boundaries. Compatible with any underlying GNN and clustering method, ECGN enhances node embeddings and achieves up to 11% higher accuracy on widely-studied benchmark datasets compared to other methods.

## Implementation Details

### Running without Cluster-aware SMOTE

1. Load the script: `baseline_experiments/baseline_ecgn_main_smote_imbalanced_main.py`
2. Modify the replication parameter in the corresponding YAML file for different datasets in the WandB config.

### Running with Cluster-aware SMOTE

1. Load the script: `baseline_experiments/baseline_ecgn_main_smote_imbalanced_main_withsmote.py`
2. Modify the replication parameter in the corresponding YAML file for different datasets in the WandB config.
