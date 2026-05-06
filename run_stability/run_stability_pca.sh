#!/bin/bash

# ============================================================================
# Stability runs for PCA-based embeddings (all versions × both columns × all layers)
#
# PCA versions:   v1_64, v2_64, v2_128, v3_64, v3_128, v4_64, v4_128
# Embedding cols: pca_logfc (PCA.logFC), pca_t (PCA.t)
# Layers:         concat, concat+dense, fixed, trainable
#
# -P sets DATA_DIR=neurips-2023-data-subsample-pca/<version>
#    and auto-sets stability subfolder to stability_pca_<version>
# ============================================================================

SEEDS="0,1,2,3,4,10,11,12,13,14"

# ============================================================================
# v1_64
# ============================================================================
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat          -P v1_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat -d       -P v1_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l fixed           -P v1_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l trainable       -P v1_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat          -P v1_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat -d       -P v1_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l fixed           -P v1_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l trainable       -P v1_64 -S ${SEEDS}

# ============================================================================
# v2_64
# ============================================================================
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat          -P v2_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat -d       -P v2_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l fixed           -P v2_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l trainable       -P v2_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat          -P v2_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat -d       -P v2_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l fixed           -P v2_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l trainable       -P v2_64 -S ${SEEDS}

# ============================================================================
# v2_128
# ============================================================================
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat          -P v2_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat -d       -P v2_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l fixed           -P v2_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l trainable       -P v2_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat          -P v2_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat -d       -P v2_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l fixed           -P v2_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l trainable       -P v2_128 -S ${SEEDS}

# ============================================================================
# v3_64
# ============================================================================
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat          -P v3_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat -d       -P v3_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l fixed           -P v3_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l trainable       -P v3_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat          -P v3_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat -d       -P v3_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l fixed           -P v3_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l trainable       -P v3_64 -S ${SEEDS}

# ============================================================================
# v3_128
# ============================================================================
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat          -P v3_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat -d       -P v3_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l fixed           -P v3_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l trainable       -P v3_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat          -P v3_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat -d       -P v3_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l fixed           -P v3_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l trainable       -P v3_128 -S ${SEEDS}

# ============================================================================
# v4_64
# ============================================================================
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat          -P v4_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat -d       -P v4_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l fixed           -P v4_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l trainable       -P v4_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat          -P v4_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat -d       -P v4_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l fixed           -P v4_64 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l trainable       -P v4_64 -S ${SEEDS}

# ============================================================================
# v4_128
# ============================================================================
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat          -P v4_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l concat -d       -P v4_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l fixed           -P v4_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_logfc -l trainable       -P v4_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat          -P v4_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l concat -d       -P v4_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l fixed           -P v4_128 -S ${SEEDS}
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e pca_t     -l trainable       -P v4_128 -S ${SEEDS}
