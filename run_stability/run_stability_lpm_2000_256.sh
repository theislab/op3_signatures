#./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e none -S 0,1,2,3,4,10,11,12,13,14 -f op3_emb_lpm_2000_256.pkl -b stability_lpm_2000_256
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e lpm -l concat -S 0,1,2,3,4,10,11,12,13,14 -f op3_emb_lpm_2000_256.pkl -b stability_lpm_2000_256
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e lpm -l concat -d -S 0,1,2,3,4,10,11,12,13,14 -f op3_emb_lpm_2000_256.pkl -b stability_lpm_2000_256
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e fp -l concat -S 0,1,2,3,4,10,11,12,13,14 -f op3_emb_lpm_2000_256.pkl -b stability_lpm_2000_256
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e fp -l concat -d -S 0,1,2,3,4,10,11,12,13,14 -f op3_emb_lpm_2000_256.pkl -b stability_lpm_2000_256
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e lpm -l fixed -S 0,1,2,3,4,10,11,12,13,14 -f op3_emb_lpm_2000_256.pkl -b stability_lpm_2000_256
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e fp -l fixed -S 0,1,2,3,4,10,11,12,13,14 -f op3_emb_lpm_2000_256.pkl -b stability_lpm_2000_256
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e lpm -l trainable -S 0,1,2,3,4,10,11,12,13,14 -f op3_emb_lpm_2000_256.pkl -b stability_lpm_2000_256
./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh -e fp -l trainable -S 0,1,2,3,4,10,11,12,13,14 -f op3_emb_lpm_2000_256.pkl -b stability_lpm_2000_256

