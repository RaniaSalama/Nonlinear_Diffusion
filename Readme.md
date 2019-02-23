The code used in Rania Ibrahim and David F. Gleich. "Nonlinear Diffusion for Community Detection and Semi-Supervised Learning" paper, accepted at The Web Conference (formally WWW), 2019.

# Synthetic Data Experiments

You can find the code for LFR synthetic data experiments in LFR_Figures folder.
To generate the conductance and F1 results, which are reported in figure 2 in the paper, run run_on_LFR_data.m

# Community Detection Experiments

You can find the code for community detection experiments in Community_Detection_Experiments folder. Inside this folder there is three subfolders:

- community_detection: contains the experiments for nonlinear diffusion using power function.
- community_detection_tanh: contains the experiments for nonlinear diffusion using tanh function.
- community_detection_p_lap: contains the experiments for nonlinear diffusion using p-Laplacian.

To generate the results in table 1 of the paper: First download the datasets from SNAP website, add them folder inside a folder called "data" and then inside each subfolder run:

- run_com-dblp.jl: which produces the DBLP results.
- run_com-amazon.jl: which produces the Amazon results.
- run_com-youtube.jl: which produces the Youtube results.
- run_com-lj: which produces the Live Journal results.

# Graph-based Semi-supervised Learning Experiments

You can find the code for graph-based semi-supervised learning experiment inside the folder Graph-based_Semi-supervised Learning_Experiment. Inside this folder, there is three subfolders:

- Nonlinear Diffusion with Growth or Decay: contains the code for running nonlinear diffusion using power function or tanh function.
- Nonlinear Diffusion via Nonlinear Transfer: contains the code for running nonlinear diffusion using p-Laplacian.
- Stability_Experiment: contains the code to generate figure 3 in the paper, where we vary the parameters for power function and tanh function and show that the classification accuracy remains highfor many choices.




