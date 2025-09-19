# Gremban Expansion for Signed Networks: Code Companion

This repository accompanies the article  
**“Gremban Expansion for Signed Networks: Algebraic and Combinatorial Foundations for Community–Faction Detection”**  
([arXiv:2509.14193](https://arxiv.org/abs/2509.14193)) by Fernando Diaz-Diaz, Karel Devriendt, and Renaud Lambiotte.

The paper develops an algebraic and combinatorial framework for analyzing **community** and **faction** structures in signed networks through the **Gremban expansion**. The code here reproduces the main constructions and experiments.

## Contents

- **`code_gremban_expansion_signed_networks.ipynb`**  
  Jupyter notebook with demonstrations from the paper:
  - Generation of signed degree–corrected stochastic block model (DC-SBM) networks.  
  - Construction of the Gremban expansion for adjacency matrices and graphs.  
  - Computation of unsigned, signed, and Gremban Laplacians (normalized and unnormalized).  
  - Spectral clustering separating communities from factions.  
  - Visualizations of original vs. expanded graphs and spectral embeddings.
  - Diffusive dynamics on original and expanded graphs. 

- **`util.py`**  
  Utility functions used in the notebook:
  - `generate_signed_dcsbm` — generate synthetic signed DC-SBM networks.  
  - `draw_network` — visualize signed networks with positive/negative edges and group markers.  
  - `gremban_expansion` — compute the Gremban expansion (matrix or NetworkX graph).  
  - `compute_laplacian` — extract unsigned, signed, or Gremban Laplacians.  
  - `pos_gremban_expansion` — layout helper for visualizing original vs. expanded graphs.  
  - `adjacency_matrix` — convert graphs to adjacency matrices.  

## Reference
If you use this code, please cite the article:  
[F. Diaz-Diaz, K. Devriendt, R. Lambiotte. *Gremban Expansion for Signed Networks: Algebraic and Combinatorial Foundations for Community–Faction Detection*. arXiv:2509.14193 (2025).](https://arxiv.org/abs/2509.14193)
