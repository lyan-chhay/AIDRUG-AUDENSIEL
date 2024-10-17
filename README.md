# AggrePred: Residue-Level Prediction of Protein and Antibody Aggregation with Mutation Suggestions for Improved Solubility and Stability
AIDRUG is a project in Audensiel Healthcare, where it aims to use AI applied in Healthcare, specifically in bioinformatics. 

AggrePred, a tool developed during my internship, aimed at predicting the aggregation propensity of proteins and antibodies based on their sequences. The model utilizes advanced deep learning techniques to analyze sequences and provide insights into aggregation risks.

![alt text](image/goal.png)


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Link](#link)
- [Contributor](#contributor)


## Installation

To install and run the AggrePred project, please follow these steps:

1. **Prerequisites**: Ensure you have Python3 installed.
2. **Clone the repository**:
```bash
git clone https://github.com/Lyan168/AIDRUG-AUDENSIEL.git
cd AIDRUG-AUDENSIEL
```

### Dependencies
This conda env is for the application and training code

```conda env create -f environment.yml```

This conda env is for preprocessing part, A3D package, as it required python 2.7.0

```conda env create -f environment_a3d.yml```


### THPLM (Model for ddg calculation)
```bash
cd application
git clone https://github.com/FPPGroup/THPLM.git
```


# Usage

To run the application , make sure you are in Application repository

```bash
cd AIDRUG-AUDENSIEL/application
streamlit run apps.py

```

#### You can follow your local-host server at http://localhost:8501

When you first launch the local server, the application interface will appear.

## Navigation
On the left side of the interface, there is a navigation menu where you can switch between **Protein and Antibody** prediction modes.

![alt text](image/interface_prot.png)


## Input Options
You can input your protein sequence in FASTA format either by typing directly into the input field or uploading a FASTA file. The file upload option allows for multiple protein predictions at once.

## Prediction
Once the input is provided, click Predict to begin the analysis. Two result tabs will appear:

### Residue Plot: 
This tab displays a line plot showing the aggregation propensity (AP) score for each residue of the protein. You can hover over the plot to view detailed information such as residue position, amino acid, and AP score.

![alt text](image/score_prot_plot.png)

### AP Score Table: 
This tab contains a table listing the AP scores. Positive AP scores (indicative of aggregation-prone residues) are highlighted in yellow. There is also a **download button** to download the AP scores in csv

![alt text](image/result_table.png)



## Advanced Options

### Auto-Mutation
The Auto-Mutate option automatically suggests mutations to reduce overall aggregation. It replaces all residues with positive AP scores (APR) with more soluble and stable amino acids **(e.g., E, D, K, R)**.

The mutated sequence with the lowest number of APRs and average AP scores is recommended.

### Free Energy Change Calculation
For further analysis, you can use the **Calculate Free Energy Change option**. This feature calculates the difference in free energy between the wild-type and mutated sequences. A negative value indicates that the mutated sequence is more stable than the original.


![alt text](image/mutation.png)





# Link

In the model training repository, I have tried multiple model with different configuration. If you are interested in exploring all the models, download them [here](https://zenodo.org/records/13943204?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjdkMzE3MzBjLTI0NzYtNDY1OS04MTA5LTBhZjQwZjczNTJjMiIsImRhdGEiOnt9LCJyYW5kb20iOiI1NDhlYTlhM2JiZDBmN2QxNGEyY2M0OWQ1ZDNkNDlhZiJ9.GZwOAgLDr9oHMNrmPqoQd8wXiLpHkbCfsKCjrg-rzdo_XNt6o-lJia-RqHJujiy5bqEimww1RNSmGF1iJQhxkg). 

# Contributor
Ly An CHHAY :  lyan.chhay@gmail.com
