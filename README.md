# CAT
[![DOI](https://zenodo.org/badge/1037222864.svg)](https://doi.org/10.5281/zenodo.16887610)
## Project Description
CAT is a machine learning framework for predicting liquid-phase rate constants. It was designed as a data-driven alternative to experimental or kinetic modelling approaches, enabling users to identify the most optimal solvent for a given reaction from a set of commonly used solvents.

The repository includes:
- Installation requirements and dependency versions to use CAT
- CAT's training architecture
- CAT's model architecture
- CAT's model weights
- A python script called Tutorial, which serves as an example on how to use CAT. 

## How CAT works
1. **Input:** A 2D reaction SMILES string is provided.
2. **Feature Extraction:** The SMILES string is parsed and converted into a set of features using the Mordred descriptor set.
3. **Latent Encoding:** The featurised SMILES string is then encoded into a latent representation.
4. **Solvent Selection:** A Gaussian surrogate model is optimised using the latent repersentation and is subsequently then used to predict the most optimal solvent for the input reaction. 

## Tutorial 
The '**tutorial**' script provides a working example of the model, where users can submit a SMILES reaction to the CAT model to yield an optimal solvent. Further instructions on how to set up the model is provided in the script.   

## Contact
If you have any queries please feel free to contact Marwan via email at marwan.zalita24@imperial.ac.uk. 
