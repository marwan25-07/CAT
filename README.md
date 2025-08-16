# CAT
[![DOI](https://zenodo.org/badge/1037222864.svg)](https://doi.org/10.5281/zenodo.16887610)
## Project Description
MOONN is a repository containing a machine learning model that was developed for the prediction of Liquid-Phase Rate Constants, using only a reaction's 2D SMILES string as input.
The model was created as a data driven alternative to traditional experimental or kinetic modelling efforts, to identify the most optimal pre-existing solvent from a set of commonly used solvents for a given reaction. 
The repository contains data in which was used to train our model, requirement installations/versions, modular code featuring the architecture used for prediction and finally a notebook that provides a walkthrough for users to trail the model. 

## How it works
To operate the model, a 2D reaction SMILES string is first provided to the model as input. The reactant SMILES string is retrieved from the input and is converted into a set of features using the Mordred feature set, available in RDKIT. The featurised reactant is than encoded into a latent space, where consequently the solvent which yields the highest rate constant is calculated via the surrogate model of a Gaussian optimisation step. 

## Tutorial 
The '**tutorial**' script provides a working example of the model in use, where uses can submit a SMILES reaction to determine the most optimal solvent. Further instructions on how to prepare the model for use is provided in the script.   

## Contact
If you have any queries please feel free to contact Marwan via [email](mz354@ic.ac.uk). 
