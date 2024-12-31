# "Responsible" data elaboration: design and testing of a selection operator for non-discrimination of protected groups

This repository contains the code for a final test project for the bachelor's degree by Nicolò Guainazzo, an Computer science student at the Università degli studi di Genova.

## Overview 👀

This project explores the impact of dataset transformations on bias, focusing on techniques like query rewriting and extended execution to meet coverage constraints and ensure fairness (using KNN based on [kd-tree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html)). It emphasizes transparency and compliance with ethical standards like GDPR while providing a tool to avoid discrimination in data processing.

## Repository Structure 📂

The repository includes the following key files:

- **1_knn_2cc.py**: Script implementing the KNN algorithm with two coverage constraints.
- **Analysis_df_.ipynb**: Jupyter Notebook containing data analysis related to the project.
- **calc_measures_sol.py**: Script for calculating solution measures.
- **create_frame_execution.py**: Script to create the execution framework.
- **create_frame_rewriting.py**: Script to create the rewriting framework.
- **knn_conv_constraint_norm.py**: Script implementing the KNN algorithm with normalized coverage constraints.
- **knn_conv_costraint(inputQCC).py**: Script implementing the KNN algorithm with input query coverage constraints.
- **knn_conv_costraint_NEW_con_norm.py**: Script implementing a new version of the KNN algorithm with normalized coverage constraints.

## Requirements

The project is primarily implemented in Python and Jupyter Notebook. Ensure you have the following packages installed:

- numpy
- pandas
- scikit-learn
- matplotlib

You can install these packages using pip:

```bash
pip install numpy pandas scikit-learn matplotlib
```
## Usage

### Clone the repository:

```bash
git clone https://github.com/Nicogs43/KNN_coverage_constraint_query.git
```
Navigate to the project directory:
```bash
cd KNN_coverage_constraint_query
```
Run the desired script:
For example, to execute the KNN algorithm with two coverage constraints:
```bash
Copy code
python 1_knn_2cc.py
```
Alternatively, open the Jupyter Notebook for data analysis:

```bash
jupyter notebook Analysis_df_.ipynb
```
