# Automated Quality Control of Time-Course Imaging from 3D in vitro cultures

<!-- Begin Badges -->
[![DOI](https://zenodo.org/badge/943608408.svg)](https://doi.org/10.5281/zenodo.15850545)
<!-- End Badges -->

This repository contains the code and notebooks associated with the manuscript "[_Automated Quality Control of Time-Course Imaging from 3D in vitro cultures_]()". It provides an algorithm for assessing and correcting technical artifacts in longitudinal imaging of 3D cell cultures, such as tumor spheroids. The algorithm relies only on image metadata, requiring no experimental modifications.

## Repository Structure

This repository is organized as follows:
```bash
.
├── README.md # This file
├── .gitignore # Specifies intentionally untracked files that Git should ignore
├── align_spheroid # Python package containing the core algorithm
│ ├── init.py # Initializes the align_spheroid package
│ ├── pycache # Compiled Python code
│ ├── alignment.py # Main alignment class and functions
│ ├── alignment_func.py # Functions related to the alignment process
│ ├── alignment_polygon.py # Alignment algorithm for 3D structures
│ ├── evaluation.py # Functions for evaluating alignment performance
│ ├── synthetic_data.py # Functions for generating synthetic data for testing
│ └── utils.py # Utility functions
├── paper # Jupyter notebooks for generating figures from the paper
│ ├── figure-1.ipynb # Notebook for creating Figure 1
│ ├── figure-2.ipynb # Notebook for creating Figure 2
│ ├── figure-3.ipynb # Notebook for creating Figure 3
│ └── tracker-comparison.ipynb # Notebook for creating tracking algorithm benachmarking figures
├── tests # Unit tests for the align_spheroid package
│ ├── test_alignment.py # Tests for alignment functions
│ ├── test_evaluation.py # Tests for evaluation functions
│ └── test_synthetic_data.py # Tests for synthetic data generation
├── synthetic-example.ipynb # Example notebook showing how to use the synthetic data generator
```

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/<your_username>/align_spheroid.git
    cd align_spheroid
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Core Algorithm

The core algorithm is implemented in the `align_spheroid` package.  Refer to the function documentation within the python package for details on use. See the `synthetic-example.ipynb` notebook for an example application of the algorithm to simulated data.

### Jupyter Notebooks for Paper Figures

The `paper` directory contains Jupyter notebooks with code used to generate the figures from the manuscript. To run these notebooks, ensure that you have installed Jupyter and all the dependencies listed in `requirements.txt`. Figures that describe findings from the biological data will need a local copy of the data set.

Example to create Figure 1:

```bash
jupyter notebook paper/figure-1.ipynb
```

The synthetic-example.ipynb notebook demonstrates the functionality of the algorithm using the synthetic data generation module included in the `align_spheroid` directory.

### Running Tests

The tests directory contains unit tests for the align_spheroid package. To run the tests, use pytest:
```bash
pytest tests
```

## Key Components:  
- align_spheroid/alignment.py: This module contains the core alignment algorithm, including the permutation-based optimization strategy and Procrustes analysis.
- align_spheroid/evaluation.py: This module provides functions for evaluating the performance of the alignment algorithm, such as the normalized Frechét distance.
- align_spheroid/synthetic_data.py: This module provides functions for generating synthetic data to test the algorithm's robustness and accuracy.
- paper/: This folder contains the Jupyter Notebooks used to generate the figures in the manuscript. These serve as useful demonstrations of how to use the code.

## License
This project is licensed under the MIT License - see the [LICENSE file](https://github.com/emcramer/align_spheroid/blob/main/LICENSE) for details.

## Citation

If you use this code in your research, please use the following citation:
> Automated Quality Control of Time-Course Imaging from 3D in vitro cultures. Eric Cramer , Tamara Lopez-Vidal , Jeanette Johnson , Vania Wang , Daniel Bergman , Ashani Weeraratna , Richard Burkhart, Elana J Fertig , Jacquelyn W Zimmerman, Laura Heiser, and Young Hwan Chang. _bioRxiv_ (2025)
