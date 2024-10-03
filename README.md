# Patient Treatment Outcome Improvement Pipeline

This project implements an algorithmic hypothesis generation pipeline to improve HIV treatment outcomes. The pipeline includes data preparation, feature engineering, predictive modeling, synthetic data generation, morphing procedures, hypothesis generation, and validation. Additionally, a Streamlit-based human interface is provided for clinical experts to interact with the generated hypotheses.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Human Interface](#human-interface)
- [Data Structures](#data-structures)
- [Contributing](#contributing)
- [License](#license)

## Overview
The pipeline aims to uncover novel factors influencing HIV treatment response by leveraging advanced machine learning techniques and synthetic data generation. The key components of the pipeline include:
1. Data Preparation
2. Feature Engineering
3. Predictive Modeling
4. Synthetic Data Generation
5. Morphing Procedure
6. Hypothesis Generation
7. Hypothesis Validation
8. Human Interface

## Installation
To install the required dependencies, run the following command:
```sh
pip install -r requirements.txt
```

## Usage
To run the entire pipeline, execute the `pipeline.py` script:
```sh
python utils/tests/pipeline.py
```

To start the Streamlit-based human interface, run:
```sh
streamlit run ui/hypothesis_ui.py
```

## Pipeline Steps
1. **Data Preparation**: Load and preprocess the dataset.
2. **Feature Engineering**: Engineer features from the raw data.
3. **Predictive Modeling**: Train a predictive model to predict treatment success.
4. **Synthetic Data Generation**: Train a Variational Autoencoder (VAE) to generate synthetic patient profiles.
5. **Morphing Procedure**: Modify synthetic profiles to investigate the impact of different features on treatment response.
6. **Hypothesis Generation**: Generate hypotheses based on synthetic and real patient profiles.
7. **Hypothesis Validation**: Validate the generated hypotheses using statistical tests.
8. **Human Interface**: Provide a Streamlit-based interface for clinical experts to interact with the generated hypotheses.

## Human Interface
The human interface allows clinical experts to:
- View and compare patient profiles
- Visualize feature importance
- Provide input on generated hypotheses
- Explore the impact of different features on treatment response
- Access educational resources and clinical guidelines
- Provide feedback on the system's usability and usefulness

## Data Structures
The pipeline uses Pydantic classes to represent various components, including:
- Patient profiles
- Hypotheses
- Evaluation results


