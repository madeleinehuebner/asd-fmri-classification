## Project structure

```None
PROJECT
├── product
│   ├── data
│   │   ├── raw                    - Downloaded .1D timeseries files
│   │   ├── external               - External Phenotypic CSV and ROI labels
│   │   └── processed              - Harmonised feature matrices and metadata
│   ├── notebooks
│   │   ├── exploratory            - Data exploration and analysis notebooks
│   │   ├── visualisation          - Plotting and visualisation notebooks
│   │   └── run_experiments.ipynb  - Run model pipelines
│   ├── results                    - Experiment artefacts
│   ├── scripts                    - Scripts for downloading and processing data
│   ├── src
│   │   ├── autoencoder            - Autoencoder definitions
│   │   ├── data_io                - Saving and loading utilities, experiement logging
│   │   ├── evaluation             - Cross validation, metrics modules
│   │   ├── gcn                    - Residual GCN and graph definitions
│   │   ├── preprocessing          - Data preprocessing modules
│   │   ├── utils                  - Path definitions
│   │   └── visualisation          - Plotting definitions
│   └── tests                      - Unit tests
├── diary.md                       - Weekly diary entries
├── pyproject.toml                 - Project configuration
└── README.md
```

## Environment requirements

- Python 3.12
- Dependencies managed via pyproject.toml

## Installation instructions

1. Clone the repo via git:

    ```bash
    git clone https://gitlab.cim.rhul.ac.uk/zlac337/PROJECT.git
    ```

    *OR Download and extract the `PROJECT.zip` file. Ensure the folder name is `PROJECT`.*

2. Navigate to the root directory `PROJECT` in your terminal

    ```bash
    cd PATH/TO/PROJECT
    ```

3. Create and activate a virtual environment. It is important to set `python` to `3.12`:

    ```bash
    conda create -n myenv python=3.12
    conda activate myenv
    ```

4. Install the project:

    ```bash
    pip install .
    ```

## Manual

### To run this project locally

#### VS Code

 1. Open this project in VS Code with `PROJECT` as the root folder.
 2. Ensure the environment is active by running `conda activate myenv` from the integrated terminal.

#### Jupyter

 1. Ensure the environment is active by running `conda activate myenv` from terminal.
 2. Launch Jupyter:

    ```bash
    jupyter lab
    ```

### To run this project in Google Colab

   1. Upload the entire `PROJECT` folder to Google Drive
   2. Run the snippet at the top of the notebooks to mount drive.

### To preprocess datasets

Run `product/scripts/run_scripts.ipynb` to process a dataset.
   > [!Warning]
   > Running this in **Google Colab** will take several minutes. It is therefore recommended to process the dataset **locally**.

### To run experiments

All experiments and visualisations are contained within `.ipynb` files in the `product/notebooks/` directory.

### To run tests

Run pytest:

```bash
pytest
```
