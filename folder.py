import os

# Define the folder structure
folders = [
    'cattle-weight-prediction/data/raw',
    'cattle-weight-prediction/data/processed',
    'cattle-weight-prediction/data/external',
    'cattle-weight-prediction/notebooks',
    'cattle-weight-prediction/src',
    'cattle-weight-prediction/models',
    'cattle-weight-prediction/reports/figures',
    'cattle-weight-prediction/tests',
    'cattle-weight-prediction/environment'
]

# Define the files to be created
files = {
    'cattle-weight-prediction/README.md': '# Cattle Weight Prediction Project\n\n',
    'cattle-weight-prediction/.gitignore': 'data/\nmodels/\n',
    'cattle-weight-prediction/data/README.md': '# Data Folder\n\nThis folder contains the raw and processed datasets.\n',
    'cattle-weight-prediction/src/__init__.py': '',
    'cattle-weight-prediction/src/data_processing.py': '# Code for data loading and preprocessing\n',
    'cattle-weight-prediction/src/model.py': '# Code for model creation, training, and evaluation\n',
    'cattle-weight-prediction/src/utils.py': '# Utility functions for the project\n',
    'cattle-weight-prediction/src/predict.py': '# Code for loading the model and making predictions\n',
    'cattle-weight-prediction/tests/test_data_processing.py': '# Tests for data processing\n',
    'cattle-weight-prediction/tests/test_model.py': '# Tests for model creation and evaluation\n',
    'cattle-weight-prediction/environment/requirements.txt': '# List of dependencies\n',
    'cattle-weight-prediction/environment/environment.yml': 'name: cattle-weight-prediction\nchannels:\n  - defaults\ndependencies:\n  - python=3.8\n  - numpy\n  - pandas\n  - scikit-learn\n  - matplotlib\n'
}

# Create the folder structure
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create the files
for filepath, content in files.items():
    with open(filepath, 'w') as f:
        f.write(content)

print("Project folder structure created successfully.")
