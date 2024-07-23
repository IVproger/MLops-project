# Source code outline
```
src
├── data.py                     # Functions to manipulate data. If ran on its own, downloads, samples, validates, and versions data
├── data_quality.py             # Unmaintained. `load_context_and_sample_data` is used in two notebooks. Keep this for archival reasons
├── data_transformations.py     # Functions for data transformation
├── evaluate.py                 # Script that validates a given model
├── main.py                     # Script runs model training (MLFlow)
├── model.py                    # MLFlow-related functions
├── utils.py                    # Utility functions
└── validate.py                 # Giskard model validation. Generates a Giskard report
```
