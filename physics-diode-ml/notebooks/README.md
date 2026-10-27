# Notebook

The analysis can be run from the script at `../src/analyze_diode.py`, including in Google Colab after uploading the project folder. The script downloads the dataset and saves all graphs, metrics, cleaned data, and predictions under `outputs/`.

For a notebook-style workflow, open Colab, create a notebook, upload `src/analyze_diode.py` and run:
```python
!pip -q install pandas numpy matplotlib scikit-learn
!python analyze_diode.py
```
Place the script in the current Colab working directory before running the command.
