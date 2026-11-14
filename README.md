# Assignment 4 · Reweighing

This notebook evaluates Reweighing from Calders, Kamiran and Pechenizkiy (2009),
[Building Classifiers with Independency Constraints](https://doi.org/10.1109/ICDMW.2009.83), §3.2.
The Python implementation calculates the paper's weights and passes them
directly to logistic regression. The paper used the weights for sampling with
replacement. No public repository for the authors' original implementation
was located.

## Reproduce the results

Install Python 3.11 and R with `Rscript` on PATH. Run the following from the
directory containing `reweighing_evaluation.ipynb`. On Windows, activate
the environment with `.venv\Scripts\activate`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.24.2 pandas==1.5.3 scikit-learn==1.2.1 notebook==7.2.2
Rscript --vanilla -e 'install.packages("remotes", repos="https://cloud.r-project.org"); remotes::install_version("caret", "7.0-1", repos="https://cloud.r-project.org", upgrade="never"); remotes::install_version("kernlab", "0.9-33", repos="https://cloud.r-project.org", upgrade="never"); remotes::install_version("titanic", "0.1.0", repos="https://cloud.r-project.org", upgrade="never")'
jupyter notebook reweighing_evaluation.ipynb
```

In Jupyter, select the Python 3 kernel from the virtual environment, then
choose **Run All Cells** and save the notebook. The tables should show 12
evaluation rows (four per dataset), four training weights per dataset and
three before/after comparisons. Executed outputs are included. Tied
nearest-neighbor distances can produce small consistency differences across platforms.

## Datasets and settings

| Dataset | Package object | Favorable label | Group 0 / reference |
|---|---|---|---|
| German Credit | `caret::GermanCredit` | Good credit risk | Age ≤25 / age >25 |
| Bay Area Income survey | `kernlab::income` | Income bin ≥$50,000 | Recorded female / male |
| Titanic | `titanic::titanic_train` | Survived | Recorded male / female |

The notebook exports the package datasets to temporary CSV files at runtime.
The joint group–label stratified split uses 20% test data and seed 42.
Both models use L2 logistic regression with C=1, the L-BFGS solver, at most
2,000 iterations and the default 0.5 threshold. Consistency uses Euclidean
distance with five neighbors including the record itself. The model and
distance calculations exclude the protected field. Weighted and unweighted
predictions use the same held-out records.
