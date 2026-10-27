# Methodology

## Aim
To examine the forward-bias current–voltage characteristic of a semiconductor diode using published measurements and investigate whether machine-learning regression models can predict measured current from voltage.

## Data source and variables
The data are taken from the public blueberrye-io diode I–V measurements repository. The analysis uses the 1N4004 forward-bias CSV file. The independent variable is measured diode voltage, `v_meas`, and the dependent variable is measured diode current, `i_meas`. The set voltage is not used for model fitting because measured voltage is the quantity directly associated with the measured current.

## Cleaning
Rows with missing or non-finite measured voltage/current values are removed. Only forward-bias measurements with voltage between 0 V and 0.80 V and positive measured current are retained. This excludes the invalid unsettled first reading noted in the source data and avoids taking logarithms of zero or negative current.

## Visualisation
The measured I–V curve is plotted with voltage on the horizontal axis and current on the vertical axis. A logarithmic current scale is also used because diode current can vary over several orders of magnitude.

## Modelling design
The observations are ordered by measured voltage. A central voltage interval between the 40th and 60th percentiles is held out as a test set. The remaining lower- and higher-voltage observations form the training set. This is a deliberate voltage-interval holdout, rather than a random split, to examine how predictions behave in a region absent from training.

Two models are compared:
- Linear Regression, a simple baseline that fits a straight-line relationship.
- Random Forest Regression, an ensemble of decision trees capable of representing nonlinear relationships.

Both models are fitted on the same training observations and evaluated on the same held-out observations.

## Evaluation
Mean Absolute Error (MAE) measures average absolute prediction error. Root Mean Squared Error (RMSE) gives greater weight to larger errors. R² compares prediction performance with a constant-mean baseline; it can be negative when predictions perform worse than that baseline.

## Limitations
This is a secondary-data computational investigation, not a newly performed laboratory experiment. It uses one diode dataset and one measurement context. The voltage-interval holdout is a specific test of interpolation across a missing region, not a guarantee of generalisation to other diodes, temperatures, devices, or circuits. A random forest can produce step-like predictions and should not be interpreted as a physical diode law. Model metrics are only valid for the actual run and must be copied from the generated output, not guessed.
