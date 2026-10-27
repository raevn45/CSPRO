"""Analyse published 1N4004 forward-bias I-V measurements with regression models."""
from pathlib import Path
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
DATA_URL = (
    "https://raw.githubusercontent.com/blueberrye-io/diode-iv-measurements/"
    "main/data/1n4004_forward.csv"
)
DATA_PATH = OUT / "1n4004_forward_source.csv"


def load_data():
    """Download source CSV and return cleaned measured voltage/current columns."""
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the dataset. Download it from {DATA_URL} "
            f"and save it as {DATA_PATH.name} in {OUT}."
        ) from exc

    raw = pd.read_csv(DATA_PATH)
    required = {"v_meas", "i_meas"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required)}; missing {sorted(missing)}. "
            f"Found: {list(raw.columns)}"
        )

    data = raw[["v_meas", "i_meas"]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[
        (data["v_meas"] >= 0)
        & (data["v_meas"] <= 0.80)
        & (data["i_meas"] > 0)
    ]
    data = data.sort_values("v_meas").reset_index(drop=True)
    if len(data) < 20:
        raise ValueError(f"Only {len(data)} valid rows remain; check the source CSV.")
    return raw, data


def main():
    raw, data = load_data()
    data.to_csv(OUT / "cleaned_measurements.csv", index=False)

    # Plot measured I-V curve on linear and logarithmic current scales.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(data["v_meas"], data["i_meas"], marker=".", linewidth=1.2)
    axes[0].set(title="Measured diode I–V characteristic", xlabel="Measured voltage (V)",
                ylabel="Measured current (A)")
    axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(data["v_meas"], data["i_meas"], marker=".", linewidth=1.2)
    axes[1].set(title="I–V characteristic (log current scale)", xlabel="Measured voltage (V)",
                ylabel="Measured current (A, log scale)")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "iv_characteristic.png", dpi=180)
    plt.close(fig)

    # Hold out the middle 40th–60th percentile voltage interval.
    low, high = data["v_meas"].quantile([0.40, 0.60]).to_numpy()
    test_mask = data["v_meas"].between(low, high, inclusive="both")
    train = data.loc[~test_mask]
    test = data.loc[test_mask]
    if len(test) < 3 or len(train) < 10:
        raise ValueError("The selected train/test split is too small for reliable comparison.")

    X_train = train[["v_meas"]].to_numpy()
    y_train = train["i_meas"].to_numpy()
    X_test = test[["v_meas"]].to_numpy()
    y_test = test["i_meas"].to_numpy()

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }
    rows = []
    predictions = pd.DataFrame({"voltage_V": X_test.ravel(), "actual_current_A": y_test})
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "MAE_A": mean_absolute_error(y_test, pred),
            "RMSE_A": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        })
        predictions[f"{name}_prediction_A"] = pred
        predictions[f"{name}_residual_A"] = y_test - pred

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    predictions.to_csv(OUT / "test_predictions.csv", index=False)

    # Prediction plot shows training observations, held-out observations, and predictions.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(train["v_meas"], train["i_meas"], s=18, alpha=0.65, label="Training data")
    ax.scatter(test["v_meas"], y_test, s=28, marker="x", label="Held-out actual data")
    order = np.argsort(X_test.ravel())
    for name in models:
        ax.plot(X_test.ravel()[order], predictions[f"{name}_prediction_A"].to_numpy()[order],
                marker=".", linewidth=1.5, label=f"{name} prediction")
    ax.axvspan(low, high, alpha=0.10, label="Held-out voltage interval")
    ax.set(title="Predictions in the held-out voltage interval",
           xlabel="Measured voltage (V)", ylabel="Current (A)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "model_predictions.png", dpi=180)
    plt.close(fig)

    # Residuals: actual minus predicted current.
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for name in models:
        ax.scatter(predictions["voltage_V"], predictions[f"{name}_residual_A"],
                   s=24, alpha=0.75, label=name)
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set(title="Residuals on held-out observations", xlabel="Measured voltage (V)",
           ylabel="Residual current (A)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "residuals.png", dpi=180)
    plt.close(fig)

    summary = [
        "DIode I-V and machine-learning analysis",
        f"Rows in source file: {len(raw)}",
        f"Rows after cleaning: {len(data)}",
        f"Training rows: {len(train)}",
        f"Held-out rows: {len(test)}",
        f"Held-out voltage interval: {low:.6f} V to {high:.6f} V",
        "",
        metrics.to_string(index=False),
        "",
        "MAE and RMSE are in amperes. R2 may be negative. Interpret these results only",
        "for this dataset and this held-out voltage interval.",
    ]
    (OUT / "run_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"\nSaved plots and CSV results in: {OUT}")


if __name__ == "__main__":
    main()
