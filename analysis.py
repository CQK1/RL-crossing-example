import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression

# Plot settings
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


def fit_gam_spline(x: np.ndarray, y: np.ndarray, df: int = 7):
    """
    Fits a Generalized Additive Model (GAM) style B-spline smoothing curve.
    
    Parameters:
        x (np.ndarray): 1D array of time steps / indices.
        y (np.ndarray): 1D array of observed traffic counts.
        df (int): Degrees of freedom for spline knots.
        
    Returns:
        np.ndarray: Predicted / fitted traffic intensity values.
    """
    n_knots = max(3, df - 3 + 1)
    model = make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            extrapolation="continue"
        ),
        LinearRegression()
    )
    x_2d = x.reshape(-1, 1)
    model.fit(x_2d, y)
    return model.predict(x_2d)


def process_movement(df_raw: pd.DataFrame, name: str, col_idx: int):
    """
    Processes and plots traffic count data for a specific movement column.
    
    Parameters:
        df_raw (pd.DataFrame): Raw Excel dataframe.
        name (str): Movement identifier name.
        col_idx (int): Column index corresponding to the movement in df_raw.
    """
    print(f"\nProcessing: {name}")
    
    # Data rows start from index 11 through 490
    date = pd.to_datetime(
        df_raw.iloc[11:491, 0],
        errors="coerce"
    )
    counts = pd.to_numeric(
        df_raw.iloc[11:491, col_idx],
        errors="coerce"
    )
    df = pd.DataFrame({
        "date": date,
        "auto": counts
    })
    df = df.dropna().reset_index(drop=True)
    
    if df.empty:
        print("No valid data found, skipping...")
        return

    all_fitted_y = []
    processed_df = pd.DataFrame()
    unique_dates = df["date"].dt.date.unique()

    # Process each day independently to avoid cross-day discontinuity
    for current_date in unique_dates:
        day_data = df[
            df["date"].dt.date == current_date
        ].copy()
        day_data = day_data.sort_values("date")
        n_points = len(day_data)
        
        if n_points < 5:
            continue
            
        t = np.arange(1, n_points + 1)
        y = day_data["auto"].values
        
        # Split fit around 13:00 (midday pattern transition)
        split_mask = (
            day_data["date"].dt.time
            <= pd.to_datetime("13:00:00").time()
        )
        split_idx = np.sum(split_mask)
        
        if split_idx == 0 or split_idx == n_points:
            fitted = fit_gam_spline(
                t,
                y,
                df=min(7, n_points - 1)
            )
        else:
            t1 = t[:split_idx]
            y1 = y[:split_idx]

            t2 = t[split_idx:]
            y2 = y[split_idx:]

            fitted1 = fit_gam_spline(
                t1,
                y1,
                df=max(3, min(7, len(y1) - 1))
            )

            fitted2 = fit_gam_spline(
                t2,
                y2,
                df=max(3, min(7, len(y2) - 1))
            )

            fitted = np.concatenate([fitted1, fitted2])

        all_fitted_y.extend(fitted)
        processed_df = pd.concat([processed_df, day_data], ignore_index=True)

    plt.figure(figsize=(14, 6))

    plt.scatter(
        processed_df["date"],
        processed_df["auto"],
        s=20,
        alpha=0.7,
        label="Observed counts"
    )

    plt.plot(
        processed_df["date"],
        all_fitted_y,
        linewidth=2,
        label="Estimated intensity"
    )

    plt.title(
        f"{name} (5-Day Overview)",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    
    filename = f"Fit_{name}.png"
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {filename}")


def main():
    file_path = r"data\Mayor Magrath Drive & 5 Avenue S_Binned_20260524170346-1.xlsx"

    print("Reading file:", file_path)

    try:
        df_raw = pd.read_excel(
            file_path,
            header=None
        )
    except Exception as e:
        print("Failed to read file:", e)
        return

    # Excel column mapping:
    #
    # North:
    # 1 Right
    # 2 Thru
    # 3 Left
    #
    # East:
    # 7 Right
    # 8 Thru
    # 9 Left
    #
    # South:
    # 13 Right
    # 14 Thru
    # 15 Left
    #
    # West:
    # 19 Right
    # 20 Thru
    # 21 Left

    movements = [
        ("North_Southbound_Right", 1),
        ("North_Southbound_Thru", 2),
        ("North_Southbound_Left", 3),

        ("East_Westbound_Right", 7),
        ("East_Westbound_Thru", 8),
        ("East_Westbound_Left", 9),

        ("South_Northbound_Right", 13),
        ("South_Northbound_Thru", 14),
        ("South_Northbound_Left", 15),

        ("West_Eastbound_Right", 19),
        ("West_Eastbound_Thru", 20),
        ("West_Eastbound_Left", 21),
    ]

    print(f"Starting processing for {len(movements)} movements...")

    for name, col in movements:
        process_movement(
            df_raw,
            name,
            col
        )

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()