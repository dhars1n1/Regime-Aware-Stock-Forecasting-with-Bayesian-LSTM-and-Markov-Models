#!/usr/bin/env python3
"""Compare two prediction CSVs and produce a set of comparison plots.

Usage:
  python compare_predictions.py \
    --pred1 data/bayesian_lstm_predictions_regularized.csv \
    --pred2 data/bayesian_lstm_predictions_regularized2.csv \
    --out output

The script will create `output/` and save PNGs and a CSV with merged data and computed metrics.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def find_common_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def prepare_df(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    # standardize column names we expect
    col_map = {}
    col_map['time_step'] = find_common_column(df, ['time_step', 'time', 'step', 'index'])
    col_map['actual'] = find_common_column(df, ['actual_log_return', 'actual', 'y_true', 'actual_return'])
    col_map['pred'] = find_common_column(df, ['predicted_mean', 'pred_mean', 'pred', 'prediction'])
    col_map['lower'] = find_common_column(df, ['lower_bound_95', 'lower', 'ci_lower'])
    col_map['upper'] = find_common_column(df, ['upper_bound_95', 'upper', 'ci_upper'])
    col_map['std'] = find_common_column(df, ['uncertainty_std', 'std', 'pred_std'])
    col_map['err'] = find_common_column(df, ['prediction_error', 'error', 'residual'])
    col_map['within'] = find_common_column(df, ['within_ci', 'in_ci'])

    rename = {}
    for logical, col in col_map.items():
        if col is not None:
            rename[col] = f"{logical}_{suffix}"

    df = df.rename(columns=rename)

    # compute missing columns if necessary
    if f'err_{suffix}' not in df.columns and f'actual_{suffix}' in df.columns and f'pred_{suffix}' in df.columns:
        df[f'err_{suffix}'] = df[f'actual_{suffix}'] - df[f'pred_{suffix}']
    if f'within_{suffix}' not in df.columns and f'lower_{suffix}' in df.columns and f'upper_{suffix}' in df.columns and f'actual_{suffix}' in df.columns:
        df[f'within_{suffix}'] = (df[f'actual_{suffix}'] >= df[f'lower_{suffix}']) & (df[f'actual_{suffix}'] <= df[f'upper_{suffix}'])
    if f'std_{suffix}' not in df.columns and f'lower_{suffix}' in df.columns and f'upper_{suffix}' in df.columns:
        # approximate std from 95% CI width: CI ~= mean +/- 1.96*std
        df[f'std_{suffix}'] = (df[f'upper_{suffix}'] - df[f'lower_{suffix}']) / (2 * 1.96)

    return df


def merge_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # try merging on time_step columns or index
    key1 = find_common_column(df1, [c for c in df1.columns if c.startswith('time_step') or c.endswith('_v1') or c.endswith('_v2')])
    # prefer explicit time_step_v1/time_step_v2
    if 'time_step_v1' in df1.columns:
        key1 = 'time_step_v1'
    if 'time_step_v2' in df2.columns:
        key2 = 'time_step_v2'
    else:
        # fallback to any column named time_step_*
        key2 = find_common_column(df2, [c for c in df2.columns if c.startswith('time_step')])

    if key1 is None or key2 is None:
        # merge by index
        df1 = df1.reset_index().rename(columns={'index': 'idx_v1'})
        df2 = df2.reset_index().rename(columns={'index': 'idx_v2'})
        merged = pd.concat([df1, df2], axis=1)
    else:
        merged = pd.merge(df1, df2, left_on=key1, right_on=key2, how='inner')
    return merged


def compute_metrics(merged: pd.DataFrame, s1: str, s2: str) -> dict:
    results = {}
    for s in (s1, s2):
        y_true = merged[f'actual_{s}']
        y_pred = merged[f'pred_{s}']
    mae = mean_absolute_error(y_true, y_pred)
    # compute RMSE without relying on the 'squared' kwarg for sklearn compatibility
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    within = None
    if f'within_{s}' in merged.columns:
        within = merged[f'within_{s}'].mean()
    results[s] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'coverage_95': within}
    return results


def plots(merged: pd.DataFrame, s1: str, s2: str, outdir: Path):
    sns.set(style='whitegrid')
    n = len(merged)

    # 1) Time series: actual and both predictions
    plt.figure(figsize=(14, 6))
    plt.plot(merged[f'time_step_{s1}'] if f'time_step_{s1}' in merged.columns else np.arange(n), merged[f'actual_{s1}'], label='actual', color='k', lw=1)
    plt.plot(merged[f'time_step_{s1}'] if f'time_step_{s1}' in merged.columns else np.arange(n), merged[f'pred_{s1}'], label=f'pred_{s1}', alpha=0.8)
    plt.plot(merged[f'time_step_{s1}'] if f'time_step_{s1}' in merged.columns else np.arange(n), merged[f'pred_{s2}'], label=f'pred_{s2}', alpha=0.8)
    plt.legend()
    plt.title('Time series: actual vs predictions')
    plt.xlabel('time_step')
    plt.tight_layout()
    plt.savefig(outdir / 'timeseries_actual_vs_preds.png', dpi=150)
    plt.close()

    # 2) Prediction errors over time
    plt.figure(figsize=(14, 5))
    plt.plot(merged[f'err_{s1}'], label=f'err_{s1}')
    plt.plot(merged[f'err_{s2}'], label=f'err_{s2}')
    plt.axhline(0, color='k', lw=0.6)
    plt.legend()
    plt.title('Prediction error (actual - pred) over time')
    plt.xlabel('index')
    plt.tight_layout()
    plt.savefig(outdir / 'errors_over_time.png', dpi=150)
    plt.close()

    # 3) Scatter: predicted vs actual
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, s in zip(axes, (s1, s2)):
        ax.scatter(merged[f'actual_{s1}'], merged[f'pred_{s}'], alpha=0.5, s=10)
        lim = np.nanpercentile(np.concatenate([merged[f'actual_{s1}'], merged[f'pred_{s}']]), [1, 99])
        ax.plot([lim[0], lim[1]], [lim[0], lim[1]], ls='--', color='gray')
        ax.set_title(f'predicted vs actual ({s})')
        ax.set_xlabel('actual')
        ax.set_ylabel('predicted')
    plt.tight_layout()
    plt.savefig(outdir / 'pred_vs_actual_scatter.png', dpi=150)
    plt.close()

    # 4) Error distributions
    plt.figure(figsize=(10, 5))
    sns.histplot(merged[f'err_{s1}'], label=s1, kde=True, stat='density', bins=50, color='C0', alpha=0.5)
    sns.histplot(merged[f'err_{s2}'], label=s2, kde=True, stat='density', bins=50, color='C1', alpha=0.5)
    plt.legend()
    plt.title('Error distribution (actual - pred)')
    plt.tight_layout()
    plt.savefig(outdir / 'error_distributions.png', dpi=150)
    plt.close()

    # 5) Absolute error boxplot / summary
    plt.figure(figsize=(8, 6))
    aes = pd.DataFrame({
        'abs_err': np.abs(merged[f'err_{s1}']),
        'model': s1
    })
    aes2 = pd.DataFrame({
        'abs_err': np.abs(merged[f'err_{s2}']),
        'model': s2
    })
    aes = pd.concat([aes, aes2], axis=0)
    sns.boxplot(x='model', y='abs_err', data=aes)
    plt.title('Absolute error boxplot')
    plt.tight_layout()
    plt.savefig(outdir / 'abs_error_boxplot.png', dpi=150)
    plt.close()

    # 6) CI width distributions
    if f'lower_{s1}' in merged.columns and f'upper_{s1}' in merged.columns:
        merged[f'ci_width_{s1}'] = merged[f'upper_{s1}'] - merged[f'lower_{s1}']
    if f'lower_{s2}' in merged.columns and f'upper_{s2}' in merged.columns:
        merged[f'ci_width_{s2}'] = merged[f'upper_{s2}'] - merged[f'lower_{s2}']

    widths = []
    if f'ci_width_{s1}' in merged.columns:
        widths.append((merged[f'ci_width_{s1}'].dropna(), s1))
    if f'ci_width_{s2}' in merged.columns:
        widths.append((merged[f'ci_width_{s2}'].dropna(), s2))
    if widths:
        plt.figure(figsize=(10, 5))
        for w, label in widths:
            sns.kdeplot(w, label=label)
        plt.title('95% CI width distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / 'ci_width_distribution.png', dpi=150)
        plt.close()

    # 7) Coverage bar
    cov = {}
    for s in (s1, s2):
        if f'within_{s}' in merged.columns:
            cov[s] = merged[f'within_{s}'].mean()
    if cov:
        plt.figure(figsize=(6, 4))
        sns.barplot(x=list(cov.keys()), y=list(cov.values()))
        plt.ylim(0, 1)
        plt.ylabel('observed coverage')
        plt.title('Observed 95% coverage')
        plt.tight_layout()
        plt.savefig(outdir / 'coverage_bar.png', dpi=150)
        plt.close()

    # 8) Uncertainty vs absolute error
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if f'std_{s1}' in merged.columns:
        plt.scatter(merged[f'std_{s1}'], np.abs(merged[f'err_{s1}']), alpha=0.5)
        plt.xlabel('uncertainty_std')
        plt.ylabel('abs(error)')
        plt.title(f'uncertainty vs abs error ({s1})')
    plt.subplot(1, 2, 2)
    if f'std_{s2}' in merged.columns:
        plt.scatter(merged[f'std_{s2}'], np.abs(merged[f'err_{s2}']), alpha=0.5, color='C1')
        plt.xlabel('uncertainty_std')
        plt.ylabel('abs(error)')
        plt.title(f'uncertainty vs abs error ({s2})')
    plt.tight_layout()
    plt.savefig(outdir / 'uncertainty_vs_abs_error.png', dpi=150)
    plt.close()

    # 9) Rolling RMSE comparison
    window = max(5, int(len(merged) * 0.05))
    merged['rmse_roll_1'] = merged[f'err_{s1}'].rolling(window).apply(lambda x: np.sqrt(np.mean(x**2)))
    merged['rmse_roll_2'] = merged[f'err_{s2}'].rolling(window).apply(lambda x: np.sqrt(np.mean(x**2)))
    plt.figure(figsize=(12, 5))
    plt.plot(merged['rmse_roll_1'], label=f'rolling_rmse_{s1}')
    plt.plot(merged['rmse_roll_2'], label=f'rolling_rmse_{s2}')
    plt.legend()
    plt.title(f'Rolling RMSE (window={window})')
    plt.tight_layout()
    plt.savefig(outdir / 'rolling_rmse.png', dpi=150)
    plt.close()

    # 10) Bland-Altman like plot (difference vs mean of predictions)
    plt.figure(figsize=(8, 6))
    mean_pred = 0.5 * (merged[f'pred_{s1}'] + merged[f'pred_{s2}'])
    diff = merged[f'pred_{s1}'] - merged[f'pred_{s2}']
    plt.scatter(mean_pred, diff, alpha=0.5)
    plt.axhline(diff.mean(), color='k', ls='--')
    plt.title('Difference between models vs mean prediction (Bland-Altman style)')
    plt.xlabel('mean(pred1, pred2)')
    plt.ylabel('pred1 - pred2')
    plt.tight_layout()
    plt.savefig(outdir / 'bland_altman.png', dpi=150)
    plt.close()

    # save merged with derived columns
    merged.to_csv(outdir / 'merged_predictions_with_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Compare two prediction CSVs and make plots')
    # Use the two prediction files directly (located next to this script)
    parser.add_argument('--out', default='output', help='output directory to save plots')
    args = parser.parse_args()

    # Resolve prediction file paths relative to this script's directory
    script_dir = Path(__file__).parent
    pred1_path = script_dir / 'bayesian_lstm_predictions_regularized.csv'
    pred2_path = script_dir / 'bayesian_lstm_predictions_regularized2.csv'

    # Validate files exist early with a clear error
    if not pred1_path.exists():
        raise FileNotFoundError(f"Expected prediction file not found: {pred1_path}")
    if not pred2_path.exists():
        raise FileNotFoundError(f"Expected prediction file not found: {pred2_path}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df1 = safe_read(str(pred1_path))
    df2 = safe_read(str(pred2_path))

    df1p = prepare_df(df1, 'v1')
    df2p = prepare_df(df2, 'v2')

    merged = merge_dfs(df1p, df2p)

    # Determine suffix names
    s1 = 'v1'
    s2 = 'v2'

    metrics = compute_metrics(merged, s1, s2)
    # print metrics and save
    print('Comparison metrics:')
    for s, m in metrics.items():
        print(s, m)
    pd.DataFrame(metrics).T.to_csv(outdir / 'summary_metrics.csv')

    plots(merged, s1, s2, outdir)

    print(f'All plots and CSVs saved to: {outdir.resolve()}')


if __name__ == '__main__':
    main()
