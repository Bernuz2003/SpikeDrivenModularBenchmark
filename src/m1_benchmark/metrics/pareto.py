from __future__ import annotations

import pandas as pd


def pareto_front(df: pd.DataFrame, acc_col: str = 'accuracy_top1', cost_cols: list[str] | None = None) -> pd.DataFrame:
    cost_cols = cost_cols or ['total_sops_proxy', 'total_state_mem_bits', 'total_spike_count']
    if df.empty:
        return df.copy()
    keep = []
    for i, a in df.iterrows():
        dominated = False
        for j, b in df.iterrows():
            if i == j:
                continue
            better_acc = b[acc_col] >= a[acc_col]
            lower_costs = all(b[c] <= a[c] for c in cost_cols if c in df.columns)
            strict = (b[acc_col] > a[acc_col]) or any(b[c] < a[c] for c in cost_cols if c in df.columns)
            if better_acc and lower_costs and strict:
                dominated = True
                break
        keep.append(not dominated)
    return df.loc[keep].copy()
