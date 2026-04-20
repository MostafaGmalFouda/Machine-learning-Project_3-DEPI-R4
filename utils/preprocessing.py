from scipy import stats

def apply_boxcox(df, lambdas):
    df_copy = df.copy()

    for col in df.columns:
        lam = lambdas[col]
        df_copy[col] = stats.boxcox(df[col], lmbda=lam)

    return df_copy