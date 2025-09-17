import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        bytes_data = uploaded_file.read()
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(bytes_data))
        else:
            df = pd.read_excel(io.BytesIO(bytes_data))
        return df
    except Exception as e:
        return None

def get_dataframe_info(df: pd.DataFrame):
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'dtypes': df.dtypes.apply(lambda x: str(x)).to_dict()
    }


def null_counts(df: pd.DataFrame):
    return df.isnull().sum().to_frame('null_count')


def column_types(df: pd.DataFrame):
    return df.dtypes.apply(lambda x: str(x)).to_frame('dtype')


def drop_columns(df: pd.DataFrame, cols):
    return df.drop(columns=cols)


def fill_column(df: pd.DataFrame, col, strategy='median', value=None):
    if strategy == 'median':
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
    elif strategy == 'mean':
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
    elif strategy == 'mode':
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else value)
    elif strategy == 'value' and value is not None:
        df[col] = df[col].fillna(value)
    return df


def corr_pvalues(df: pd.DataFrame):
    # compute correlation p-values for numeric columns
    num = df.select_dtypes(include=[np.number])
    cols = num.columns
    pvals = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            try:
                _, p = stats.pearsonr(num[cols[i]].dropna(), num[cols[j]].dropna())
            except Exception:
                p = np.nan
            pvals.iloc[i, j] = p
            pvals.iloc[j, i] = p
    return pvals


def feature_pvalues_vs_target(df: pd.DataFrame, target: str):
    # For numeric features vs numeric target: Pearson p-values
    num = df.select_dtypes(include=[np.number])
    if target not in num.columns:
        return None
    pvals = {}
    for col in num.columns:
        if col == target:
            continue
        try:
            _, p = stats.pearsonr(num[col].dropna(), num[target].dropna())
        except Exception:
            p = np.nan
        pvals[col] = p
    return pd.Series(pvals).to_frame('p_value')


def correlation_matrix(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).corr()


def pairplot_fig(df: pd.DataFrame, cols=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
    fig = plt.figure(figsize=(8,8))
    sns.pairplot(df[cols].dropna())
    return fig

def basic_cleaning(df: pd.DataFrame):
    # fill numeric NA with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    return df

def describe_df(df: pd.DataFrame):
    return df.describe(include='all')

def plot_histogram(df, col):
    fig, ax = plt.subplots()
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    return fig

def plot_boxplot(df, col):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    return fig

def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, ax=ax)
    return fig

def plot_time_series(df, time_col, value_col):
    fig, ax = plt.subplots()
    df2 = df.copy()
    df2[time_col] = pd.to_datetime(df2[time_col])
    df2 = df2.sort_values(time_col)
    ax.plot(df2[time_col], df2[value_col])
    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    return fig

def plot_bubble(df, x, y, size):
    fig, ax = plt.subplots()
    sc = ax.scatter(df[x], df[y], s=(df[size].fillna(0).astype(float).abs()+1)*10, alpha=0.6)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return fig

def calculate_vif(df):
    X = df.select_dtypes(include=[np.number]).dropna()
    if X.shape[1] < 2:
        return pd.DataFrame({'variable':[], 'vif':[]})
    vif_data = pd.DataFrame()
    vif_data['variable'] = X.columns
    vif_data['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data
