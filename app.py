import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_app.utils import (load_data, get_dataframe_info, basic_cleaning, calculate_vif,
                   describe_df, plot_histogram, plot_boxplot, plot_heatmap,
                   plot_time_series, plot_bubble, null_counts, column_types,
                   drop_columns, fill_column, corr_pvalues,
                   feature_pvalues_vs_target, correlation_matrix, pairplot_fig)
from streamlit_app.models import ModelRunner


st.set_page_config(page_title="NoCode ML Studio", layout="wide")

def main():
    st.title("NoCode Low-Code ML Studio")

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])
        sample = st.checkbox("Use sample dataset (iris)")
        menu = option_menu(None, ["Data","EDA","Modeling"],
                           icons=["file-earmark","bar-chart","gear"], menu_icon="cast",
                           default_index=0)

    # session state for dataframe
    if 'df' not in st.session_state:
        st.session_state['df'] = None

    if sample and uploaded is None:
        import seaborn as sns
        df = sns.load_dataset('iris')
    else:
        df = load_data(uploaded)

    # if a new df is loaded, store in session_state
    if df is not None:
        st.session_state['df'] = df

    if st.session_state.get('df') is None:
        st.info("Upload a dataset to begin or check 'Use sample dataset'")
        return

    df = st.session_state['df']

    if menu == "Data":
        st.header("Dataset preview & cleaning")
        info = get_dataframe_info(df)
        st.metric("Rows", info['rows'])
        st.metric("Columns", info['columns'])
        st.subheader("Dataframe dtypes")
        st.dataframe(column_types(df))
        st.subheader("Null counts")
        st.dataframe(null_counts(df))
        st.subheader("Preview (first 50 rows)")
        st.dataframe(df.head(50))

        st.subheader("Cleaning operations")
        cols = list(df.columns)
        to_drop = st.multiselect("Select columns to drop", cols)
        if st.button("Drop selected columns"):
            if to_drop:
                df = drop_columns(df, to_drop)
                st.session_state['df'] = df
                st.success(f"Dropped columns: {to_drop}")

        st.write("Fill missing values for a column")
        col_fill = st.selectbox("Column to fill", [None]+cols)
        if col_fill:
            strat = st.selectbox("Strategy", ['median','mean','mode','value'])
            val = None
            if strat == 'value':
                val = st.text_input("Value to use (as string)")
            if st.button("Apply fill"):
                df = fill_column(df, col_fill, strategy=strat, value=val)
                st.session_state['df'] = df
                st.success(f"Filled column {col_fill} using {strat}")

        if st.button("Fill all numeric NAs with median"):
            df = basic_cleaning(df)
            st.session_state['df'] = df
            st.success("Filled numeric NAs with median")

        st.write("Advanced helpers")
        if st.button("Show VIF (numeric features)"):
            vif = calculate_vif(df)
            st.dataframe(vif)
        if st.button("Show correlation p-values (numeric)"):
            pvals = corr_pvalues(df)
            st.dataframe(pvals)

    elif menu == "EDA":
        st.header("Exploratory Data Analysis")
        cols = st.multiselect("Columns to describe", list(df.columns), default=list(df.columns)[:5])
        st.write(describe_df(df[cols]))
        st.subheader("Visualizations")
        if st.checkbox("Histogram"):
            c = st.selectbox("Column", cols)
            st.pyplot(plot_histogram(df, c))
        if st.checkbox("Boxplot"):
            c = st.selectbox("Column for boxplot", cols)
            st.pyplot(plot_boxplot(df, c))
        if st.checkbox("Heatmap (correlation)"):
            st.pyplot(plot_heatmap(df[cols]))
        if st.checkbox("Correlation matrix"):
            st.dataframe(correlation_matrix(df))
        if st.checkbox("Feature p-values vs target"):
            target = st.selectbox("Select numeric target for p-values", options=[None]+list(df.columns))
            if target:
                pv = feature_pvalues_vs_target(df, target)
                st.dataframe(pv)
        if st.checkbox("Pairplot (first up to 5 numeric cols)"):
            st.pyplot(pairplot_fig(df))
        if st.checkbox("Time Series"):
            ts_col = st.selectbox("Time column", options=[None]+list(df.columns))
            val_col = st.selectbox("Value column", cols)
            if ts_col:
                st.pyplot(plot_time_series(df, ts_col, val_col))
        if st.checkbox("Bubble plot"):
            x = st.selectbox("X", cols)
            y = st.selectbox("Y", cols)
            size = st.selectbox("Size", cols)
            st.pyplot(plot_bubble(df, x, y, size))

    elif menu == "Modeling":
        st.header("Modeling")
        target = st.selectbox("Target column", df.columns)
        features = st.multiselect("Features", [c for c in df.columns if c!=target], default=[c for c in df.columns if c!=target][:5])
        task = st.radio("Task type", ["Regression","Classification","Time Series"])
        test_size = st.slider("Test size (fraction)", 0.05, 0.5, 0.2)
        run = st.button("Run models")
        if run:
            runner = ModelRunner(df[features+[target]], target, task)
            res = runner.run_all(test_size=test_size)
            st.write(res)


if __name__ == '__main__':
    main()
