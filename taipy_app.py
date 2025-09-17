from taipy.gui import Gui
from streamlit_app.utils import load_data, describe_df, get_dataframe_info

page = """
# NoCode Taipy ML Studio (minimal)

Enter a local CSV/XLSX file path and click Load. (This simple Taipy UI reads files from the server filesystem.)

<|Enter file path:|file_path|input|>
<|Load dataset|load_button|button|on_action=load_action|>

<|{message}|text|>

<|Dataframe preview|df_preview|table|data={df_head}|>

<|Data info|df_info|markdown|>
"""


def load_action(state, var_name=None):
    path = state.file_path
    try:
        df = load_data_from_path(path)
        state.df = df
        state.df_head = df.head(50).to_dict(orient='records')
        info = get_dataframe_info(df)
        state.df_info = f"Rows: {info['rows']}  \\n+Columns: {info['columns']}"
        state.message = f"Loaded {info['rows']} rows and {info['columns']} columns"
    except Exception as e:
        state.message = f"Error: {e}"


def load_data_from_path(path):
    # small helper that mirrors utils.load_data but from a filesystem path
    import pandas as pd
    if path.lower().endswith('.csv'):
        return pd.read_csv(path)
    else:
        return pd.read_excel(path)


if __name__ == '__main__':
    gui = Gui(page)
    gui.run()
