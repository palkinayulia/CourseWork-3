import pandas as pd

def read_candidates_file(file_path):
    name = str(file_path).lower()

    if name.endswith(".xlsx"):
        return pd.read_excel(file_path)

    if name.endswith(".csv"):
        # 1) пробуем автоопределение разделителя
        try:
            return pd.read_csv(file_path, sep=None, engine="python")
        except Exception:
            pass

        # 2) пробуем ; (часто в русской локали)
        try:
            return pd.read_csv(file_path, sep=";")
        except Exception:
            pass

        # 3) пробуем , (классика)
        return pd.read_csv(file_path, sep=",")

    raise ValueError("Поддерживаются только CSV и XLSX")