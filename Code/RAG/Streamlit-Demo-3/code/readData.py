import pandas as pd

def loadData(path, fileType):
    match fileType:
        case "excel":
            return pd.read_excel(path, na_values=['NA'], skiprows=1)
        case "json":
            return pd.read_json(path)
        case "jsonl":
            return pd.read_json(path, lines=True)
        case "csv": 
            return pd.read_csv(path)