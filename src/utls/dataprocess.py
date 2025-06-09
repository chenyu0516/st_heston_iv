import pandas as pd
import ast

def datatype_trans(df, time_list, data_list):
    for col in time_list:
        df[col] = df[col].apply(pd.to_datetime)
    
    for col in data_list:
        df[col] = df[col].apply(ast.literal_eval)
        
    return df