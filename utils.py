import pandas as pd
import json

# Takes input of attribute-values, and returns a dict of example to feed the model
def create_example(data: dict) -> dict:
    example = {}
    n = len(data['Attribute'])
    example['question'] = data['Attribute']
    example['context'] = [' '.join(data['Value'])] * n
    example['id'] = list(range(n))
    return example

# Takes result output by the model and input data, and generates a DataFrame table to visualize results
def display_result(res: dict, data: dict) -> pd.DataFrame:
    new_data = {}
    new_data['Attribute'] = data['Attribute']
    new_data['Value'] = res.values()
    return pd.DataFrame(new_data)

def read_examples(file_name: str) -> list:
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_changes(original_df, predicted_df):
    predicted_df['changes'] = (original_df['Value'] != predicted_df['Value'])
    return predicted_df

def get_all_changes(original_df: pd.DataFrame, predicted_df: pd.DataFrame):
    changes = []
    for i, row in predicted_df.iterrows():
        if row['Value'] == '':
            changes.append(1)
        elif row['Value'] != original_df.iloc[i]['Value']:
            changes.append(2)
        else:
            changes.append(0)
    predicted_df['changes'] = changes
    return predicted_df

def highlight_changes(x):
    c = f"background-color:darkred" 
    #condition
    m = x["changes"]
    # DataFrame of styles
    df1 = pd.DataFrame('', index=x.index, columns=['Attribute', 'Value'])
    # set columns by condition
    df1.loc[m, 'Attribute'] = c
    x.drop(columns=['changes'], inplace=True)
    return df1

def highlight_all_changes(x):
    r = f"background-color:red" 
    b = f"background-color:green" 
    #condition
    m = x["changes"] == 1
    n = x['changes'] == 2
    # DataFrame of styles
    df1 = pd.DataFrame('', index=x.index, columns=['Attribute', 'Value'])
    # set columns by condition
    df1.loc[m, 'Attribute'] = r
    df1.loc[n, 'Attribute'] = b
    x.drop(columns=['changes'], inplace=True)
    return df1

    