import pandas as pd

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
    r = f"background-color:orange" 
    b = f"color:green"
    f = f'background-color:yellow'
    #condition
    m = x["changes"] == 1
    n = x['changes'] == 2
    # DataFrame of styles
    df1 = pd.DataFrame('', index=x.index, columns=['Attribute', 'Value'])
    # set columns by condition
    df1.loc[m, 'Attribute'] = r
    df1.loc[n, 'Attribute'] = f
    df1.loc[n, 'Value'] = b   
    return df1