import json
import pandas as pd


# Takes input of attribute-values, and returns a dict of example to feed the model
def create_example(data: dict) -> dict:
    example = {}
    n = len(data['Attribute'])
    example['question'] = data['Attribute']
    example['context'] = [' '.join(data['Value'])] * n
    example['id'] = list(range(n))
    return example

def example_from_title(data: dict) -> dict:
    example = {}
    n = len(data['Attribute'])
    example['question'] = data['Attribute']
    example['context'] = [data['title']] * n
    example['id'] = list(range(n))
    return example

def create_new_example(data: dict) -> dict:
    example = {}
    n = len(data['new_attributes'])
    example['question'] = data['new_attributes']
    example['context'] = [data['title']] * n
    example['id'] = list(range(n))
    return example

# Takes result output by the model and input data, and generates a DataFrame table to visualize results
def display_result(res: dict, data: dict) -> pd.DataFrame:
    new_data = {}
    new_data['Attribute'] = data['Attribute']
    new_data['Value'] = res.values()
    return pd.DataFrame(new_data)

def display_new_result(res: dict, data:dict) -> pd.DataFrame:
    new_data = {}
    new_data['Attribute'] = data['new_attributes']
    new_data['Value'] = res.values()
    new_data['dump'] = res.values()
    new_df = pd.DataFrame(new_data)
    return new_df.loc[new_df['Value'] != '']

def read_examples(file_name: str) -> list:
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def combine_results(res_df1: pd.DataFrame, res_df2: pd.DataFrame ,new_df: pd.DataFrame) -> pd.DataFrame:
    list_of_lists = []
    for index, row in res_df1.iterrows():
        my_list = []
        value_list = []
        my_list.append(row['Attribute'])
        value_list.append(row['Value'])
        if row['Value'] == res_df2.iloc[index]['Value']:
            my_list.append(row['Value'])
        else:
            value_list.append(res_df2.iloc[index]['Value'])
            try:
                value_list.remove('')
            except:
                pass
            my_list.append(",".join(value_list))
        list_of_lists.append(my_list)
    
    my_df = pd.DataFrame(list_of_lists, columns=['Attribute', 'Value'])
    return pd.concat([my_df, new_df], ignore_index=True)

# Function to aggregate tables after normalisation
def aggregate_normalised_tables(df1: pd.DataFrame, df2: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    result_df = pd.merge(df1.drop(columns='dummy'), df2.drop(columns='dummy'), on='Attribute', how='outer')
    result_df['dummy'] = result_df['Attribute']
    result_df.rename(columns={'Value_x': col1, 'Value_y': col2}, inplace=True)
    return result_df


    