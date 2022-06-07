import json
import requests 

import pandas as pd
from selectorlib import Extractor

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
    new_df = pd.DataFrame(new_data)
    return new_df.loc[new_df['Value'] != '']

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
    b = f"color:green"
    f = f'background-color:darkorange'
    #condition
    m = x["changes"] == 1
    n = x['changes'] == 2
    # DataFrame of styles
    df1 = pd.DataFrame('', index=x.index, columns=['Attribute', 'Value'])
    # set columns by condition
    df1.loc[m, 'Attribute'] = r
    df1.loc[n, 'Attribute'] = f
    df1.loc[n, 'Value'] = b   
    x.drop(columns=['changes'], inplace=True)
    return df1

# Get data from an amazon url
def scrape(url):  
    e = Extractor.from_yaml_file('template.yml')
    headers = {
        'dnt': '1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-user': '?1',
        'sec-fetch-dest': 'document',
        'referer': 'https://www.amazon.com/',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    }

    # Download the page using requests
    print("Downloading %s"%url)
    r = requests.get(url, headers=headers)
    # Simple check to check if page was blocked (Usually 503)
    if r.status_code > 500:
        if "To discuss automated access to Amazon data please contact" in r.text:
            print("Page %s was blocked by Amazon. Please try using better proxies\n"%url)
        else:
            print("Page %s must have been blocked by Amazon as the status code was %d"%(url,r.status_code))
        return None
    # Pass the HTML of the page and create 
    return e.extract(r.text)

# Function to process results
def extract_data(url):
    data = scrape(url)
    new_data = {}
    new_data['image'] = data['Image']
    new_data['Attribute'] = []
    new_data['Value'] = []
    new_data['title'] = data['Info']['Title']
    for i in range(1, 10):
        attr = 'attr' + str(i)
        val = 'val' + str(i)
        if data['Info'][attr] != None:
            new_data['Attribute'].append(data['Info'][attr])
            new_data['Value'].append(data['Info'][val])
    return new_data

    