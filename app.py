from attribute_extractor import AttributeExtractor
from utils import *
from highlighter import get_all_changes, highlight_all_changes
from scraper import extract_data
from examples import EXAMPLES
from normaliser import AttributeNormaliser


import requests
from PIL import Image
import re

import streamlit as st
from annotated_text import annotated_text
import numpy as np
import pandas as pd

_DISTILBERT = 'ksabeh/distilbert-attribute-correction'
_BERT = 'ksabeh/bert-base-uncased-attribute-correction'
_ROBERTA = 'ksabeh/roberta-base-attribute-correction'
_ALBERT = 'ksabeh/albert-base-v2-attribute-correction'
_XLNET = 'ksabeh/xlnet-base-cased-attribute-correction'

prompts = list(EXAMPLES.keys())

@st.cache(allow_output_mutation=True)
def load_model(model_ckpt: str) -> AttributeExtractor:
    return AttributeExtractor(model_ckpt)

def load_data(path: str) -> list:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def choose_example(raw_data: dict) -> dict:
    return {key: value for key, value in raw_data.items() if key in ['Attribute', 'Value']}

def create_ner_list(result_df: pd.DataFrame, title: str) -> list:
    my_title = title.replace(',', '')
    my_title = my_title.replace('"', '')
    title_list = my_title.split()
    new_list = []
    my_dict = dict(zip(result_df['Value'].tolist(), result_df['Attribute'].tolist()))
    for element in title_list:
        if element in my_dict.keys():
            new_list.append((element, my_dict[element]))
        else:
            new_list.append(element + ' ')
    return new_list

def display_product(raw_data: dict) -> None:
    st.markdown(f"#### {raw_data['title']}")
    st.markdown('')
    image_url = raw_data['image']
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((400,300))
    st.image(image)

def data_from_prompt(raw_data: dict) -> dict:
    title_example = example_from_title(raw_data)
    new_example = create_new_example(raw_data)
    selected_example = choose_example(raw_data)
    example = create_example(selected_example)
    df = pd.DataFrame.from_dict(selected_example)
    display_product(raw_data)
    st.table(df)
    return {'df': df, 'example': example, 'selected_example': selected_example,
    'raw_data': raw_data, 'title_example': title_example, 'new_example': new_example}

def data_from_link(amazon_link: str) -> dict:
    if amazon_link:
        try:
            with st.spinner('Collecting data..'):
                raw_data = extract_data(amazon_link)
        except:
            st.warning('Please input a valid link..')
            st.stop()
    else:
        st.stop()
    raw_data['new_attributes'] = ['Type']
    title_example = example_from_title(raw_data)
    selected_example = choose_example(raw_data)
    example = create_example(selected_example)
    df = pd.DataFrame.from_dict(selected_example)
    display_product(raw_data)
    st.table(df)
    return {'df': df, 'example': example, 'selected_example': selected_example, 
    'raw_data': raw_data, 'title_example': title_example}

def choose_model(config: dict) -> AttributeExtractor:
    if  config['chosen_model'] == 'DistilBERT':
        model_ckpt = _DISTILBERT
    elif config['chosen_model'] == 'RoBERTa':
        model_ckpt = _ROBERTA
    elif config['chosen_model'] == 'BERT':
        model_ckpt = _BERT
    elif config['chosen_model'] == 'ALBERT':
        model_ckpt = _ALBERT
    elif config['chosen_model'] == 'XLNET':
        model_ckpt = _XLNET
    
    if config['masked_language']:
        model_ckpt += '-mlm'

    return load_model(model_ckpt), model_ckpt

def align_text():
    for _ in range(9):
        st.markdown('')
    
def get_results(model: AttributeExtractor, res: dict, config: dict) -> pd.DataFrame:
    results = model.predict(res['example'], negative= not config['correct'], min_null_score=config['null_score'])
    res_df = display_result(results, res['selected_example'])
    return get_all_changes(res['df'], res_df)

def get_title_results(model: AttributeExtractor, res: dict) -> pd.DataFrame:
    results = model.predict(res['title_example'])
    res_df = display_result(results, res['selected_example'])
    return get_all_changes(res['df'], res_df)

def get_new_results(model: AttributeExtractor, res: dict) -> dict:
    results = model.predict(res['new_example'])
    return display_new_result(results, res['raw_data'])

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

def submit_on_callback():
    st.session_state.submit_clicked = True

def submit_off_callback():
    st.session_state.submit_clicked = False

def submit_compare_on_callback():
    st.session_state.compare_submit_clicked = True

def use_mlm_callback():
    st.session_state.mlm = True

def new_link_callback():
    st.session_state.new_link = True

def submit_compare_off_callback():
    st.session_state.compare_submit_clicked = False


def compare_products(config: dict) -> None:
    if 'compare_submit_clicked' not in st.session_state:
        st.session_state.compare_submit_clicked = False

    attribute_data = load_data('resources/attributes.jsonl')
    with st.container():
        st.markdown("## 📌 Compare Two Products")
        col1, _, col2 = st.columns([1, 0.2, 1])
        with col1:
            prompt = st.selectbox(
            'Examples (select from this list)',
            prompts,
            index=9,
            help='Choose an example from the list or input your own example',
            key='product_1',
            on_change=submit_compare_off_callback
            )
            if prompt == 'Custom':
                amazon_link = st.text_area('Insert a link for the first product:')
                if 'res_1' not in st.session_state:
                    st.session_state.res_1 = data_from_link(amazon_link)
                else:
                    display_product(st.session_state.res_1['raw_data'])
                    st.table(st.session_state.res_1['df'])
                my_res_1 = st.session_state.res_1
            else:
                # align_text()
                raw_data_1 = EXAMPLES[prompt]
                my_res_1 = data_from_prompt(raw_data_1)
        
        with col2:
            prompt = st.selectbox(
            'Examples (select from this list)',
            prompts,
            index=10,
            help='Choose an example from the list or input your own example',
            key='product_2  '
            )

            if prompt == 'Custom':
                amazon_link = st.text_area('Insert a link for the second product:')
                if 'res_2' not in st.session_state:
                    st.session_state.res_2 = data_from_link(amazon_link)
                else:
                    display_product(st.session_state.res_2['raw_data'])
                    st.table(st.session_state.res_2['df'])
                my_res_2 = st.session_state.res_2
            else:
                # align_text()
                raw_data_2 = EXAMPLES[prompt]
                my_res_2 = data_from_prompt(raw_data_2)
        submit = st.button('✨ Correct and compare data!', on_click=submit_compare_on_callback)
    with st.container():
        if st.session_state.compare_submit_clicked:
            st.markdown("## 🎈 Check & download results")
            with st.spinner("Correcting data..."):
                    model, _ = choose_model(config)
                    # st.text(f"Using model {model.return_checkpoint()}")
                    res_df_1 = get_results(model, my_res_1, config)
                    res_df_2 = get_results(model, my_res_2, config)

                    st.markdown("## Attribute correction")
                    col_1, _, col_2 = st.columns([1, 0.2, 1])
                    
                    with col_1:
                        st.table(res_df_1.style.apply(highlight_all_changes, axis=None))

                    with col_2:
                        st.table(res_df_2.style.apply(highlight_all_changes, axis=None))

                    if config['normalise_attributes']:

                        st.markdown("## Attribute normalisation")
                        c1, _, c2 = st.columns([1, 0.2, 1])
                        with c1:
                            attribute_normaliser_1 = AttributeNormaliser(res_df_1, attribute_data)
                            normalised_df_1 = attribute_normaliser_1.normalise_attributes(algorithm=config['simalirity_alg'], threshold=config['threshold'])
                            st.table(normalised_df_1)
                        with c2:
                            attribute_normaliser_2 = AttributeNormaliser(res_df_2, attribute_data)
                            normalised_df_2 = attribute_normaliser_2.normalise_attributes(algorithm=config['simalirity_alg'], threshold=config['threshold'])
                            st.table(normalised_df_2)
                        
                        # st.markdown('## Attribute comparison')
                        # res_df = pd.DataFrame.from_dict(RES)
                        # st.table(res_df)
                        # csv = convert_df(res_df)
                        # st.download_button(
                        # "📥 Download (.csv)",
                        # csv,
                        # f'comparison.csv',
                        # "text/csv",
                        # key='download_csv',
                        # )


def correct_products(config: dict) -> None:
    if 'submit_clicked' not in st.session_state:
        st.session_state.submit_clicked = False
    if 'new_link' not in st.session_state:
        st.session_state.new_link = True
    with st.container():
        st.markdown("## 📌 An Example")
        # cl1, _, cl2, = st.columns([1, 0.2, 1])
        # with cl1:
        prompt = st.selectbox(
            'Examples (select from this list)',
            prompts,
            index=0,
            help='Choose an example from the list or input your own example',
            key='select_example',
            on_change=submit_off_callback
        )

        if prompt == 'Custom':
            amazon_link = st.text_area('Insert a valid amazon link here:', key='amazon_link', on_change=new_link_callback())
            if 'res' not in st.session_state:
                st.session_state.res = data_from_link(amazon_link)
            else:
                display_product(st.session_state.res['raw_data'])
                st.table(st.session_state.res['df'])
            my_res = st.session_state.res
        else:
            raw_data = EXAMPLES[prompt]
            my_res = data_from_prompt(raw_data)
    model, model_ckpt = choose_model(config)
    if config['use_title']:
        title_model = load_model(model_ckpt + '-titles')
    submit = st.button('✨ Correct data!', key='submit_button', on_click=submit_on_callback)
    with st.container():
        if(st.session_state.submit_clicked):
            st.markdown("## 🎈 Check & download results")
            with st.spinner("Correcting data..."):
                
                # st.text(f"Using model {model.return_checkpoint()}")
                if config['use_title']:
                    c1, _, c2 = st.columns([1, 0.2, 1])
                    with c1:
                        st.markdown("### 📋 Using attributes")
                        res_df = get_results(model, my_res, config)
                        st.table(res_df.style.apply(highlight_all_changes, axis=None))
                    with c2:
                        st.markdown("### 🔍 Using product titles")
                        res_title_df = get_title_results(title_model, my_res)
                        st.table(res_title_df.style.apply(highlight_all_changes, axis=None))
                
                else:
                    res_df = get_results(model, my_res, config)
                    st.table(res_df.style.apply(highlight_all_changes, axis=None))

                if (config['generate_attributes']):
                    st.markdown("### 🤔 Extract new attributes")
                    extracted_attributes = get_new_results(title_model, my_res)
                    my_list = create_ner_list(extracted_attributes, raw_data['title'])
                    annotated_text(*my_list)
                    st.table(extracted_attributes)

                    st.markdown("### 📃 All attributes")
                    res_all_df = combine_results(res_df, res_title_df, extracted_attributes)
                    csv = convert_df(res_all_df)
                    st.table(res_all_df)
                    st.download_button(
                     "📥 Download (.csv)",
                    csv,
                    f'{prompt}.csv',
                    "text/csv",
                    key='download_csv',
                    )
                    

st.set_page_config(
    layout='wide',
    page_title='CAVE: Correcting Attribute Values in E-commerce Profiles',
    page_icon='🛒'
)
st.markdown("<h1 style='text-align: center; color: black;'>⛰️ CAVE: Correcting Attribute Values in E-commerce Profiles</h1>", unsafe_allow_html=True)


with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   CAVE corrects attribute values by exploiting information from both titles and attribute tables.
- It supplements the attribute tables with newly extracted attributes and their corresponding values.
- It normalises product attributes to compare between product profiles.
        """
        )


                    
                

def main():
    st.sidebar.markdown("## 💡 Mode")
    config = {}
    _MODE = st.sidebar.selectbox("App mode",
            index=0,
            options=['Correct attributes', 'Compare products'],
            help='Tick this box if you want to compare between two products.'
        )
    config['mode'] = _MODE
    st.sidebar.markdown("## ⚙️ Model Configuration")

    _CHOSEN_MODEL = st.sidebar.selectbox("Choose model", 
        index=0, 
        options=['DistilBERT', 'BERT', 'RoBERTa', 'ALBERT', 'XLNET'],
        help="At present, you can choose between 2 models (RoBERTa or DistilBERT) to embed your text. More to come!")
    config['chosen_model'] = _CHOSEN_MODEL

    _MASKED_LANGUAGE = st.sidebar.checkbox(
        "Use language model",
        help="Tick this box if you want to use the models trained on a masked language task.",
        value= True,
        key='mlm'
    )
    config['masked_language'] = _MASKED_LANGUAGE

    # _NUM_RES = st.sidebar.slider(
    #     'Number of outputs',
    #     min_value=1,
    #     max_value=5,
    #     value=1,
    #     help="""You can choose the number of predictions to display."""
    #         )
    # config['num_res'] = _NUM_RES

    _CORRECT = st.sidebar.checkbox(
        "Always correct",
        help="Tick this box if you want to force the model to generate a prediction."
        )
    config['correct'] = _CORRECT
    
    _NULL_SCORE = st.sidebar.slider(
        'Minimum null score',
        min_value=0,
        max_value=20,
        value=7,
        help="""You can set the sensitivity of the model to the error displays."""
            )
    config['null_score'] = _NULL_SCORE

    st.sidebar.markdown("## 🔧 Options")

    _USE_TITLES = st.sidebar.checkbox(
    "Use title",
    help="Tick this box if you want the model to use the information in the title for correction.",
    key='use_title',
    on_change=use_mlm_callback
    )
    config['use_title'] = _USE_TITLES

    _GENERATE_ATTRIBUTES = st.sidebar.checkbox(
    "Generate new attributes",
    help="Tick this box if you want the model to automatically extract new attributes."
    )
    config['generate_attributes'] = _GENERATE_ATTRIBUTES

    _NORMALISE_ATTRIBUTES = st.sidebar.checkbox(
    "Normalise attributes",
    help="Tick this box if you want the model to automatically normalise the attributes."
    )
    config['normalise_attributes'] = _NORMALISE_ATTRIBUTES

    _STRING_SIM = st.sidebar.selectbox("Normalisation algorithm", 
        index=0, 
        options=['Cosine similarity', 'Jaccard index', 'Sorensen–Dice coefficient'],
        help="Choose the normalisation algorithm to normalise the attributes.")
    config['simalirity_alg'] = _STRING_SIM

    _NORM_THRESHOLD = st.sidebar.slider(
    'Normalisation threshold',
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    help="""You can set the threshold for the normalisation of the attributes."""
        )
    config['threshold'] = _NORM_THRESHOLD
    if _MODE == 'Compare products':
        compare_products(config)
    else:
        correct_products(config)



if __name__ == '__main__':
    main()