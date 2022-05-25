from attribute_extractor import AttributeExtractor
from utils import create_example, display_result, read_examples, highlight_changes
from utils import get_changes, get_all_changes, highlight_all_changes, extract_data
from examples import EXAMPLES

import json
import os
import requests
from PIL import Image

import streamlit as st
import numpy as np
import pandas as pd

@st.cache(allow_output_mutation=True)
def load_model(model_ckpt):
    return AttributeExtractor(model_ckpt)


def main():
    prompts = list(EXAMPLES.keys()) + ["Custom"]
    st.set_page_config(
        layout='wide',
        page_title='Attributes in E-commerce: The good, The wrong and The ugly',
        page_icon='🛒'
    )

    def _max_width_():
        max_width_str = f"max-width: 800px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>    
        """,
            unsafe_allow_html=True,
        )

    _max_width_()
    st.markdown("<h1 style='text-align: center; color: black;'>Attributes in E-commerce: The good, The wrong and The ugly</h1>", unsafe_allow_html=True)


    with st.expander("ℹ️ - About this app", expanded=True):

        st.write(
            """     
    -   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for the amazing [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
    -   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers] (https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
            """
        )

    st.markdown("## ⚙️ Model Configuration")

    with st.container():

        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns([2, 0.3, 2, 2, 0.3, 1.5, 1.5, 1.5, 1.5])
        with c1:
            model = st.selectbox("Choose model", 
            index=0, 
            options=['DistilBERT', 'BERT', 'RoBERTa'],
            help="At present, you can choose between 2 models (RoBERTa or DistilBERT) to embed your text. More to come!")

        with c3:
            num_results = st.slider(
                    'Number of outputs',
                    min_value=1,
                    max_value=5,
                    value=1,
                    help="""You can choose the number of predictions to display."""
                )
        
        with c4:
            null_score = st.slider(
                    'Minimum score',
                    min_value=0,
                    max_value=10,
                    value=10,
                    help="""You can set the sensitivity of the model to the error displays."""
                )
            
        with c6:
            correct = st.checkbox(
                    "Allow correction",
                    help="Tick this box if you want to allow the app to predict a substitute for wrong attributes."
                )

        with c7:
            masked_language = st.checkbox(
                    "Use language model",
                    help="Tick this box if you want to use the models trained on a masked language task."
                ) 

        with c8:
            generate_attributes = st.checkbox(
                "Generate attributes",
                help="Tick this box if you want the model to automatically extract new attributes."
            )

        with c9:
            compare_products = st.checkbox(
                "Compare products",
                help='Tick this box if you want to compare between two products.'
            )
    
    if compare_products:
        st.markdown("## 📌 Compare Two Products")
        with st.container():
            col1, col_1, col2 = st.columns([1, 0.2, 1])

            with col1:
                amazon_link = st.text_area('Insert a link for first product:')
                if amazon_link:
                        try:
                            with st.spinner('Collecting data..'):
                                raw_data_1 = extract_data(amazon_link)
                        except:
                            st.warning('Please input a valid link...')
                            st.stop()
                else:
                    st.stop()
                
                selected_example_1 = {key: value for key, value in raw_data_1.items() if key in ['Attribute', 'Value']}
                example_1 = create_example(selected_example_1)
                df_1 = pd.DataFrame.from_dict(selected_example_1)
                st.markdown(f"#### {raw_data_1['title']}")
                st.markdown('')
                image_url = raw_data_1['image']
                image = Image.open(requests.get(image_url, stream=True).raw)
                image = image.resize((400,300))
                st.image(image)
                st.table(df_1)
            
            with col2:
                amazon_link = st.text_area('Insert a link for the second product:')
                if amazon_link:
                        try:
                            with st.spinner('Collecting data..'):
                                raw_data_2 = extract_data(amazon_link)
                        except:
                            st.warning('Please input a valid link...')
                            st.stop()
                else:
                    st.stop()
                
                selected_example_2 = {key: value for key, value in raw_data_2.items() if key in ['Attribute', 'Value']}
                example_2 = create_example(selected_example_2)
                df_2 = pd.DataFrame.from_dict(selected_example_2)
                st.markdown(f"#### {raw_data_2['title']}")
                st.markdown('')
                image_url = raw_data_1['image']
                image = Image.open(requests.get(image_url, stream=True).raw)
                image = image.resize((500,400))
                st.image(image)
                st.table(df_2)


    else:
        st.markdown("## 📌 An Example")
        with st.container():
            cl1, cl_1, cl2, = st.columns([1, 0.2, 1])

            with cl1:
                prompt = st.selectbox(
                    'Examples (select from this list)',
                    prompts,
                    index=0,
                    help="At present, you can only choose between 4 examples. More to come!"
                )
                if prompt == 'Custom':
                    amazon_link = st.text_area('Insert a valid amazon link here:')
                    if amazon_link:
                        try:
                            with st.spinner('Collecting data..'):
                                raw_data = extract_data(amazon_link)
                        except:
                            st.warning('Please input a valid link...')
                            st.stop()
                    else:
                        st.stop()
                else:
                    raw_data = EXAMPLES[prompt]
                selected_example = {key: value for key, value in raw_data.items() if key in ['Attribute', 'Value']}
                example = create_example(selected_example)
                df = pd.DataFrame.from_dict(selected_example)
                st.table(df)
            
            with cl2:
                st.markdown(f"#### {raw_data['title']}")
                st.markdown('')
                image_url = raw_data['image']
                image = Image.open(requests.get(image_url, stream=True).raw)
                image = image.resize((500,400))
                st.image(image)
            
            correct_button = st.button(label='✨ Correct the data!')

        
            if correct_button:
                st.markdown("## 🎈 Check & download results")
                with st.spinner("Correcting data..."):
                    if model == "DistilBERT":
                        model_ckpt = 'ksabeh/distilbert-attribute-correction'
                    else:
                        model_ckpt = 'ksabeh/distilbert-base-uncased-finetuned-attributes-qa'
                    
                    if masked_language:
                        model_ckpt += '-mlm'
                    

                    model = load_model(model_ckpt)
                    results = model.predict(example)
                    res_df = display_result(results, selected_example)
                    res_df = get_all_changes(df, res_df)
                    st.table(res_df.style.apply(highlight_all_changes, axis=None))
            





    #         image_name = 'data/example' + str(index) + '.jpg'
    #         image = Image.open(image_name)
    #         image = image.resize((400,300))
    #         st.image(image)
    #         st.table(df)


    # with st.form(key="my_form"):
    #     col1, col2, col3 = st.columns([1, 1, 3])
        
    #     with col1:
    #         ModelType = st.radio(
    #             'Choose model',
    #             ['DistilBERT', 'RoBERTa', 'BERT', 'minilm', 'albert'],
    #             help="At present, you can choose between 2 models (RoBERTa or DistilBERT) to embed your text. More to come!"
    #         )

    #         if ModelType == "DistilBERT":
    #             model_ckpt = 'ksabeh/distilbert-attribute-correction'
    #         else:
    #             model_ckpt = 'ksabeh/distilbert-base-uncased-finetuned-attributes-qa'
            
    #         TopResults = st.slider(
    #             '# of outputs',
    #             min_value=1,
    #             max_value=5,
    #             value=1,
    #             help="""You can choose the number of predictions to display."""
    #         )

    #         NullScore = st.slider(
    #             'Minimum score',
    #             min_value=0,
    #             max_value=10,
    #             value=10,
    #             help="""You can set the sensitivity of the model to the error displays."""
    #         )

    #         Correction = st.checkbox(
    #             "Allow correction",
    #             help="Tick this box if you want to allow the app to predict a substitute for wrong attributes."
    #         )

    #         MasekdLanguage = st.checkbox(
    #             "Use language model",
    #             help="Tick this box if you want to use the models trained on a masked language task."
    #         )

    #         if MasekdLanguage:
    #             model_ckpt += '-mlm'

    #         model = load_model(model_ckpt)

    #     with col2:
    #         Example = st.radio(
    #             'Choose example',
    #             ['Example 1', 'Example 2', 'Example 3', 'Example 4'],
    #             help="At present, you can only choose between 4 examples. More to come!"

    #         )

    #         if Example == 'Example 1':
    #             example = create_example(data[0])
    #             index = 0
    #         elif Example == 'Example 2':
    #             example = create_example(data[1])
    #             index = 1
    #         elif Example == 'Example 3':
    #             example = create_example(data[2])
    #             index = 2
    #         elif Example == 'Example 4':
    #             example = create_example(data[3])
    #             index = 3
            
    #     with col3:
    #         df = pd.DataFrame.from_dict(data[index])
    #         image_name = 'data/example' + str(index) + '.jpg'
    #         image = Image.open(image_name)
    #         image = image.resize((400,300))
    #         st.image(image)
    #         st.table(df)
    #     submit_button = st.form_submit_button(label="✨ Correct the data!")
        
    # if not submit_button:
    #     st.stop()

    # st.markdown("## **🎈 Check & download results **")
    # st.header("")

    # results = model.predict(example)
    # res_df = display_result(results, data[index])
    # res_df = get_all_changes(df, res_df)
    # st.table(res_df.style.apply(highlight_all_changes, axis=None))

if __name__ == '__main__':
    main()