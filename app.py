from attribute_extractor import AttributeExtractor
from utils import create_example, display_result, read_examples, highlight_changes, get_changes, get_all_changes, highlight_all_changes

import json
import os

import streamlit as st
import numpy as np
import pandas as pd


data = read_examples('data/examples.jsonl')


st.set_page_config(
    layout='wide',
    page_title='E-commerce Attribute Corrector',
    page_icon='🛒'
)

def _max_width_():
    max_width_str = f"max-width: 1000px;"
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
st.markdown("<h1 style='text-align: center; color: black;'>E-commerce Attributes Corrector</h1>", unsafe_allow_html=True)


with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for the amazing [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
-   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers] (https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
	    """
    )

st.markdown("")
st.markdown("## **📌 An Example **")

with st.form(key="my_form"):
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        ModelType = st.radio(
            'Choose model',
            ['DistilBERT', 'RoBERTa', 'BERT', 'minilm', 'albert'],
            help="At present, you can choose between 2 models (RoBERTa or DistilBERT) to embed your text. More to come!"
        )

        if ModelType == "DistilBERT":
            model_ckpt = 'ksabeh/distilbert-attribute-correction'
        else:
           model_ckpt = 'ksabeh/distilbert-base-uncased-finetuned-attributes-qa'
        
        TopResults = st.slider(
            '# of outputs',
            min_value=1,
            max_value=5,
            value=1,
            help="""You can choose the number of predictions to display."""
        )

        NullScore = st.slider(
            'Minimum score',
            min_value=0,
            max_value=10,
            value=10,
            help="""You can set the sensitivity of the model to the error displays."""
        )

        Correction = st.checkbox(
            "Allow correction",
            help="Tick this box if you want to allow the app to predict a substitute for wrong attributes."
        )

        MasekdLanguage = st.checkbox(
            "Use language model",
            help="Tick this box if you want to use the models trained on a masked language task."
        )

        if MasekdLanguage:
            model_ckpt += '-mlm'
        
        @st.cache(allow_output_mutation=True)
        def load_model(model_ckpt):
            return AttributeExtractor(model_ckpt)
        model = load_model(model_ckpt)


        with col2:
            Example = st.radio(
                'Choose example',
                ['Example 1', 'Example 2', 'Example 3', 'Example 4'],
                help="At present, you can only choose between 4 examples. More to come!"

            )

            if Example == 'Example 1':
                example = create_example(data[0])
                index = 0
            elif Example == 'Example 2':
                example = create_example(data[1])
                index = 1
            elif Example == 'Example 3':
                example = create_example(data[2])
                index = 2
            elif Example == 'Example 4':
                example = create_example(data[3])
                index = 3
        
        with col3:
            df = pd.DataFrame.from_dict(data[index])
            st.table(df)
            print("runnnn")
        submit_button = st.form_submit_button(label="✨ Correct the data!")
    
if not submit_button:
    st.stop()

st.markdown("## **🎈 Check & download results **")
st.header("")

results = model.predict(example)
res_df = display_result(results, data[index])
res_df = get_all_changes(df, res_df)
st.table(res_df.style.apply(highlight_all_changes, axis=None))






# st.markdown("")
# st.markdown("")

# with st.form(key="my_form"):


#     ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
#     with c1:
#         ModelType = st.radio(
#             "Choose your model",
#             ["DistilBERT (Default)", "RoBERTa", "BERT"],
#             help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
#         )

#         if ModelType == "DistilBERT (Default)":
#             @st.cache(allow_output_mutation = True)
#             def load_model():
#                 return AttributeExtractor('ksabeh/distilbert-base-uncased-attribute-correction-qa')
#             extractor = load_model()
        
#         elif ModelType == 'RoBERTa':
#             @st.cache(allow_output_mutation = True)
#             def load_model():
#                 return AttributeExtractor('ksabeh/distilbert-base-uncased-finetuned-attributes-qa')
#             extractor = load_model()
        
#         results = st.slider(
#             'min null score',
#             min_value=1,
#             max_value=10,
#             value=5
#         )

#         negative = st.checkbox(
#             'allow negative',
#             value = True,
#         )

        
#     submit_button = st.form_submit_button(label="✨ Get me the data!")
    
# if not submit_button:
#     st.stop()

# results = extractor.predict(example)

# st.markdown("## **🎈 Check & download results **")
# st.header("")

# st.write(str(results))