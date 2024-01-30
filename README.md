# CAVE: Correcting Attribute Values in E-commerce Profiles
CAVE is a tool for attribute value correction and enrichment in E-commerce. Given a product profile consisting of attributes table and description, CAVE: (1) detects and corrects wrong attribute values (2) enriches the attribute tables with newly extracted attributes and their corresponding values (3) normalises product attributes for product comparison.

CAVE relies on a Question-Answering system for the attribute value correction and extraction tasks. It learns information from both descriptions and attribute tables, using language encoder models to correct attribute values. For this purpose, CAVE is trained on datasets generated from the [Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html).

More details can be found in our [CIKM 2022 paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557161).

## Installation
Python 3 is required, and [Streamlit](https://streamlit.io/):
```
pip install streamlit
```
Please refer to `requirements.txt` for all relevent dependencies of the application.

## Usage
1. Clone the repo
2. Run the command below in your terminal
```
streamlit run app.py
```
The above command runs the python script `app.py` which is a [Streamlit](https://streamlit.io/) web application. Now you can see CAVE web interface:

<center>
<img src=".graphs/web_app.jpg" alt="drawing"/>
</center

The application can be used in two modes:

#### Correct Attributes
In this mode, the application is used to correct existing attribute values and discover new attributes and their corresponding values using information from the product profiles. The extracted information is highlighted as shown below:

<center>
<img src=".graphs/attr_correction.jpg" alt="drawing"/>
</center

#### Compare Products
In this mode, the application is used to compare between two products. As above, the attributes are first corrected, then the attributes are normalised to their canonical form and displayed side by side for comparison. 

<center>
<img src=".graphs/product_comp.jpg" alt="drawing"/>
</center

For more information about using the application, please refer to the [video demonstration](https://bit.ly/3xW1W3E) of our system.

------

If you found this work useful, please cite it as follows:

```
@inproceedings{10.1145/3511808.3557161,
author = {Sabeh, Kassem and Kacimi, Mouna and Gamper, Johann},
title = {CAVE: Correcting Attribute Values in E-commerce Profiles},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557161},
doi = {10.1145/3511808.3557161},
booktitle = {Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
pages = {4965–4969},
numpages = {5},
keywords = {question answering, data generation, attribute value extraction},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}
```
