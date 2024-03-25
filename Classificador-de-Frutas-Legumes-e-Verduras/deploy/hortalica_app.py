# streamlit run hortalica_app.py
# pip install --upgrade streamlit

import streamlit as st
import numpy as np
import re
import pickle
from scipy.sparse import csr_matrix
import os
import sklearn
from unidecode import unidecode

st.set_page_config(page_title="Vegetable classifier")

dir_path = os.path.dirname(os.path.realpath(__file__))


def processador_de_texto(data):
    data = unidecode(data)
    data = data.replace("-", " ")
    data = data.replace("  ", " ")
    data = re.sub('[^\w\s]', '', str(data.lower()))
    data = data.replace("\r", " ")
    data = data.replace("\n", " ")
    data = data.replace("  ", " ")
    data = data.split(" ")
    return data


def classify_new(input_, myvectorizer):

    input_vect = myvectorizer.transform(input_)

    input_vect_array = np.array(input_vect)
    input_vect_array_csr_matrix = csr_matrix((input_vect_array).all())

    return input_vect_array_csr_matrix


with open(f"{dir_path}/hortalicia_vectorizer.pickle", "rb") as f:
    h_vectorizer = pickle.load(f)

with open(f"{dir_path}/hortalicia_model.pkl", "rb") as f:
    model = pickle.load(f)

h_classes = {0.0: "Fruta", 1.0: "Legume", 2.0: "Verdura", }

st.title("My Fruit and Vegetable Classifier")

data_load_state = st.text('Este é um projeto de classificação de frutas, legumes e verduras que usa\n' +
                          'machine learning. O modelo de ml por trás do mesmo é o de Árvore de Decisão (Forest Classifier).\n' +
                          '\n'
                          'Forneça o nome de uma fruta, legume ou verdura e veja o resultado da\n' +
                          'classificação em tempo real!')

with st.form(key='my_form'):
    # st.text("Digite ou selecione algumas das opções")

    input_value = st.text_input("Digite o nome de uma fruta, legume ou verdura",
                                placeholder='Após digitar, clique em "Executar modelo"')

    submit_button = st.form_submit_button(label='Executar modelo')

if input_value:
    # if st.button("Executar modelo"):
    if submit_button:

        classify_vector = classify_new(
            processador_de_texto(input_value), h_vectorizer)
        
        input_validation = True

        if not str(classify_vector):
            input_validation = False

        if input_validation:
            proba = model.predict_proba(classify_vector)
            classification = model.predict(classify_vector)

            result = f"'{input_value}' é classificado como '{h_classes[classification[0]]}'!!!"
            st.success(result)

        else:
            error_r = (
                f"""
            Desculpe-me, mas o pedido para '{input_value}' foi negado!\n
            Pode ser que tenha ocorrido alguma falha na digitação ou o termo pode não se enquadrar nas categorias permitidas!
            """
            )

            st.error(error_r)
            result = (
                """
                Caso acredite que tenha ocorrido um erro durante sua experiência de uso,
                por favor, reporte ao desenvolvedor.
                """
            )
            st.info(result)
elif submit_button:
    st.error("Digite alguma coisa ou selecione um item!")

