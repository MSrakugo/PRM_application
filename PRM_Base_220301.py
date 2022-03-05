#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 23:06:08 2022

@author: satoshi_matsuno
"""

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import matplotlib.pyplot as plt

import PRM_liblary_220301 as prm
import PRM_Predict_liblary_220303 as prm_predict

class Models:
    def __init__(self, model, sc, pca, ICA):
        self.model = model
        self.sc = sc
        self.pca = pca
        self.ICA = ICA

class Model_datas:
    def __init__(self, Score_all, test_error_all, test_data_all):
        self.Score_all = Score_all
        self.test_error_all = test_error_all
        self.test_data_all = test_data_all

class Model_feature_setting:
    def __init__(self, setting_X_raw, setting_X_log, setting_X_product, setting_X_product_log, setting_X_ratio, setting_X_ratio_log, setting_NORMAL_OR_LOG, \
                 setting_PCA, setting_ICA, setting_standard_scaler):
        self.setting_X_raw = setting_X_raw
        self.setting_X_log = setting_X_log
        self.setting_X_product = setting_X_product
        self.setting_X_product_log = setting_X_product_log

        self.setting_X_ratio = setting_X_ratio
        self.setting_X_ratio_log = setting_X_ratio_log

        #standard_scalerに与える時，X_log or Xを選択
        self.setting_NORMAL_OR_LOG = setting_NORMAL_OR_LOG
        self.setting_PCA = setting_PCA
        self.setting_ICA = setting_ICA
        self.setting_standard_scaler = setting_standard_scaler

st.header("Protolith reconstruction model for Metabasalt")
st.caption("This is a prototype application. If you find any errors, please point them out to us through Github.")
###### Data read
uploaded_file = st.sidebar.file_uploader("Choose a file (Excel or CSV)")
index_col = age = st.sidebar.slider('Input index_columns number', 0, 10, 0)
header = age = st.sidebar.slider('Input header number', 0, 10, 0)
DataBase=st.sidebar.text_input("Write database", "example: Kelley et al. 2003")
SAMPLE_INFO=st.sidebar.text_input("Write sample infomation", "example: Seafloor altered basalt")

if uploaded_file is not None:
    raw_data = prm.read_Raw_data(uploaded_file, index_col, header, DataBase, SAMPLE_INFO)
else:
    raw_data=None
    #st.write(raw_data.astype(str))
###### Data read
if raw_data is None:
    pass
else:
    # select location info in uproad dataset
    location_info = st.sidebar.multiselect("Choose the location columns", raw_data.columns, ["DataBase", "SAMPLE_INFO"])

    ###### Data check/preprocessing
    PM, Location_Ref_Data = prm.preprocessing_normalize(raw_data, DataBase, SAMPLE_INFO, location_info)
    #st.write(PM)

    ###### estimation by PRM
    # model folder
    now_model_folder_name = 'models_220203_ALL_ALL'

    # elements list
    elements_list = ['Rb', 'Ba', 'Th', 'U', 'Nb', 'K', 'La', 'Ce', 'Pb', 'Sr', 'Nd', 'Zr', 'Ti', 'Y', 'Yb','Lu', 'SiO2', 'Al2O3', 'MgO', 'Na2O', 'P2O5', 'CaO', 'MnO', 'FeO', 'K2O']
    immobile_elem=st.sidebar.multiselect("Choose the immobile elements (Not Change)", elements_list, ["Zr", "Th", "Ti", "Nb"])
    mobile_elem=st.sidebar.multiselect("Choose the mobile elements", elements_list, ['Rb', 'Ba', 'Th', 'U', 'Nb', 'K', 'La', 'Ce', 'Pb', 'Sr', 'Nd', 'Zr', 'Ti', 'Y', 'Yb','Lu',])

    mobile_data_compile, spidergram_data_compile = prm_predict.predict_protolith(mobile_elem, immobile_elem, PM, Location_Ref_Data, now_model_folder_name)

    ###### Data visualization
    choice_sample = st.selectbox('Choice sample',spidergram_data_compile.index, )
    st.write('You selected:', choice_sample)


    fig, ax = plt.subplots(constrained_layout=True)
    pred_data = pd.DataFrame(spidergram_data_compile.loc[choice_sample]).T.dropna()
    now_col=pred_data.columns

    fig=prm.Spidergram_simple(pd.DataFrame(PM.loc[choice_sample]).T[now_col], "log", "off","#344c5c", "--", "off", fig, ax)
    fig=prm.Spidergram_simple(pred_data, "log", "on","#f08575", "-", "off", fig, ax)
    plt.title(choice_sample)
    plt.legend(["Metabasalt comp.", "Protolith comp."])
    plt.ylim(0.1, 1000)
    st.pyplot(fig)
    ###### Data visualization


    protolith_data = spidergram_data_compile.copy()
    protolith_data_ppm = prm.PM_to_ppm(protolith_data)
    element_mobility = PM[protolith_data.columns]/protolith_data


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            label="Protolith composition predicted by PRM (PM_normalized)",
            data=protolith_data.to_csv().encode('utf-8'),
            file_name="Protolith_comp_by_PRM_(PM_normalized)_" + DataBase +".csv",
            mime='text/csv',
            )
    with col2:
        st.download_button(
            label="Protolith composition predicted by PRM (ppm)",
            data=protolith_data_ppm.to_csv().encode('utf-8'),
            file_name="Protolith_comp_by_PRM_(ppm)" + DataBase +".csv",
            mime='text/csv',
            )

    with col3:
        st.download_button(
            label="Element mobile predicted by PRM (ppm)",
            data=mobile_data_compile.to_csv().encode('utf-8'),
            file_name="Element_mobile_by_PRM_" + DataBase +".csv",
            mime='text/csv',
            )

    with col4:
        st.download_button(
            label="Element mobility predicted by PRM (Metabasalt/Protolith)",
            data=element_mobility.to_csv().encode('utf-8'),
            file_name="Element_mobility_by_PRM_" + DataBase +".csv",
            mime='text/csv',
            )


st.subheader("Cite this article")
st.caption("Matsuno, S., Uno, M., Okamoto, A. Tsuchiya, N. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022). https://doi.org/10.1038/s41598-022-05109-x")
st.caption("Press release in Japanese: https://www.tohoku.ac.jp/japanese/2022/02/press20220210-01-machine.html")
st.caption("Press release in English: Coming soon...")
st.caption("Made by Satoshi Matsuno (Graduate School of Environmental Studies, Tohoku univ.)")
