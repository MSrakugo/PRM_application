#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 23:06:08 2022

@author: satoshi_matsuno
"""

import streamlit as st
import pandas as pd
import numpy as np
import openpyxl

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


#################################################################################### sidebar
st.header("Protolith reconstruction model (PRM) for Metabasalt")
st.caption("This is a prototype application. If you find any errors, please point them out to us through Github: https://github.com/MSrakugo/PRM_application")
st.caption("Protolith included in training data: Mid-ocean ridge (MORB), Ocean island basalt (OIB), Volcanic arc basalt (VAB), Back-arc basin basalt (BAB)")

#### read example dataset
example_data = pd.read_csv("Example_Dataset/Example_dataset(Kelley_2003_Seafloor_altered_basalt).csv", index_col=0)
st.sidebar.download_button(
    label="Example dataset (Quoted from PetDB)",
    data=example_data.to_csv().encode('utf-8'),
    file_name="Example_dataset(Kelley_2003).csv",
    mime='text/csv',
    )

###### Data input
uploaded_file = st.sidebar.file_uploader("Choose a file (Excel or CSV)")
index_col = age = st.sidebar.slider('Input index_columns number', 0, 10, 0)
header = age = st.sidebar.slider('Input header number', 0, 10, 0)

###### Add info
flag_add_info=0
if st.sidebar.checkbox('Additional information'):
    DataBase=st.sidebar.text_input("Write database", "example: Kelley et al. 2003")
    SAMPLE_INFO=st.sidebar.text_input("Write sample infomation", "example: Seafloor altered basalt")
    flag_add_info=1
else:
    DataBase="No database info"
    SAMPLE_INFO="No sample info"
    location_info = ["DataBase", "SAMPLE_INFO"]
###### Add info

###### Data read
if uploaded_file is not None:
    raw_data = prm.read_Raw_data(uploaded_file, index_col, header, DataBase, SAMPLE_INFO)
else:
    raw_data=None
###### Data read

###### select location if st.sidebar.checkbox('Additional information')==True
if flag_add_info==1:
   if raw_data is None:
        pass
   else:
        location_info = st.sidebar.multiselect("Choose the location columns", raw_data.columns, ["DataBase", "SAMPLE_INFO"])
###### select location if st.sidebar.checkbox('Additional information')==True

###### Select element
if raw_data is None:
    pass
else:
    elements_list = ['Rb', 'Ba', 'Th', 'U', 'Nb', 'K', 'La', 'Ce', 'Pb', 'Sr', 'Nd', 'Zr', 'Ti', 'Y', 'Yb','Lu', 'SiO2', 'Al2O3', 'MgO', 'Na2O', 'P2O5', 'CaO', 'MnO', 'FeO', 'K2O']
    immobile_elem=st.sidebar.multiselect("Choose the immobile elements (Do Not Change)", elements_list, ["Zr", "Th", "Ti", "Nb"])
    mobile_elem=st.sidebar.multiselect("Choose the mobile elements", elements_list, ['Rb', 'Ba', 'Th', 'U', 'Nb', 'K', 'La', 'Ce', 'Pb', 'Sr', 'Nd', 'Zr', 'Ti', 'Y', 'Yb','Lu',])
###### Select element

###### モデル推定開始のフラグ
#flag_MODEL_RUN=1
###### モデル推定開始のフラグ
#################################################################################### sidebar

#################################################################################### Main

###### Data read
if raw_data is None:
    pass
else:
    try:
        st.write(raw_data)
    except:
        pass
    ###### Data check/preprocessing
    PM, Location_Ref_Data = prm.preprocessing_normalize(raw_data, DataBase, SAMPLE_INFO, location_info)
    #st.write(PM)

    ###### estimation by PRM
    # model folder
    now_model_folder_name = 'models_220203_ALL_ALL'
    # model score read
    now_model_folder_score_name = now_model_folder_name+"/"+str(immobile_elem).strip("[").strip("]").strip("'")+"/Score_all.xlsx"
    model_score = pd.read_excel(now_model_folder_score_name, index_col=0)
    model_score[immobile_elem]=0
    model_score=model_score[elements_list]
    model_score = model_score.loc["Optune_Test_mean"]
    
    # estimate
    mobile_data_compile, spidergram_data_compile = prm_predict.predict_protolith(mobile_elem, immobile_elem, PM, Location_Ref_Data, now_model_folder_name)

    ###### Data visualization
    st.subheader("Visualize your data")
    with st.expander("See figures"):   
        # select sample
        choice_sample = st.selectbox('Select sample',spidergram_data_compile.index, )
        
        st.subheader("Spidergram")
        ###### figure 
        fig, ax = plt.subplots(constrained_layout=True)
        # road data
        pred_data_now = pd.DataFrame(spidergram_data_compile.loc[choice_sample]).T.dropna(axis=1)
        now_col=pred_data_now.columns
        raw_data_now=pd.DataFrame(PM.loc[choice_sample]).T[now_col]
        model_score_now=model_score[now_col]
        values = st.slider('Select y axis range in log scale for spidergram',-10.0, 10.0, (-1.0, 3.0))
        
        # figure control
        #fig=prm.Spidergram_fill_immobile(now_col, immobile_elem, '#ecc06f', 0.18, fig, ax)
        fig=prm.Spidergram_simple(raw_data_now, "log", "off","#344c5c", "--", "off", fig, ax)
        fig=prm.Spidergram_error(pred_data_now, model_score_now,"log", "on","#f08575", "-", "off", fig, ax)
        fig=prm.Spidergram_marker(raw_data_now, immobile_elem, '#f08575', '#344c5c', 'd', 16, fig, ax)
        # figure control
        # figure setting
        plt.title(choice_sample)
        plt.ylabel("Sample / PM")
        plt.legend(["Metabasalt comp.", "Protolith comp."])
        plt.ylim(10**values[0], 10**values[1])
        plt.tick_params(which='both', direction='in',bottom=True, left=True, top=True, right=True)
        plt.tick_params(which = 'major', length = 7.5, width = 2)
        plt.tick_params(which = 'minor', length = 4, width = 1)
        # figure setting
        st.pyplot(fig)
        
        
        st.subheader("Element mobility")
        fig, ax = plt.subplots(constrained_layout=True)
        # road data
        pred_mobility_now = raw_data_now/pred_data_now
        values_m = st.slider('Select y axis range in log scale for mobility figure',-10.0, 10.0, (-1.0, 2.0))
        # figure control
        fig=prm.Spidergram_error(pred_mobility_now, model_score_now,"log", "off","#f08575", "-", "off", fig, ax)
        fig=prm.Spidergram_marker(pred_mobility_now, immobile_elem, '#f08575', '#344c5c', 'd', 16, fig, ax)
        ax.axhline(y=1, xmin=0, xmax=len(pred_mobility_now.columns)-1, color = "#344c5c", linestyle='--',)
        # figure control
        # figure setting
        plt.title(choice_sample)
        plt.ylabel("Metabasalt/Protolith")
        plt.ylim(10**values_m[0], 10**values_m[1])
        plt.tick_params(which='both', direction='in',bottom=True, left=True, top=True, right=True)
        plt.tick_params(which = 'major', length = 7.5, width = 2)
        plt.tick_params(which = 'minor', length = 4, width = 1)
        # figure setting
        st.pyplot(fig)        
        
        
    ###### Data visualization


    protolith_data = spidergram_data_compile.copy()
    protolith_data_ppm = prm.PM_to_ppm(protolith_data)
    element_mobility = PM[protolith_data.columns]/protolith_data

    st.subheader("Download PRM results")
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
#################################################################################### Main

st.subheader("Cite this article")
st.caption("Matsuno, S., Uno, M., Okamoto, A. Tsuchiya, N. Machine-learning techniques for quantifying the protolith composition and mass transfer history of metabasalt. Sci Rep 12, 1385 (2022). https://doi.org/10.1038/s41598-022-05109-x")
st.caption("Press release in Japanese: https://www.tohoku.ac.jp/japanese/2022/02/press20220210-01-machine.html")
st.caption("Press release in English: Coming soon...")
st.caption("Made by Satoshi Matsuno (Graduate School of Environmental Studies, Tohoku univ.)")

