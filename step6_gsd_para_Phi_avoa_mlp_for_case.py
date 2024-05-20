# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from DataPostPrecessingControl import post_date_processing_regressor_for_predic
import DataPreDrocessingControl as dpdCL
import ExcelControl as excelCL
import joblib as jblib


# 从excel中导出训练样本的矩阵
def get_data_from_excel(df_excel, row_size,colum_size,list_fields_names,list_label,flag_traindata):
    # 属性表字段名
    # list_fields_names = ["f_suscep", "f_dem", "f_ic", "f_rootdep", "f_soildep", "f_soiltrength", "f_vegload", "Gully_J_Sd",
    #               "流速Sd", "流深Sd"]
    # list_fields_names = ["f_suscep", "f_dem", "f_ic", "f_soiltrength", "f_vegload", "Gully_J_Sd", "流速Sd", "流深Sd"]
    # list_label = ["label"]
    # 初始化全流域矩阵
    X_total = np.zeros(shape=[row_size, colum_size], dtype='float32')
    Y_total = np.zeros(shape=[row_size, 1], dtype="bool")
    X_total = excelCL.convertDFtoArray(df_excel, list_fields_names)
    # print(X_yangben)
    if flag_traindata:
        Y_total = excelCL.convertDFtoArray(df_excel, list_label)
    return X_total, Y_total

def get_data_from_excel_for_GSD(df_excel, row_size,list_fields_names,list_label,flag_traindata):
    # 属性表字段名
    # list_fields_names = ["f_suscep", "f_dem", "f_ic", "f_rootdep", "f_soildep", "f_soiltrength", "f_vegload", "Gully_J_Sd",
    #               "流速Sd", "流深Sd"]
    # list_fields_names = ["f_suscep", "f_dem", "f_ic", "f_soiltrength", "f_vegload", "Gully_J_Sd", "流速Sd", "流深Sd"]
    # list_label = ["label"]
    # 初始化全流域矩阵
    # X_total = np.zeros(shape=[row_size, colum_size], dtype='float32')
    Y_total = np.zeros(shape=[row_size, 1], dtype='float32')
    X_total = excelCL.convertDFtoArray(df_excel, list_fields_names)
    # print(X_yangben)
    if flag_traindata:
        Y_total = excelCL.convertDFtoArray(df_excel, list_label)
    return X_total, Y_total

if __name__ == '__main__':

    # controlling flag
    flag_evaluation = True
    modelNmae = "AVOA_MPL_for_Phi"
    flagSimle = False
    flag_fit_scaler = False
    flag_prediction = True

    # parameters detial info
    keyUseFlag = "Use_Flag"  # 0为不可用，1为可用
    df_totalyangben_environment = pd.read_excel('input/GSD_Samples/Samples_ALLSoil_ForTest_V1_with_gsd.xlsx')
    df_totalyangben_environment = df_totalyangben_environment[df_totalyangben_environment[keyUseFlag] == 1]

    # parameters detial info
    row_size = df_totalyangben_environment.shape[0]
    str_para_vary = ""

    # reading samples
    list_fields_names = ["gsd_miu","gsd_dc","Chara_m"]
    list_fields_names_plot = ["gsd_miu","gsd_dc","Chara_m"]

    list_label = ["Para_Phi"]
    field_name_for_opt = "Para_Phi_pre"
    X_total, Y_total = get_data_from_excel_for_GSD(df_totalyangben_environment, row_size,  list_fields_names,
                                           list_label, False)

    # (1)samples preprocessing
    saved_scaler_filepath = "saved_para/saved_for_gsd_scaler_1.0.pkl"
    if flag_fit_scaler:
        df_totalyangben_environment_forStandard = pd.read_excel('input/GSD_Samples/TrainingSamples_ALLSoil_ForTest_V1_with_gsd.xlsx')
        df_totalyangben_environment_forStandard = df_totalyangben_environment_forStandard[df_totalyangben_environment_forStandard[keyUseFlag] == 1]
        row_size_for_sd = df_totalyangben_environment_forStandard.shape[0]
        X_total_for_sd, Y_total_for_sd = get_data_from_excel_for_GSD(df_totalyangben_environment, row_size_for_sd,  list_fields_names,
                                           list_label, False)
        sc_first = dpdCL.dataprocessing_first_getStandardScaler(X_total_for_sd)
        dpdCL.save_scaler(sc_first, saved_scaler_filepath)
    sc = dpdCL.load_scaler(saved_scaler_filepath)
    X_total_std = dpdCL.dataprocessing_second_StandardScaler_transform(sc, X_total)

    if flag_prediction:
        # obtaining trainted model
        model_saved_path = "saved_model/trained_model_mlp_gsd_curve_Phi_v1.0.pkl"
        # model loading
        model = jblib.load(model_saved_path)
        # prediction output
        post_date_processing_regressor_for_predic(model, modelNmae, X_total_std, str_para_vary, flagSimle,
                                                  df_totalyangben_environment, field_name_for_opt)



