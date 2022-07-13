import pickle

import numpy as np
import tensorflow as tf

from tfrbm import GBRBM

def APIcodefeaturelistinput():
    gbrbm_1 = GBRBM(n_visible=128, n_hidden=500))
    data = np.load("APIcodefeaturedemo.npy")
    codefeaturelist = []
    for i in range(len(data)):
        x = list(data[i])
        x_re = []
        tempt = []
        for j in x:
            tempt.append(float(j))
        x_re.append(tempt)
        y = gbrbm_1.compute_hidden(x_re)
        y_re = y.numpy()
        codefeaturelist.append(list(y_re[0]))
    np.save("APIcodefeaturedemo_re.npy", codefeaturelist)


def APItextfeaturelistinput():
    gbrbm_1 = GBRBM(n_visible=128, n_hidden=500)
    data = np.load("APItextfeaturedemo.npy")
    textfeaturelist = []
    for i in range(len(data)):
        x = list(data[i])
        x_re = []
        tempt = []
        for j in x:
            tempt.append(float(j))
        x_re.append(tempt)
        y = gbrbm_1.compute_hidden(x_re)
        y_re = y.numpy()
        textfeaturelist.append(list(y_re[0]))
    np.save("APItextfeaturedemo_re.npy", textfeaturelist)


def reconstruct_code_to_text_vec():
    gbrbm_1 = GBRBM(n_visible=128, n_hidden=500)
    hidden_data = np.load('demo_code_to_text_demo.npy')
    list_tempt = []
    for i in range(len(hidden_data)):
        list_tempt.append(float(hidden_data[i]))
    hidden_data_re = [list_tempt]
    visible_data = gbrbm_1.compute_visible(hidden_data_re)
    visible_data = visible_data.numpy()
    visible_data_list = list(visible_data[0])
    return  visible_data_list

def reconstruct_text_to_code_vec():
    gbrbm_1 = GBRBM(n_visible=128, n_hidden=500)
    hidden_data = np.load('demo_text_to_code_demo_25.npy')
    list_tempt = []
    for i in range(len(hidden_data)):
        list_tempt.append(float(hidden_data[i]))
    hidden_data_re = [list_tempt]
    visible_data = gbrbm_1.compute_visible(hidden_data_re)
    visible_data = visible_data.numpy()
    visible_data_list = list(visible_data[0])
    return visible_data_list
