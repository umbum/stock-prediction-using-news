"""
뉴스 데이터를 이용하여 학습된 신경망으로 주가 예측하는 모듈

4bert_vectors.pickle 파일이 같은 디렉토리 안에 있어야함
"""

from __future__ import print_function

import copy
import pickle
import yaml
import numpy as np
import random
import codecs, json
import tensorflow as tf
import os, sys

sys.path.append("../data")
from news_labeling_util import labelingNewsIterArr

from keras.models import Model, load_model
from train_ntn import NeuralTensorLayer, custom_loss, get_data
from train_cnn import ConvolutionalNN, binary_classification_loss
from tensorflow.contrib.data import sliding_window_batch

with open('EB-CNN.yaml', 'r', encoding='UTF8') as f_yaml:
    parser = yaml.load(f_yaml)
ntn_oup = parser['ntn_output']
ntn_bs = parser['ntn_bs']
cnn_inp = parser['cnn_input']
cnn_bs = parser['cnn_bs']
l = parser['cnn_window']
dd = parser['data_day']
ul = parser['long_term']  # U - Long-term
um = parser['mid_term']  # U - mid-term
us = parser['short_term']  # U - short-term


def main():
    """
    X_train, Y_train, X_test, Y_test = get_data(ntn_bs)
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    #print(np.shape(X_test))

    w_day = []
    w_sub = []
    w_act = []
    w_obj = []


    vec_len = 0

    with open('6news_vectors_100.pickle', 'rb') as f:
        while True:
            try:
                pic = pickle.load(f)
                w_day.append(pic['day'])
                w_sub.append(pic['subject'].astype(np.float32))
                w_act.append(pic['action'].astype(np.float32))
                w_obj.append(pic['object'].astype(np.float32))
                vec_len += 1
            except (EOFError):
                break

    w_subr = copy.deepcopy(w_obj)
    random.shuffle(w_subr)


    ntn_model = load_model("01-103.4483.hdf5",custom_objects={'NeuralTensorLayer':NeuralTensorLayer, 'custom_loss':custom_loss})
    ntn_out = ntn_model.predict([w_sub, w_act, w_obj,w_subr],batch_size=ntn_bs)
    print(np.shape(ntn_out[:,:,0]))
    u_vec =ntn_out[:,:,0]

    u_vec_d = []
    for i in range(int(len(u_vec))):
        u_vec_d.append([w_day[i], u_vec[i].tolist()])   # U-vector
    print(u_vec_d)
    print(np.array(u_vec_d).shape)
    json_u = json.dumps(u_vec_d)

    # Saving U-vector
    with open(parser['u_vector_path'], 'wb'):
        json.dump(json_u, codecs.open(parser['u_vector_path'], 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        print("U-vector saved.")

    with open(parser['u_vector_path'], encoding='UTF8') as f:
        json_u = json.load(f)

    u_vec_d = np.asarray(json.loads(json_u))  # ('cnn_data_num','cnn_input')
    #print(u_vec,":u_vec")
    """

    """        
    w_day = []
    u_vec = []
    label = []
    #s_idx = -1
    for i in range(len(u_vec_d)):
        #tempLabel, s_idx = labelingNewsIterArr("SK하이닉스_data.npy", u_vec_d[i][0], s_idx)
        #if tempLabel is None:
        #    continue
        w_day.append(u_vec_d[i][0])
        u_vec.append(u_vec_d[i][1])
        label.append(u_vec_d[i][2])
    """

    # u_vec_d = []
    w_day = []
    u_vec = []
    label = []

    with open('4bert_vectors.pickle', 'rb') as f:
        dst = open("./5avg_vec.pickle", "wb")
        pic = pickle.load(f)
        w_day.append(pic['day'])
        u_vec.append(pic['u-vector'].tolist())
        label.append(pic['label'])
        temp = pic['day']
        cnt = 0
        idx = 0
        while True:
            try:
                pic = pickle.load(f)
                # u_vec_d.append([pic['day'],pic['u-vector'].tolist(),pic['label']])
                if pic['day'] == temp:
                    cnt += 1
                    idx += 1

                else:
                    pickle_entry = {
                        "day": temp,
                        "u-vector": np.mean(u_vec[idx - cnt:idx + 1], axis=0),
                        "label": label[idx]
                    }
                    pickle.dump(pickle_entry, dst)
                    cnt = 0
                    idx = 0
                    w_day = []
                    u_vec = []
                    label = []
                    temp = pic['day']

                w_day.append(pic['day'])
                u_vec.append(pic['u-vector'].tolist())
                label.append(pic['label'])
            except (EOFError):
                break

    """
    w_day_avg = []
    u_vec_avg = []
    label_avg = []

    temp = w_day[0]
    cnt = 0
    idx = 0
    labelled = []
    for day in w_day[1:]:
        if day == temp:
            cnt += 1
            idx += 1
            continue
        elif day != temp:
            w_day_avg.append(temp)
            u_vec_avg.append(np.mean(u_vec[idx - cnt:idx + 1], axis=0))
            label_avg.append(label[idx])
            labelled.append([temp, np.mean(u_vec[idx - cnt:idx + 1], axis=0).tolist(), label[idx]])
            temp = day
            cnt = 0
            idx += 1
    w_day_avg.append(temp)
    u_vec_avg.append(np.mean(u_vec[idx - cnt:idx + 1], axis=0))
    label_avg.append(label[idx])
    labelled.append([temp, np.mean(u_vec[idx - cnt:idx + 1], axis=0).tolist(), label[idx]])

    json_labelled = json.dumps(labelled)

    with open('labelled.json', 'wb'):
      json.dump(json_labelled, codecs.open('labelled.json', 'w', encoding='utf-8'), 
              separators=(',', ':'), sort_keys=True, indent=4)
    print("labelled data saved.")
    """

    """
    with open('labelled.json',encoding='utf-8') as f:
       json_labelled = json.load(f)

    labelled = np.asarray(json.loads(json_labelled))
    w_day_avg = []
    u_vec_avg = []
    label_avg = [] 
    for i in range(len(labelled)):
      w_day_avg.append(labelled[i][0])
      u_vec_avg.append(labelled[i][1])
      label_avg.append(labelled[i][2])

    w_day_avg = np.array(w_day_avg)
    u_vec_avg = np.array(u_vec_avg)
    label_avg = np.array(label_avg)
    m = len(u_vec_avg) // ul

    ln = m * ul - ul + 1  # subtract 1 to make spare batch
    ln = ln - divmod(ln, cnn_bs)[1] + ul -1 # make multiple of batch size
    #print(np.array(u_vec[:ln , :, :].shape))

    dataset = tf.data.Dataset.from_tensor_slices((np.array(u_vec_avg[:ln][:])))
    dataset = dataset.apply(sliding_window_batch(window_size=ul))

    iterator = dataset.make_one_shot_iterator()  # iterator
    next_element = iterator.get_next()
    numpy_data = np.zeros((ln - ul + 1, ul, cnn_inp))

    sess = tf.Session()  # tensorflow session
    for i in range(ln - ul + 1):
        data_ = sess.run(
            next_element)  # data_ contains the data and label_ contains the labels that we fed in the previous step
        numpy_data[i, :, :] = data_

    cnn_model = load_model(parser['cnn_mod_name'], custom_objects={'ConvolutionalNN': ConvolutionalNN, 'binary_classification_loss':binary_classification_loss})
    cnn_out = cnn_model.predict(numpy_data,
                                batch_size=cnn_bs)

    print(np.array(cnn_out).shape)


    pred = []
    for i in range(ln-ul+1):
        pred.append([w_day_avg[i],cnn_out[i].tolist()])

    json_bin_class = json.dumps(pred)

    # Saving Binary-classification result
    with open(parser['bin_class_path'], 'wb'):
        json.dump(json_bin_class, codecs.open(parser['bin_class_path'], 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        print("Binary-classification result saved.")
    """


if __name__ == "__main__":
    main()