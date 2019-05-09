from __future__ import print_function

import copy
import pickle
import yaml
import numpy as np
import random
import codecs,json
import tensorflow as tf

from keras.models import Model, load_model
from train_ntn import NeuralTensorLayer, custom_loss, get_data
from train_cnn import ConvolutionalNN, binary_classification_loss
from tensorflow.contrib.data import sliding_window_batch



with open('EB-CNN.yaml', 'r', encoding='UTF8') as f_yaml:
    parser = yaml.load(f_yaml)
ntn_oup = parser['ntn_output']
ntn_bs = parser['data_day']
cnn_inp = parser['cnn_input']
cnn_bs = parser['cnn_bs']
l= parser['cnn_window']
dd = parser['data_day']
ul = parser['long_term']      # U - Long-term
um = parser['mid_term']       # U - mid-term
us = parser['short_term']     # U - short-term

def main():
    X_train, Y_train, X_test, Y_test = get_data(ntn_bs)
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    #print(np.shape(X_test))

    word_vec = []
    w_sub = []
    w_act = []
    w_obj = []


    vec_len = 0

    with open('6news_vectors.pickle', 'rb') as f:
        for i in range(0, vec_len):
            try:
                pic = pickle.load(f)
                # word_vec.append(pic)
                w_sub.append(pic['subject'].astype(np.float32))
                w_act.append(pic['action'].astype(np.float32))
                w_obj.append(pic['object'].astype(np.float32))
                vec_len += 1
            except (EOFError):
                break

    w_subr = copy.deepcopy(w_obj)
    random.shuffle(w_subr)


    ntn_model = load_model(parser['ntn_mod_name'],custom_objects={'NeuralTensorLayer':NeuralTensorLayer, 'contrastive_loss':custom_loss(ntn_oup,ntn_bs)})
    ntn_out = ntn_model.predict([w_sub, w_act, w_obj,w_subr],batch_size=ntn_bs)
    print(np.shape(ntn_out[:,:,0]))
    u_vec =ntn_out[:,:,0]

    u_vec_d = []
    for i in range(int(len(u_vec)/ntn_bs)):
        u_vec_d.append(u_vec[i*ntn_bs:(i+1)*ntn_bs].tolist())   # U-vector
    print(np.array(u_vec_d).shape)
    json_u = json.dumps(u_vec_d)

    # Saving U-vector
    with open(parser['u_vector_path'], 'wb'):
        json.dump(json_u, codecs.open(parser['u_vector_path'], 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        print("U-vector saved.")

    """
    u_vec = np.asarray(u_vec_d)
    m = len(u_vec) // ul

    ln = m * ul - ul + 1  # subtract 1 to make spare batch
    ln = ln - divmod(ln, cnn_bs)[1] + ul -1 # make multiple of batch size
    print(np.array(u_vec[:ln , :, :].shape))
    dataset = tf.data.Dataset.from_tensor_slices((u_vec[:ln , :, :]))
    dataset = dataset.apply(sliding_window_batch(window_size=ul))

    iterator = dataset.make_one_shot_iterator()  # iterator
    next_element = iterator.get_next()
    numpy_data = np.zeros((ln - ul + 1, ul, dd, cnn_inp))

    sess = tf.Session()  # tensorflow session
    for i in range(ln  - ul + 1):
        data_ = sess.run(
            next_element)  # data_ contains the data and label_ contains the labels that we fed in the previous step
        numpy_data[i, :, :, :] = data_

    cnn_model = load_model(parser['cnn_mod_name'], custom_objects={'ConvolutionalNN': ConvolutionalNN, 'binary_classification_loss':binary_classification_loss})
    cnn_out = cnn_model.predict(numpy_data,
                                batch_size=cnn_bs)

    print(np.array(cnn_out).shape)
    json_bin_class = json.dumps(cnn_out.tolist())

    # Saving Binary-classification result
    with open(parser['bin_class_path'], 'wb'):
        json.dump(json_bin_class, codecs.open(parser['bin_class_path'], 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        print("Binary-classification result saved.")
    """

if __name__ == "__main__" :
    main()