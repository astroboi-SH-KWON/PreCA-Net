import os
import sys
import platform
import numpy as np
from time import time
# from scipy import stats
# from sklearn import metrics
import random as py_random
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras

is_in_google_colab = True
if 'google.colab' not in sys.modules:
    import DataGenerator
    import Util
    import ModelGenerator
    import CustomLoss

    is_in_google_colab = False

# to obtain reproducible results
np.random.seed(1)
py_random.seed(1)
tf.random.set_seed(1)

SYSTEM_NM = platform.system()
# TOTAL_CPU = mp.cpu_count()
# MULTI_CNT = int(TOTAL_CPU*0.8)
# tf.config.threading.set_intra_op_parallelism_threads(MULTI_CNT)
# tf.config.threading.set_inter_op_parallelism_threads(MULTI_CNT)


MODEL_NM = "20210517_" + "Fine_Tune" + "_model"
limit_mdl_cnt = 1
mem_limit = 1024 * 9
if __name__ == '__main__':
    start_time = time()
    print("start >>>>>>>>>>>>>>>>>>")
    if is_in_google_colab:
        print("is_in_google_colab\nmodel_nm :", MODEL_NM)
        data_generator = DataGenerators()  # colab
        util = Utils()  # colab
        custm_loss = CustomLosses()  # colab
        deep_model = DeepModels()  # colab

    else:
        data_generator = DataGenerator.DataGenerators()
        util = Util.Utils()
        custm_loss = CustomLoss.CustomLosses()
        deep_model = ModelGenerator.DeepModels()

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)])
            except RuntimeError as err:
                print("[ERROR]", err)

    ################# st hyper param ############################
    if SYSTEM_NM == 'Linux':
        # REAL
        # batch_epoch_list = [[32, 50], [64, 100], [128, 150], [256, 200], [512, 250], [1024, 300], [2048, 350]][::-1]  # linux
        batch_epoch_list = [[512, 250]][::-1]  # linux
        pre_model_path = "./models/Pre_Train_Model/"
    else:
        # DEV
        batch_epoch_list = [[64, 2]][::-1]
        pre_model_path = "D:/000_WORK/mine/SpCas9variants/models/"

    fc_1_arr = [128, 256, 512, 1024]  # fine-tuning
    drp_arr = [0.0]  # use BN not dropout
    loss_nms_list = [("mse", "pearson")]
    loss_func_list = [("mse", custm_loss.pearson_loss)]

    filt_stride = 1
    layer_f_arr = [128, 256, 512, 1024]  # fine-tuning
    layer_last_arr = [128, 256, 512, 1024]  # fine-tuning
    pad = 'VALID'

    load_model_epoch = 0  # 0 is_training
    # mntr = 'val_loss'  # 'val_pearson_loss' 'val_output_pearson_loss' 'val_output_1_pearson_loss'
    mntr_arr = ['val_output_pearson_loss']
    lr_arr = [0.001]
    ################# en hyper param ############################

    init_drp = 1.0

    for i in range(9, 0, -1):
    # for i in [9]:
        print("load Train_" + str(i) + ".xlsx >>>>>>>>>>>>>>>>>>")
        train_X, train_Feature, _, train_Y = data_generator.get_excel_w_trgt(
            "./input/190925_Variant_Train/Train_" + str(i) + ".xlsx", 0, 2, 1)
        train_Cat = np.sign(train_Y)
        print(train_X.shape, train_Y.shape, train_Feature.shape)

        # min-Max Normalization
        min_max_scaler = MinMaxScaler()
        print("min_max_scaler.fit(train_Y.reshape(-1, 1))", min_max_scaler.fit(train_Y.reshape(-1, 1)))
        train_Y = min_max_scaler.transform(train_Y.reshape(-1, 1)).reshape(train_Y.shape[0], )

        test_x, test_f, seq_x, ans_y = data_generator.get_excel_w_trgt(
            "./input/190925_Variant_Test/test_" + str(i) + ".xlsx", 0, 1, 2)
        test_cat = np.sign(ans_y)
        # ans_y = (ans_y - min(train_Y)) / (max(train_Y) - min(train_Y)) * 0.99 + 0.01
        ans_y = min_max_scaler.transform(ans_y.reshape(-1, 1)).reshape(ans_y.shape[0], )
        print("loading Train_" + str(i) + ".xlsx is DONE")

        for learning_rate in lr_arr:
            MODEL_NM = MODEL_NM + str(learning_rate) + "_model"
            print("make model option dict")
            options_list = []

            for drp in drp_arr:
                cnv1_krnl_num = 256
                cnv2_krnl_num = 256
                for fc_1_num in fc_1_arr:
                    for layer_f_num in layer_f_arr:
                        for layer_last_num in layer_last_arr:
                            options_dict = {
                                "fc_1": {
                                    "w_num": fc_1_num
                                    , "acti": "relu"
                                    , "drp": drp
                                }
                                , "feat": {
                                    "w_num": layer_f_num
                                    , "acti": "relu"
                                    , "drp": drp
                                }
                                , "last": {
                                    "w_num": layer_last_num
                                    , "acti": "relu"
                                    , "drp": drp
                                }
                            }
                            options_list.append(options_dict)

            print("DONE : make model option dict")

            for opts_dict in options_list:
                fc_1 = opts_dict['fc_1']['w_num']
                feat = opts_dict['feat']['w_num']
                last = opts_dict['last']['w_num']

                for func_idx in range(len(loss_func_list)):
                    for mntr_tmp in mntr_arr:
                        for batch_epoch_arr in batch_epoch_list:
                            batch_size = batch_epoch_arr[0]
                            epoch = batch_epoch_arr[1]

                            ################# st path ############################
                            fl_nm = ""
                            if is_in_google_colab:
                                fl_nm += MODEL_NM
                            else:
                                fl_nm += os.path.basename(__file__).replace(".py", "")

                            model_dir = './models/' + fl_nm + '/'
                            ouput_path = "./output/" + fl_nm + '/'
                            ################# en path ############################

                            # # If you load model for fine tuning with custom objects
                            with keras.utils.CustomObjectScope(
                                    {'pearson_loss': custm_loss.pearson_loss}):

                                sngl_oput = deep_model.model_1cnv_incptn_w_mx({
                                                                                "conv1": {
                                                                                    "knl_num": 256
                                                                                    , "filt_shp": (4, 3)
                                                                                    , "strd": filt_stride
                                                                                    , "pad": pad
                                                                                    , "acti": "relu"
                                                                                    , "drp": 0.0
                                                                                    , "bn_flag": True
                                                                                }
                                                                                , "conv2": {
                                                                                    "knl_num": 256
                                                                                    , "filt_shp": (1, 2)
                                                                                    , "strd": filt_stride
                                                                                    , "pad": pad
                                                                                    , "acti": "relu"
                                                                                    , "drp": 0.0
                                                                                    , "bn_flag": True
                                                                                }
                                                                                , "conv2_1": {
                                                                                    "knl_num": 256
                                                                                    , "filt_shp": (1, 2)
                                                                                    , "strd": filt_stride
                                                                                    , "pad": pad
                                                                                    , "acti": "relu"
                                                                                    , "drp": 0.0
                                                                                    , "bn_flag": True
                                                                                }
                                                                                , "conv3": {
                                                                                    "knl_num": 256
                                                                                    , "filt_shp": (1, 3)
                                                                                    , "strd": filt_stride
                                                                                    , "pad": pad
                                                                                    , "acti": "relu"
                                                                                    , "drp": 0.0
                                                                                    , "bn_flag": True
                                                                                }
                                                                                , "conv3_1": {
                                                                                    "knl_num": 256
                                                                                    , "filt_shp": (1, 3)
                                                                                    , "strd": filt_stride
                                                                                    , "pad": pad
                                                                                    , "acti": "relu"
                                                                                    , "drp": 0.0
                                                                                    , "bn_flag": True
                                                                                }
                                                                                , "conv5": {
                                                                                    "knl_num": 256
                                                                                    , "filt_shp": (1, 5)
                                                                                    , "strd": filt_stride
                                                                                    , "pad": pad
                                                                                    , "acti": "relu"
                                                                                    , "drp": 0.0
                                                                                    , "bn_flag": True
                                                                                }
                                                                                , "conv_concate": {
                                                                                    "knl_num": 256
                                                                                    , "filt_shp": (1, 2)
                                                                                    , "strd": filt_stride
                                                                                    , "pad": pad
                                                                                    , "acti": "relu"
                                                                                    , "drp": 0.0
                                                                                    , "bn_flag": True
                                                                                }
                                                                                , "fc_1": {
                                                                                    "w_num": 1024
                                                                                    , "acti": "relu"
                                                                                    , "drp": 0.0
                                                                                    , "bn_flag": True
                                                                                }
                                                                                , "feat": {
                                                                                    "w_num": 128
                                                                                    , "acti": "relu"
                                                                                    , "drp": 0.0
                                                                                    , "bn_flag": True
                                                                                }
                                                                                , "last": {
                                                                                    "w_num": 128
                                                                                    , "acti": "relu"
                                                                                    , "drp": 0.0
                                                                                    , "bn_flag": True
                                                                                }
                                                                            })
                                ################# st data array ############################
                                loss_num = len(loss_nms_list[func_idx])
                                if isinstance(loss_nms_list[func_idx], str) or loss_num == 1:
                                    if mntr_tmp == 'val_output_pearson_loss':
                                        mntr_tmp = 'val_pearson_loss'
                                    mdl_trgt_arr = sngl_oput
                                    trn_trgt_arr = train_Y
                                    tst_trgt_arr = ans_y
                                elif loss_num == 2:
                                    mdl_trgt_arr = [sngl_oput, sngl_oput]
                                    trn_trgt_arr = [train_Y, train_Y]
                                    tst_trgt_arr = [ans_y, ans_y]
                                else:
                                    mdl_trgt_arr = [sngl_oput, sngl_oput, sngl_oput]
                                    trn_trgt_arr = [train_Y, train_Y, train_Y]
                                    tst_trgt_arr = [ans_y, ans_y, ans_y]

                                if "category" in loss_nms_list[func_idx]:
                                    crsentp_idx = loss_nms_list[func_idx].index("category")
                                    mdl_trgt_arr[crsentp_idx] = tf.sign(sngl_oput)
                                    trn_trgt_arr[crsentp_idx] = train_Cat
                                    tst_trgt_arr[crsentp_idx] = test_cat
                                ################# en data array ############################

                                pre_model = keras.Model([deep_model.main_input, deep_model.feature_input], mdl_trgt_arr)
                                pre_model.load_weights(pre_model_path + "Pre_Train_" + str(i) + ".h5")

                                x = pre_model.layers[30].output  # flatten 30
                                print("flatten", x)

                                sngl_oput = deep_model.fine_tune_model(x, opts_dict)

                                # freeze pre_model
                                cnt = 0
                                for layer in pre_model.layers:
                                    if cnt <= 30:
                                        layer.trainable = False
                                        # print(layer.name, cnt)
                                    cnt += 1
                                    # print(layer.name, cnt)
                                # print(pre_model.layers[30].name)


                                ################# st data array ############################
                                loss_num = len(loss_nms_list[func_idx])
                                if isinstance(loss_nms_list[func_idx], str) or loss_num == 1:
                                    if mntr_tmp == 'val_output_pearson_loss':
                                        mntr_tmp = 'val_pearson_loss'
                                    mdl_trgt_arr = sngl_oput
                                    trn_trgt_arr = train_Y
                                    tst_trgt_arr = ans_y
                                elif loss_num == 2:
                                    mdl_trgt_arr = [sngl_oput, sngl_oput]
                                    trn_trgt_arr = [train_Y, train_Y]
                                    tst_trgt_arr = [ans_y, ans_y]
                                else:
                                    mdl_trgt_arr = [sngl_oput, sngl_oput, sngl_oput]
                                    trn_trgt_arr = [train_Y, train_Y, train_Y]
                                    tst_trgt_arr = [ans_y, ans_y, ans_y]

                                if "category" in loss_nms_list[func_idx]:
                                    crsentp_idx = loss_nms_list[func_idx].index("category")
                                    mdl_trgt_arr[crsentp_idx] = tf.sign(sngl_oput)
                                    trn_trgt_arr[crsentp_idx] = train_Cat
                                    tst_trgt_arr[crsentp_idx] = test_cat
                                ################# en data array ############################

                                fine_model = keras.Model([deep_model.main_input, deep_model.feature_input], mdl_trgt_arr)

                                fine_model.compile(
                                    loss=loss_func_list[func_idx]
                                    , optimizer=keras.optimizers.Adam(lr=learning_rate, clipvalue=0.5, clipnorm=1.0,
                                                                      beta_1=0.9, beta_2=0.999,
                                                                      epsilon=None, decay=0.0)
                                    , metrics=["mse", custm_loss.pearson_loss]
                                )

                                os.makedirs(model_dir, exist_ok=True)
                                opt = 'Fine_' + str(i) + '_fc_{}_feat_{}_last_{}'.format(fc_1, feat, last)
                                print("[OPT]", opt)
                                model_path = model_dir + opt + '.h5'

                                early_stopping = keras.callbacks.EarlyStopping(monitor=mntr_tmp, patience=50)

                                checkpointer = keras.callbacks.ModelCheckpoint(filepath=model_path
                                                                               , monitor=mntr_tmp
                                                                               , verbose=1
                                                                               , mode='min'
                                                                               , save_best_only=True
                                                                               )

                                fine_model.summary()

                                history = fine_model.fit(
                                    {"main_input": train_X, "feature_input": train_Feature}
                                    , trn_trgt_arr
                                    , epochs=epoch
                                    , batch_size=batch_size
                                    , verbose=1
                                    , shuffle=True
                                    , validation_data=([test_x, test_f], tst_trgt_arr)
                                    , callbacks=[early_stopping, checkpointer]
                                )

                                os.makedirs(ouput_path, exist_ok=True)
                                y_pred_last, _ = fine_model.predict([test_x, test_f])
                                last_model_path = ouput_path + opt + ".txt"
                                test_f_list = list(test_f)
                                util.make_trgt_result_file(last_model_path, ans_y, y_pred_last, seq_x, test_f_list)


