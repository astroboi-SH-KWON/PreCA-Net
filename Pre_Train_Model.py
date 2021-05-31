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


MODEL_NM = "20210517_" + "Pre_Train_1cnv_incptn_w_mx" + "_model"
limit_mdl_cnt = 0
mem_limit = 1024 * 9
if __name__ == '__main__':
    start_time = time()
    print("start >>>>>>>>>>>>>>>>>>")
    if is_in_google_colab:
        print("is_in_google_colab\nmodel_nm :", MODEL_NM)
        data_generator = DataGenerators()  # colab
        util = Utils()  # colab
        custm_loss = CustomLosses()  # colab
    else:
        data_generator = DataGenerator.DataGenerators()
        util = Util.Utils()
        custm_loss = CustomLoss.CustomLosses()

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
        batch_epoch_list = [[32, 100], [64, 150], [128, 200], [256, 250], [512, 300], [1024, 350]][::-1]  # linux
    else:
        # DEV
        batch_epoch_list = [[32, 2], [64, 2]][::-1]
    ############### st krnl
    max_krnl = 512
    krnl_arr = []
    while True:
        krnl_arr.append(int(max_krnl))
        if max_krnl == 64:
            break
        max_krnl /= 2
    cnv1_krnl_arr = krnl_arr
    ############### en krnl
    fc_1_arr = [1024]  # before fine-tuning
    bn_arr = [True, False]
    drp_arr = [0.0, 0.2, 0.3, 0.4, 0.5]
    loss_nms_list = [("mse", "pearson")]
    loss_func_list = [("mse", custm_loss.pearson_loss)]
    # loss_nms_list = [("mse", "pearson"), ("pearson", "mse")]
    # loss_func_list = [("mse", custm_loss.pearson_loss), (custm_loss.pearson_loss, "mse")]

    filt_stride = 1
    layer_f_num = 128
    layer_last_num = 128
    pad = 'VALID'

    load_model_epoch = 0  # 0 is_training
    # mntr = 'val_loss'  # 'val_pearson_loss' 'val_output_pearson_loss' 'val_output_1_pearson_loss'
    mntr_arr = ['val_output_pearson_loss']
    lr_arr = [0.0075, 0.01, 0.0125, 0.005, 0.015][::-1]
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

        print("st : make model option dict")
        options_list = []
        for drp in drp_arr:
            for bn_flag in bn_arr:
                if not bn_flag and drp == 0.0:
                    continue
                for cnv1_krnl_num in cnv1_krnl_arr:
                    for krnl_dvd in [1, 2]:
                        cnv2_krnl_num = cnv1_krnl_num // krnl_dvd
                        for fc_1_num in fc_1_arr:
                            for lr in lr_arr:
                                if not bn_flag:
                                    if lr_arr[0] != lr:
                                        continue
                                    lr = 0.001
                                options_dict = {
                                    "conv1": {
                                        "knl_num": cnv1_krnl_num
                                        , "filt_shp": (4, 3)
                                        , "strd": filt_stride
                                        , "pad": pad
                                        , "acti": "relu"
                                        , "drp": drp
                                        , "bn_flag": bn_flag
                                        , "lr": lr
                                    }
                                    , "conv2": {
                                        "knl_num": cnv2_krnl_num
                                        , "filt_shp": (1, 2)
                                        , "strd": filt_stride
                                        , "pad": pad
                                        , "acti": "relu"
                                        , "drp": drp
                                        , "bn_flag": bn_flag
                                    }
                                    , "conv2_1": {
                                        "knl_num": cnv2_krnl_num
                                        , "filt_shp": (1, 2)
                                        , "strd": filt_stride
                                        , "pad": pad
                                        , "acti": "relu"
                                        , "drp": drp
                                        , "bn_flag": bn_flag
                                    }
                                    , "conv3": {
                                        "knl_num": cnv2_krnl_num
                                        , "filt_shp": (1, 3)
                                        , "strd": filt_stride
                                        , "pad": pad
                                        , "acti": "relu"
                                        , "drp": drp
                                        , "bn_flag": bn_flag
                                    }
                                    , "conv3_1": {
                                        "knl_num": cnv2_krnl_num
                                        , "filt_shp": (1, 3)
                                        , "strd": filt_stride
                                        , "pad": pad
                                        , "acti": "relu"
                                        , "drp": drp
                                        , "bn_flag": bn_flag
                                    }
                                    , "conv5": {
                                        "knl_num": cnv2_krnl_num
                                        , "filt_shp": (1, 5)
                                        , "strd": filt_stride
                                        , "pad": pad
                                        , "acti": "relu"
                                        , "drp": drp
                                        , "bn_flag": bn_flag
                                    }
                                    , "conv_concate": {
                                        "knl_num": cnv1_krnl_num
                                        , "filt_shp": (1, 2)
                                        , "strd": filt_stride
                                        , "pad": pad
                                        , "acti": "relu"
                                        , "drp": drp
                                        , "bn_flag": bn_flag
                                    }
                                    , "fc_1": {
                                        "w_num": fc_1_num
                                        , "acti": "relu"
                                        , "drp": drp
                                        , "bn_flag": bn_flag
                                    }
                                    , "feat": {
                                        "w_num": layer_f_num
                                        , "acti": "relu"
                                        , "drp": drp
                                        , "bn_flag": bn_flag
                                    }
                                    , "last": {
                                        "w_num": layer_last_num
                                        , "acti": "relu"
                                        , "drp": drp
                                        , "bn_flag": bn_flag
                                    }
                                }
                                options_list.append(options_dict)
        print("DONE : make model option dict")

        for opts_dict in options_list:
            learning_rate = opts_dict["conv1"]["lr"]
            cnv1 = opts_dict["conv1"]["knl_num"]
            cnv2 = opts_dict["conv2"]["knl_num"]
            cnvCon = opts_dict["conv_concate"]["knl_num"]
            fc_1 = opts_dict["fc_1"]["w_num"]
            bn_f = opts_dict["fc_1"]["bn_flag"]

            mdl_strctr_flag = False
            tmp_drp = opts_dict["conv1"]["drp"]
            if tmp_drp != init_drp:
                mdl_strctr_flag = True
                init_drp = tmp_drp

            for func_idx in range(len(loss_func_list)):

                for mntr_tmp in mntr_arr:
                    if loss_nms_list[func_idx] == 'pearson' and mntr_tmp == 'val_loss':
                        continue

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

                        model_opt = 'lr_{}_bn_{}_drp_{}_bat_{}_k1_{}_k2_{}_cnvCon_{}'.format(learning_rate, bn_f,
                                                                                             tmp_drp, batch_size, cnv1,
                                                                                             cnv2, cnvCon)

                        chkpnt_path = '/{}'.format(model_opt)
                        chkpnt_path_tmp = chkpnt_path
                        chkpnt_path = chkpnt_path_tmp + "/Train_" + str(i) + "/"
                        ouput_path = "./output/"
                        ################# en path ############################

                        if is_in_google_colab:
                            deep_model = DeepModels()  # colab
                        else:
                            deep_model = ModelGenerator.DeepModels()

                        sngl_oput = deep_model.model_1cnv_incptn_w_mx(opts_dict)

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

                        model = keras.Model([deep_model.main_input, deep_model.feature_input], mdl_trgt_arr)

                        model.summary()
                        print("model_opt :::", model_opt)

                        model.compile(
                            loss=loss_func_list[func_idx]
                            , optimizer=keras.optimizers.Adam(lr=learning_rate, clipvalue=0.5, clipnorm=1.0,
                                                              beta_1=0.9, beta_2=0.999,
                                                              epsilon=None, decay=0.0)
                            , metrics=["mse", custm_loss.pearson_loss]
                        )

                        os.makedirs(model_dir + chkpnt_path, exist_ok=True)
                        # if mdl_strctr_flag:
                        #     tf.keras.utils.plot_model(model, to_file=model_dir + model_opt + '.png')
                        #     mdl_strctr_flag = False

                        model_path = model_dir + chkpnt_path + '/{epoch}' + '_fc_1_{}_feat_{}_last_{}.h5'.format(
                            fc_1, layer_f_num, layer_last_num)

                        early_stopping = keras.callbacks.EarlyStopping(monitor=mntr_tmp, patience=30)

                        checkpointer = keras.callbacks.ModelCheckpoint(filepath=model_path
                                                                       # , save_weights_only=True
                                                                       , monitor=mntr_tmp
                                                                       , verbose=1
                                                                       , mode='min'
                                                                       , save_best_only=True
                                                                       )

                        if load_model_epoch == 0:
                            if batch_size == 0:
                                batch_size = train_X.shape[0]

                            history = model.fit(
                                {"main_input": train_X, "feature_input": train_Feature}
                                , trn_trgt_arr
                                , epochs=epoch
                                , batch_size=batch_size
                                , verbose=1
                                , shuffle=True
                                , validation_data=([test_x, test_f], tst_trgt_arr)
                                , callbacks=[early_stopping, checkpointer]
                            )

                            if batch_size == train_X.shape[0]:
                                batch_size = 0

                            # import matplotlib.pyplot as plt
                            #
                            # hist_nm_0 = 'loss'
                            # plt.plot(history.history[hist_nm_0])
                            # plt.plot(history.history['val_' + hist_nm_0])
                            # plt.title('Model_' + hist_nm_0)
                            # plt.ylabel(hist_nm_0)
                            # plt.xlabel('Epoch')
                            # plt.legend(['Train', 'Test'], loc='upper left')
                            # # if is_in_google_colab:
                            # #     plt.show()
                            # # else:
                            # #     plt.savefig(model_dir + chkpnt_path + model_opt + "_" + hist_nm_0 + ".png")
                            # plt.savefig(model_dir + chkpnt_path + hist_nm_0 + ".png")
                            # plt.close()

                            test_f_list = list(test_f)

                            print("DONE opt [ {} ]".format(model_opt))

                            # get best model results_list
                            h5_file_list = util.get_files_from_dir(model_dir + chkpnt_path + "/*.h5")
                            if SYSTEM_NM == 'Linux':
                                # REAL
                                new_h5_list = [[tmp_path, int(tmp_path.split("/")[-1].split("_")[0])] for tmp_path
                                               in
                                               h5_file_list]  # linux
                            else:
                                # DEV
                                new_h5_list = [
                                    [tmp_path, int(tmp_path[tmp_path.index("\\") + len("\\"):].split("_")[0])]
                                    for tmp_path in h5_file_list]  # window
                            sorted_h5_list = sorted(new_h5_list, key=lambda new_h5_list: new_h5_list[-1])

                            # # If you load model only for prediction (w_out training), you need to set compile flag to False
                            try:
                                last_model = keras.models.load_model(sorted_h5_list[-1][0], compile=False)
                                if isinstance(loss_nms_list[func_idx], str) or loss_num == 1:
                                    y_pred_last = last_model.predict([test_x, test_f])
                                elif "mse" in loss_nms_list[func_idx]:
                                    y_pred_last = last_model.predict([test_x, test_f])[
                                        loss_nms_list[func_idx].index("mse")]
                                else:
                                    y_pred_last = last_model.predict([test_x, test_f])[
                                        loss_nms_list[func_idx].index("pearson")]

                                os.makedirs(ouput_path, exist_ok=True)
                                append_path = ouput_path + "/append_" + fl_nm + "_Train_" + str(i) + "_lr_" + str(
                                    learning_rate) + ".txt"
                                # append_path = ouput_path + "/append_" + fl_nm + "_Train_" + str(i) + ".txt"
                                str_epoch = str(sorted_h5_list[-1][1])
                                mdl_op_sum = model_opt
                                mdl_op_sum += "_bestEpoch_" + str_epoch

                                print("y_pred_last", y_pred_last.shape)
                                print("ans_y", ans_y.shape)
                                sprmn = util.append_model_pearsonr_spearmanr(append_path, ans_y, y_pred_last,
                                                                             start_time,
                                                                             mdl_op_sum)
                            except Exception as err:
                                print(err)
                                pass

    print("::::::::::: %.2f seconds ::::::::::::::" % (time() - start_time))
