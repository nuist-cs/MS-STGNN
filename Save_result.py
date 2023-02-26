import numpy as np
import matplotlib.pyplot as plt
import math
from metrics import MAE, MAPE, RMSE
import csv

def show_pred(all_y_true, all_predict_values):
    # all_y_true = all_y_true.reshape(all_y_true.shape[0], int(math.sqrt(all_y_true.shape[1])), -1)
    # all_predict_values = all_predict_values.reshape(all_predict_values.shape[0], int(math.sqrt(all_predict_values.shape[1])), -1)
    header = ['test_y', 'predicted_values']
    ele_values = np.concatenate((all_y_true[:, :1], all_predict_values[:, :1]), axis=1)
    with open('./result/ele.csv', 'a', encoding='utf-8', newline='') as fp:
        # 写
        writer = csv.writer(fp)
        # 设置第一行标题头
        writer.writerow(header)
        # 将数据写入
        writer.writerows(ele_values)

    cooling_values = np.concatenate((all_y_true[:, 1:2], all_predict_values[:, 1:2]), axis=1)
    with open('./result/cooling.csv', 'a', encoding='utf-8', newline='') as fp:
        # 写
        writer = csv.writer(fp)
        # 设置第一行标题头
        writer.writerow(header)
        # 将数据写入
        writer.writerows(cooling_values)

    heating_values = np.concatenate((all_y_true[:, 2:3], all_predict_values[:, 2:3]), axis=1)
    with open('./result/heating.csv', 'a', encoding='utf-8', newline='') as fp:
        # 写
        writer = csv.writer(fp)
        # 设置第一行标题头
        writer.writerow(header)
        # 将数据写入
        writer.writerows(heating_values)


    mae1 = MAE(all_y_true[:,:1], all_predict_values[:,:1])
    mape1 = MAPE(all_y_true[:,:1], all_predict_values[:,:1])
    rmase1 = RMSE(all_y_true[:,:1], all_predict_values[:,:1])
    predict = all_predict_values[:,:1]
    Ytest = all_y_true[:,:1]
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation1 = (correlation[index]).mean()
    print("============电负荷===================")
    print("MAE = " + str(mae1))
    print("MAPE = " + str(mape1))
    print("RMSE = " + str(rmase1))
    print("acc = " + str(correlation1))
    print("============cooling==================")
    mae2 = MAE(all_y_true[:,1:2], all_predict_values[:,1:2])
    mape2 = MAPE(all_y_true[:,1:2], all_predict_values[:,1:2])
    rmase2 = RMSE(all_y_true[:,1:2], all_predict_values[:,1:2])
    predict = all_predict_values[:,1:2]
    Ytest = all_y_true[:,1:2]
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation2= (correlation[index]).mean()
    print("MAE = " + str(mae2))
    print("MAPE = " + str(mape2))
    print("RMSE = " + str(rmase2))
    print("acc = " + str(correlation2))

    print("==============heating=================")
    mae3 = MAE(all_y_true[:,2:3], all_predict_values[:,2:3])
    mape3 = MAPE(all_y_true[:,2:3], all_predict_values[:,2:3])
    rmase3 = RMSE(all_y_true[:,2:3], all_predict_values[:,2:3])

    predict = all_predict_values[:,2:3]
    Ytest = all_y_true[:,2:3]
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation3 = (correlation[index]).mean()

    print("MAE = " + str(mae3))
    print("MAPE = " + str(mape3))
    print("RMSE = " + str(rmase3))
    print("acc = " + str(correlation3))

    with open('./result/data.txt', 'a') as f:  # 设置文件对象
        print("============电负荷===================", file=f, flush = True)
        print("MAE = " + str(mae1), file=f, flush = True)
        print("MAPE = " + str(mape1), file=f, flush = True)
        print("RMSE = " + str(rmase1), file=f, flush = True)
        print("acc = " + str(correlation1), file=f, flush = True)
        print("============cooling==================", file=f, flush = True)
        print("MAE = " + str(mae2), file=f, flush = True)
        print("MAPE = " + str(mape2), file=f, flush = True)
        print("RMSE = " + str(rmase2), file=f, flush = True)
        print("acc = " + str(correlation2), file=f, flush = True)
        print("==============heating=================", file=f, flush = True)
        print("MAE = " + str(mae3), file=f, flush = True)
        print("MAPE = " + str(mape3), file=f, flush = True)
        print("RMSE = " + str(rmase3), file=f, flush = True)
        print("acc = " + str(correlation3), file=f, flush = True)



    node_id = 0
    time = 24 * 31
    plt.figure(figsize=(20, 10)) # 宽度、高度
    plt.title("electricity")
    plt.xlabel("time/one_hour")
    plt.ylabel("electricity")
    plt.plot(all_y_true[:time, node_id], linewidth=6.0, label='true')
    plt.plot(all_predict_values[:time, node_id], linewidth=1.0, label='pred')
    plt.legend()
    plt.savefig("./assets/the first month pred electricity.png")
    # plt.show()


    node_id = 1
    time = 24 * 31
    plt.figure(figsize=(20, 10)) # 宽度、高度
    plt.title("cooling")
    plt.xlabel("time/one_hour")
    plt.ylabel("electricity")
    plt.plot(all_y_true[:time, node_id], linewidth=6.0, label='true')
    plt.plot(all_predict_values[:time, node_id], linewidth=1.0, label='pred')
    plt.legend()
    plt.savefig("./assets/the first month pred cooling.png")
    # plt.show()



    node_id = 2
    time = 24 * 31
    plt.figure(figsize=(20, 10)) # 宽度、高度
    plt.title("heating")
    plt.xlabel("time/one_hour")
    plt.ylabel("heating")
    plt.plot(all_y_true[:time, node_id], linewidth=6.0, label='true')
    plt.plot(all_predict_values[:time, node_id], linewidth=1.0, label='pred')
    plt.legend()
    plt.savefig("./assets/the first month pred heating.png")
    # plt.show()


    mae = MAE(all_y_true,all_predict_values)
    rmse = RMSE(all_y_true,all_predict_values)
    mape = MAPE(all_y_true,all_predict_values)

    print("ST-GCN基于原始值的精度指标  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse, mape))
    with open('./result/data.txt', 'a') as f:  # 设置文件对象
        print("ST-GCN基于原始值的精度指标  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse, mape), file=f)

