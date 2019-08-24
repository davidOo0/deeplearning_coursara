#引入必要的包
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# 必要函数
def sigmoid(z):
    a=1/(1+np.exp(-z))
    return a

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b=0
    return w,b

def propagate(w,b,X,Y):
    m=X.shape[1] #一共多少图像
    A=sigmoid(np.dot(w.T,X)+b)
    cost=-1/m * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

    dw = 1/m * np.dot(X,(A-Y).T)
    db=1/m * np.sum(A-Y)
    
    assert(dw.shape == w.shape)#assert断言 保证矩阵正确
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):#梯度下降算法
    costs = []
    for i in range(num_iterations):#迭代次数
        (grads, cost) = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return ( params, grads, costs)


def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1   
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    (w, b) = initialize_with_zeros(X_train.shape[0])#一共多少张图片
    (parameters, grads, costs) = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)#
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

#主函数
if __name__=='__main__':
    # 导入/加载数据集dataset (cat/non-cat)
    (train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes)= load_dataset()
    #train_set_x_orig 训练的图像  test_set_x_orig 测试的图像

    '''
    index = 2
    plt.imshow(test_set_x_orig[index])
    plt.show()
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
    # 测试可视化
    '''
    m_train = train_set_x_orig.shape[0]   #shape[0]计算第一维（m ）
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]      #shape[1]计算第二维（nx ）
    '''
    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    '''
    #dataset 数组reshape
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  #-1是让电脑帮你算一共多少行
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    '''
    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
    '''
    #标准化数据集  /255 RBG通道
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.

    #建立逻辑回归模型
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 3000,learning_rate = 0.005, print_cost = True)

