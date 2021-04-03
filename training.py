from keras import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils

# 定义全连接网络模型
model = Sequential()
model.add(Dense(units=512, input_shape=(784,), activation='relu'))  # 第一层隐藏层
model.add(Dense(units=10, activation='softmax'))                    # 输出层

# 编译静态图
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics='accuracy')

# 加载并处理 mnist 数据集，图片是 28*28=784 分辨率，对图片进行 reshape 展平
(train_x, train_y), (test_x, test_y) = mnist.load_data()
X_train = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2]).astype('float32') / 255
Y_train = np_utils.to_categorical(train_y, num_classes=10)
X_test = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2]).astype('float32') / 255
Y_test = np_utils.to_categorical(test_y, num_classes=10)

# 模型训练
model.fit(X_train, Y_train, epochs=5, batch_size=32)

# 精度验证
loss, accuracy = model.evaluate(X_test, Y_test)
print('Test loss:', loss)
print('Accuracy:', accuracy)
predict = model.predict(X_test)
print('预测值：', predict.argmax())
print('实际值：', test_y[0])