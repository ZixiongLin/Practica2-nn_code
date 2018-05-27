import tensorflow as tf
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plot
# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
"""x_data cogen las 4 primeras filas despues de barajarlas"""
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data

y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code
"""Seleccion de datos"""
rango_x = int(np.floor(len(x_data) * 7 / 10))
rango_y = int(np.floor(len(y_data) * 7 / 10))

x_train = x_data[:rango_x]
y_train = y_data[:rango_y]

validacion_x = rango_x + int(np.floor(len(x_data)*0.15))
validacion_y = rango_y + int(np.floor(len(y_data)*0.15))

x_valid = x_data[rango_x:validacion_x]
y_valid = y_data[rango_y:validacion_y]

x_test = x_data[validacion_x:]
y_test = y_data[validacion_y:]

"""======================="""
print ("\nSome samples...")
for i in range(20):
    print (x_data[i], " -> ", y_data[i])
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))
print(loss)
train =tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

entrenamiento = []
validacion = []

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 20
epoch = 0
diferencia = 100.0

"""Primer entrenamiento"""
epoch += 1
for jj in range(int(len(x_train) / batch_size)):
    batch_xs = x_train[jj * batch_size: jj * batch_size + batch_size]
    batch_ys = y_train[jj * batch_size: jj * batch_size + batch_size]
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
data_train = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
entrenamiento.append(data_train)
print("Epoca: ", epoch, "Error: ", data_train)
result = sess.run(y, feed_dict={x: batch_xs})
for b, r in zip(batch_ys, result):
    print(b, "-->", r)
print("----------------------------------------------------------------------------------")
data_valid = sess.run(loss, feed_dict={x: x_valid, y_: y_valid})
validacion.append(data_valid)
print("Epoca validacion: ", epoch, " Error: ", data_valid)
while diferencia > 0.0001:
    epoch += 1
    for jj in range(int(len(x_train) / batch_size)):
        batch_xs = x_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    data_train = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    entrenamiento.append(data_train)
    print("Epoca: ", epoch, "Error: ",data_train)
    result = sess.run(y,feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print (b, "-->", r)
    print ("----------------------------------------------------------------------------------")

    data_valid = sess.run(loss, feed_dict={x: x_valid, y_: y_valid})
    validacion.append(data_valid)
    print("Epoca validacion: ", epoch," Error: ", data_valid)
    diferencia=abs(validacion[-2]-data_valid)
    print("Diferencia de: ", diferencia)
    print("====================")

print("=====================")
print("=====Resultado=======")
print("=====================")

total = 0.0
error = 0.0
test_data = sess.run(y, feed_dict={x:x_test})
for b, r in zip (y_test, test_data):
    if(np.argmax(b)!=np.argmax(r)):
        error +=1
    total+=1
fallo = error /total *100
print("El porcentaje de error es: ",fallo,"5 y el de exito ", (100-fallo),"%")
print(entrenamiento[-1])
plot.ylabel('Errores')
plot.xlabel('Epocas')
tr_handle, = plot.plot(entrenamiento)
vl_handle, = plot.plot(validacion)
plot.legend(handles=[tr_handle, vl_handle],
            labels=['Error entrenamiento', 'Error validacion'])
plot.show()
plot.savefig('Grafica_entrenamiento_validacion_while.png')








