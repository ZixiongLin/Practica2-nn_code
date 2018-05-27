import gzip
import pickle as cPickle

import tensorflow as tf
import numpy as np


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


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print(train_y[57])

# TODO: the neural net!!
"""Convertimos en one_hot()"""
"""NOTA: one_hot, es un array relleno de 0's y un 1"""
train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)

"""Matriz x de Tensores, filas indefinidas y 784 columnas"""
x = tf.placeholder(tf.float32, [None, 784])  # samples
""""Matriz y_ de Tnesores, filas indefinidas y de 10 columnas"""
y_ = tf.placeholder(tf.float32, [None, 10])  # labels

"""Neurona 1, su peso w1 y su valor b1"""
W1 = tf.Variable(np.float32(np.random.rand(784, 100)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(100)) * 0.1)

"""Neurona 2, su pedo w2 y su valor b2"""
W2 = tf.Variable(np.float32(np.random.rand(100, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

"""Aplicamos la sigmoide"""
h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!

"""Error"""
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

"""Perdida"""
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

"""Descenso del gradiente, minimizando al valor de loss"""
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

"""Inicializar las variables"""
init = tf.initialize_all_variables()

"""Obejto session...."""
sess = tf.Session()
"""... activa las funciones"""
sess.run(init)

entrenamiento = []
validacion = []

sess = tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 20
epoch = 0
diferencia = 100.0

while diferencia > 0.001:
    epoch += 1
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    data_train = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    entrenamiento.append(data_train)
    print("Epoca: ", epoch, "Error: ", data_train)
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print(b, "-->", r)
    print("----------------------------------------------------------------------------------")

    data_valid = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    validacion.append(data_valid)
    print("Epoca validacion: ", epoch, " Error: ", data_valid)
    if epoch > 1:
        diferencia = abs(validacion[-2] - data_valid)
    print("Diferencia de: ", diferencia)
    print("====================")

print("=====================")
print("=====Resultado=======")
print("=====================")

total = 0.0
error = 0.0
test_data = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, test_data):
    if np.argmax(b) != np.argmax(r):
        error += 1
    total += 1
fallo = error / total * 100
print("El porcentaje de error es: ", fallo, "5 y el de exito ", (100 - fallo), "%")
print(entrenamiento[-1])
plt.ylabel('Errores')
plt.xlabel('Epocas')
tr_handle, = plt.plot(entrenamiento)
vl_handle, = plt.plot(validacion)
plt.legend(handles=[tr_handle, vl_handle],
           labels=['Error entrenamiento', 'Error validacion'])
plt.show()
plt.savefig('Grafica.png')
