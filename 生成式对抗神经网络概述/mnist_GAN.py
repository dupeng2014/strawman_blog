import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

BATCH_SIZE = 64
UNITS_SIZE = 128
LEARNING_RATE = 0.001
EPOCH = 300
SMOOTH = 0.1

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data',one_hot=True)


# 生成模型
def generatorModel(noise_img, units_size, out_size, alpha=0.01):
    with tf.variable_scope('generator'):
        FC = tf.layers.dense(noise_img, units_size)
        reLu = tf.nn.leaky_relu(FC, alpha)
        drop = tf.layers.dropout(reLu, rate=0.2)
        logits = tf.layers.dense(drop, out_size)
        outputs = tf.tanh(logits)
        return logits, outputs


# 判别模型
def discriminatorModel(images, units_size, alpha=0.01, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        FC = tf.layers.dense(images, units_size)
        reLu = tf.nn.leaky_relu(FC, alpha)
        logits = tf.layers.dense(reLu, 1)
        outputs = tf.sigmoid(logits)
        return logits, outputs


# 损失函数
"""
判别器的目的是：
1. 对于真实图片，D要为其打上标签1
2. 对于生成图片，D要为其打上标签0
生成器的目的是：对于生成的图片，G希望D打上标签1
"""


def loss_function(real_logits, fake_logits, smooth):
    # 生成器希望判别器判别出来的标签为1; tf.ones_like()创建一个将所有元素都设置为1的张量
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                    labels=tf.ones_like(fake_logits) * (1 - smooth)))
    # 判别器识别生成器产出的图片，希望识别出来的标签为0
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                       labels=tf.zeros_like(fake_logits)))
    # 判别器判别真实图片，希望判别出来的标签为1
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
                                                                       labels=tf.ones_like(real_logits) * (1 - smooth)))
    # 判别器总loss
    D_loss = tf.add(fake_loss, real_loss)
    return G_loss, fake_loss, real_loss, D_loss


# 优化器
def optimizer(G_loss, D_loss, learning_rate):
    train_var = tf.trainable_variables()
    G_var = [var for var in train_var if var.name.startswith('generator')]
    D_var = [var for var in train_var if var.name.startswith('discriminator')]
    # 因为GAN中一共训练了两个网络，所以分别对G和D进行优化
    G_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=G_var)
    D_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=D_var)
    return G_optimizer, D_optimizer


# 训练
def train(mnist):
    image_size = mnist.train.images[0].shape[0]
    real_images = tf.placeholder(tf.float32, [None, image_size])
    fake_images = tf.placeholder(tf.float32, [None, image_size])

    # 调用生成模型生成图像G_output
    G_logits, G_output = generatorModel(fake_images, UNITS_SIZE, image_size)
    # D对真实图像的判别
    real_logits, real_output = discriminatorModel(real_images, UNITS_SIZE)
    # D对G生成图像的判别
    fake_logits, fake_output = discriminatorModel(G_output, UNITS_SIZE, reuse=True)
    # 计算损失函数
    G_loss, real_loss, fake_loss, D_loss = loss_function(real_logits, fake_logits, SMOOTH)
    # 优化
    G_optimizer, D_optimizer = optimizer(G_loss, D_loss, LEARNING_RATE)

    saver = tf.train.Saver()
    step = 0
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(EPOCH):
            for batch_i in range(mnist.train.num_examples // BATCH_SIZE):
                batch_image, _ = mnist.train.next_batch(BATCH_SIZE)
                # 对图像像素进行scale，tanh的输出结果为(-1,1)
                batch_image = batch_image * 2 - 1
                # 生成模型的输入噪声
                noise_image = np.random.uniform(-1, 1, size=(BATCH_SIZE, image_size))
                #
                session.run(G_optimizer, feed_dict={fake_images: noise_image})
                session.run(D_optimizer, feed_dict={real_images: batch_image, fake_images: noise_image})
                step = step + 1
            # 判别器D的损失
            loss_D = session.run(D_loss, feed_dict={real_images: batch_image, fake_images: noise_image})
            # D对真实图片
            loss_real = session.run(real_loss, feed_dict={real_images: batch_image, fake_images: noise_image})
            # D对生成图片
            loss_fake = session.run(fake_loss, feed_dict={real_images: batch_image, fake_images: noise_image})
            # 生成模型G的损失
            loss_G = session.run(G_loss, feed_dict={fake_images: noise_image})
            print('epoch:', epoch, 'loss_D:', loss_D, ' loss_real', loss_real, ' loss_fake', loss_fake, ' loss_G',
                  loss_G)
            model_path = './Model/' + "mnist.model"
            print(model_path)
            saver.save(session, model_path, global_step=step)


def main(argv=None):
    train(mnist)


if __name__ == '__main__':
    tf.app.run()