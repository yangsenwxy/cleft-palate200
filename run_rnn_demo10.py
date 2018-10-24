#在run_rnn_demo07的基础上做改进，


# 加入多层LSTM看结果
#还有dropout DropoutWrapper

#对 a 语音数据进行测试

import tensorflow as tf 
from tensorflow.contrib import rnn
import numpy as np
import random

from func_labels_to_onehot import func_labels_to_onehot
from func_write_data_to_mat import func_write_data_to_mat

def run_rnn_demo10(train_data, train_label, test_data, test_label, param_dict, num):
    
    tf.reset_default_graph() 

    lr = param_dict['lr']
    training_iters = param_dict['training_iters']
    #batch_size = param_dict['batch_size']

    n_inputs = param_dict['n_inputs']   # data input
    n_steps = param_dict['n_steps']    # time steps
    n_hidden_units = param_dict['n_hidden_units']   # neurons in hidden layer
    n_classes = param_dict['n_classes']     # classes 

    layer_num = param_dict['layer_num']  #layers num

#    train_data_len = param_dict['train_data_len']
#    train_data_width = param_dict['train_data_width']
#
#    choice_num_not = param_dict['choice_num_not']

    batch_size = tf.placeholder(tf.int32, [])
    keep_prob = tf.placeholder(tf.float32)

    _x = tf.placeholder(tf.float32, [None, train_data_width])
    y = tf.placeholder(tf.float32, [None, n_classes])
    x = tf.reshape(_x, [-1, n_steps, n_inputs])
    
    # tf Graph input
    #x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    #y = tf.placeholder(tf.float32, [None, n_classes])

    # Define weights
    weights = {
        # (28, 128)
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
        # (128, 10)
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
    }
    biases = {
        # (128, )
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
        # (10, )
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    }

    def RNN(X, weights, biases):
        # hidden layer for input to cell
        ########################################

        # transpose the inputs shape from
        # X ==> (128 batch * 28 steps, 28 inputs)
        X = tf.reshape(X, [-1, n_inputs])

        # into hidden
        # X_in = (128 batch * 28 steps, 128 hidden)
        X_in = tf.matmul(X, weights['in']) + biases['in']
        # X_in ==> (128 batch, 28 steps, 128 hidden)
        X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

        # cell
        ##########################################

        # basic LSTM Cell.
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        else:
            cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
        # lstm cell is divided into two parts (c_state, h_state)
        #init_state = cell.zero_state(batch_size, dtype=tf.float32)

        #添加 dropout layer, 一般只设置 output_keep_prob
        cell = rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        #调用 MultiRNNCell 来实现多层 LSTM
        cell = rnn.MultiRNNCell([cell] * layer_num, state_is_tuple=True)

        # lstm cell is divided into two parts (c_state, h_state)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)

        # You have 2 options for following step.
        # 1: tf.nn.rnn(cell, inputs);
        # 2: tf.nn.dynamic_rnn(cell, inputs).
        # If use option 1, you have to modified the shape of X_in, go and check out this:
        # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
        # In here, we go for option 2.
        # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
        # Make sure the time_major is changed accordingly.
        outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

        # hidden layer for output as the final results
        #############################################
        # results = tf.matmul(final_state[1], weights['out']) + biases['out']

        # # or
        # unpack to list [(batch, outputs)..] * steps
        # unpack to list [(batch, outputs)..] * steps
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
        else:
            outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
        results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

        return results


    pred = RNN(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    pred = tf.nn.softmax(pred)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    with tf.Session() as sess:
        # tf.initialize_all_variables() no long valid from
        # 2017-03-02 if using tensorflow >= 0.12
        _batch_size = param_dict['batch_size']

        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

        step = 0
        train_accuracy_list = []
        test_accuracy_list = []
        every_batch_train_accuracy_lsit = []
        every_train_cost_list = []
        every_test_cost_list = []

        ###########################
#        every_batch_train_cost_list = []

        while step < training_iters:
            
            #batch_xs = train_data
            #batch_ys = train_label
            batch_xs = []
            batch_ys = []
            index1= []
            num_of_batch = 0
            #batch_xs = random.sample(train_data, _batch_size)
            #batch_ys = random.sample(train_label, _batch_size)
            while num_of_batch < _batch_size:
                temp = random.randint(0, (train_data_len - 1))
                if temp not in index1:
                    index1.append(temp)
                    num_of_batch = num_of_batch + 1

            for temp in index1:
                batch_xs.append(train_data[temp])
                batch_ys.append(train_label[temp])

            sess.run([train_op], feed_dict={
                _x: batch_xs,
                y: batch_ys,
                keep_prob: 0.5, 
                batch_size: _batch_size
            })

            if step % 1 == 0:
                every_batch_train_accuracy = sess.run(accuracy, feed_dict={
                    _x: batch_xs,
                    y: batch_ys,
                    keep_prob: 1.0, 
                    batch_size: _batch_size
                    })
                every_batch_train_accuracy_lsit.append(every_batch_train_accuracy)
                print('step %d, every_batch_train_accuracy: %g' % ( step, every_batch_train_accuracy ) )

                #loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen}) #计算每个batch训练时的损失Minibatch Loss

                ###################################
                ###### 得到损失函数的输出
#                every_batch_train_cost = sess.run(cost, feed_dict={
#                    _x: batch_xs,
#                    y: batch_ys,
#                    keep_prob: 1.0, 
#                    batch_size: _batch_size
#                    })
#                every_batch_train_cost_list.append(every_batch_train_cost)

                train_accuracy = sess.run(accuracy, feed_dict={
                    _x: train_data,
                    y: train_label,
                    keep_prob: 1.0, 
                    batch_size: train_data_len
                    })
                train_accuracy_list.append(train_accuracy)
                print('step %d, training accuracy: %g' % ( step, train_accuracy ) )

                test_accuracy_ = sess.run(accuracy, feed_dict={
                    _x: test_data,
                    y: test_label,
                    keep_prob: 1.0, 
                    batch_size: choice_num_not
                    })
                test_accuracy_list.append(test_accuracy_)
                print("Testing Accuracy: %g" % ( test_accuracy_ ) )
                
                ##### 得到损失函数的输出
                every_train_cost = sess.run(cost, feed_dict={
                    _x: train_data,
                    y: train_label,
                    keep_prob: 1.0, 
                    batch_size: train_data_len
                    })
                every_train_cost_list.append(every_train_cost)
                
                every_test_cost = sess.run(cost, feed_dict={
                    _x: train_data,
                    y: train_label,
                    keep_prob: 1.0, 
                    batch_size: train_data_len
                    })
                every_test_cost_list.append(every_test_cost)

            step += 1

        print("Optimization Finished!")

        #将每次step的正确率存储在一个accuracy_list中，并保存在mat中
        mat_name3 = "every_batch_train_accuracy_lsit" + str(num) + ".mat"
        data_name3 = "every_batch_train_accuracy_lsit"
        func_write_data_to_mat(mat_name3, data_name3, every_batch_train_accuracy_lsit) #保存输出到mat文件
        
        mat_name4 = "train_accuracy_list" + str(num) + ".mat"
        data_name4 = "train_accuracy_list"
        func_write_data_to_mat(mat_name4, data_name4, train_accuracy_list) #保存输出到mat文件
        
        mat_name5 = "test_accuracy_list" + str(num) + ".mat"
        data_name5 = "test_accuracy_list"
        func_write_data_to_mat(mat_name5, data_name5, test_accuracy_list) #保存输出到mat文件
        
        #保存每次训练过程的损失
        mat_name5 = "every_train_cost_list" + str(num) + ".mat"
        data_name5 = "every_train_cost_list"
        func_write_data_to_mat(mat_name5, data_name5, every_train_cost_list) #保存输出到mat文件
        
        #保存每次训练过程的损失
        mat_name5 = "every_test_cost_list" + str(num) + ".mat"
        data_name5 = "every_test_cost_list"
        func_write_data_to_mat(mat_name5, data_name5, every_test_cost_list) #保存输出到mat文件

        all_data_train_accuracy = sess.run(accuracy, feed_dict={
                    _x: train_data,
                    y: train_label,
                    keep_prob: 1.0, 
                    batch_size: train_data_len
                    })
#        mat_name4 = "all_data_train_accuracy" + ".mat"
#        data_name4 = "all_data_train_accuracy"
#        func_write_data_to_mat(mat_name4, data_name4, all_data_train_accuracy) #保存输出到mat文件
        print('train done, all data training accuracy: %g' % ( all_data_train_accuracy ) )

        test_accuracy = sess.run(accuracy, feed_dict={
                    _x: test_data,
                    y: test_label,
                    keep_prob: 1.0, 
                    batch_size: choice_num_not
                    })

#        mat_name5 = "test_accuracy" +".mat"
#        data_name5 = "test_accuracy"
#        func_write_data_to_mat(mat_name5, data_name5, test_accuracy) #保存输出到mat文件
        print("Testing Accuracy: %g" % ( test_accuracy ) )
        
        #处理测试数据标签并保存
        out_y_test = sess.run(pred, feed_dict={
            _x: test_data,
            #y: batch_ys,
            keep_prob: 1.0, 
            batch_size: choice_num_not
            })
        out_y_test = sess.run(tf.argmax(out_y_test, 1))
        out_y_test = out_y_test.tolist()
        #out_y = (tf.argmax(out_y, 1)).eval() #使用这种形式转换成的是 [2 1 3 1 2 3 2 3 1 0] 形式，没有办法编码成one-hot形式
        out_y_test = func_labels_to_onehot(out_y_test, n_classes)
        out_y_test = (out_y_test.eval())
        mat_name6 = "out_y_test" + str(num) +".mat"
        data_name6 = "out_y_test"
        func_write_data_to_mat(mat_name6, data_name6, out_y_test) #保存输出到mat文件
        
        return all_data_train_accuracy, test_accuracy

        #下面处理将得到的结果和原始的标签 保存在两个不同的 mat 文件中
#        out_y_all = sess.run(pred, feed_dict={
#            _x: train_data,
#            #y: batch_ys,
#            keep_prob: 1.0, 
#            batch_size: train_data_len
#            })
        # mat_name1 = "batch_ys" + str(step) + ".mat"
        # data_name1 = "batch_ys" + str(step)
        # func_write_data_to_mat(mat_name1, data_name1, batch_ys) #保存输出到mat文件

#        out_y_all = sess.run(tf.argmax(out_y_all, 1))
#        out_y_all = out_y_all.tolist()
#        #out_y = (tf.argmax(out_y, 1)).eval() #使用这种形式转换成的是 [2 1 3 1 2 3 2 3 1 0] 形式，没有办法编码成one-hot形式
#        out_y_all = func_labels_to_onehot(out_y_all, n_classes)
#        out_y_all = (out_y_all.eval())
#        mat_name2 = "out_y_all" +".mat"
#        data_name2 = "out_y_all"
#        func_write_data_to_mat(mat_name2, data_name2, out_y_all) #保存输出到mat文件
        #print(out_y)
        
        #处理测试数据标签并保存
#        out_y_test = sess.run(pred, feed_dict={
#            _x: test_data,
#            #y: batch_ys,
#            keep_prob: 1.0, 
#            batch_size: choice_num_not
#            })
#        out_y_test = sess.run(tf.argmax(out_y_test, 1))
#        out_y_test = out_y_test.tolist()
#        #out_y = (tf.argmax(out_y, 1)).eval() #使用这种形式转换成的是 [2 1 3 1 2 3 2 3 1 0] 形式，没有办法编码成one-hot形式
#        out_y_test = func_labels_to_onehot(out_y_test, n_classes)
#        out_y_test = (out_y_test.eval())
#        mat_name6 = "out_y_test" +".mat"
#        data_name6 = "out_y_test"
#        func_write_data_to_mat(mat_name6, data_name6, out_y_test) #保存输出到mat文件

#        mat_name7 = "train_label" +".mat"
#        data_name7 = "train_label"
#        func_write_data_to_mat(mat_name7, data_name7, train_label) #保存输出到mat文件
#        mat_name8 = "test_label" +".mat"
#        data_name8 = "test_label"
#        func_write_data_to_mat(mat_name8, data_name8, test_label) #保存输出到mat文件

        ########################
        #############保存every_batch_train_cost_list
#        mat_name9 = "every_batch_train_cost_list" +".mat"
#        data_name9 = "every_batch_train_cost_list"
#        func_write_data_to_mat(mat_name9, data_name9, every_batch_train_cost_list) #保存输出到mat文件
    
