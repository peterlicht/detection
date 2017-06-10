def train_classes(stage, train_coils, train_set, train_labels, test_coils, test_set, test_labels, train_labels_n, test_labels_n, features, learning_rate, max_steps, rng_str, dropout, summaries_dir, variables_dir, mode, anew=False):
    from functions import get_stochastic_batch
    import os
    from matplotlib import pyplot as plt
    import tensorflow as tf
    from classification_functions import bias_variable, weight_variable, max_pool_2x2, conv2d

    stagex = 2 ** stage
    stagey = 2 ** (stage + 1)
    tiles = stagex * stagey
    stage_str = str(stage)

    variables_path = variables_dir + '/' + stage_str + '/' + rng_str
    variables_file = variables_path + '/variables'

    sess = tf.InteractiveSession()

    tf.histogram_summary('train_labels', train_labels_n)
    tf.histogram_summary('test_labes', test_labels_n)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, tiles], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        tf.histogram_summary('batch', tf.argmax(y_, 1))

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1, stagey, stagex, 1])
        tf.image_summary('input', x_image, 10)
        print(x_image.get_shape())

    with tf.name_scope('conv_pool1'):
        W_conv1 = weight_variable([4, 4, 1, features])
        b_conv1 = bias_variable([features])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        print('h_c1:', h_conv1.get_shape())
        print('h_p1:', h_pool1.get_shape())

    with tf.name_scope('conv_pool2'):
        W_conv2 = weight_variable([4, 4, features, 2 * features])
        b_conv2 = bias_variable([2 * features])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        print('h_c2:', h_conv2.get_shape())
        print('h_p2:', h_pool2.get_shape())

    with tf.name_scope('layer1'):
        W_fc1 = weight_variable([int(tiles / 16) * 2 * features, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, int(tiles / 16) * 2 * features])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        print('h_p2f:', h_pool2_flat.get_shape())
        print('h_fc1:', h_fc1.get_shape())

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(h_fc1, keep_prob)
        print('drop:', dropped.get_shape())
    with tf.name_scope('layer2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.nn.softmax(tf.matmul(dropped, W_fc2) + b_fc2)
        print('y_c:', y_conv.get_shape())
        print('y_:', y_.get_shape())

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv + 1e-10))
        tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    merged = tf.merge_all_summaries()

    train_writer = tf.train.SummaryWriter(summaries_dir + '/' + stage_str + '/' + rng_str + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(summaries_dir + '/' + stage_str + '/' + rng_str + '/test')

    saver = tf.train.Saver()
    # Restores training variables or creates new file for it
    if not anew:
        if os.path.isfile(variables_file):
            saver.restore(sess, variables_file)
        else:
            init = tf.initialize_all_variables()
            sess.run(init)
    else:
        init = tf.initialize_all_variables()
        sess.run(init)

    # Testing coil: 13536638


    if mode == 'train' or mode == 'both':
        for i in range(max_steps):

            if i % 10 == 0:  # Record summaries and test-set accuracy
                batch_coils, batch_xs, batch_ys = get_stochastic_batch(test_coils, test_set, test_labels, size=80)
                k = 1.0
                summary, acc = sess.run([merged, accuracy], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: k})
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, acc))
            else:  # Record train set summaries, and train
                if i % 100 == 99:  # Record execution stats
                    batch_coils, batch_xs, batch_ys = get_stochastic_batch(train_coils, train_set, train_labels,
                                                                           size=240)
                    k = dropout
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict={x: batch_xs, y_: batch_ys, keep_prob: k},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)
                else:  # Record a summary
                    batch_coils, batch_xs, batch_ys = get_stochastic_batch(train_coils, train_set, train_labels,
                                                                           size=240)
                    k = dropout
                    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: k})
                    train_writer.add_summary(summary, i)

        train_writer.close()
        test_writer.close()

        if not os.path.exists(variables_path):
            os.makedirs(variables_path)
        save_path = saver.save(sess, variables_file)
        print("Model saved in file: %s" % save_path)
    elif mode == 'evaluate' or mode == 'both':
        evaluate = ''
        while evaluate != 'end':
            evaluate = ''
            try:
                evaluate = input('Enter a CoilID to evaluate: ')
                if not evaluate == 'end':
                    evaluate = int(evaluate)
                    eval_index = np.where(train_coils == evaluate)
                    batch_x = np.squeeze(train_set[eval_index, :], axis=1)
                    batch_y = np.squeeze(train_labels[eval_index, :], axis=1)
            except ValueError:
                print('Not a number or end, or ID not found in train set')
            if not evaluate == 'end':
                k = 1.0
                eval_dict = {x: batch_x, y_: batch_y, keep_prob: k}
                classification = sess.run(tf.argmax(y_conv, 1), feed_dict=eval_dict)
                class_real = sess.run(y_conv, feed_dict=eval_dict)
                print(class_real)
                class_sum = np.sum(class_real)
                class_value = class_real[0, classification]
                class_prob = class_value / class_sum * 100
                print('Predicted class:', classification)
                print('Probability:', class_prob)
                print('True class:', np.argmax(batch_y, 1))
                plt.imshow(batch_x.reshape(stagey, stagex), cmap=plt.cm.binary)
                plt.show()
    else:
        print('No training or eval done')
    return