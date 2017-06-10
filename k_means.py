import tensorflow as tf
import time

def k_m_tf(defect_tensor, clusters, max_iters, summaries_dir, stage_str, name_str, go_to_max = False):
    length = len(defect_tensor[:,0])
    num_clus = clusters
    MAX_ITERS = max_iters
    tiles = len(defect_tensor[0,:])
    start = time.time()

    sess = tf.InteractiveSession()
    with tf.name_scope('input'):
        points = tf.Variable(tf.random_uniform([length,tiles]), dtype = tf.float32)
    with tf.name_scope('cluster_assigns'):
        cluster_assignments = tf.Variable(tf.zeros([length], dtype = tf.float32))


    with tf.name_scope('cents'):
        centroids = tf.Variable(tf.random_crop(points.initialized_value(), [num_clus,tiles]), dtype = tf.float32)
    # centroids = tf.Print(centroids,[centroids], summarize = 16, message = 'centroids')

    # Replicate to N copies of each centroid and K copies of each
    # point, then subtract and compute the sum of squared distances.
    with tf.name_scope('Replicate'):
        rep_centroids = tf.reshape(tf.tile(centroids, [length, 1]), [length, num_clus, tiles])
        # rep_centroids = tf.Print(rep_centroids,[tf.shape(rep_centroids)],message='shape_rep_centroids')
        rep_points = tf.reshape(tf.tile(points, [1, num_clus]), [length, num_clus, tiles])

    with tf.name_scope('Sum_squares'):
        squares = tf.square(rep_points - rep_centroids)
        sum_squares = tf.reduce_sum(tf.square(squares),
                                    reduction_indices=2)
        squares_1d = tf.scalar_summary('sum_squares', tf.reduce_mean(sum_squares))
        # sum_squares = tf.Print(sum_squares,[sum_squares], summarize = 40, message = 'sum_squares')
        # sum_squares = tf.Print(sum_squares,[tf.shape(sum_squares)], summarize = 16, message = 'sum_squares_shape')

        # Use argmin to select the lowest-distance point
    with tf.name_scope('argmin'):
        best_centroids = tf.argmin(sum_squares, 1)
        # best_centroids = tf.Print(best_centroids,[best_centroids], summarize = 40,  message = ' best_cents')
    did_assignments_change = tf.reduce_any(tf.not_equal(tf.cast(best_centroids, tf.float32),
                                                        cluster_assignments))

## This part exists for counting purposes, since I can't simply access the count in the means part
    with tf.name_scope('counting'):
        const_1d = {}
        num_1d = {}
        found_1d = {}
        scalar_1d = {}

        for i in range(0,num_clus):
             const_1d[i] = tf.constant(i, shape = [320,1],dtype = tf.int64)
            # string_1d[i] = tf.constant(str[i], shape =[320,1], dtype = tf.string)

        for i in range(0,num_clus):
            num_1d[i] = tf.equal(tf.reshape(best_centroids,[320,1]), const_1d[i])
            found_1d[i] = tf.reduce_sum(tf.cast(num_1d[i], tf.int32))
            found_1d[i] = tf.expand_dims(found_1d[i],-1)
            scalar_1d[i] = tf.scalar_summary(str(i),tf.squeeze(found_1d[i]))
            # found_1d[i] = tf.Print(found_1d[i], [found_1d[i]], summarize=40, message=str(i))
            # found_1d[i] = tf.Print(found_1d[i], [tf.shape(found_1d[i])], summarize=40, message=str(i))
            # found_1d[i] = tf.Print(found_1d[i],[tf.expand_dims(found_1d[i],0)], summarize = 40, message =str(i))
            # found_1d[i] = tf.Print(found_1d[i],[tf.shape(tf.expand_dims(found_1d[i],0))], summarize = 40, message =str(i))
            # found_1d[i] = tf.Print(found_1d[i], [tf.shape(tf.reshape(found_1d[i],[1,1]))], summarize=40, message=str(i))

        found_tensor = tf.concat(0, [found_1d[i] for i in range(0, num_clus)])
        distro = tf.histogram_summary('Distribution', found_tensor)






## calculate the means at the indices of best_centroids.
    with tf.name_scope('means'):
        total = tf.unsorted_segment_sum(points, best_centroids, num_clus)
        count = tf.unsorted_segment_sum(tf.ones_like(points), best_centroids, num_clus)
        # count = tf.Print(count, [tf.shape(count)])
        means = total / count
        means = tf.select(tf.is_nan(means), tf.ones_like(means) * 0, means)
        means_1d = tf.scalar_summary('means', tf.reduce_mean(means))
        # means = tf.Print(means,[means],summarize = 16, message = 'MEANS')
        # means = tf.Print(means,[tf.shape(means)], message = 'm_shape')
    # Do not write to the assigned clusters variable until after
    # computing whether the assignments have changed - hence with_dependencies
    with tf.name_scope('Do_updates'):
        with tf.control_dependencies([did_assignments_change]):
            do_updates = tf.group(
                centroids.assign(means),
                cluster_assignments.assign(tf.cast (best_centroids,tf.float32)))



    changed = True
    iters = 0
    found_numerical = {}
    # found_1d = tf.Print(found_1d,[found_1d])

    # Merge summaries
    scalar_summary = tf.merge_summary([scalar_1d[i] for i in range(0,num_clus)])
    other_summary = tf.merge_summary([means_1d, squares_1d])
    histogram_summary = tf.merge_summary([distro])

    writer = tf.train.SummaryWriter(summaries_dir + '/' + stage_str + '/kmeans/' + name_str, sess.graph)
    init = tf.initialize_all_variables()

    sess.run(init)
    # loop
    # check for assignment changes and assign new based on new means. If assignments didnt change, stop.
    while changed and iters < MAX_ITERS:
        iters += 1
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        # if iters%10 == 1:
        [changed, _, histogram_sum_run, scalar_sum_run, other_sum_run] = sess.run([did_assignments_change, do_updates, histogram_summary, scalar_summary, other_summary], feed_dict={points: defect_tensor})
        writer.add_run_metadata(run_metadata, 'step%03d' % iters)
        writer.add_summary(histogram_sum_run, iters)
        writer.add_summary(scalar_sum_run, iters)
        writer.add_summary(other_sum_run,iters)
        # else:
        #     [changed, _, scalar_sum_run] = sess.run([did_assignments_change, do_updates, scalar_summary], feed_dict={points: defect_tensor})
        #     writer.add_run_metadata(run_metadata, 'step%03d' % iters)
        #     writer.add_summary(scalar_sum_run, iters)


        ## Note: due to the interconnectivity of found_1d, it seems as you need to run it ALONG the session a couple lines before in order to get numerical results
        ## Can't do that in a seperate run. Weirdly enough it works for found_tensor, which is simply a concat of found_1d. I don't know why.
        # found_numerical[0] = sess.run([found_1d[0]], feed_dict={points:defect_tensor})
        found_numerical[1] = sess.run([found_1d[1]], feed_dict={points:defect_tensor})
        found_numerical[3] = sess.run([found_1d[3]], feed_dict={points:defect_tensor})
        found_numerical[4] = sess.run([found_1d[4]], feed_dict={points:defect_tensor})

        if  go_to_max == True:
            changed = True
    writer.close()
    [centers, assignments] = sess.run([centroids, cluster_assignments])

    end = time.time()

    print("Found in %.2f seconds" % (end-start), iters, "iterations")
    print('Distribution:', sess.run(found_tensor, feed_dict={points:defect_tensor}))

    tf.reset_default_graph()
    sess.close()
    return centers, assignments


def k_means(defect_tensor,labels, stage, sum_dir, clustering, dim_red):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale, MinMaxScaler
    from clustering_functions import assign_new_detailed, analysis, plot_2D, pca_var, pca_ana

    import numpy as np

    # Parameters
    np.set_printoptions(suppress=True, threshold=np.inf, precision=4)
    num_clusters = len(np.unique(labels))  # num_clusters = FLAGS.clusters  alte
    np.random.seed(42)
    stage_str = str(stage)

    # Dataset
    dataset_params = [False, False]
    scaled = MinMaxScaler().fit_transform(defect_tensor)  # from 0 to 1
    cum_var = pca_var(scaled, up_to_components=60, plot_it=dataset_params[0], verbose=dataset_params[1])
    pca_ana(scaled, var_lim=0.99, verbose=dataset_params[1])

    # KM with TF
    if clustering == 'tensorflow':
        centroids, assignments = k_m_tf(scaled, clusters=num_clusters, max_iters=1000,
                                        summaries_dir=sum_dir, stage_str=stage_str, name_str='kmtf_raw',
                                        go_to_max=False)
        new_assignments, labels_map, countarray, relative_countarray = assign_new_detailed(labels, assignments, powr=4)
        analysis('Raw Data', labels, assignments, mapping=True, acc=True, powr=4)

    # KM with TF with PCA
    if dim_red == 'sklearn' and clustering == 'tensorflow':
        reduced_data = PCA(n_components=5).fit_transform(scaled)
        centroids_PCA, assignments_PCA = k_m_tf(reduced_data, clusters=num_clusters, max_iters=1000,
                                                summaries_dir=sum_dir, stage_str=stage_str,
                                                name_str='kmtf_pca', go_to_max=False)
        new_assignments_PCA, labels_map_PCA, countarray_PCA, relative_countarray_PCA = assign_new_detailed(labels,
                                                                                                           assignments_PCA,
                                                                                                           powr=4)
        analysis('PCA Data', labels, assignments_PCA, mapping=True, acc=True, powr=4)

    # KM with SKL
    if dim_red == 'sklearn' and clustering == 'sklearn':
        from sklearn.cluster import KMeans
        reduced_data = PCA(n_components=5).fit_transform(scaled)
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
        kmeans.fit(reduced_data)
        centroids_skl = kmeans.cluster_centers_


        # if len(reduced_data[0,:]) == 2:
        #     plot_2D(reduced_data,centroids_skl, kmeans)
        #     plot_2D(reduced_data, centroids_PCA, kmeans)