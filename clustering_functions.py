import numpy as np

def create_assignment_matrix(labels, assignments, relative = False, power = 1):
    dims = len(np.unique(labels))
    countarray = np.zeros((dims, dims))
    relative_countarray = np.zeros((dims,dims))
    for i in range(0, len(labels[:,0])):
        for j in range(0, dims):
            if assignments[i] == j:
                countarray[labels.astype(int)[i], j] = countarray[labels.astype(int)[i], j] + 1

    countarray = np.power(countarray,power)

    if relative == True:
        for i in range(0,dims):
            for j in range(0,dims):
                if np.sum(countarray[:,j]) != 0:
                    relative_countarray[i,j] = countarray[i,j]/np.sum(countarray[:,j])
        countarray = relative_countarray

    return countarray

def map_cluster_to_label(countarray):
    labelsmap = np.zeros((len(countarray[0,:]), 1))
    for i in range(0, len(labelsmap)):
        labelsmap[i, 0] = np.argmax(countarray[i, :])

    return labelsmap.astype(int)


def get_new_assignments(labels, labels_map, assignments):
    labels_map = labels_map.astype(int)
    new_assignments = np.zeros((len(labels), 1))
    used_before = np.zeros((len(labels_map), 1)).astype(int)
    for i in range(0, len(labels_map)):
        # I'm an idiot I couldn't figure out how to use it without a dummy
        label_indices = np.where(assignments == labels_map[i])[0]

        if not used_before[labels_map[i,0]] == 1:
            new_assignments[label_indices] = i
            used_before[labels_map[i,0]] = 1
        # else:
        #     print(np.where(used_before == 0))
    return new_assignments


def get_accuracy(labels, labels_map, new_assignments):
    accuracy = np.zeros((len(labels_map), 1))
    for i in range(0, len(labels_map)):
        label_indices = np.where(labels == i)
        boolarr = labels[label_indices] == new_assignments[label_indices]
        accuracy[i] = np.sum(boolarr) / len(boolarr)
    return accuracy

def get_accuracy_detail(labels, labels_map, new_assignments):
    ## Labels_MAP is not necessarily needed but maybe it helps with array errors?
    accuracy_detail = np.zeros((len(labels_map), len(labels_map)))
    for i in range(0, len(labels_map)):
        label_indices = np.where(labels == i)
        for j in range(0, len(labels_map)):
            boolarr = new_assignments[label_indices] == j
            accuracy_detail[i, j] = np.sum(boolarr) / len(label_indices[1])
    return accuracy_detail



def bench_k_means(estimator, name, data, labels, sample_size):
    from sklearn import metrics
    from time import time
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

def plot_2D(reduced_data, centroids, kmeans):
    # Plot the decision boundary. For that, we will assign a color to each

    import matplotlib.pyplot as plt
    x_min, x_max = 1.1 * reduced_data[:, 0].min(), 1.1 * reduced_data[:, 0].max()
    y_min, y_max = 1.1 * reduced_data[:, 1].min(), 1.1 * reduced_data[:, 1].max()

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = ((abs(x_max) - abs(x_min)) + (abs(y_max) - abs(y_min))) / 200  # point in the mesh [x_min, x_max]x[y_min, y_max].
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), sparse=False)
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X

    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the dataset \n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    return

class results(object):
    def _init_(self, name, data,):
        self.name = name


def assign_new_detailed(labels, assignments, powa = 1, powr = 3):


    countarray = create_assignment_matrix(labels , assignments , relative = False, power = powa)
    relative_countarray = create_assignment_matrix(labels, assignments, relative = True, power = powr)
    labels_map = map_cluster_to_label(relative_countarray)
    new_assignments = get_new_assignments(labels, labels_map, assignments)

    return new_assignments, labels_map, countarray, relative_countarray





def analysis(name_str, labels, assignments, mapping = False, acc = False, powa = 1, powr = 3):
    ## Overhead warning: assign_new_detailed runs for every analysis, whereas it could be passed once established/ run only once

    new_assignments, labels_map, countarray, relative_countarray = assign_new_detailed(labels, assignments, powa,powr)
    accuracy = get_accuracy(labels, labels_map, new_assignments)
    accuracy_detail = get_accuracy_detail(labels, labels_map, new_assignments)
    print(name_str, 'Dataset')
    print(79 * '_')
    for i in range(0, len(labels_map)):
        if mapping == True and acc == False:
            print('Cluster', labels_map[i,0], 'to label', i, '\t Relative occurence', countarray[i, labels_map[i,0]], '/', np.sum(countarray[:, labels_map[i,0]]))
            #print('Cluster', labels_map[i, 0], 'to label', i, '\t Relative occurence', countarray[i, labels_map[i, 0]], '/', np.sum(countarray[:, labels_map[i, 0]]), '\t Relative value', relative_countarray[i, labels_map[i, 0]])
        elif mapping == True and acc == True:
            print('Cluster', labels_map[i,0], 'to label', i, '\t Relative occurence', countarray[i, labels_map[i,0]], '/', np.sum(countarray[:, labels_map[i,0]]), '\t Accuracy:', accuracy[i])
        elif mapping == False and acc == True:
            print('Accuracy:', accuracy[i])
    print('')

    return accuracy_detail

def pca_var(data, up_to_components, plot_it = False, verbose = False):
    ## Simple function that gives explained cumulative variance and maybe plots it



    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np

    cumulative_var = np.zeros(up_to_components)
    reduced_data = PCA(n_components=up_to_components).fit_transform(data)

    variance = np.var(reduced_data, axis=0)
    variance_ratio = variance / np.sum(variance)

    for i in range(0,up_to_components):
        cumulative_var[i] = np.sum(variance_ratio[:i],axis=0)
    if verbose == True:
        print(cumulative_var[up_to_components-1],'% variance explained by', up_to_components, 'components' )


    if plot_it == True:
        print('Plotting variance for', up_to_components,'components..')
        plt.plot(np.arange(up_to_components),cumulative_var)
        plt.show()

    return cumulative_var

def pca_ana(data, var_lim, verbose = False):
    for i in range(0, len(data[0,:])-1):
        check_var = pca_var(data, i + 1)
        if check_var[i] > var_lim:
            if verbose == True:
                print('Threshold of', var_lim, 'reached at', i + 1, 'components')
            break

    return