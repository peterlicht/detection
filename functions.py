def read_csv_to_array(file,delim,stage,side, limit,verbose):
    import csv
    import numpy as np

    with open(file) as csvfile:

        counter=int(0)
        reader = csv.reader(csvfile, delimiter=delim)
        header = next(reader)
        columns = len(header)

        array = np.empty((0,columns),int)
        row_placeholder = np.empty([1,columns],int)
        stageyes=0
        stagecolumn=-1
        sideyes=0
        sidecolumn=-1
        lengthyes=0
        lengthcolumn=-1
        widthyes=0
        widthcolumn=-1

       # print(array)
        #print(row_placeholder)
        #print(columns)

        # TEMPORAL fix for quotation marks in CSV files. REVISE LATER
        for i in range(0,columns):
            header[i] = str.replace(header[i], "'", "")

        print('')
        print('Header Parameters:')
        print(header)
        print('')


        # TEMPORAL fix because the CSV files display width and length with too many zeros. REVISE LATER
        # The HEADER also does not have quotation marks. Depends on the way it is saved and so on
        for i in range(0,columns):
            if header[i] in ["Width"]:
                widthyes=1
                widthcolumn=i

        for i in range(0, columns):
            if header[i] in ["Length"]:
                lengthyes = 1
                lengthcolumn = i


        # Check for Stage or Side in the header. If there is none, stageyes=0 and sideyes=0 will result in NO filtering of added lines to the array
        for i in range(0,columns):
            if header[i] in ["Stage"]:
                stageyes=1
                stagecolumn=i

        for i in range(0,columns):
            if header[i] in ["Side"]:
                sideyes=1
                sidecolumn=i

        if stageyes==1:
            print('Stageheader detected at column', stagecolumn)
        else:
            print('No Stageheader detected')

        if sideyes == 1:
            print('Sideheader detected at column', sidecolumn)
        else:
            print('No Sideheader detected')
        print('')

        # Remember array is a dict, so I assign empty numpy arrays to every line in dict
 #       for i in range(0, columns):
 #          array[i] = np.array([])

        print('Reading csv into array..')
        for row in reader:

            counter = counter + 1
            if counter/limit==0.5 or counter/(limit+1)==0.5:
                print("50%")
            if counter/limit==0.25 or counter/(limit+1)==0.25:
                print("25%")
            if counter/limit==0.75 or counter/(limit+1)==0.75:
                print("75%")
            if counter/limit==1 or counter/(limit+1)==1:
                print("100%")
            if counter>=limit:
                print('Stopping at limit..', limit)
                break

            if verbose==1:
                print(row)

            # Creating a placeholder for a row
            # and converting the format to ints without punctuation
            # maybe a bit superfluos and straining on computation. REVISE LATER
            for i in range(0, columns):
                if row[0]:
                    a = str(row[i])
                    b = str.replace(a, ',', '')
                    # This is a stupid temporal fix to the length and width format of the CSV files REVISE LATER
                    if (lengthyes==1 and lengthcolumn==i):
                        d = int(float(b)/1000000000000)
                    elif (widthyes==1 and widthcolumn==i):
                        d = int(float(b)/10000000000000000)
                    else:
                        d = int(float(b))
                    row_placeholder[0,i]=d
                    # print(row_placeholder)

            # Only save if there is no stage or stage is equal to wanted stage
            if stageyes==0 and sideyes==0:
                array=np.append(array,row_placeholder, axis=0)
            if stageyes==1 and row_placeholder[0,stagecolumn]==stage and sideyes==1 and row_placeholder[0,sidecolumn]==side:
                array=np.append(array,row_placeholder,axis=0)

        if counter<limit:
            print("Reached end of file")

        print('Done')
        print("")
        print(len(array),"out of",counter, "instances matched the respecting parameters")
        print("")
        data=[header,]
        data.append(array)

    return data


def create_defectmatrix(header,array,coilID,stage):

    import numpy as np

    coilcolumn=int
    tilecolumn=int
    countcolumn=int

    stagex = 2**stage
    stagey = 2**(stage+1)

    matrix = np.zeros([stagey, stagex], int)

    for i in range(0,len(header)):
        if header[i] in ["CoilID"]:
            coilcolumn=i
        if header[i] in ["TileID"]:
            tilecolumn=i
        if header[i] in ["Count"]:
            countcolumn=i


    for i in range(0,len(array)):
        if array[i,coilcolumn] == coilID:
            matrix[array[i, tilecolumn]//stagex,array[i, tilecolumn]%stagex]= matrix[array[i, tilecolumn]//stagex,array[i, tilecolumn]%stagex] + array[i, countcolumn]

    return matrix




# Actually a superfluous function, but allows for COIL filtering and sums up errors
def create_defectvector(header, array, coilID, stage):
    import numpy as np

    coilcolumn=int
    tilecolumn=int
    countcolumn=int
    tiles = (2 ** (stage)) * (2 ** (stage + 1))
    defectvector = np.zeros(tiles)

    for i in range(0,len(header)):
        if header[i] in ["CoilID"]:
            coilcolumn=i
        if header[i] in ["TileID"]:
            tilecolumn=i
        if header[i] in ["Count"]:
            countcolumn=i


    for i in range(0, len(array)):
        if array[i, coilcolumn] == coilID:
            defectvector[array[i, tilecolumn]] = defectvector[array[i, tilecolumn]] + array[i, countcolumn]

    return defectvector


def create_defect_heatmap(defect_header, defect_array, coils, savepath_str, stage):
    """Creates the matrix and also the heatmap of a defect array. It only works if there is a 'CoilID', 'TileID' and 'Count' header present, so it is not a versatile function.
    Savepath must be a string, coils can be an array or a single coil."""
    from functions import create_defectmatrix
    import os.path
    import matplotlib.pyplot as plt

    savepath = savepath_str + '/' + str(stage)
    png = '.png'
    length = len(coils)
    counter = 0

    print('Creating defect heatmaps in', savepath_str, '..')

    for coilid in coils[:, 0]:
        defectmatrix = create_defectmatrix(defect_header, defect_array, coilid, stage)
        fig, ax = plt.subplots()
        cmap = plt.cm.jet
        im = ax.imshow(defectmatrix, cmap)
        fig.colorbar(im, ax=ax)
        coil_file = str(coilid) + png
        savepath_file = os.path.join(savepath, coil_file)
        print('Created',savepath_file)

        counter = counter + 1
        if counter / length == 0.5 or counter / (length + 1) == 0.5:
            print("50%")
        if counter / length == 0.25 or counter / (length + 1) == 0.25:
            print("25%")
        if counter / length == 0.75 or counter / (length + 1) == 0.75:
            print("75%")
        if counter / length == 1 or counter / (length + 1) == 1:
            print("100%")

        if not os.path.exists(savepath):
            os.makedirs(savepath)
        plt.savefig(savepath_file)
        plt.clf()

    print('Done')
    print('')
    return


def create_defect_tensor(coils_array, defect_header, defect_array, stage):
    import numpy as np
    counter = 0
    tiles = (2 ** (stage)) * (2 ** (stage + 1))
    defect_tensor = np.zeros((len(coils_array), tiles))
    for coilid in coils_array[:, 0]:
        defect_vector = create_defectvector(defect_header, defect_array, coilid, stage)
        defect_tensor[counter, :] = defect_vector[:]
        counter = counter + 1
    return defect_tensor


def create_classifier(classifier_file, coils_header, coils_array, one_hot, classes):
    import numpy as np

    coilcolumn=int
    for i in range(0,len(coils_header)):
        if coils_header[i] in ["CoilID"]:
            coilcolumn=i


    previousline = 0

    if one_hot:
        class_matrix = np.zeros([len(coils_array[:,coilcolumn]),classes])
    else:
        class_vector = np.zeros(len(coils_array[:,coilcolumn]))

# This Algorithm could behave more dynamically (i.e. an indexing vector for classes or a dict or something. Some loops. REVISE LATER
# It could also not loop through the entire file each time
# It could also set the ZERO position to 1 as a condition instead of replacing the ones with zeros. REVISE LATER


    if one_hot:
        for i in range(0, len(coils_array[:, coilcolumn])):
            class_matrix[i, 0] = 1
            list = open(classifier_file, 'r')
            for line in list:
                if int(line) == coils_array[i, coilcolumn]:
                    if previousline == 1:
                        class_matrix[i,1] = 1
                        class_matrix[i, 0] = 0
                    if previousline == 2:
                        class_matrix[i,2] = 1
                        class_matrix[i, 0] = 0
                    if previousline == 3:
                        class_matrix[i, 3] = 1
                        class_matrix[i, 0] = 0
                    if previousline == 4:
                        class_matrix[i,4] = 1
                        class_matrix[i, 0] = 0
                    if previousline == 5:
                        class_matrix[i, 5] = 1
                        class_matrix[i, 0] = 0
                    if previousline == 6:
                        class_matrix[i, 6] = 1
                        class_matrix[i, 0] = 0
                    if previousline == 7:
                        class_matrix[i,7] = 1
                        class_matrix[i, 0] = 0
                    if previousline == 8:
                        class_matrix[i, 8] = 1
                        class_matrix[i, 0] = 0
                    if previousline == 9:
                        class_matrix[i, 9] = 1
                        class_matrix[i, 0] = 0
                previousline = int(line)
    else:
        for i in range(0, len(coils_array[:, coilcolumn])):
            list = open(classifier_file, 'r')
            for line in list:
                if int(line) == coils_array[i, coilcolumn]:
                    if previousline == 1:
                        class_vector[i] = 1
                    if previousline == 2:
                        class_vector[i] = 2
                    if previousline == 3:
                        class_vector[i] = 3
                    if previousline == 4:
                        class_vector[i] = 4
                    if previousline == 5:
                        class_vector[i] = 5
                    if previousline == 6:
                        class_vector[i] = 6
                    if previousline == 7:
                        class_vector[i] = 7
                    if previousline == 8:
                        class_vector[i] = 8
                    if previousline == 9:
                        class_vector[i] = 9
                previousline = int(line)

    if one_hot:
        print('Created', len(class_matrix[:,0]), 'x', len(class_matrix[0,:]), 'labels matrix.')
        print('')
        return class_matrix
    else:
        print('Created', len(class_vector), 'x', '1', 'labels vector.')
        print('')
        return class_vector



def split_by_y(coils, input_vec, label_vec, ratio, random, randomize):
    from sklearn.cross_validation import train_test_split
    import numpy as np

    if randomize:
        train_coils, test_coils, train_set, test_set, train_label, test_label = train_test_split(coils, input_vec, label_vec, train_size=ratio,
                                                                        random_state=random)
    else:
        limit = ratio*len(input_vec[:,0])
        if limit%1 != 0:
            raise ValueError('Not a valid ratio. Ratio multiplied with vector length must yield integer value, but was', limit)
        limit = int(limit)
        input_x = len(input_vec[0,:])
        input_y = len(input_vec[:,0])
        input_xl = len(label_vec[0, :])
        input_yl = len(label_vec[:, 0])
        rest_y = input_y - limit

        train_set = np.zeros((limit,input_x))
        train_label = np.zeros((limit, input_xl))

        test_set= np.zeros((rest_y, input_x))
        test_label = np.zeros((rest_y, input_xl))

        # Randomize
        if randomize:
            rand_input_vec, rand_label_vec = np.random.shuffle(zip(input_vec, label_vec))
            np.split(rand_input_vec)




        if input_y == input_yl:
            train_set[:,:] = input_vec[0:limit, :]
            test_set[:,:] = input_vec[limit:input_y, :]
            train_label[:,:] = label_vec[0:limit, :]
            test_label[:,:] = label_vec[limit:input_yl, :]
        else:
            raise ValueError('Y dimensions of labels and input must match, but are' , input_y, 'x', input_x, 'and', input_yl, 'x', input_xl, '.')

    print('train set:', len(train_set[:,0]), 'entries' )
    print('test set:', len(test_set[:,0]), 'entries')



    return train_coils, test_coils, train_set, train_label, test_set, test_label




def save_data(savepath_str, stage, defect_tensor, coils_array, labels_array, onehot_array, overwrite = ''):
    import os.path
    import numpy as np

    stage_str = str(stage)
    overwrite_defects = overwrite
    overwrite_labels = overwrite
    overwrite_onehot = overwrite


    filename_defects= 'defect_data.csv'
    filepath_defects = savepath_str + '/' + stage_str
    file_defects = filepath_defects + '/' + filename_defects

    filename_labels = 'labels.csv'
    filepath_labels =  savepath_str + '/' + stage_str
    file_labels = filepath_labels + '/' + filename_labels

    filename_onehot = 'one_hot.csv'
    filepath_onehot = savepath_str + '/' + stage_str
    file_onehot = filepath_onehot  + '/' + filename_onehot

    coils = np.reshape(coils_array[:,0],(len(defect_tensor[:,0]),1))
    labels = np.reshape(labels_array,(len(labels_array),1))

    if os.path.isfile(filepath_defects):
        while overwrite_defects != 'y' and overwrite_defects != 'n':
            overwrite_defects = input('A defect file for that stage already exists. Overwrite? y/n  ')
        if overwrite_defects == 'y':
            joined_defects = np.concatenate((coils, defect_tensor), axis=1)
            np.savetxt(file_defects, joined_defects, fmt='%10.5f', delimiter=",")
            print('Overwriting file', file_defects, '..')
    else:
        joined_defects = np.concatenate((coils, defect_tensor), axis=1)
        if not os.path.exists(filepath_defects):
            os.makedirs(filepath_defects)
        np.savetxt(file_defects, joined_defects, fmt='%10.5f', delimiter=",")
        print('Creating new file', file_defects, '..')


    if os.path.isfile(filepath_labels):
        while overwrite_labels != 'y' and overwrite_labels != 'n':
            overwrite_labels = input('A labels file for that stage already exists. Overwrite? y/n  ')
        if overwrite_labels == 'y':
            joined_labels = np.concatenate((coils, labels), axis=1)
            np.savetxt(file_labels, joined_labels, fmt='%10.5f', delimiter=",")
            print('Overwriting file', file_labels, '..')
    else:
        joined_labels = np.concatenate((coils, labels), axis=1)
        if not os.path.exists(filepath_labels):
            os.makedirs(filepath_labels)
        np.savetxt(file_labels, joined_labels, fmt='%10.5f', delimiter=",")
        print('Creating new file', file_labels, '..')


    if os.path.isfile(filepath_onehot):
        while overwrite_onehot != 'y' and overwrite_onehot != 'n':
            overwrite_onehot = input('A onehot file for that stage already exists. Overwrite? y/n  ')
        if overwrite_onehot == 'y':
            joined_onehot = np.concatenate((coils, onehot_array), axis=1)
            np.savetxt(file_onehot, joined_onehot, fmt='%10.5f', delimiter=",")
            print('Overwriting file', file_onehot, '..')
    else:
        joined_onehot = np.concatenate((coils, onehot_array), axis=1)
        if not os.path.exists(filepath_onehot):
            os.makedirs(filepath_onehot)
        np.savetxt(file_onehot, joined_onehot, fmt='%10.5f', delimiter=",")
        print('Creating new file', file_onehot, '..')

    return


def get_data(savepath_str, stage, type_str):
    import numpy as np

    stage_str = str(stage)
    if type_str == 'defects':
        filename = 'defect_data.csv'
    elif type_str == 'labels':
        filename = 'labels.csv'
    elif type_str == 'one_hot':
        filename = 'one_hot.csv'
    else:
        raise ValueError('Label must either be defects, labels or one_hot.')

    filepath = savepath_str + '/' + stage_str + '/' + filename
    print('Reading out data from', filepath, '..')
    data = np.genfromtxt(filepath, delimiter=',')
    
    return data

def get_stochastic_batch(coils, set, labels, size = 20):
    import numpy as np
    indices = np.random.choice(len(set), size, replace= False)
    batch_coils = np.take(coils,indices, axis=0)
    batch_set = np.take(set,indices, axis=0)
    batch_labels = np.take(labels, indices, axis=0)

    return batch_coils, batch_set, batch_labels

def store_from_csv(mode_str, stage, side, defect_str='/home/patrick/resources/GR_Defects.csv', coil_str='/home/patrick/resources/COILS.csv', limit = 2000000) :
    """mode_str sets the mode to either storing -data, storing -maps or -both"""

    # Everything is stored in a 2-tuple list with the first entry being HEADER and the second entry being the DATA ARRAY

    defects = read_csv_to_array(defect_str,';',stage, side , limit, verbose=0)
    coils = read_csv_to_array(coil_str,';',stage, side, 1000, verbose = 0)

    # defect_array stores all the Info from GR_defects. It has the dynamic length of the filtered results
    # and the dynamic width of however many headers/columns there are.

    # t.ex. GR_Defects for stage=3 and side=0 has 7 headers so it yields a 2768 x 7 matrix

    # defects: [headers, 3.000.000 x 6]
    # defect_array: 3.000.000 x 1
    defect_header=defects[0]
    defect_array=defects[1]

    # coils: [headers, 320 x 3]
    # coils_array: 320 x 3
    coils_header=coils[0]
    coils_array=coils[1]

    # 320 x 128
    defect_tensor = create_defect_tensor(coils_array, defect_header, defect_array, stage)

    labels_matrix = create_classifier('/home/patrick/resources/coilmaps/111List.txt', coils_header, coils_array, one_hot=True, classes=10)
    labels = create_classifier('/home/patrick/resources/coilmaps/111List.txt', coils_header, coils_array, one_hot=False, classes=10)
    if mode_str == 'data' or mode_str == 'both':
        save_data('/home/patrick/resources/defects', stage, defect_tensor, coils_array, labels, labels_matrix, overwrite = 'y')
    elif mode_str == 'maps' or mode_str == 'both':
        create_defect_heatmap(defect_header,defect_array, coils_array, '/home/patrick/resources/coilmaps', stage)
    else:
        raise ValueError('Data mode must either be "data", "maps, or "both"')
    return

def get_all_summary_data(stage):
    stagex = 2 ** stage
    stagey = 2 ** (stage + 1)
    tiles = stagex * stagey
    stage_str = str(stage)


    defects = get_data('/home/patrick/resources/defects', stage, 'defects')
    defect_tensor = defects[:, 1:len(defects[0, :])]
    coils = defects[:, 0]
    labels_array = get_data('/home/patrick/resources/defects', stage, 'labels')
    labels = labels_array[:, 1:len(labels_array[0, :])]
    onehot_array = get_data('/home/patrick/resources/defects', stage, 'one_hot')
    labels_matrix = onehot_array[:, 1:len(onehot_array[0, :])]

    print('Input Data:', stagey, 'x', stagex, 'for stage', stage)
    return coils, defect_tensor, labels, labels_matrix