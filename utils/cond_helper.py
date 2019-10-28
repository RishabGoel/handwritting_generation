import numpy as np
import string
import random

SPECIAL_CHAR = '#'

def make_equal_len(arr, lens):
    max_len = max(lens)
    feats_dim = arr[0].shape[-1]
    
    new_seq = np.zeros((len(lens), max_len, feats_dim))
    seq_mask = np.zeros((len(lens), max_len))
    
    for point_idx in range(arr.shape[0]):   
        new_seq[point_idx, :lens[point_idx]] = np.copy(arr[point_idx])
        seq_mask[point_idx, :lens[point_idx]] = 1.0

    return new_seq, seq_mask

def get_chr2d(text):
    chr2id = {SPECIAL_CHAR:0}
    chr2remove = {}

    for line_id in range(len(text)):
        line = text[line_id]
        for char_id in range(len(line)):
            char = line[char_id]
            if char.isdigit()==False and char not in string.punctuation:

                if char not in chr2id :
                    chr2id[char] = len(chr2id)  
            else:

                if char not in chr2remove:
                    chr2remove[char] = len(chr2remove)

                text[line_id][char_id] = SPECIAL_CHAR
    return text, chr2id, chr2remove

def process_text(text_orig):
    
    text, chr2id, chr2remove = get_chr2d(text_orig)

    text_vector = []
    for line in text:
        line_vec = np.zeros((len(line), len(chr2id)))
        for char_id in range(len(line)):

            line_vec[char_id, chr2id[line[char_id]]] = 1.0

            # import pdb;pdb.set_trace()
        text_vector.append(line_vec)

            
    return np.array(text_vector), chr2id, chr2remove

def create_splits(path_to_data, path_to_text, create_valid = False):
    
    data = np.load(path_to_data, allow_pickle=True, encoding = 'latin1')
    data_lens = [point.shape[0] for point in data]
    data_eql, data_mask = make_equal_len(data, data_lens)

    text = [list(line.strip()) for line in open(path_to_text, "r").readlines()]
    text_lens = [len(line) for line in text]
    one_hot_text, chr2id, chr2remove = process_text(text)
    text_eql, text_mask = make_equal_len(one_hot_text, text_lens)
    # import pdb; pdb.set_trace()
    idxs = list(range(text_eql.shape[0]))
    random.shuffle(idxs)

    data_eql = data_eql[idxs]
    data_mask = data_mask[idxs]
    
    text_eql = text_eql[idxs]
    text_mask = text_mask[idxs]
    # import pdb; pdb.set_trace()
    split_pt = int(0.8 * data_mask.shape[0])
    
    train_data, test_data = data_eql[:split_pt], data_eql[split_pt:]
    train_data_mask, test_data_mask = data_mask[:split_pt], data_mask[split_pt:]
    
    train_text, test_text = text_eql[:split_pt], text_eql[split_pt:]
    train_text_mask, test_text_mask = text_mask[:split_pt], text_mask[split_pt:]

    return train_data, train_data_mask, train_text, \
            train_text_mask, test_data, test_data_mask, test_text, test_text_mask  

def create_xy(data, data_mask):
    return data[:,:-1], data[:, 1:], data_mask[:, 1:]

def get_stats(data):
    x = []
    y =  []
    for i in range(data.shape[0]):
        x.append(data[i][:,1])
        y.append(data[i][:,2])

    assert(len(x)==data.shape[0])
    assert(len(y)==data.shape[0])

    print(len(x))
    
    x = np.concatenate(x)
    y = np.concatenate(y)

    assert(x.shape==y.shape)
    
    print(x.shape)

    # import pdb;pdb.set_trace()
    
    return [np.mean(x), np.mean(y)], [np.std(x), np.std(y)]

def normalize_data(data, mean, std):
    norm_data = []
    
    for i in range(data.shape[0]):
        tmp = np.copy(data[i])
        # import pdb; pdb.set_trace()
        tmp[:,1] -= mean[0]
        tmp[:,1] /= std[0]

        tmp[:,2] -= mean[1]
        tmp[:,2] /= std[1]
        # import pdb; pdb.set_trace()
        norm_data.append(tmp)
        # import pdb; pdb.set_trace()
    
    return np.array(norm_data)

def un_normalize_data(data, mean, std):
    norm_data = []
    
    for i in range(data.shape[0]):
        tmp = np.copy(data[i])
        tmp[:, 1] *= std[0]
        tmp[:, 1] += mean[0]

        tmp[:, 2] *= std[1]
        tmp[:, 2] += mean[1]

        norm_data.append(tmp)

    return np.array(norm_data)


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()

def get_data(path_to_data = '..\\data\\strokes.npy', path_to_text = '..\\data\\sentences.txt'):
    train_data, train_data_mask, train_text, \
            train_text_mask, test_data, test_data_mask, test_text, test_text_mask = create_splits(path_to_data, path_to_text)
    
    train_x, train_y, train_y_mask = create_xy(train_data, train_data_mask)
    test_x, test_y, test_y_mask = create_xy(test_data, test_data_mask)
    
    return train_x, train_y, train_y_mask, train_text, \
            train_text_mask, test_x, test_y, test_y_mask, test_text, test_text_mask
    # import pdb;pdb.set_trace()
    # train, test = create_splits(path_to_data, path_to_text)
    # train_mean, train_std = get_stats(np.copy(train))

    # train_norm, test_norm = normalize_data(train, train_mean, train_std),normalize_data(test, train_mean, train_std) 

    # train_un_norm, test_un_norm = un_normalize_data(train_norm, train_mean, train_std),un_normalize_data(test_norm, train_mean, train_std)
    # # import pdb; pdb.set_trace()
    # # assert(np.array_equal(train, train_un_norm))
    # # import pdb; pdb.set_trace()
    # # assert(np.array_equal(test, test_un_norm))
    # # import pdb; pdb.set_trace()
    # train_x, train_y = create_xy(train_norm)
    # test_x, test_y = create_xy(test_norm)
    # print(train.shape, test.shape, train_x.shape, train_y.shape, test_x.shape, test_y.shape, train[0].shape, train_x[0].shape)
    # return train_x, train_y, test_x, test_y, train_mean, train_std

if __name__ == '__main__':
    get_data()