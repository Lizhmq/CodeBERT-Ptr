import os
import torch
import numpy as np
import pickle
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.functional import norm
from codebert import codebert
from tqdm import tqdm


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def main():
    device = torch.device("cuda", 0)
    # path = "./save/varmis/checkpoint-52000-1.0726/model.bin"
    # path = "./save/varmis/checkpoint-112000-1.008/model.bin"
    # path = "./save/java/checkpoint-24000-0.352/model.bin"
    path = "./save/pyop/checkpoint-32000-0.4451/model.bin"
    # orig_model = torch.load(path)
    # torch.save(orig_model.state_dict(), "tmp112.bin")
    # del orig_model
    model = codebert("../.code-bert-cache/codebert-base", device)
    model.load_state_dict(torch.load(path))
    # model.load_state_dict(torch.load("tmp112.bin"))
    # model.load_state_dict(torch.load("tmp52.bin"))
    model = model.to(device)

    # data_path = "../great/test.pkl"
    # data_path = "../bigJava/datasets/test.pkl"
    data_path = "../CuBert/wrong_op/test.pkl"
    batch_size = 16
    data = load_data(data_path)

    test_size = len(data["norm"])
    import math
    batch_num = math.ceil(test_size / batch_size) // 4 # too slow evaluation
    # batch_num = 10

    all_pred, all_label = [], []
    with torch.no_grad():
        for i in tqdm(range(batch_num)):
            batch_in = data["norm"][i * batch_size:(i + 1) * batch_size]
            # batch_in = [" ".join(tokens) for tokens in batch_in]
            labels = data["label"][i * batch_size:(i + 1) * batch_size]
            # idxs = data["error"][i * batch_size:(i + 1) * batch_size]
            idxs = data["idx"][i * batch_size:(i + 1) * batch_size]
            batch_in, batch_out = model.preprocess(batch_in, labels, idxs)
            batch_size = len(batch_in)
            predicted = model._run_batch(batch_in).cpu().data.numpy()
            predicted = np.argmax(predicted, axis=1)
            lens = len(batch_in)
            for i in range(lens):
                if labels[i] == 1:
                    all_label.append(batch_out[i])
                else:
                    all_label.append(0)
                all_pred.append(predicted[i])

    normal_label = [0 if item == 0 else 1 for item in all_label]
    normal_pred = [0 if item == 0 else 1 for item in all_pred]
    all_pos = 0
    true_loc = 0
    for i in range(len(all_pred)):
        if all_label[i] != 0:
            all_pos += 1
            if all_pred[i] >= all_label[i][0] and all_pred[i] < all_label[i][1]:
                true_loc += 1
    print("Total num:", len(normal_pred))
    print("Positive num, Correct Localization, Loc Accuracy:", all_pos, true_loc, true_loc / all_pos)
    print("Confusion matrix")
    print(confusion_matrix(normal_label, normal_pred))

main()