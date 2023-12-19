import random
import time

from PIL import ImageFilter
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering,spectral_clustering
import copy

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)



def cifar_partition_by_grad_nogroup_bcl(partition_loader, model, criterion_ce, optimizer, args):

    n_labells = len(args.cls_num_list)
    model.eval()

    all_labels_grads_dict = {i: [0, 0] for i in range(n_labells)}
    batch_time_list = []
    for batch_i, data in enumerate(partition_loader):
        batch_start = time.time()
        inputs, targets = data
        batch_size = targets.size(0)
        if (isinstance(inputs, list)):
            cls_inputs = inputs[0]
        else:
            cls_inputs = inputs

        cls_inputs, targets = cls_inputs.cuda(), targets.cuda()
        # get gradient of each class
        _, logits, _ = model(cls_inputs)
        param_dict = {'outputs': logits, 'targets': targets, 'args': args}
        label_lib, label_dict, label_count, outputs_classified_by_label, targets_classified_by_label = get_classified_data(
            param_dict)

        loss_list = []
        for i, output_each_class in enumerate(outputs_classified_by_label):
            temp_loss = criterion_ce(outputs_classified_by_label[i], targets_classified_by_label[i])
            loss_i = temp_loss * label_count[label_lib[i]]
            loss_list.append(loss_i)


        for i, label in enumerate(label_lib):
            optimizer.zero_grad()
            loss_i = loss_list[i]
            loss_i.backward(retain_graph=True)
            count = 0
            for name, parms in model.named_parameters():
                name_list = name.split('.')
                if args.dataset in ['imagenet']:
                    if 'fc' in name_list or 'head' in name_list or 'head_fc' in name_list:
                        continue
                else:
                    if 'fc' in name_list or 'head' in name_list or 'head_center' in name_list:  # 确保不要把分类器包括进来 ！
                        continue
                if (parms.grad == None):
                    print('here')
                parms_deepcopy = copy.deepcopy(parms.grad).cpu()
                reshaped_tensor = parms_deepcopy.reshape(-1)  # reshape into vector
                if count == 0:
                    reshaped_vector_of_model = reshaped_tensor
                else:
                    reshaped_vector_of_model = torch.cat((reshaped_vector_of_model, reshaped_tensor), 0)
                count = count + 1
            all_labels_grads_dict[label][0] += reshaped_vector_of_model
            all_labels_grads_dict[label][1] += label_count[label]
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_time_list.append(batch_time)

    all_labels_avg_grads_dict = {i: 0 for i in range(n_labells)}
    # get avg gradiet
    all_label_grads_list = []
    for i in range(n_labells):
        all_labels_avg_grads_dict[i] = all_labels_grads_dict[i][0] / all_labels_grads_dict[i][1]
        all_label_grads_list.append(all_labels_avg_grads_dict[i].unsqueeze(0))
    all_label_grads_matrix = torch.cat(all_label_grads_list, dim=0)

    sim_cal_start_time = time.time()
    # get sim matrix:：a·b/|a|*|b|
    A = torch.mm(all_label_grads_matrix, all_label_grads_matrix.T)
    a_norm = torch.norm(all_label_grads_matrix, dim=1, keepdim=True)
    B = torch.mm(a_norm, a_norm.T)
    affinity_matrix = torch.div(A, B)
    affinity_matrix = affinity_matrix.numpy()

    affinity_matrix = (affinity_matrix + 1)

    # 使用谱聚类进行划分
    n_groups = args.n_groups
    partition_results = spectral_clustering(affinity=affinity_matrix, n_clusters=n_groups, assign_labels='cluster_qr')
    new_groups = [[] for _ in range(n_groups)]
    for idx, group_idx in enumerate(partition_results):
        new_groups[group_idx].append(idx)

    return new_groups


def imagenet_partition_by_grad_nogroup_bcl(partition_loader, model, criterion_ce, optimizer, args):#这是针对我的Group BCL写的
    n_labells = len(args.cls_num_list)
    model.eval()

    all_labels_grads_dict = {i: [0, 0] for i in range(n_labells)}

    for batch_i, data in enumerate(partition_loader):
        inputs, targets = data
        batch_size = targets.size(0)
        if(isinstance(inputs,list)):
            cls_inputs = inputs[0]
        else:
            cls_inputs = inputs

        cls_inputs, targets = cls_inputs.cuda(), targets.cuda()
        _,logits,_ = model(cls_inputs)#todo check shape
        param_dict = {'outputs': logits, 'targets': targets, 'args': args}
        label_lib, label_dict, label_count, outputs_classified_by_label, targets_classified_by_label = get_classified_data(param_dict)

        loss_list = []
        for i, output_each_class in enumerate(outputs_classified_by_label):
            temp_loss = criterion_ce(outputs_classified_by_label[i], targets_classified_by_label[i])
            loss_i = temp_loss * label_count[label_lib[i]]
            loss_list.append(loss_i)

        for i, label in enumerate(label_lib):
            optimizer.zero_grad()
            loss_i = loss_list[i]
            loss_i.backward(retain_graph=True)
            count = 0
            for name, parms in model.named_parameters():
                name_list = name.split('.')
                if args.dataset in ['imagenet']:
                    if 'fc' in name_list or 'head' in name_list or 'head_fc' in name_list:
                        continue
                else:
                    if 'fc' in name_list or 'head' in name_list or 'head_center' in name_list:
                        continue
                parms_deepcopy = copy.deepcopy(parms.grad).cpu()
                reshaped_tensor = parms_deepcopy.reshape(-1)
                if count == 0:
                    reshaped_vector_of_model = reshaped_tensor
                else:
                    reshaped_vector_of_model = torch.cat((reshaped_vector_of_model, reshaped_tensor), 0)
                del parms_deepcopy
                del reshaped_tensor
                count = count + 1

            # 将该batch中各类别的梯度向量及其样本数添加到dict中
            all_labels_grads_dict[label][0] += reshaped_vector_of_model
            all_labels_grads_dict[label][1] += label_count[label]
    all_labels_avg_grads_dict = {i:0 for i in range(n_labells)}
    all_label_grads_list = []
    for i in range(n_labells):
        all_labels_avg_grads_dict[i] = all_labels_grads_dict[i][0] / all_labels_grads_dict[i][1]
        all_label_grads_list.append(all_labels_avg_grads_dict[i].unsqueeze(0))#变成（0，vector_size)的shape
    all_label_grads_matrix = torch.cat(all_label_grads_list,dim=0)

    A = torch.mm(all_label_grads_matrix,all_label_grads_matrix.T)
    a_norm = torch.norm(all_label_grads_matrix,dim=1,keepdim=True)
    B = torch.mm(a_norm,a_norm.T)
    affinity_matrix = torch.div(A,B)
    affinity_matrix = affinity_matrix.numpy()


    affinity_matrix = (affinity_matrix + 1)
    #使用谱聚类进行划分
    n_groups = args.n_groups

    partition_results = spectral_clustering(affinity=affinity_matrix,n_clusters=n_groups, assign_labels='cluster_qr')
    new_groups = [[] for _ in range(n_groups)]
    for idx,group_idx in enumerate(partition_results):
        new_groups[group_idx].append(idx)
    exit()
    return new_groups

def get_classified_data(param_dict):
    outputs = param_dict['outputs']
    targets = param_dict['targets']
    args = param_dict['args']
    targets_classified_by_label = []
    outputs_classified_by_label = []
    label_lib = []
    label_dict = {}
    label_count = {}
    for i, ori_label in enumerate(targets):
        label = ori_label.item()
        if label not in label_lib:
            # 在label列表中加入新的类别
            label_lib.append(label)
            label_dict.update({label: len(label_lib) - 1})
            label_count.update({label:1})
            target_templist = [label]
            output_templist = [outputs[i]]
            targets_classified_by_label.append(target_templist)
            outputs_classified_by_label.append(output_templist)
        else:
            insert_index = label_dict[label]
            label_count[label] += 1
            targets_classified_by_label[insert_index].append(label)
            outputs_classified_by_label[insert_index].append(outputs[i])

    for i, target_each_class in enumerate(targets_classified_by_label):
        temp_tensor = torch.tensor(target_each_class, dtype=torch.int64)
        targets_classified_by_label[i] = temp_tensor.long().cuda(args.gpu, non_blocking=True)

    for i, input_each_class in enumerate(outputs_classified_by_label):
        temp_tensor = torch.stack(input_each_class)
        outputs_classified_by_label[i] = temp_tensor


    return label_lib,label_dict,label_count,outputs_classified_by_label,targets_classified_by_label



