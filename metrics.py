from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import torch
import json

def save_t2v_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]

    metrics = {}
    metrics['R1'] = round(float(np.sum(ind == 0)) * 100 / len(ind), 1)
    metrics['R5'] = round(float(np.sum(ind < 5)) * 100 / len(ind), 1)
    metrics['R10'] = round(float(np.sum(ind < 10)) * 100 / len(ind), 1)
    metrics['R15'] = round(float(np.sum(ind < 15)) * 100 / len(ind), 1)
    metrics['R20'] = round(float(np.sum(ind < 20)) * 100 / len(ind), 1)
    metrics['R25'] = round(float(np.sum(ind < 25)) * 100 / len(ind), 1)
    metrics['R30'] = round(float(np.sum(ind < 30)) * 100 / len(ind), 1)
    metrics['R35'] = round(float(np.sum(ind < 35)) * 100 / len(ind), 1)
    metrics['R40'] = round(float(np.sum(ind < 40)) * 100 / len(ind), 1)
    metrics['R45'] = round(float(np.sum(ind < 45)) * 100 / len(ind), 1)
    metrics['R50'] = round(float(np.sum(ind < 50)) * 100 / len(ind), 1)
    metrics['R55'] = round(float(np.sum(ind < 55)) * 100 / len(ind), 1)
    metrics['R60'] = round(float(np.sum(ind < 60)) * 100 / len(ind), 1)
    metrics['R65'] = round(float(np.sum(ind < 65)) * 100 / len(ind), 1)
    metrics['R70'] = round(float(np.sum(ind < 70)) * 100 / len(ind), 1)
    metrics['R75'] = round(float(np.sum(ind < 75)) * 100 / len(ind), 1)
    metrics['R80'] = round(float(np.sum(ind < 80)) * 100 / len(ind), 1)
    
    f_w = open('cmc/t2v_cmc_res.json', 'w')
    json.dump(metrics, f_w, indent=4)

def save_v2t_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]

    metrics = {}
    metrics['R1'] = round(float(np.sum(ind == 0)) * 100 / len(ind), 1)
    metrics['R5'] = round(float(np.sum(ind < 5)) * 100 / len(ind), 1)
    metrics['R10'] = round(float(np.sum(ind < 10)) * 100 / len(ind), 1)
    metrics['R15'] = round(float(np.sum(ind < 15)) * 100 / len(ind), 1)
    metrics['R20'] = round(float(np.sum(ind < 20)) * 100 / len(ind), 1)
    metrics['R25'] = round(float(np.sum(ind < 25)) * 100 / len(ind), 1)
    metrics['R30'] = round(float(np.sum(ind < 30)) * 100 / len(ind), 1)
    metrics['R35'] = round(float(np.sum(ind < 35)) * 100 / len(ind), 1)
    metrics['R40'] = round(float(np.sum(ind < 40)) * 100 / len(ind), 1)
    metrics['R45'] = round(float(np.sum(ind < 45)) * 100 / len(ind), 1)
    metrics['R50'] = round(float(np.sum(ind < 50)) * 100 / len(ind), 1)
    metrics['R55'] = round(float(np.sum(ind < 55)) * 100 / len(ind), 1)
    metrics['R60'] = round(float(np.sum(ind < 60)) * 100 / len(ind), 1)
    metrics['R65'] = round(float(np.sum(ind < 65)) * 100 / len(ind), 1)
    metrics['R70'] = round(float(np.sum(ind < 70)) * 100 / len(ind), 1)
    metrics['R75'] = round(float(np.sum(ind < 75)) * 100 / len(ind), 1)
    metrics['R80'] = round(float(np.sum(ind < 80)) * 100 / len(ind), 1)

    f_w = open('cmc/v2t_cmc_res.json', 'w')
    json.dump(metrics, f_w, indent=4)

def save_vatex_t2v_metrics(sim_tensor, top_k = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)

    # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2) # 10 1354 1354
    first_argsort = torch.argsort(stacked_sim_matrices, dim = -1, descending= True)
    second_argsort = torch.argsort(first_argsort, dim = -1, descending= False)

    # Extracts ranks i.e diagonals
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1 = 1, dim2 = 2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]
    # A quick dimension check validates our results, there may be other correctness tests pending
    # Such as dot product localization, but that is for other time.
    #assert int(valid_ranks.shape[0]) ==  sum([len(text_dict[k]) for k in text_dict])
    if not torch.is_tensor(valid_ranks):
      valid_ranks = torch.tensor(valid_ranks)
    metrics = {f"R{k}": round(float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)), 1) for k in top_k}

    f_w_ = open('cmc/VATEX/t2v_cmc_res.json', 'w')
    json.dump(metrics, f_w_, indent=4)

def save_vatex_v2t_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]

    metrics = {}
    metrics['R1'] = round(float(np.sum(ind == 0)) * 100 / len(ind), 1)
    metrics['R5'] = round(float(np.sum(ind < 5)) * 100 / len(ind), 1)
    metrics['R10'] = round(float(np.sum(ind < 10)) * 100 / len(ind), 1)
    metrics['R15'] = round(float(np.sum(ind < 15)) * 100 / len(ind), 1)
    metrics['R20'] = round(float(np.sum(ind < 20)) * 100 / len(ind), 1)
    metrics['R25'] = round(float(np.sum(ind < 25)) * 100 / len(ind), 1)
    metrics['R30'] = round(float(np.sum(ind < 30)) * 100 / len(ind), 1)
    metrics['R35'] = round(float(np.sum(ind < 35)) * 100 / len(ind), 1)
    metrics['R40'] = round(float(np.sum(ind < 40)) * 100 / len(ind), 1)
    metrics['R45'] = round(float(np.sum(ind < 45)) * 100 / len(ind), 1)
    metrics['R50'] = round(float(np.sum(ind < 50)) * 100 / len(ind), 1)
    metrics['R55'] = round(float(np.sum(ind < 55)) * 100 / len(ind), 1)
    metrics['R60'] = round(float(np.sum(ind < 60)) * 100 / len(ind), 1)
    metrics['R65'] = round(float(np.sum(ind < 65)) * 100 / len(ind), 1)
    metrics['R70'] = round(float(np.sum(ind < 70)) * 100 / len(ind), 1)
    metrics['R75'] = round(float(np.sum(ind < 75)) * 100 / len(ind), 1)
    metrics['R80'] = round(float(np.sum(ind < 80)) * 100 / len(ind), 1)

    f_w_ = open('cmc/VATEX/v2t_cmc_res.json', 'w')
    json.dump(metrics, f_w_, indent=4)

def save_t2v_retrieved_res(x):
    # step1: obtain top-10 retrieved results index
    sx_ = np.argsort(-x, axis=1)
    sx = sx_[:, :10]

    # step2: load val_data.json
    f_r = open('/public/home/jinxl/jinxl/research/Experiment/DataSet/MSRVTT/data/compressed/split_file/val_data.json', 'r')
    val_data_dict = json.load(f_r)

    # step3: write t2v_retrieved_res.json
    retrieved_dict = dict()
    f_w = open('retrieved/t2v_retrieved_res.json', 'w')

    for index in range(sx.shape[0]):
        sub_index = val_data_dict[str(index)]['sentence'] + '(' + val_data_dict[str(index)]['video'] + ')'

        sx_index = sx[index]
        sub_retrieved_dict_value = [val_data_dict[str(sub_sx_index)]['video'] for sub_sx_index in list(sx_index)]
        
        retrieved_dict[sub_index] = sub_retrieved_dict_value 
        
    json.dump(retrieved_dict, f_w, indent=4)

def save_v2t_retrieved_res(x):
    # step1: obtain top-10 retrieved results index
    sx_ = np.argsort(-x, axis=1)
    sx = sx_[:, :10]

    # step2: load val_data.json
    f_r = open('/public/home/jinxl/jinxl/research/Experiment/DataSet/MSRVTT/data/compressed/split_file/val_data.json', 'r')
    val_data_dict = json.load(f_r)

    # step3: write v2t_retrieved_res.json
    retrieved_dict = dict()
    f_w = open('retrieved/v2t_retrieved_res.json', 'w')

    for index in range(sx.shape[0]):
        sub_index = val_data_dict[str(index)]['video'] + '(' + val_data_dict[str(index)]['sentence'] + ')'

        sx_index = sx[index]
        sub_retrieved_dict_value = [val_data_dict[str(sub_sx_index)]['sentence'] for sub_sx_index in list(sx_index)]
        
        retrieved_dict[sub_index] = sub_retrieved_dict_value 
        
    json.dump(retrieved_dict, f_w, indent=4)

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

# below two functions directly come from: https://github.com/Deferf/Experiments
def tensor_text_to_video_metrics(sim_tensor, top_k = [1,5,10]):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)

    # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim = -1, descending= True)
    second_argsort = torch.argsort(first_argsort, dim = -1, descending= False)

    # Extracts ranks i.e diagonals
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1 = 1, dim2 = 2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]
    # A quick dimension check validates our results, there may be other correctness tests pending
    # Such as dot product localization, but that is for other time.
    #assert int(valid_ranks.shape[0]) ==  sum([len(text_dict[k]) for k in text_dict])
    if not torch.is_tensor(valid_ranks):
      valid_ranks = torch.tensor(valid_ranks)
    results = {f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
    results['MR'] = results["MedianR"]
    return results

def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)
    # Code to avoid nans
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    # Forms a similarity matrix for use with rank at k
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
    return torch.squeeze(values).T
