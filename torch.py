import torch


def remove_rows(list_x, row_id_list):
    if len(list_x) == 0:
        raise Exception("Empty list.")

    all_id_list = list(range(list_x[0].shape[0]))

    return get_rows(list_x, subtract_list(all_id_list, row_id_list))


def get_rows(list_x, row_id_list):    
    return [torch.index_select(x, dim=0, index=torch.tensor(row_id_list)) for x in list_x]


def subtract_list(l1, l2):    
    return list(set(l1) - set(l2))
