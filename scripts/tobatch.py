def list2batch(mylist, istag=False):
    max_l = 0
    for s in mylist:
        if max_l < len(s):
            max_l = len(s)
    target_array = []
    if istag:
        minus = -1
    else:
        minus = 0
    for k in range(max_l):
        dmy = []
        for _d in range(len(mylist)):
            if len(mylist[_d]) > k:
                dmy.append(mylist[_d][k])
            else:
                dmy.append(0 + minus)
        target_array.append(dmy)
    return target_array


def list2batch_char(mylist):
    max_ele = 0
    max_l = 0
    for ele in mylist:
        for _ele in ele:
            if max_ele < len(_ele):
                max_ele = len(_ele)
        if max_l < len(ele):
            max_l = len(ele)
    new_list = []
    ele_list = []
    for ele in mylist:
        for _ele in ele:
            diff = max_ele - len(_ele)
            a = _ele + [0] * diff
            ele_list.append(a)
        new_list.append(ele_list)
        ele_list = []
    out_list = []
    for ele in range(max_l):
        dmy = []
        for e in range(len(new_list)):
            if len(new_list[e]) > ele:
                dmy.append(new_list[e][ele])
            else:
                dmy.append([0] * max_ele)
        out_list.append(dmy)
    return out_list
