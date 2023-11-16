import numpy as np
import re

def count_nonzero(matrix, axis, index):
    """
    Count the number of non-zero elements in the specified row or column of the matrix.

    Parameters:
    - matrix (list of lists): The input matrix.
    - axis (str): Specify 'row' or 'col' to indicate whether to check a row or a column.
    - index (int): The index of the row or column to check.

    Returns:
    - int: The number of non-zero elements.
    """

    if axis == 'row':
        # Count non-zero elements in the specified row
        return sum(1 for element in matrix[index] if element != 0)
    elif axis == 'col':
        # Count non-zero elements in the specified column
        return sum(1 for row in matrix if row[index] != 0)
    else:
        raise ValueError("Invalid axis. Use 'row' or 'col'.")

def nonzero_indices(arr, axis, index):
    """
    返回指定数组的行或列的非零值的下标。

    参数:
    - arr (numpy.ndarray): 输入数组。
    - axis (str): 指定 'row' 或 'col'，表示要查找的是行还是列。
    - index (int): 要查找的行或列的索引。

    返回:
    - numpy.ndarray: 非零值的下标数组。
    """
    if axis == 'row':
        return np.nonzero(arr[index, :])[0]
    elif axis == 'col':
        return np.nonzero(arr[:, index])[0]
    else:
        raise ValueError("Invalid axis. Use 'row' or 'col'.")


def find_u_v_max_path(u,v,w):
    res_mul_num = []
    res_add_num = []
    # 搜索每条res的路径
    for i in range(w.shape[0]):
        mul_num = 0
        add_num = count_nonzero(w, 'row', i) - 1 
        for j in range(w.shape[1]):
            if w[i,j] != 0:
                # 查找res的相关数目
                mul_num += 1
                add_num += count_nonzero(u, 'col', j) - 1 
                add_num += count_nonzero(v, 'col', j) - 1
        res_mul_num.append(mul_num)
        res_add_num.append(add_num)
    #print("\r\nres_mul_num:", res_mul_num)
    #print("\r\nres_add_num:", res_add_num)    

    # 按照复杂程度排序
    sorted_indices = sorted(range(len(res_mul_num)), key=lambda i: res_mul_num[i] + res_add_num[i],reverse=True)
    #print("\r\nSorted indices:", sorted_indices)

    max_path = sorted_indices[0]
    path_num_max = res_mul_num[max_path] + res_add_num[max_path]

    return (path_num_max,max_path)

def find_w_max_path(arr):
    non_zero_counts = np.count_nonzero(arr, axis=1)
    row_with_max_non_zero = np.argmax(non_zero_counts)
    return row_with_max_non_zero

def find_most_significant_adder(arr,arr_coe,index,axis):
    arr_row = arr.shape[0]
    arr_col = arr.shape[1]
    bias_dict = {}
    if axis == 'col':
        for i0 in range(arr_row):
            if arr[i0][index] != 0:
                for i1 in range(i0+1,arr_row):
                    if arr[i1][index] != 0:
                        pe_num0 = arr[i0][index]
                        pe_num1 = arr[i1][index]
                        coe_ref0 = arr_coe[i0][index]
                        coe_ref1 = arr_coe[i1][index]
                        bias_dict[(i0,i1)] = 0
                        for j in range(arr_col):
                            if arr[i0][j] == pe_num0 and arr[i1][j] == pe_num1:
                                ratio0 = arr_coe[i0][j]/coe_ref0
                                ratio1 = arr_coe[i1][j]/coe_ref1
                                if(ratio0 == ratio1):
                                    bias_dict[(i0,i1)] += 1
    elif axis == 'row':
       for i0 in range(arr_col):
            if arr[index][i0] != 0:
                for i1 in range(i0+1,arr_col):
                    if arr[index][i1] != 0:
                        pe_num0 = arr[index][i0]
                        pe_num1 = arr[index][i1]
                        coe_ref0 = arr_coe[index][i0]
                        coe_ref1 = arr_coe[index][i1]
                        bias_dict[(i0,i1)] = 0
                        for j in range(arr_row):
                            if arr[j][i0] == pe_num0 and arr[j][i1] == pe_num1:
                                ratio0 = arr_coe[j][i0]/coe_ref0
                                ratio1 = arr_coe[j][i1]/coe_ref1
                                if(ratio0 == ratio1):
                                    bias_dict[(i0,i1)] += 1
    #print(f"index:{index} bias：")
    #for key, value in bias_dict.items():
    #    print(f"key: {key}, bias: {value}")

    return max(bias_dict, key=bias_dict.get)
                    
                    



class PE:
    pe_counter = 2  # 类变量，用于跟踪PE的编号
    def __init__(self,index,level,typeofPE,in0,in1,pe_number0,pe_number1):
        self.index = index
        self.level = level
        self.in0 = in0
        self.in1 = in1
        self.coe0 = 0.0
        self.coe1 = 0.0
        i0 = ''
        i1 = ''
        if typeofPE == 'u_add':
            i0 = f'a{in0+1}'
            i1 = f'a{in1+1}'
        elif typeofPE == 'v_add':
            i0 = f'b{in0+1}'
            i1 = f'b{in1+1}'
        elif typeofPE == 'mul':
            i0 = f'a{in0+1}'
            i1 = f'b{in1+1}'
        elif typeofPE == 'u_ratio' or typeofPE == 'v_ratio' or typeofPE == 'w_ratio':
            i0 = f'{pe_number0}'
        # w_add的输入一定来自其他pe的输出
        #elif typeofPE == 'w_add':
        #    i0 = f'm{in0}'
        #    i1 = f'm{in1}'

        if pe_number0 >= 2:
            # pe的输入是上一个pe的输出
            i0 = f'pe{pe_number0}'
        #else:
            # pe的输入来自总线

        if pe_number1 >= 2:
            i1 = f'pe{pe_number1}'
        #else:
            # pe的输入来自总线

        self.typeofPE = typeofPE
        self.name = f"{typeofPE}_{i0}_{i1}"
        self.pe_number = PE.pe_counter # 保存PE的编号
        PE.pe_counter += 1  # 递增编号



def distribute_pe(u,v,w,w_act,index,arch,level,typeofPE,in0,in1):
    # 如果level不存在，创建一个空列表
    if level not in arch:
        arch[level] = []

    # u的加法
    if typeofPE == 'u_add':
        pe_number0 = u[in0][index]
        pe_number1 = u[in1][index]

    # v的加法
    if typeofPE == 'v_add':
        pe_number0 = v[in0][index]
        pe_number1 = v[in1][index]

    # 乘法
    if typeofPE == 'mul':

        pe_number0 = u[in0][index]
        pe_number1 = v[in1][index]

    # w的加法
    if typeofPE == 'w_add':
        pe_number0 = w_act[index][in0]
        pe_number1 = w_act[index][in1]


    # 创建PE
    new_pe = PE(index,level,typeofPE,in0,in1,pe_number0,pe_number1)

    # 检查新创建的 PE 对象的 name 是否已经存在于系统结构中
    name_exists = any(pe.name == new_pe.name for pe in arch[level])

    if name_exists :
        PE.pe_counter -= 1
    else:
        # 将 PE 对象添加到相应级别的列表中
        arch[level].append(new_pe)

    return new_pe

def coefficient_backup(u,v,w,u_coe,v_coe,w_coe):
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            u_coe[i][j] = u[i][j]
            if u[i][j] != 0:
                u[i][j] = 1
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            v_coe[i][j] = v[i][j]
            if v[i][j] != 0 :
                v[i][j] = 1
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w_coe[i][j] = w[i][j]
            if w[i][j] != 0 :
                w[i][j] = 1 


# 每次level分配完pe，刷新一下矩阵，用于下个level分配
# 不同于u、v可以直接参与运算，w需要等待m算好才激活,w_act用于记录m的完成情况 
def refresh_calculation(u,v,w,u_coe,v_coe,w_coe,w_act,arch,level):

    # 刷新已分配的运算，两个元素，其中一个清零，另一个使用PE.number进行占位 
    if level not in arch:
        return

    for pe in arch[level]:
        # refresh u
        in0 = pe.in0
        in1 = pe.in1
        index = pe.index

        #print(f"refresh: pe{pe.pe_number}")
        #print(f"{pe.level}")
        #print(f"name: {pe.name}")
        #print(f"index: {pe.index}")
        #print(f"in0: {pe.in0}")
        #print(f"in1: {pe.in1}")

        if pe.typeofPE == 'u_add':
            for i in range(u.shape[1]):
                if i != index:
                    if(u[in0][index] == u[in0][i]):
                        coe0 = u_coe[in0][i] 
                        coe_ref0 = u_coe[in0][index]
                    else:
                        continue
                    if(u[in1][index] == u[in1][i]):
                        coe1 = u_coe[in1][i] 
                        coe_ref1 = u_coe[in1][index]
                    else:
                        continue

                    ratio0 = coe0/coe_ref0
                    ratio1 = coe1/coe_ref1

                    if(ratio0 == ratio1 and ratio0 != 0):
                        u[in0][i] = pe.pe_number
                        u[in1][i] = 0
                        u_coe[in0][i] = ratio0
                        u_coe[in1][i] = 0
            # 最后更新 index 位置
            pe.coe0 = u_coe[in0][index]
            pe.coe1 = u_coe[in1][index]
            u[in0][index] = pe.pe_number
            u[in1][index] = 0
            u_coe[in0][index] = 1
            u_coe[in1][index] = 0


        if pe.typeofPE == 'v_add':
            for i in range(v.shape[1]):
                if i != index:
                    if(v[in0][index] == v[in0][i]):
                        coe0 = v_coe[in0][i] 
                        coe_ref0 = v_coe[in0][index]
                    else:
                        continue
                    if(v[in1][index] == v[in1][i]):
                        coe1 = v_coe[in1][i] 
                        coe_ref1 = v_coe[in1][index]
                    else:
                        continue

                    ratio0 = coe0/coe_ref0
                    ratio1 = coe1/coe_ref1

                    if(ratio0 == ratio1 and ratio0 != 0):
                        v[in0][i] = pe.pe_number
                        v[in1][i] = 0
                        v_coe[in0][i] = ratio0
                        v_coe[in1][i] = 0
            # 最后更新 index 位置
            pe.coe0 = v_coe[in0][index]
            pe.coe1 = v_coe[in1][index]
            v[in0][index] = pe.pe_number
            v[in1][index] = 0
            v_coe[in0][index] = 1 
            v_coe[in1][index] = 0

        if pe.typeofPE == 'mul':
            pe.coe0 = u_coe[in0][index]
            pe.coe1 = v_coe[in1][index]
            u[in0][index] = 0
            v[in1][index] = 0
            u_coe[in0][index] = 0
            v_coe[in1][index] = 0
            for i in range(w.shape[0]):
                ratio = w_coe[i][index]
                if (ratio != 0):
                    # 激活w
                    w_act[i][index] = pe.pe_number
        if pe.typeofPE == 'w_add':
            for i in range(w.shape[0]):
                if i != index :
                    if(w_act[i][in0] != 0 and w_act[i][in1] != 0) and (w_act[index][in0] == w_act[i][in0] and w_act[index][in1] == w_act[i][in1]):
                        coe0 = w_coe[i][in0];
                        coe1 = w_coe[i][in1];
                        coe_ref0 = w_coe[index][in0]
                        coe_ref1 = w_coe[index][in1]
                    else:
                        continue
                    
                    ratio0 = coe0/coe_ref0
                    ratio1 = coe1/coe_ref1

                    if(ratio0 == ratio1 and ratio0 != 0):
                        w_act[i][in0] = pe.pe_number
                        w_act[i][in1] = 0
                        w_coe[i][in0] = ratio0
                        w_coe[i][in1] = 0

            pe.coe0 = w_coe[index][in0]
            pe.coe1 = w_coe[index][in1]
            w_act[index][in0] = pe.pe_number
            w_act[index][in1] = 0
            w_coe[index][in0] = 1
            w_coe[index][in1] = 0
            print(f"{pe.level}")
            print(f'refresh {pe.name}')
            print(f'w_act\r\n{w_act}')


def algorithm_verify(u,v,w,arr0,arr1):
    
    arr0_flatten = arr0.flatten()
    arr1_flatten = arr1.flatten()
    m = np.zeros(u.shape[1])
    for i in range(u.shape[1]):
        # col
        u_col = u[:, i]
        v_col = v[:, i]
        u_equation = 0
        v_equation = 0
        for j in range(u.shape[0]):
            # row
            u_equation += u_col[j]*arr0_flatten[j]
            #if u_col[j] != 1 and u_col[j] != -1 and u_col[j] != 0:
            #    print(f'Unexpected coefficient: u({j},{i}):{u_col[j]}')

        for j in range(v.shape[0]):
            # row
            v_equation += v_col[j]*arr1_flatten[j]
            #if v_col[j] != 1 and v_col[j] != -1 and v_col[j] != 0:
            #    print(f'Unexpected coefficient: v({j},{i}):{v_col[j]}')
        m[i] = u_equation * v_equation


    res_flatten = np.zeros(w.shape[0])
    for i in range(w.shape[0]):
        w_row = w[i,:]
        w_equation = 0
        for j in range(w.shape[1]):
            w_equation += w_row[j]*m[j]
            #if w_row[j] != 1 and w_row[j] != -1 and w_row[j] != 0:
            #    print(f'Unexpected coefficient: w({i},{j}):{w_row[j]}')
        res_flatten[i] = w_equation
    return   res_flatten.reshape(arr1.shape[1],arr0.shape[0]).T


def print_equations(u, v, w):

    u_coe = np.zeros(u.shape)
    v_coe = np.zeros(v.shape)
    w_coe = np.zeros(w.shape)

    w_act = np.zeros(w.shape,dtype=int)
    pe_arch  = {}

    coefficient_backup(u,v,w,u_coe,v_coe,w_coe)
    
    #print(u)
    #print(v)
    #print(w)

    #print(u_coe)
    #print(v_coe)
    #print(w_coe)
    
    (total_level,max_path_col) = find_u_v_max_path(u,v,w)
    print(f'total_level:{total_level}')

    for level_n in range(total_level):

        (this_total_path,max_path_col) = find_u_v_max_path(u,v,w)
        max_path_row = find_w_max_path(w_act)

        u_nonzero_count = count_nonzero(u, 'col', max_path_col)
        v_nonzero_count = count_nonzero(v, 'col', max_path_col)
        w_nonzero_count = count_nonzero(w_act, 'row', max_path_row)

        if(u_nonzero_count > 2):
            (u_in0,u_in1) = find_most_significant_adder(u,u_coe,max_path_col,'col')
            distribute_pe(u,v,w,w_act,max_path_col,pe_arch,f'level{level_n}','u_add',u_in0,u_in1)
            print(f'u most significant:{u_in0},{u_in1}')
        if(v_nonzero_count > 2):
            (v_in0,v_in1) = find_most_significant_adder(v,v_coe,max_path_col,'col')
            distribute_pe(u,v,w,w_act,max_path_col,pe_arch,f'level{level_n}','v_add',v_in0,v_in1)
            print(f'v most significant:{v_in0},{v_in1}')
        if(w_nonzero_count > 2):
            (w_in0,w_in1) = find_most_significant_adder(w_act,w_coe,max_path_row,'row')
            distribute_pe(u,v,w,w_act,max_path_row,pe_arch,f'level{level_n}','w_add',w_in0,w_in1)
            print(f'w most significant:{w_in0},{w_in1}')

        for i in range(u.shape[1]):
            u_nonzero_index = nonzero_indices(u, 'col', i)
            v_nonzero_index = nonzero_indices(v, 'col', i)
            u_nonzero_count = count_nonzero(u, 'col', i)
            v_nonzero_count = count_nonzero(v, 'col', i)
            if (u_nonzero_count == 1) and (v_nonzero_count == 1):
                in0 = u_nonzero_index[0]
                in1 = v_nonzero_index[0]
                distribute_pe(u,v,w,w_act,i,pe_arch,f'level{level_n}','mul',in0,in1)

            if (u_nonzero_count == 2):
                in0 = u_nonzero_index[0]
                in1 = u_nonzero_index[1]
                distribute_pe(u,v,w,w_act,i,pe_arch,f'level{level_n}','u_add',in0,in1)

            if (v_nonzero_count == 2):
                in0 = v_nonzero_index[0]
                in1 = v_nonzero_index[1]
                distribute_pe(u,v,w,w_act,i,pe_arch,f'level{level_n}','v_add',in0,in1)

        for i in range(w.shape[0]):
            w_act_nonzero_index = nonzero_indices(w_act, 'row', i) 
            w_act_nonzero_count = count_nonzero(w_act, 'row', i) 
            if(w_act_nonzero_count == 2):
                in0 = w_act_nonzero_index[0]
                in1 = w_act_nonzero_index[1]
                distribute_pe(u,v,w,w_act,i,pe_arch,f'level{level_n}','w_add',in0,in1)
                #print(f'distribute w_add:{i},{in0},{in1}')

        refresh_calculation(u,v,w,u_coe,v_coe,w_coe,w_act,pe_arch,f'level{level_n}')
        #print(f"u:\r\n{u}")
        #print(f"v:\r\n{v}")
        #print(f"w_act:\r\n{w_act}")
        #print(f"u_coe:\r\n{u_coe}")
        #print(f"v_coe:\r\n{v_coe}")
        #print(f"w_coe:\r\n{w_coe}")


    # 打印arch
    for level, pe_list in pe_arch.items():
        print(f"{level}:")
        for pe in pe_list:
            print(f"  PE Name: {pe.name}, Number: {pe.pe_number}, Type: {pe.typeofPE}, Inputs: {pe.in0}, {pe.in1}, coe: {pe.coe0}, {pe.coe1}")




