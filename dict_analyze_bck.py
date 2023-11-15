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

def column_merge(arr, res_dict, mask_col):
    # 查找同类项
    num_cols = arr.shape[1]
    for i in range(num_cols):
        for j in range(i + 1, num_cols):
            # 遍历任意两列组合，相等为1,相反为-1,零保持不变
            arr_col_nonzero_idices_i_eq_j = (arr[:,i] == arr[:,j]) & (arr[:, i] != 0)
            arr_col_nonzero_idices_i_eqn_j = (arr[:,i] == -arr[:,j]) & (arr[:, i] != 0)
            arr_col_nonzero_idices_i_j = arr_col_nonzero_idices_i_eq_j.astype(int) - arr_col_nonzero_idices_i_eqn_j.astype(int)
            if (i not in mask_col) & (j not in mask_col):
                # 只有 >= 2才有意义,可合并,加入字典
                if sum(1 for element in arr_col_nonzero_idices_i_j if element != 0) >= 2:
                    res_dict[(i, j)] = arr_col_nonzero_idices_i_j

def find_most_significant_adder(arch,arr,cmd,index):
    arr_row = arr.shape[0]
    arr_col = arr.shape[1]
    for i in range(arr_row):
        if arr[i][index] != 0:
            for j in range(i+1,arr_row):
                if arr[j][index] != 0:
                    for k in range(arr_col):
                        if arr[i][index] > 10:
                            if cmd == 'u'
                                # find pe
                                pe0 = [pe for pe_list in arch for pe in pe_list if (pe.typeofPE == 'u_ratio' and pe.in0 = ]
                            ref_i = pe0.coe0
                    
                    


    ## 查找同类项
    #num_cols = arr.shape[1]
    ##res = [[0 for _ in range(num_cols)] for _ in range(arr.shape[0])] 
    #res = np.zeros((arr.shape[0], num_cols))
    #for j in range(num_cols):
    #    if j != col:
    #        # 相等为1,相反为-1,零保持不变
    #        arr_col_nonzero_idices_i_eq_j = (arr[:,col] == arr[:,j]) & (arr[:, col] != 0)
    #        arr_col_nonzero_idices_i_eqn_j = (arr[:,col] == -arr[:,j]) & (arr[:, col] != 0)
    #        arr_col_nonzero_idices_i_j = arr_col_nonzero_idices_i_eq_j.astype(int) - arr_col_nonzero_idices_i_eqn_j.astype(int)
    #        # 只有 >= 2才有意义,可合并,加入字典
    #        if sum(1 for element in arr_col_nonzero_idices_i_j if element != 0) >= 2:
    #            for i in range(len(res)):
    #                res[i][j] = arr_col_nonzero_idices_i_j[i]
    #
    #bias_dict = {}
    #for i in range(res.shape[1]):            
    #    res_col_nonzero_indces = nonzero_indices(res,'col',i)
    #    for m in range(res_col_nonzero_indces.shape[0]):
    #        for n in range(m+1,res_col_nonzero_indces.shape[0]):
    #            ind0 = res_col_nonzero_indces[m]
    #            ind1 = res_col_nonzero_indces[n]
    #            if (ind0,ind1) not in bias_dict: 
    #                bias_dict[(ind0,ind1)] = 2
    #                for j in range(i+1,res.shape[1]):
    #                    if (res[ind0][i] == res[ind0][j]) & (res[ind1][i] == res[ind1][j]) | (res[ind0][i] == -res[ind0][j]) & (res[ind1][i] == -res[ind1][j]): 
    #                        bias_dict[(ind0,ind1)] += 1


    #print("最高优先级合并：")
    #print(res)
    print("bias：")
    for key, value in bias_dict.items():
        print(f"key: {key}, bias: {value}")

    #for j in :
    #    print(f"列: {key}, 同类项: {value}")



class PE:
    pe_counter = 10  # 类变量，用于跟踪PE的编号,此处避开0 1 0.125 0.5 2 3
    def __init__(self,index,level,typeofPE,in0,in1,pe_number0,pe_number1):
        self.index = index
        self.level = level
        self.in0 = in0
        self.in1 = in1
        i0 = ''
        i1 = ''
        if typeofPE == 'u_add':
            i0 = f'a{in0}'
            i1 = f'a{in1}'
        elif typeofPE == 'v_add':
            i0 = f'b{in0}'
            i1 = f'b{in1}'
        elif typeofPE == 'mul':
            i0 = f'a{in0}'
            i1 = f'b{in1}'
        elif typeofPE == 'u_ratio' or typeofPE == 'v_ratio' or typeofPE == 'w_ratio':
            i0 = f'{pe_number0}'
        # w_add的输入一定来自其他pe的输出
        #elif typeofPE == 'w_add':
        #    i0 = f'm{in0}'
        #    i1 = f'm{in1}'

        if pe_number0 >= 10:
            # pe的输入是上一个pe的输出
            i0 = f'pe{pe_number0}'
            #self.coe0 = 1 if pe_number0 > 0 else -1
            self.coe0 = 1
        else:
            # pe的输入来自总线
            self.coe0 = pe_number0

        if pe_number1 >= 10:
            i1 = f'pe{pe_number1}'
            self.coe1 = 1
        else:
            # pe的输入来自总线
            self.coe1 = pe_number1

        self.typeofPE = typeofPE
        self.name = f"{typeofPE}_{i0}_{i1}"
        self.pe_number = PE.pe_counter # 保存PE的编号
        PE.pe_counter += 1  # 递增编号



def distribute_pe(u,v,w,index,arch,level,typeofPE,in0,in1):
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
        pe_number0 = w[index][in0]
        pe_number1 = w[index][in1]


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

# 每次level分配完pe，刷新一下矩阵，用于下个level分配
# 不同于u、v可以直接参与运算，w需要等待m算好才激活,w_act用于记录m的完成情况 
def refresh_calculation(u,v,w,w_act,arch,level):

    # 刷新已分配的运算，两个元素，其中一个清零，另一个使用PE.number进行占位 
    for pe in arch[level]:
        # refresh u
        in0 = pe.in0
        in1 = pe.in1

        if pe.typeofPE == 'u_add':
            for i in range(u.shape[1]):
                coe0 = u[in0][i] 
                coe1 = u[in1][i] 
                ratio0 = coe0/pe.coe0
                ratio1 = coe1/pe.coe1
                if(ratio0 == ratio1 and ratio0 != 0):
                    new_pe = PE(i,level,'u_ratio',in0,i,ratio0,pe.pe_number) # ratio作为系数coe0, pe.number作为输入, in0是行, i是列
                    name_exists = any(pe_this.name == new_pe.name for pe_this in arch[level])
                    if name_exists :
                        PE.pe_counter -= 1
                    else:
                        # 将 PE 对象添加到相应级别的列表中
                        arch[level].append(new_pe)
                    u[in0][i] = pe.pe_number
                    u[in1][i] = 0
        if pe.typeofPE == 'v_add':
            for i in range(v.shape[1]):
                coe0 = v[in0][i] 
                coe1 = v[in1][i] 
                ratio0 = coe0/pe.coe0
                ratio1 = coe1/pe.coe1
                if(ratio0 == ratio1 and ratio0 != 0):
                    new_pe = PE(i,level,'v_ratio',in0,i,ratio0,pe.pe_number) # ratio作为系数coe0, pe.number作为输入, in0是行, i是列
                    name_exists = any(pe_this.name == new_pe.name for pe_this in arch[level])
                    if name_exists :
                        PE.pe_counter -= 1
                    else:
                        # 将 PE 对象添加到相应级别的列表中
                        arch[level].append(new_pe)
                    v[in0][i] = pe.pe_number
                    v[in1][i] = 0                
        if pe.typeofPE == 'mul':
            for i in range(w.shape[0]):
                index = pe.index
                ratio = w[i][index]
                if (ratio != 0):
                    new_pe = PE(index,level,'w_ratio',i,index,ratio,pe.pe_number) # ratio作为系数coe0, pe.number作为输入, in0是行, i是列
                    name_exists = any(pe_this.name == new_pe.name for pe_this in arch[level])
                    if name_exists :
                        PE.pe_counter -= 1
                    else:
                        # 将 PE 对象添加到相应级别的列表中
                        arch[level].append(new_pe)
                    # 激活w
                    w_act[i][index] = pe.pe_number
        if pe.typeofPE == 'w_add':
            for i in range(w.shape[0]):
                coe0 = w_act[i][in0] 
                coe1 = w_act[i][in1] 
                ratio0 = coe0/pe.coe0
                ratio1 = coe1/pe.coe1
                if(ratio0 == ratio1 and ratio0 != 0):
                    new_pe = PE(i,level,'w_ratio',in0,i,ratio0,pe.pe_number) # ratio作为系数coe0, pe.number作为输入, in0是列, i是行
                    name_exists = any(pe_this.name == new_pe.name for pe_this in arch[level])
                    if name_exists :
                        PE.pe_counter -= 1
                    else:
                        # 将 PE 对象添加到相应级别的列表中
                        arch[level].append(new_pe)
                    w_act[i][in0] = pe.pe_number
                    w_act[i][in1] = 0               


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
            if u_col[j] != 1 and u_col[j] != -1 and u_col[j] != 0:
                print(f'Unexpected coefficient: u({j},{i}):{u_col[j]}')
            #if u_col[j] == 1:
            #    u_equation += arr0_flatten[j]
            #elif u_col[j] == -1:
            #    u_equation -= arr0_flatten[j]

        for j in range(v.shape[0]):
            # row
            v_equation += v_col[j]*arr1_flatten[j]
            if v_col[j] != 1 and v_col[j] != -1 and v_col[j] != 0:
                print(f'Unexpected coefficient: v({j},{i}):{v_col[j]}')
            #if v_col[j] == 1:
            #    v_equation += arr1_flatten[j]
            #elif v_col[j] == -1:
            #    v_equation -= arr1_flatten[j]
        m[i] = u_equation * v_equation
        #print(f'm{i+1}={m[i]}')


    res_flatten = np.zeros(w.shape[0])
    for i in range(w.shape[0]):
        w_row = w[i,:]
        w_equation = 0
        for j in range(w.shape[1]):
            w_equation += w_row[j]*m[j]
            if w_row[j] != 1 and w_row[j] != -1 and w_row[j] != 0:
                print(f'Unexpected coefficient: w({i},{j}):{w_row[j]}')
            #if w_row[j] == 1:
            #    w_equation += m[j]
            #elif w_row[j] == -1:
            #    w_equation -= m[j]
        res_flatten[i] = w_equation
        #print(f'c{i+1}={w_equation}')
    return   res_flatten.reshape(arr1.shape[1],arr0.shape[0]).T


def print_equations(u, v, w):
    #for i in range(u.shape[1]):
    #    u_col = u[:, i]
    #    v_col = v[:, i]
    #    #w_row = w[:, i]

    #    u_equation = ""
    #    v_equation = ""

    #    #print(f"u_row_num = {u.shape[0]}")
    #    #print(f"u_col_num = {u.shape[1]}")
    #    #print(f"v_row_num = {v.shape[0]}")
    #    #print(f"v_col_num = {v.shape[1]}")

    #    for j in range(u.shape[0]):
    #        if u_col[j] == 1:
    #            u_equation += f" + a{j+1}"
    #        elif u_col[j] == -1:
    #            u_equation += f" - a{j+1}"

    #    #if len(u_equation) != 0 :
    #    if u_equation[1] == '+':
    #        u_equation = u_equation[3:]  # Remove leading space and plus sign

    #    for j in range(v.shape[0]):
    #        if v_col[j] == 1:
    #            v_equation += f" + b{j+1}"
    #        elif v_col[j] == -1:
    #            v_equation += f" - b{j+1}"
    #    #if len(v_equation) != 0 :
    #    if v_equation[1] == '+':
    #        v_equation = v_equation[3:]

    #    equation = f"m{i+1} = ({u_equation}) * ({v_equation})"
    #    print(equation)

    #for i in range(w.shape[0]):
    #    w_row = w[i,:]
    #    w_equation = ""
    #    for j in range(w.shape[1]):
    #        if w_row[j] == 1:
    #            w_equation += f" + m{j+1}"
    #        elif w_row[j] == -1:
    #            w_equation += f" - m{j+1}"
    #    if len(w_equation) != 0 :
    #        if w_equation[1] == '+':
    #            w_equation = w_equation[3:]
    #    equation = f"c{i+1} = ({w_equation})"
    #    print(equation)

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
    print("\r\nres_mul_num:", res_mul_num)
    print("\r\nres_add_num:", res_add_num)
    #print(f"Element at ({i}, {j}) : {add_num}")

    # 找到 res_mul_num 与 res_add_num 之和最大的下标
    #most_deep_res = max(range(len(res_mul_num)), key=lambda i: res_mul_num[i] + res_add_num[i]) + 1
    #print(f"The most deep res: c{most_deep_res}")

    # 按照复杂程度排序
    sorted_indices = sorted(range(len(res_mul_num)), key=lambda i: res_mul_num[i] + res_add_num[i],reverse=True)
    print("\r\nSorted indices:", sorted_indices)

 


    most_deep_res = sorted_indices[0]

    # 获取数组的列数
    #num_cols = v.shape[1]
    # 用于记录任意两列相同元素个数的矩阵
    #same_elements_count = np.zeros((num_cols, num_cols), dtype=int)
    #
    ## 遍历每一对列
    #for i in range(num_cols):
    #    for j in range(i + 1, num_cols):
    #        # 计算相同元素的个数
    #        count = np.sum((v[:, i] == v[:, j]) & (v[:, i] != 0))
    #        same_elements_count[i, j] = count
    #        same_elements_count[j, i] = count  # 对称性，因为相同的元素数量是一样的
    #        print(f"same_element: {i},{j}:{count}")
    #
    ## 打印结果
    #print("相同元素的个数矩阵:")
    #print(same_elements_count)


    # 预处理，找到可以在前两层完成运算的m，直接分配
    add_index = 0
    mul_index = 0
    level0_calc = ""
    level1_calc = ""
    u_nonzero_index = ""
    v_nonzero_index = ""

    w_act = np.zeros(w.shape)

    pe_arch  = {}

    # 记录每个m的完成情况，1完成，0未完成
    # m = np.zeros(u.shape[1]), dtype=int)

    # 每层m的完成情况：u v矩阵会消去，如果u v某一列全为零，则对应m完成
    
    col_distributed = []

    print("\r\nSome calculation can be distributed directly.")
    print("Notice that \'+\' and \'*\' has coefficient actrually,here we omit.")
    for i in range(u.shape[1]):
        u_nonzero_index = nonzero_indices(u, 'col', i)
        v_nonzero_index = nonzero_indices(v, 'col', i)
        if (count_nonzero(u, 'col', i) == 1) & (count_nonzero(v, 'col', i) == 1):
            in0 = u_nonzero_index[0]
            in1 = v_nonzero_index[0]

            distribute_pe(u,v,w,i,pe_arch,'level1','mul',in0,in1)
            refresh_calculation(u,v,w,w_act,pe_arch,'level1')

            # level0 分配一个乘法器
            level0_calc += f"mul{mul_index}: a{u_nonzero_index[0]} * b{v_nonzero_index[0]}\r\n"

            mul_index += 1
            col_distributed.append(i) 
        elif (count_nonzero(u, 'col', i) == 1) & (count_nonzero(v, 'col', i) == 2):
            # level0 分配一个加法器
            # level1 分配一个乘法器
            level0_calc += f"add{add_index}: b{v_nonzero_index[0]} + b{v_nonzero_index[1]}\r\n"
            level1_calc += f"mul{mul_index}: b{u_nonzero_index[0]} * add{add_index}\r\n"
            mul_index += 1
            add_index += 1
            col_distributed.append(i) 
        elif (count_nonzero(u, 'col', i) == 2) & (count_nonzero(v, 'col', i) == 1):
            # level0 分配一个加法器
            # level1 分配一个乘法器
            level0_calc += f"add{add_index}: b{u_nonzero_index[0]} + b{u_nonzero_index[1]}\r\n"
            level1_calc += f"mul{mul_index}: b{v_nonzero_index[0]} * add{add_index}\r\n"
            mul_index += 1
            add_index += 1
            col_distributed.append(i) 
        elif (count_nonzero(u, 'col', i) == 2) & (count_nonzero(v, 'col', i) == 2):
            # level0 分配两个加法器
            # level1 分配一个乘法器
            level0_calc += f"add{add_index}: b{u_nonzero_index[0]} + b{u_nonzero_index[1]}\r\n"
            add_index += 1
            level0_calc += f"add{add_index}: b{v_nonzero_index[0]} + b{v_nonzero_index[1]}\r\n"
            level1_calc += f"mul{mul_index}: add{add_index - 1} * add{add_index}\r\n"
            mul_index += 1
            add_index += 1
            col_distributed.append(i) 
    
    print(f"level0:\r\n{level0_calc}")
    print(f"level1:\r\n{level1_calc}")


    level_num = res_mul_num[sorted_indices[0]] + res_add_num[sorted_indices[0]]
    print(f'total level num: {level_num}')

    if 

    # 查找同类项
    u_col_merge_matrix = {}
    v_col_merge_matrix = {}
    #column_merge(u,u_col_merge_matrix,col_distributed)
    #column_merge(v,v_col_merge_matrix,col_distributed)
    column_merge(u,u_col_merge_matrix,[])
    column_merge(v,v_col_merge_matrix,[])

    find_most_significant_adder(u,sorted_indices[0])

    # 逐行打印字典的键值对
    # print("u可化简的运算：")
    # for key, value in u_col_merge_matrix.items():
    #     print(f"列: {key}, 同类项: {value}")

    # print("v可化简的运算：")
    # for key, value in v_col_merge_matrix.items():
    #     print(f"列: {key}, 同类项: {value}")


    # 打印arch

    for level, pe_list in pe_arch.items():
        print(f"Level: {level}")
        for pe in pe_list:
            print(f"  PE Name: {pe.name}, Number: {pe.pe_number}, Type: {pe.typeofPE}, Inputs: {pe.in0}, {pe.in1}, coe: {pe.coe0}, {pe.coe1}")











        
    #for i in range(w.shape[0]):
    #    mul_num = 0
    #    add_num = count_nonzero(w, 'row', i) - 1 
    #    for j in range(w.shape[1]):
    #        if w[i,j] != 0:
    #            # 查找res的相关数目
    #            mul_num += 1
    #            if count_nonzero(u, 'col', j) > 2:
    #                
    #            add_num += count_nonzero(u, 'col', j) - 1 
    #            add_num += count_nonzero(v, 'col', j) - 1
    #    res_mul_num.append(mul_num)
    #    res_add_num.append(add_num)
    #print("res_mul_num:", res_mul_num)
    #print("res_add_num:", res_add_num)



# Given matrices u, v, w
#u = np.array([[0	, 1	, 1	, 0	, 1	, 1	, 0]	,
#              [0	, 0	, -1	, 1	, 0	, 0	, 0]	,
#              [1	, 1	, 1	, 0	, 1	, 0	, 0]	,
#              [-1	, -1	, -1	, 0	, 0	, 0	, 1]])
#
#v = np.array([[0	, 0	, 0	, 0	, 1	, 1	, 0]	,
#              [1	, 1	, 0	, 0	, 1	, 0	, 1]	,
#              [0	, 1	, 1	, 1	, 1	, 0	, 0]	,
#              [0	, 1	, 1	, 0	, 1	, 0	, 1]])
#
#w = np.array([[0	, 0	, 0	, 1	, 0	, 1	, 0]	,
#              [0	, -1	, 0	, 0	, 1	, -1	, -1]	,
#              [-1	, 1	, -1	, -1	, 0	, 0	, 0]	,
#              [1	, 0	, 0	, 0	, 0	, 0	, 1]])

# Print equations m1 to m7
#print_equations(u, v, w)


