import re
from sympy import symbols, simplify, sympify

# 定义符号
#a_vars = symbols('a1 a2 a3 a4 a5 a6 a7 a8 a9')
#b_vars = symbols('b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12')

# 读取文件
filename = 'equation.txt'  # 替换为你的文件路径
with open(filename, 'r') as file:
    file_content = file.read()

# 使用正则表达式匹配所有m、a、b表达式
m_matches = re.findall(r'm\d+\s*=\s*(.*)', file_content)
a_matches = re.findall(r'a\d+\s*=\s*(.*)', file_content)
b_matches = re.findall(r'b\d+\s*=\s*(.*)', file_content)

# 创建符号表示的a和b
a_dict = {}
for i, a_match in enumerate(a_matches, start=1):
    a_dict[f'a{i}'] = sympify(a_match)

b_dict = {}
for i, b_match in enumerate(b_matches, start=1):
    b_dict[f'b{i}'] = sympify(b_match)

# 创建符号表示的m
m_dict = {}
for i, m_match in enumerate(m_matches, start=1):
    m_dict[f'm{i}'] = sympify(m_match).subs(a_dict).subs(b_dict)

# 遍历c表达式并输出化简结果
c_matches = re.findall(r'c\d+\s*=\s*(.*)', file_content)
for i, c_match in enumerate(c_matches, start=1):
    c_expr = sympify(c_match).subs(m_dict)
    simplified_c_expr = simplify(c_expr)
   #print(f"c{i} (Original): {c_expr}")
    print(f"c{i} = {simplified_c_expr}\n")

