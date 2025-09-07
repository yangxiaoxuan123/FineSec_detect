import json
import os
import re
from collections import defaultdict

def load_json_data(file_path):
    """加载JSON格式的数据集"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式")
        return None

def save_json_data(data, file_path):
    """保存预处理后的数据集为JSON格式"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"预处理后的数据已保存至 {file_path}")
        return True
    except Exception as e:
        print(f"保存文件失败：{str(e)}")
        return False

def filter_by_length(data, max_bytes=32765, min_lines=3):
    """
    长度过滤：排除大于32765字节的代码文件和少于3行的函数
    """
    filtered_data = []
    removed_too_long = 0
    removed_too_short = 0
    
    for item in data:
        code = item.get('code', '') or item.get('input', '')
        
        # 检查代码长度（字节数）
        code_bytes = len(code.encode('utf-8'))
        if code_bytes > max_bytes:
            removed_too_long += 1
            continue
        
        # 检查代码行数
        code_lines = code.split('\n')
        # 过滤空行后检查行数
        non_empty_lines = [line.strip() for line in code_lines if line.strip()]
        if len(non_empty_lines) < min_lines:
            removed_too_short += 1
            continue
        
        filtered_data.append(item)
    
    print(f"长度过滤完成：")
    print(f"  移除过长代码（>32765字节）：{removed_too_long} 条")
    print(f"  移除过短代码（<3行有效代码）：{removed_too_short} 条")
    print(f"  保留数据：{len(filtered_data)} 条")
    return filtered_data

def remove_duplicates(data):
    """
    重复数据删除：移除重复的样本
    """
    seen = set()
    unique_data = []
    duplicate_count = 0
    
    for item in data:
        # 使用code/input和cwe的组合作为唯一标识
        code = item.get('code', '') or item.get('input', '')
        cwe = item.get('cwe', '')
        
        # 将cwe转换为字符串，处理可能的列表形式
        if isinstance(cwe, list):
            cwe_str = ','.join(cwe)
        else:
            cwe_str = str(cwe)
            
        # 创建唯一标识符
        identifier = f"{code}|{cwe_str}"
        
        if identifier in seen:
            duplicate_count += 1
            continue
        
        seen.add(identifier)
        unique_data.append(item)
    
    print(f"重复数据删除完成：")
    print(f"  移除重复样本：{duplicate_count} 条")
    print(f"  保留唯一样本：{len(unique_data)} 条")
    return unique_data

def hide_information(code):
    """
    信息隐藏：移除注释和包含CWE相关信息的内容
    """
    # 移除单行注释 (// ...)
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    
    # 移除多行注释 (/* ... */)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # 移除包含CWE相关信息的内容（如CWE-XXX）
    code = re.sub(r'CWE-\d+', '[REMOVED_CWE]', code, flags=re.IGNORECASE)
    
    # 移除可能包含漏洞提示的注释关键词
    vulnerability_keywords = [
        'buffer', 'overflow', 'injection', 'xss', 'sql', 
        'vulnerability', 'exploit', 'attack', 'danger'
    ]
    for keyword in vulnerability_keywords:
        # 只在可能是注释的位置替换（这里简化处理）
        code = re.sub(r'\b' + re.escape(keyword) + r'\b', '[KEYWORD]', code, flags=re.IGNORECASE)
    
    return code

def replace_keywords(code):
    """
    关键字替换：将BAD、GOOD、VULN、PATCHED等函数名替换为func
    """
    # 替换函数名中的关键字
    keywords = ['BAD', 'GOOD', 'VULN', 'PATCHED']
    for keyword in keywords:
        # 匹配函数定义中的关键字（如void BAD_function() -> void func_function()）
        pattern = re.compile(r'\b(' + re.escape(keyword) + r')\b', re.IGNORECASE)
        code = pattern.sub('func', code)
    
    return code

def process_information_hiding_and_keywords(data):
    """
    处理信息隐藏和关键字替换
    """
    processed_data = []
    
    for item in data:
        new_item = item.copy()
        # 获取代码内容
        code = item.get('code', '') or item.get('input', '')
        
        # 信息隐藏
        code = hide_information(code)
        
        # 关键字替换
        code = replace_keywords(code)
        
        # 更新代码字段
        if 'code' in new_item:
            new_item['code'] = code
        elif 'input' in new_item:
            new_item['input'] = code
            
        processed_data.append(new_item)
    
    print(f"信息隐藏和关键字替换完成：处理了 {len(processed_data)} 条数据")
    return processed_data

def preprocess_dataset(original_file, output_file):
    """
    完整的数据集预处理流程
    """
    # 1. 加载数据
    print("加载原始数据...")
    data = load_json_data(original_file)
    if not data:
        return False
    
    original_count = len(data)
    print(f"原始数据量：{original_count} 条")
    
    # 2. 长度过滤
    print("\n=== 执行长度过滤 ===")
    data = filter_by_length(data)
    
    # 3. 重复数据删除
    print("\n=== 执行重复数据删除 ===")
    data = remove_duplicates(data)
    
    # 4. 信息隐藏和关键字替换
    print("\n=== 执行信息隐藏和关键字替换 ===")
    data = process_information_hiding_and_keywords(data)
    
    # 5. 保存结果
    print("\n=== 保存预处理结果 ===")
    save_json_data(data, output_file)
    
    # 6. 输出总体统计
    print("\n预处理总体统计：")
    print(f"原始数据量：{original_count} 条")
    print(f"预处理后数据量：{len(data)} 条")
    print(f"数据保留比例：{len(data)/original_count*100:.2f}%")
    
    return True

if __name__ == "__main__":
    # 配置文件路径
    INPUT_FILE = "original_dataset.json"    # 原始数据集路径
    OUTPUT_FILE = "preprocessed_dataset.json"  # 预处理后数据集路径
    
    # 执行预处理
    preprocess_dataset(INPUT_FILE, OUTPUT_FILE)
