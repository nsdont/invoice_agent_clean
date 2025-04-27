#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
产品数据库创建脚本
将Excel产品清单转换为CSV和JSON格式，并进行必要的数据清洗和标准化
"""

import os
import sys
import pandas as pd
import json
import re
from pathlib import Path

def clean_product_name(name):
    """清理产品名称，统一格式"""
    if not isinstance(name, str):
        return str(name)
    
    # 去除多余空格
    name = re.sub(r'\s+', ' ', name.strip())
    return name

def generate_aliases(df):
    """为产品生成可能的别名，用于模糊匹配"""
    if 'aliases' not in df.columns:
        print("生成产品别名列...")
        
        def create_aliases(name):
            name_str = str(name)
            aliases = [name_str]  # 原始名称始终作为一个别名
            
            # 处理分隔符，不包括括号
            delimiters = ['/', '-', '、', '，', ',']
            
            # 对每个分隔符进行分割并收集所有部分
            current_parts = [name_str]
            for delimiter in delimiters:
                new_parts = []
                for part in current_parts:
                    if delimiter in part:
                        split_parts = [p.strip() for p in part.split(delimiter)]
                        new_parts.extend([p for p in split_parts if p])
                    else:
                        new_parts.append(part)
                current_parts = new_parts
            
            # 添加所有分割后的部分作为别名
            for part in current_parts:
                if part and part != name_str and part not in aliases:
                    aliases.append(part)
            
            return aliases
            
        df['aliases'] = df['name'].apply(create_aliases)
    return df

def standardize_price(df):
    """标准化价格数据"""
    if 'price' in df.columns:
        # 确保价格是数值类型
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        # 填充缺失值为0
        df['price'] = df['price'].fillna(0)
    return df

def process_excel_to_database(excel_path, output_dir='data'):
    """
    处理Excel产品清单，转换为CSV和JSON格式
    
    参数:
    excel_path: Excel文件路径
    output_dir: 输出目录
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"正在读取Excel文件: {excel_path}")
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path)
        
        # 显示原始数据信息
        print(f"读取到 {len(df)} 条产品记录")
        print("原始列名:", df.columns.tolist())
        
        # 检查并重命名关键列
        column_mapping = {
            # 示例映射，根据实际Excel调整
            '产品名称': 'name',
            '价格': 'price',
            '类别': 'category',
            # 中文键名映射为英文
            '品號': 'product_code',
            '品名': 'name',
            '單位': 'unit',
            '幣別': 'currency',
            '數量': 'quantity',
            '金額': 'amount',
            '單價': 'price',
            '備註': 'remarks',
            '日期': 'date',
            '客戶': 'customer',
            '供應商': 'supplier',
            # 添加更多映射...
        }
        
        # 应用列名映射（如果列存在）
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # 确保所有列名都是英文（处理未在映射中指定的中文列名）
        for col in df.columns:
            if any('\u4e00' <= c <= '\u9fff' for c in col):  # 检测是否包含中文字符
                # 将未映射的中文列名转换为拼音或其他英文表示
                # 这里简单地用col_X替代，X是列的序号
                col_index = list(df.columns).index(col)
                new_col_name = f"column_{col_index}"
                print(f"将列名 '{col}' 重命名为 '{new_col_name}'")
                df = df.rename(columns={col: new_col_name})
        
        # 确保必要的列存在
        required_columns = ['name', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"警告: 缺少必要的列: {missing_columns}")
            # 如果缺少name列，尝试找到可能的替代列
            if 'name' in missing_columns:
                # 查找可能包含产品名称的列
                possible_name_cols = [col for col in df.columns if '名' in col or 'name' in col.lower()]
                if possible_name_cols:
                    print(f"使用 '{possible_name_cols[0]}' 作为产品名称列")
                    df = df.rename(columns={possible_name_cols[0]: 'name'})
                    missing_columns.remove('name')
            
            # 对于仍然缺失的列，创建默认值
            for col in missing_columns:
                if col == 'name':
                    df['name'] = f"未命名产品_{df.index}"
                elif col == 'price':
                    df['price'] = 0.0
        
        # 数据清洗和标准化
        print("正在进行数据清洗和标准化...")
        df['name'] = df['name'].apply(clean_product_name)
        df = generate_aliases(df)
        df = standardize_price(df)
        
        # 输出CSV
        csv_path = os.path.join(output_dir, 'products.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV文件已保存至: {csv_path}")
        
        # 输出JSON
        json_path = os.path.join(output_dir, 'products.json')
        
        # 将DataFrame转换为字典列表
        products_list = df.to_dict(orient='records')
        
        # 确保JSON中的别名是实际的列表而不是字符串
        for product in products_list:
            if 'aliases' in product and isinstance(product['aliases'], str):
                try:
                    product['aliases'] = eval(product['aliases'])
                except:
                    product['aliases'] = [product['aliases']]
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(products_list, f, ensure_ascii=False, indent=2)
        
        print(f"JSON文件已保存至: {json_path}")
        
        return True, f"处理完成。产品数据已保存为CSV和JSON格式。共处理{len(df)}条记录。"
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return False, f"处理失败: {str(e)}"

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python create_product_database.py <Excel文件路径> [输出目录]")
        print("示例: python create_product_database.py ../data/products.xlsx ../data")
        sys.exit(1)
    
    excel_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'data'
    
    success, message = process_excel_to_database(excel_path, output_dir)
    
    if success:
        print("\n✅ " + message)
        sys.exit(0)
    else:
        print("\n❌ " + message)
        sys.exit(1)

if __name__ == "__main__":
    main() 