import Get_neo4j_2
import Data_Extractor_7
import sys
import io
import os
import json
import pandas as pd



if __name__ == '__main__':

    '''
    必备操作：
    1，数据抽取类的初始化
    2，图数据库连接的初始化

    使用方法：
    对于输入的问题，使用（在此代码70行左右）
        answer = extractor.extract_keywords_with_deepseek("爱因斯坦发现了相对论") 
        prompt = kg.semantic_search(answer)                                    
        print(prompt)         
    第一行代码，输入问题（字符串），转化成关键词列表
    第二行：对关键词列表，调用查询，返回“主谓宾句子”列表
    第三行：展示结果，可有可无

    '''
    
    # 初始化抽取器 (启用CLIP需要先安装transformers和torch)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # 获取DeepSeek API Key
    api_key = os.getenv("sk-c28ec338b39e4552b9e6bded47466442")
    if not api_key:
        api_key = "sk-c28ec338b39e4552b9e6bded47466442"  # 你的DeepSeek API Key

    extractor = Data_Extractor_7.EnhancedMultimodalExtractor(
        use_clip=True,
        domain_config_path=r"E:\Neo4j\neo4j-community-5.26.0-windows\neo4j-community-5.26.0-windows\neo4j-community-5.26.0\import\domain_config.json",
        use_deepseek_api=True,  # 启用DeepSeek云端API
        deepseek_model="deepseek-chat",  # 使用DeepSeek的chat模型
        api_key=api_key  # 传入API Key
    )

    # 上面这些是完成数据抽取代码的初始化（包括调用deepseek大模型）

    
    # 下面这个注释是数据抽取的过程，暂时可以不管
    '''
    file_path = r"E:/Neo4j/neo4j-community-5.26.0-windows/neo4j-community-5.26.0-windows/neo4j-community-5.26.0\import\week4_p65.json"
    # 执行抽取
    result = extractor.process_multimodal_data([file_path])

    entities = result.entities
    relations = result.relationships
    attributes = result.attributes

    # 转换数据
    nodes_df, rels_df = Data_Extractor_7.convert_extracted_to_dataframes(entities, relations)
    '''
    try:
        kg = Get_neo4j_2.Neo4jTeamCollaborator(
            uri="bolt://101.132.130.25:7687",
            user="neo4j",
            password="wangshuxvan@1"
        )
        # 图数据库连接初始化
        
        # 从DataFrame直接导入数据
        # kg.import_from_dataframe(nodes_df=nodes_df, rels_df=rels_df)
        answer = extractor.extract_keywords_with_deepseek("爱因斯坦发现了相对论") # 输入问题（字符串），使用数据抽取那个类里面的deepseek做关键词抽取工作，返回关键词列表
        prompt = kg.semantic_search(answer)                                     # 对于关键词列表查找，返回“主谓宾句子”的列表
        print(prompt)          # 展示结果，可有可无

    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if 'kg' in locals():
            kg.close()
            print("数据库连接已关闭")



