import Get_neo4j_2
import Data_Extractor_7
import sys
import io
import os
import json
import pandas as pd



if __name__ == '__main__':

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
    '''
    file_path = r"E:/Neo4j/neo4j-community-5.26.0-windows/neo4j-community-5.26.0-windows/neo4j-community-5.26.0\import\week4_p65.json"
    # 执行抽取
    result = extractor.process_multimodal_data([file_path])

    entities = result.entities
    relations = result.relationships
    attributes = result.attributes



    # 转换数据
    nodes_df, rels_df = Data_Extractor_7.convert_extracted_to_dataframes(entities, relations)
    
    print("节点数据:")
    print(nodes_df.head())
    print("\n关系数据:")
    print(rels_df.head())
    '''

    try:
        kg = Get_neo4j_2.Neo4jTeamCollaborator(
            uri="bolt://101.132.130.25:7687",
            user="neo4j",
            password="wangshuxvan@1"
        )

        # 从DataFrame直接导入数据
        # kg.import_from_dataframe(nodes_df=nodes_df, rels_df=rels_df)
        answer = extractor.extract_keywords_with_deepseek("爱因斯坦发现了相对论")
        prompt = kg.semantic_search(answer)
        print(prompt)

    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if 'kg' in locals():
            kg.close()
            print("数据库连接已关闭")
