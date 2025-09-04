from neo4j import GraphDatabase
import pandas as pd
from IPython.display import display
import os
import chardet
from typing import Optional, Dict, List,Tuple
import requests
import json
from collections import defaultdict



def get_encoding(file_paths: List[str]) -> None:
    """检测文件编码格式"""
    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read())
            print(f"{file_path}: {result['encoding']}")
        except Exception as e:
            print(f"检测文件 {file_path} 编码时出错: {e}")


def file_exist(file_paths: List[str]) -> None:
    """检查文件是否存在"""
    for file_path in file_paths:
        print(f"{file_path}: {'存在' if os.path.exists(file_path) else '不存在'}")


class Neo4jTeamCollaborator:
    def __init__(self, uri: str, user: str, password: str):
        """初始化neo4j"""
        self.driver = None
        try:
            config = {
                "keep_alive": True,
                "max_connection_lifetime": 3600,
                "max_connection_pool_size": 100
            }
            self.driver = GraphDatabase.driver(uri, auth=(user, password), **config)
            self._check_connection()
        except Exception as e:
            print(f"初始化失败: {e}")
            raise

    def _check_connection(self) -> bool:
        """测试数据库连接"""
        try:
            if self.driver:
                with self.driver.session() as session:
                    session.run("RETURN 1 AS status")
                print("Neo4j连接成功")
                return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def close(self) -> None:
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            print("数据库连接已关闭")

    def clear_database(self) -> None:
        """清空现有数据"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("数据已清空")
        except Exception as e:
            print(f"清空数据库时出错: {e}")

    def _detect_encoding(self, file_path: str) -> str:
        """检测本地文件编码"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception as e:
            print(f"检测文件编码失败 {file_path}: {e}")
            return 'utf-8'

    def _detect_encoding_from_url(self, url: str) -> str:
        """从HTTP URL检测文件编码"""
        try:
            raw_data = requests.get(url).content[:10000]
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
        except Exception as e:
            print(f"检测URL编码失败 {url}: {e}")
            return 'utf-8'

    def _read_csv_data(self, file_path: str) -> pd.DataFrame:
        """读取CSV数据"""
        if file_path.startswith("http"):
            encoding = self._detect_encoding_from_url(file_path)
            response = requests.get(file_path)
            response.encoding = encoding
            return pd.read_csv(response.text)
        else:
            encoding = self._detect_encoding(file_path)
            return pd.read_csv(file_path, encoding=encoding)

    def _read_json_data(self, file_path: str) -> pd.DataFrame:
        """读取JSON数据"""
        if file_path.startswith("http"):
            encoding = self._detect_encoding_from_url(file_path)
            response = requests.get(file_path)
            response.encoding = encoding
            return pd.read_json(response.text)
        else:
            encoding = self._detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                return pd.read_json(f)

    def import_from_csv(self, node_file: str = None, rel_file: str = None, batch_size: int = 1000) -> None:
        """支持HTTP和本地路径的CSV导入函数，自动检测编码"""
        try:
            # 参数检查：至少要有一个文件
            if not node_file and not rel_file:
                raise ValueError("至少需要提供 node_file 或 rel_file 中的一个")

            # 读取节点数据（如果提供了节点文件）
            nodes = None
            if node_file:
                nodes = self._read_csv_data(node_file)
                print(f"读取到 {len(nodes)} 个节点")

            # 读取关系数据（如果提供了关系文件）
            rels = None
            if rel_file:
                rels = self._read_csv_data(rel_file)
                print(f"读取到 {len(rels)} 个关系")

            with self.driver.session() as session:
                if nodes is not None:
                    self._import_nodes_batch(session, nodes, batch_size)
                if rels is not None:
                    self._import_relationships_batch(session, rels, batch_size)

            if nodes is not None or rels is not None:
                total_nodes = len(nodes) if nodes is not None else 0
                total_rels = len(rels) if rels is not None else 0
                print(f"导入完成: {total_nodes}节点, {total_rels}关系")

        except Exception as e:
            print(f"导入CSV数据时出错: {e}")
            raise

    def import_from_json(self, node_file: str = None, rel_file: str = None, batch_size: int = 1000) -> None:
        """从JSON文件批量导入数据"""
        try:
            # 参数检查：至少要有一个文件
            if not node_file and not rel_file:
                raise ValueError("至少需要提供 node_file 或 rel_file 中的一个")

            # 读取节点数据（如果提供了节点文件）
            nodes_data = None
            if node_file:
                nodes_data = self._read_json_data(node_file)
                print(f"读取到 {len(nodes_data)} 个节点")

            # 读取关系数据（如果提供了关系文件）
            rels_data = None
            if rel_file:
                rels_data = self._read_json_data(rel_file)
                print(f"读取到 {len(rels_data)} 个关系")

            with self.driver.session() as session:
                if nodes_data is not None:
                    self._import_nodes_batch(session, nodes_data, batch_size)
                if rels_data is not None:
                    self._import_relationships_batch(session, rels_data, batch_size)

            if nodes_data is not None or rels_data is not None:
                total_nodes = len(nodes_data) if nodes_data is not None else 0
                total_rels = len(rels_data) if rels_data is not None else 0
                print(f"导入完成: {total_nodes}节点, {total_rels}关系")

        except Exception as e:
            print(f"导入JSON数据时出错: {e}")
            raise

    def import_data(self, node_file: str =  None, rel_file: str = None, format_type: str = "csv", batch_size: int = 1000) -> None:
        """统一的数据导入接口"""
        if format_type.lower() == "json":
            self.import_from_json(node_file, rel_file, batch_size)
        elif format_type.lower() == "csv":
            self.import_from_csv(node_file, rel_file, batch_size)
        else:
            raise ValueError("不支持的文件格式，请使用 'csv' 或 'json'")

    def import_from_dataframe(self, nodes_df: pd.DataFrame = None, rels_df: pd.DataFrame = None, batch_size: int = 1000) -> None:
        """直接从DataFrame导入数据，无需文件读取"""
        try:
            with self.driver.session() as session:
                if nodes_df is not None:
                    self._import_nodes_batch(session, nodes_df, batch_size)
                    print(f"导入完成: {len(nodes_df)}个节点")
                if rels_df is not None:
                    self._import_relationships_batch(session, rels_df, batch_size)
                    print(f"导入完成: {len(rels_df)}个关系")
        except Exception as e:
            print(f"从DataFrame导入数据时出错: {e}")
            raise

    def _import_nodes_batch(self, session, nodes: pd.DataFrame, batch_size: int) -> None:
        """批量导入节点"""
        for i in range(0, len(nodes), batch_size):
            batch = nodes.iloc[i:i + batch_size]
            self._process_node_batch(session, batch)

    def _process_node_batch(self, session, batch: pd.DataFrame) -> None:
        """处理节点批次"""
        for _, row in batch.iterrows(): # pandas库里面DataFrame的一个迭代方法，返回下标和行两个结果，这里忽略下标只考虑行

            node_type = str(row.get('type', 'Concept')).strip()
            node_id = str(row.get('id', '')).strip()
            if not node_id:
                print("发现空ID的节点，跳过")
                continue

            props = {k: v for k, v in row.items()
                     if k not in ['id', 'type'] and pd.notna(v) and v != ''}
            props = {k: str(v) if not isinstance(v, (int, float, bool)) else v
                     for k, v in props.items()}
            # 数据清洗
            node_labels = self._get_node_labels(node_type)
            label_str = ':'.join(node_labels)
            query = f"""
                MERGE (n:{label_str} {{id: $id}})
                SET n += $props 
            """

            try:
                session.run(query, id=node_id, props=props)
            except Exception as e:
                print(f"导入节点 {node_id} 时出错: {e}")

    def _get_node_labels(self, node_type: str) -> List[str]:
        """获取节点标签"""
        type_mapping = {
            'Concept': ['Concept'],
            'Theorem': ['Theorem'],
            'Formula': ['Formula'],
            'Scientist': ['Scientist'],
            'Example': ['Example'],
            'Person': ['Person', 'Entity'],
            'Organization': ['Organization', 'Entity'],
            'Event': ['Event', 'Entity']
        }
        return type_mapping.get(node_type, [node_type, 'Entity'])

    def _import_relationships_batch(self, session, rels: pd.DataFrame, batch_size: int) -> None:
        """批量导入关系"""
        for i in range(0, len(rels), batch_size):
            batch = rels.iloc[i:i + batch_size]
            self._process_relationship_batch(session, batch)

    def _process_relationship_batch(self, session, batch: pd.DataFrame) -> None:
        """处理关系批次"""
        for _, row in batch.iterrows():
            source_id = str(row.get('source', '')).strip()
            target_id = str(row.get('target', '')).strip()
            rel_type = str(row.get('type', 'RELATED')).strip().upper()

            if not source_id or not target_id:
                print("发现空ID的关系，跳过")
                continue

            if not rel_type.replace('_', '').isalnum():
                print(f"无效的关系类型: {rel_type}，使用默认类型")
                rel_type = 'RELATED'

            props = {k: v for k, v in row.items()
                     if k not in ['source', 'target', 'type'] and pd.notna(v) and v != ''}
            props = {k: str(v) if not isinstance(v, (int, float, bool)) else v
                     for k, v in props.items()}

            query = """
                MATCH (a {id: $source_id}), (b {id: $target_id})
                MERGE (a)-[r:%s]->(b)
                SET r += $props
            """ % rel_type

            try:
                session.run(query,
                            source_id=source_id,
                            target_id=target_id,
                            props=props)
            except Exception as e:
                print(f"导入关系 {source_id}->{target_id} 时出错: {e}")

    def semantic_search(self, keyword: str, limit: int = 5) -> pd.DataFrame:
        """语义搜索"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH path=(c)-[r]-(related)
                    WHERE toLower(c.name) CONTAINS toLower($keyword)
                    RETURN 
                        c.name AS concept,
                        type(r) AS relation_type,
                        related.name AS related_to,
                        labels(related) AS target_type
                    LIMIT $limit
                """, keyword=keyword, limit=limit)

                records = [dict(record) for record in result]

                # df = pd.DataFrame(records)
                # print(f"搜索关键词 '{keyword}' 返回 {len(df)} 条结果")
                return records
        except Exception as e:
            print(f"搜索时出错: {e}")
            return pd.DataFrame()

    def advanced_search(self, keyword: str, node_types: Optional[List[str]] = None,
                        rel_types: Optional[List[str]] = None, limit: int = 10) -> pd.DataFrame:
        """高级搜索功能"""
        try:
            with self.driver.session() as session:
                query_parts = [
                    "MATCH (n)-[r]->(m)",
                    "WHERE (toLower(n.name) CONTAINS toLower($keyword) OR toLower(m.name) CONTAINS toLower($keyword))"
                ]
                params = {"keyword": keyword, "limit": limit}

                if node_types:
                    type_conditions = " OR ".join([f"n:{nt}" for nt in node_types])
                    query_parts.append(f"AND ({type_conditions})")

                if rel_types:
                    rel_conditions = " OR ".join([f"type(r) = '{rt}'" for rt in rel_types])
                    query_parts.append(f"AND ({rel_conditions})")

                query_parts.extend([
                    "RETURN",
                    "n.name AS source_name,",
                    "labels(n) AS source_type,",
                    "type(r) AS relation_type,",
                    "m.name AS target_name,",
                    "labels(m) AS target_type",
                    "LIMIT $limit"
                ])

                query = " ".join(query_parts)
                result = session.run(query, **params)

                records = [dict(record) for record in result]
                return pd.DataFrame(records)
        except Exception as e:
            print(f"高级搜索时出错: {e}")
            return pd.DataFrame()

    def get_node_statistics(self) -> Dict[str, int]:
        """获取知识图谱中节点统计信息"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    RETURN labels(n) AS node_type, count(*) AS count
                """)
                stats = {}
                for record in result:
                    labels = record["node_type"]
                    count = record["count"]
                    for label in labels:
                        stats[label] = stats.get(label, 0) + count
                return stats
        except Exception as e:
            print(f"获取节点统计时出错: {e}")
            return {}

    def get_relationship_statistics(self) -> Dict[str, int]:
        """获取知识图谱中关系统计信息"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) AS rel_type, count(*) AS count
                """)
                stats = {record["rel_type"]: record["count"] for record in result}
                return stats
        except Exception as e:
            print(f"获取关系统计时出错: {e}")
            return {}

def fetch_data_from_url(url):
    try:
        # 发送 HTTP GET 请求
        response = requests.get(url , verify = False)

        # 检查请求是否成功
        if response.status_code == 200:
            # 返回文件内容
            return response.text
        else:
            print(f"请求失败，状态码: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")
        return None


# 示例用法
if __name__ == "__main__":
    try:

        kg = Neo4jTeamCollaborator(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="Wangshuxvan@1"
        )

        # 测试CSV节点文件路径
        # node_csv_path = r"E:\Neo4j\neo4j-community-5.26.0-windows\neo4j-community-5.26.0-windows\neo4j-community-5.26.0\import\get_node_csv.csv"

        # 测试JSON文件路径
        #　node_json_path = r"E:\Neo4j\neo4j-community-5.26.0-windows\neo4j-community-5.26.0-windows\neo4j-community-5.26.0\import\get_node_json.json"

        # 测试csv关系文件路径
        rel_csv_path = r"E:\Neo4j\neo4j-community-5.26.0-windows\neo4j-community-5.26.0-windows\neo4j-community-5.26.0\import\get_rel_csv.csv"

        # 测试json关系文件路径
        #　rel_json_path = r"E:\Neo4j\neo4j-community-5.26.0-windows\neo4j-community-5.26.0-windows\neo4j-community-5.26.0\import\get_rel_json.json"

        # print("=== 检查文件存在性 ===")
        # file_exist([node_csv_path, node_json_path, rel_csv_path])


        # 导入数据
        kg.import_data(node_file = None , rel_file = rel_csv_path ,format_type="csv")

        # print("\n=== 导入JSON节点文件 ===")
        # 导入JSON节点（使用相同的关系文件，或者可以创建单独的JSON关系文件）
        # kg.import_data(rel_file = rel_csv_path, format_type="json")


        print("\n=== 执行搜索测试 ===")
        results = kg.semantic_search("爱因斯坦")
        print("搜索的结果:")
        # display(results)

        for line in results:
            print(line['concept'] + line['relation_type'] + line['related_to'])

        # print("\n=== 获取统计信息 ===")
        # node_stats = kg.get_node_statistics()
        # rel_stats = kg.get_relationship_statistics()
        # print("节点统计:", node_stats)
        # print("关系统计:", rel_stats)
        '''
        print("\n=== 测试高级搜索 ===")
        # 搜索新导入的泰勒定理相关节点
        advanced_results = kg.advanced_search("神经" )
        print("搜索的结果:")
        display(advanced_results)
        '''

    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'kg' in locals():
            kg.close()
            print("数据库连接已关闭")
