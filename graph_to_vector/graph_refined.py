import os
import argparse
from neo4j import GraphDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm
import jieba
import jieba.analyse
import re
from typing import List

# 如果你想使用不同的模型，可以修改这里
OLLAMA_MODEL_NAME = "kingzeus/llama-3-chinese-8b-instruct-v3:latest"

def fetch_graph_data(uri: str, user: str, password: str) -> list:
    """
    从 Neo4j 数据库中抓取所有节点及其关系，并将数据转换为简要文本形式。
    返回一个包含文本字符串的列表，每个字符串可以被视为一个知识片段。
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    text_data_list = []

    with driver.session() as session:
        # 1) 获取所有节点的信息
        #    可以根据业务需求灵活修改查询。如只获取某些标签的节点或过滤一些属性。
        query_nodes = """
        MATCH (n)
        RETURN n
        """
        node_records = session.run(query_nodes)
        for record in node_records:
            node = record["n"]
            # node.id 只有在较新版本驱动中才可用，也可使用 element_id()
            # labels = list(node.labels)
            labels = list(node.labels)
            properties = dict(node)
            # 构造一个描述该节点的文本
            node_text = f"[Node] Labels: {labels}, Properties: {properties}"
            text_data_list.append(node_text)

        # 2) 获取所有节点之间的关系信息
        #    同理，可根据实际需求修改查询。
        query_rels = """
        MATCH (a)-[r]->(b)
        RETURN a, r, b
        """
        rel_records = session.run(query_rels)
        for record in rel_records:
            a = record["a"]
            r = record["r"]
            b = record["b"]
            start_labels = list(a.labels)
            end_labels = list(b.labels)
            rel_type = r.type
            rel_props = dict(r)

            # 拼接关系描述文本
            rel_text = (
                f"[Relation] StartNode: Labels={start_labels}, Properties={dict(a)} "
                f"--> RelType: {rel_type}, RelProperties={rel_props} "
                f"--> EndNode: Labels={end_labels}, Properties={dict(b)}"
            )
            text_data_list.append(rel_text)

    driver.close()
    return text_data_list

def extract_keywords(text: str, topK: int = 5) -> List[str]:
    """
    从文本中提取关键词
    """
    # 使用 jieba TF-IDF 提取关键词
    keywords = jieba.analyse.extract_tags(text, topK=topK)
    return keywords

def get_text_summary(text: str, max_length: int = 100) -> str:
    """
    生成文本摘要（简单实现，仅取开头部分作为摘要）
    """
    # 清理文本
    text = re.sub(r'\s+', ' ', text).strip()
    # 取第一句话或前max_length个字符作为摘要
    sentences = re.split(r'([。！？])', text)
    if len(sentences) > 1:
        summary = sentences[0] + sentences[1]  # 包含标点符号
    else:
        summary = text[:max_length]
    return summary.strip()

def main():
    parser = argparse.ArgumentParser(description="从Neo4j知识图谱拉取数据并加入Chroma向量数据库")
    parser.add_argument('--persist_directory', required=True, help="Chroma向量数据库的持久化目录")
    parser.add_argument('--neo4j_uri', default="bolt://localhost:7687", help="Neo4j数据库连接URI，默认为bolt://localhost:7687")
    parser.add_argument('--neo4j_user', default="neo4j", help="Neo4j数据库用户名，默认为neo4j")
    parser.add_argument('--neo4j_password', default="Sly654321", help="Neo4j数据库密码，默认为Sly654321")

    args = parser.parse_args()
    persist_directory = args.persist_directory
    neo4j_uri = args.neo4j_uri
    neo4j_user = args.neo4j_user
    neo4j_password = args.neo4j_password

    # 1) 从 Neo4j 获取知识图谱数据（节点和关系）
    print("Fetching data from Neo4j ...")
    graph_data_list = fetch_graph_data(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    print(f"Total pieces of data fetched: {len(graph_data_list)}")

    if not graph_data_list:
        print("No data found in Neo4j graph. Exiting.")
        return

    # 2) 初始化向量数据库 & Embedding 模型
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL_NAME)
    vectorstore = Chroma(
        collection_name="txt_documents",  # 可以与原有的 collection_name 相同，也可自定义
        persist_directory=persist_directory,
        embedding_function=embeddings,
        # 如果需要与已有数据库设置保持一致，可以在这里继续传递相同的 collection_metadata
        collection_metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 200,
            "hnsw:M": 64,
        },
    )

    # 3) 将知识图谱文本分割并加入向量数据库
    #    为与 `vector.py` 脚本一致，可以使用相同的文本分割方式。
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n", " ", ""],
        length_function=len,
    )

    # 将所有知识图谱文本转换为 Document 对象
    docs = []
    for idx, text_item in enumerate(graph_data_list):
        # 提取关键词和生成摘要
        keywords = extract_keywords(text_item)
        summary = get_text_summary(text_item)
        
        docs.append(Document(
            page_content=text_item,
            metadata={
                "source": "neo4j",
                "chunk_type": "graph_data",
                "record_index": idx,
                "content_length": len(text_item),
                "keywords": ','.join(keywords),  # 使用逗号分隔的字符串存储关键词
                "summary": summary,
            }
        ))

    # 执行分割
    chunked_docs = []
    for doc in docs:
        new_docs = text_splitter.split_documents([doc])
        chunked_docs.extend(new_docs)

    # 分批次插入到向量数据库
    batch_size = 30
    print("Adding graph data to vectorstore ...")
    for i in tqdm(range(0, len(chunked_docs), batch_size), desc="Indexing"):
        batch = chunked_docs[i:i+batch_size]
        vectorstore.add_documents(batch)

    print(f"Graph data indexing complete. Persist directory: {persist_directory}")

if __name__ == "__main__":
    main()
