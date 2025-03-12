import os
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 优化后的模板
TEMPLATE = """你是一位专业的世界遗产知识讲解员，对中国的世界遗产知识了如指掌。你需要以一个友好的学习伙伴的身份来分享知识。
知识图谱信息：
{neo4j_info}

请严格按照以下步骤处理用户的世界遗产相关问题：

第一步 - 信息筛选与整理：
1. 仔细分析检索到的每条信息，判断其与用户问题的相关性
2. 对于相关性不高的信息，请直接忽略不要使用
3. 将筛选后的信息按来源分类：
   - 来自Neo4j知识图谱的核心信息（以【Neo4j知识图谱信息】开头的部分）
   - 来自向量数据库的补充信息

第二步 - 信息分析与组织：
1. 优先分析Neo4j知识图谱中的信息：
   - Heritage Site Properties：遗产的基本属性
   - Categories：遗产的分类
   - Criteria：符合的世界遗产标准
   - Cultures：相关的文化背景
   - Dynasties：所属朝代
   - Links：相关链接
2. 重点分析该遗产符合的世界遗产标准，以及入选世界遗产名录的原因
3. 分析其他来源的补充信息，提取有价值的深度内容
4. 确保所使用的每条信息都与问题直接相关
5. 将信息整合成结构化的知识点

回答要求：
1. 以友好、生动的语气进行讲解，就像在和朋友分享有趣的发现
2. 采用清晰的层次结构，使用标题和分点来组织内容
3. 内容必须包含以下方面：
   - 基本概况（位置、类型、规模等）
   - 符合的世界遗产标准（详细解释为什么符合这些标准）
   - 入选世界遗产名录的重要价值
   - 历史渊源
   - 文化/自然价值
   - 独特特征
   - 有趣的故事或小知识
4. 适当融入自己的知识，但要确保准确性
5. 结尾可以加入一个有趣的互动建议或参观提示

注意事项：
1. 如果发现信息相互矛盾，优先使用Neo4j知识图谱中的信息
2. 要确保答案具有层次结构，并且内容丰富
3. 对于可能过时或不准确的信息要谨慎使用
4. 如果某个方面信息不足，可以根据你自己的已知信息补充答案

已知信息：
{context}

历史对话：
{chat_history}

用户问题：{question}

请按照上述要求，用生动活泼的语气，用中文回答："""


class ChatBot:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        if not os.path.exists(persist_directory):
            raise ValueError(f"向量数据库目录不存在: {persist_directory}")
        self.chain = self._create_chain()
        self.memory = None  # 添加内存属性

    def _create_chain(self) -> ConversationalRetrievalChain:
        # 初始化 embeddings
        embeddings = OllamaEmbeddings(model="llama3.1")

        # 初始化向量数据库
        vectorstore = Chroma(
            collection_name="txt_documents",
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )

        # 检查向量数据库是否为空
        if vectorstore._collection.count() == 0:
            raise ValueError("向量数据库为空，请先添加文档")

        # 配置 LLM
        llm = Ollama(
            model="llama3.1",
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.8,
        )

        # 配置 memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        self.memory = memory  # 保存内存引用

        # 创建自定义提示模板
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=TEMPLATE
        )
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 8,
                "lambda_mult": 0.8,
            }
        )

        # 创建对话链
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )

        print(f"向量数据库中的文档数量: {vectorstore._collection.count()}")


        test_docs = retriever.get_relevant_documents("测试查询")
        print(f"测试检索结果数量: {len(test_docs)}")

        return chain

    def get_response(self, question: str) -> Tuple[str, List[str]]:
        """
        获取对问题的回答
        返回: (回答内容, 相关文档来源列表)
        """
        try:
            result = self.chain({"question": question})

            # 提取源文档信息
            sources = []
            if result.get("source_documents"):
                sources = [doc.metadata.get("source", "未知来源")
                           for doc in result["source_documents"]]
                sources = list(set(sources))  # 去重
                print(f"找到的相关文档数量: {len(sources)}")  # 添加调试信息
            else:
                print("警告：没有找到相关文档")

            return result["answer"], sources

        except Exception as e:
            print(f"获取回答时出错: {str(e)}")
            raise

    def clear_memory(self):
        """清除对话历史记录"""
        if self.memory:
            self.memory.clear()

# ... 其他代码 ...
