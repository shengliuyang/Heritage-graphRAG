import os
import warnings
from typing import List, Tuple
import jieba

try:
    import jieba.analyse
except ImportError:
    # 如果没有安装 jieba.analyse，使用简单的分词作为备选方案
    def simple_extract_keywords(text: str, topK: int = 5) -> List[str]:
        words = jieba.cut(text)
        # 过滤掉停用词和单字词
        valid_words = [w for w in words if len(w) > 1]
        # 返回前 topK 个词
        return valid_words[:topK]


    jieba.analyse = type('DummyAnalyse', (), {'extract_tags': simple_extract_keywords})()

# 忽略所有警告
warnings.filterwarnings('ignore')

# 忽略特定类型的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 设置环境变量来禁用某些警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from neo4j_query import Neo4jQuerier
from langchain.schema import Document

# 提示模板
TEMPLATE = """
你是一位专业的世界遗产学习规划师，对中国的世界遗产知识了如指掌。你需要以一个知心朋友的身份来设计学习计划。
知识图谱信息：
{neo4j_info}
补充信息：
{context}
请按照以下步骤设计学习计划：
第一步 - 信息整理与分析：
1. 分析知识图谱信息（注意分行和突出重点，这部分内容严格按照知识图谱信息回答）：
   1) 基本信息(主要是建造时间与文化特色)
   2) 遗产分类
   3) 符合哪些世界遗产评选标准
   4) 文化背景和内涵
   5) 所属朝代
   6) 访问链接
2. 分析向量数据库中的补充信息：
   - 相关遗产背景和信息
   - 教学大纲内容
   - 参考文献资料
3. 将信息按照难度和逻辑关系进行分类整理
第二步 - 学习计划设计：
1. 设定学习目标：
   - 核心知识掌握目标
   - 技能培养目标
   - 文化理解目标
   - 实践体验目标
2. 规划学习周期：
   - 总体学习时长
   - 阶段性学习安排
   - 每周学习时间建议
3. 设计学习内容：
   - 基础知识学习
   - 深度主题探究
   - 实地考察建议
   - 延伸阅读推荐
4. 制定复习计划：
   - 阶段性复习重点
   - 知识巩固方法
   - 学习效果评估
回答要求：
1. 开场白要亲切自然，像老朋友一样交谈
2. 回答要有标题，分条回答，逻辑清晰
3. 学习计划必须包含：
   - 整体学习目标
   - 预计学习周期（建议2-4周）
   - 每周具体学习安排
   - 学习资源推荐
   - 复习与巩固方案
   - 趣味性学习建议
4. 内容编排要注意：
   - 循序渐进，由浅入深
   - 理论与实践结合
   - 知识点之间要有联系
   - 适当加入有趣的互动元素
注意事项：
1. 优先使用知识图谱中的核心信息设计学习路径
2. 设计学习进度
3. 保持学术严谨性的同时要有趣味性
4. 适当加入激励性的话语和学习建议
5. 可以根据遗产特点设计特色学习活动
6. 要有标题，有格式设计

历史对话：{chat_history}
用户问题：{question}
请按照上述要求，以朋友的口吻，生成一份生动有趣且专业的学习计划：
"""


class ChatBot:
    def __init__(self, persist_directory: str):
        if not os.path.exists(persist_directory):
            raise ValueError(f"向量数据库目录不存在: {persist_directory}")

        self.neo4j_querier = Neo4jQuerier()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
        self.vectorstore = Chroma(  # 保存vectorstore实例以便后续使用
            collection_name="txt_documents",
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        llm = Ollama(
            model="llama3.1",
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.15,
        )

        # 基础检索器配置
        self.base_retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.5,
                "k": 10,  # 增加初始检索数量以便后续过滤
                "filter": None
            }
        )

        prompt = PromptTemplate(
            input_variables=["neo4j_info", "context", "chat_history", "question"],
            template=TEMPLATE
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.create_enhanced_retriever(),  # 使用增强的检索器
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )

    def filter_by_metadata(self, docs: List[Document], query: str) -> List[Document]:
        """基于元数据对检索结果进行过滤和排序，并去重"""
        if not docs:
            return []

        try:
            # 提取查询关键词
            query_keywords = set(jieba.analyse.extract_tags(query, topK=20))
            scored_docs = []
            seen_contents = set()  # 用于去重

            # 对每个文档计算相关性分数
            for doc in docs:
                # 检查内容是否重复
                content_hash = hash(doc.page_content)
                if content_hash in seen_contents:
                    continue
                seen_contents.add(content_hash)

                score = 0
                metadata = doc.metadata

                # 1. 关键词匹配度 (增加权重)
                keywords_str = metadata.get('keywords', '')
                doc_keywords = set(keywords_str.split(',')) if keywords_str else set()
                keyword_overlap = len(query_keywords & doc_keywords)
                score += keyword_overlap * 3  # 增加关键词匹配的权重

                # 2. 摘要相关性 (增加权重)
                summary = metadata.get('summary', '')
                summary_matches = sum(1 for keyword in query_keywords if keyword in summary)
                score += summary_matches * 2  # 增加摘要匹配的权重

                # 3. 文档来源权重
                source = metadata.get('source', '').lower()
                if source == 'neo4j':
                    score += 0.5  # 降低 neo4j 数据的权重
                elif 'txt' in source or 'pdf' in source:
                    score += 2.0  # 增加文本数据的权重

                # 4. 内容长度适中性
                content_length = metadata.get('content_length', 0)
                if 300 <= content_length <= 1000:  # 优先选择中等长度的内容
                    score += 1.5
                elif content_length > 1000:
                    score += 1.0

                scored_docs.append((doc, score))

            # 按分数排序并返回文档
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # 只返回分数大于0的文档，最多返回8个
            filtered_docs = [doc for doc, score in scored_docs if score > 0][:5]

            return filtered_docs if filtered_docs else docs[:5]

        except Exception as e:
            print(f"关键词提取过程出现错误: {e}")
            return docs[:3]

    def create_enhanced_retriever(self):
        """
        创建带有元数据过滤的增强检索器
        """

        def enhanced_get_relevant_documents(query: str) -> List[Document]:
            # 使用基础检索器获取初始结果
            docs = self.base_retriever.get_relevant_documents(query)
            # 应用元数据过滤
            filtered_docs = self.filter_by_metadata(docs, query)
            return filtered_docs

        # 创建自定义检索器
        from langchain.schema import BaseRetriever

        class EnhancedRetriever(BaseRetriever):
            def get_relevant_documents(self, query: str) -> List[Document]:
                return enhanced_get_relevant_documents(query)

            async def aget_relevant_documents(self, query: str) -> List[Document]:
                raise NotImplementedError("Async retrieval not implemented")

        return EnhancedRetriever()

    def get_response(self, question: str) -> Tuple[str, List[str]]:
        try:
            # 获取Neo4j信息
            neo4j_info = self.neo4j_querier.query_heritage_info(question)
            if len(neo4j_info) > 50000:
                neo4j_info = neo4j_info[:50000] + "\n...[信息过长，已截断]"

            # 使用增强的检索器获取文档
            retrieved_docs = self.chain.retriever.get_relevant_documents(question)
            
            # 去重处理
            seen_contents = set()
            unique_docs = []
            for doc in retrieved_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_docs.append(doc)
            
            # 使用去重后的文档
            context = "\n".join(doc.page_content for doc in unique_docs)
            chat_history = self.memory.load_memory_variables({}).get("chat_history", [])

            result = self.chain.combine_docs_chain.invoke({
                "input_documents": unique_docs,  # 使用去重后的文档
                "question": question,
                "chat_history": chat_history,
                "context": context,
                "neo4j_info": neo4j_info
            })

            # 获取更详细的来源信息（同样使用去重后的文档）
            sources = []
            for doc in unique_docs:
                source = doc.metadata.get('source', '未知来源')
                doc_type = doc.metadata.get('document_type', '')
                if doc_type == 'pdf':
                    page = doc.metadata.get('page_number', '')
                    source = f"{source} (PDF第{page}页)"
                sources.append(source)

            return result, list(set(sources))

        except Exception as e:
            print(f"获取回答时出现异常: {e}")
            import traceback
            traceback.print_exc()
            return "抱歉，处理您的问题时发生了一些错误，请稍后重试！", []

    def clear_memory(self):
        self.memory.clear()

    def __del__(self):
        self.neo4j_querier.driver.close()


# 用于独立测试
if __name__ == "__main__":
    persist_directory = "/mnt/f/chroma_data"
    chatbot = ChatBot(persist_directory)

    print("欢迎使用世界遗产智能问答系统！")
    while True:
        user_input = input("请输入您的问题(输入'exit'退出)：")
        if user_input.lower() == 'exit':
            break

        response, sources = chatbot.get_response(user_input)
        print("\nAI助手的回答：")
        print(response)

        if sources:
            print("\n相关来源：")
            for src in sources:
                print(f"- {src}")
        print("=" * 50)
