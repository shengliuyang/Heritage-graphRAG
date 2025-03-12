# neo4j_query.py
from neo4j import GraphDatabase
from langchain_community.llms import Ollama

class Neo4jQuerier:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "Sly654321")
        )
        self.llm = Ollama(model="llama3.1", temperature=0.8)

    def translate_heritage_name(self, question: str) -> str:
        prompt = f"""请从下面的问题中找出包含的中国的世界遗产名称，并将该名称翻译为英文，注意：仅生成英文名称，不要生成其他任何内容。
        例如用户提问：我想了解颐和园的知识点， 回答：Summer Palace
        提示：
# 中国世界遗产名称汇总

截至目前，中国共有 57 处世界遗产，分为文化遗产、自然遗产和混合遗产三大类。以下是详细列表：

---

## 一、文化遗产（39 处）

1. 北京故宫  
2. 天坛  
3. 颐和园  
4. 长城  
5. 秦始皇陵及兵马俑  
6. 莫高窟  
7. 龙门石窟  
8. 云冈石窟  
9. 平遥古城  
10. 苏州古典园林  
11. 福建土楼  
12. 丽江古城  
13. 布达拉宫  
14. 开平碉楼与村落  
15. 泰山  
16. 殷墟  
17. 曲阜孔庙、孔府、孔林  
18. 清东陵  
19. 宏村和西递古村落  
20. 泉州：“宋代海外贸易中心”  
21. 交河故城  
22. 高昌故城  
23. 明清皇家陵寝  
24. 承德避暑山庄及周围寺庙  
25. 武当山  
26. 澳门历史城区  
27. 大足石刻  
28. 北京周口店遗址  
29. 明孝陵  
30. 大运河  
31. 良渚古城遗址  
32. 其他8项（因资料统计略有差异）

---

## 二、自然遗产（14 处）

1. 黄山风景区  
2. 九寨沟风景区  
3. 武陵源风景名胜区  
4. 三江并流  
5. 中国南方喀斯特  
6. 四川大熊猫栖息地  
7. 庐山  
8. 梵净山  
9. 石林风景区  
10. 神农架  
11. 张掖丹霞地貌  
12. 三清山  
13. 长白山  
14. 黄龙风景名胜区

---

## 三、混合遗产（4 处）

1. 都江堰－青城山  
2. 峨眉山－乐山大佛  
3. 武夷山  
4. 红河哈尼梯田
```
        用户问题：{question}
"""
        heritage_en_name = self.llm.invoke(prompt).strip()
        print(f"翻译得到的遗产英文名称：{heritage_en_name}")
        return heritage_en_name

    def format_heritage_info(self, data: dict) -> str:
        """格式化遗产信息，使其更易读"""
        if not data:
            return "未找到相关遗产信息"
        
        formatted_text = []
        
        # 基本属性部分
        if '基本属性' in data:
            props = data['基本属性']
            formatted_text.append("【基本信息】")
            
            # 名称
            if 'name' in props:
                formatted_text.append(f"名称：{props['name']}")
            
            # 建造时间
            if 'construction_time' in props and props['construction_time']:
                formatted_text.append(f"建造时间：{props['construction_time']}")
            
            # 完整性
            if 'integrity' in props and props['integrity']:
                formatted_text.append(f"完整性：{props['integrity']}")
            
            # 真实性
            if 'authenticity' in props and props['authenticity']:
                formatted_text.append(f"真实性：{props['authenticity']}")
            
            # 文化特色
            if 'culture' in props and props['culture']:
                formatted_text.append(f"文化特色：{', '.join(props['culture'])}")
            
            # 保护管理
            if 'protection_and_management' in props and props['protection_and_management']:
                formatted_text.append(f"保护管理：{props['protection_and_management']}")
            
            # 相关链接
            if 'links' in props and props['links']:
                formatted_text.append(f"相关链接：{', '.join(props['links'])}")
        
        # 分类信息
        if '分类' in data and data['分类']:
            formatted_text.append("\n【遗产分类】")
            formatted_text.append(', '.join(data['分类']))
        
        # 世界遗产标准
        if '世界遗产标准' in data and data['世界遗产标准']:
            formatted_text.append("\n【世界遗产标准】")
            formatted_text.append(', '.join(data['世界遗产标准']))
        
        # 相关文化
        if '相关文化' in data and data['相关文化']:
            formatted_text.append("\n【相关文化】")
            formatted_text.append(', '.join(data['相关文化']))
        
        # 所属朝代
        if '所属朝代' in data and data['所属朝代']:
            formatted_text.append("\n【所属朝代】")
            formatted_text.append(', '.join(data['所属朝代']))
        
        return '\n'.join(formatted_text)

    def query_heritage_info(self, question: str) -> str:
        """查询遗产信息并返回格式化的结果"""
        try:
            heritage_en_name = self.translate_heritage_name(question)
            if not heritage_en_name:
                return "未找到相关遗产信息"

            with self.driver.session() as session:
                query = """
                MATCH (h:HeritageSite)
                WHERE toLower(h.name) CONTAINS toLower($name)
                OPTIONAL MATCH (h)-[:HAS_CATEGORY]->(cat:Category)
                OPTIONAL MATCH (h)-[:HAS_CRITERIA]->(cri:Criteria)
                OPTIONAL MATCH (h)-[:HAS_CULTURE]->(cul:Culture)
                OPTIONAL MATCH (h)-[:HAS_DYNASTY]->(dyn:Dynasty)
                OPTIONAL MATCH (h)-[:HAS_LINK]->(l:Link)
                WITH h, 
                     collect(DISTINCT cat.name) AS categories,
                     collect(DISTINCT cri.name) AS criteria,
                     collect(DISTINCT cul.name) AS cultures,
                     collect(DISTINCT dyn.name) AS dynasties,
                     collect(DISTINCT l.links) AS links
                RETURN {
                    properties: properties(h),
                    categories: categories,
                    criteria: criteria,
                    cultures: cultures,
                    dynasties: dynasties,
                    links: links
                } AS result
                LIMIT 1
                """

                result = session.run(query, name=heritage_en_name).single()

                if not result:
                    print("精确查询无结果，尝试模糊匹配...")
                    fuzzy_query = """
                    MATCH (h:HeritageSite)
                    WHERE h.name =~ '(?i).*' + replace($name, ' ', '.*') + '.*'
                    OPTIONAL MATCH (h)-[:HAS_CATEGORY]->(cat:Category)
                    OPTIONAL MATCH (h)-[:HAS_CRITERIA]->(cri:Criteria)
                    OPTIONAL MATCH (h)-[:HAS_CULTURE]->(cul:Culture)
                    OPTIONAL MATCH (h)-[:HAS_DYNASTY]->(dyn:Dynasty)
                    OPTIONAL MATCH (h)-[:HAS_LINK]->(l:Link)
                    WITH h, 
                         collect(DISTINCT cat.name) AS categories,
                         collect(DISTINCT cri.name) AS criteria,
                         collect(DISTINCT cul.name) AS cultures,
                         collect(DISTINCT dyn.name) AS dynasties,
                         collect(DISTINCT l.links) AS links
                    RETURN {
                        properties: properties(h),
                        categories: categories,
                        criteria: criteria,
                        cultures: cultures,
                        dynasties: dynasties,
                        links: links
                    } AS result
                    LIMIT 1
                    """
                    result = session.run(fuzzy_query, name=heritage_en_name).single()

                    if not result:
                        print("模糊查询也未找到匹配的遗产。")
                        return "未找到相关遗产信息"

                data = result['result']

                # 格式化数据
                formatted_data = {
                    "基本属性": data['properties'],
                    "分类": data['categories'],
                    "世界遗产标准": data['criteria'],
                    "相关文化": data['cultures'],
                    "所属朝代": data['dynasties'],
                    "相关链接": data['links']
                }

                # 使用格式化函数处理数据
                formatted_result = self.format_heritage_info(formatted_data)
                print("\n=== 查询结果 ===")
                print(formatted_result)
                print("===============\n")
                
                return formatted_result

        except Exception as e:
            error_msg = f"查询知识图谱时出错: {str(e)}"
            print(error_msg)
            return error_msg

    def close(self):
        self.driver.close()

    def __del__(self):
        self.close()

# 测试main函数
if __name__ == "__main__":
    neo4j_querier = Neo4jQuerier()
    question = "我想了解颐和园的知识点"
    result = neo4j_querier.query_heritage_info(question)
    print(result)
    neo4j_querier.close()
