from flask import Flask, render_template, request, jsonify
from chat_process_1_refined import ChatBot
import os

app = Flask(__name__)

# 初始化聊天机器人
persist_directory = "./chroma_data"  # 替换为你的向量数据库路径
chatbot = ChatBot(persist_directory)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': '请输入问题'})
    
    try:
        response, sources = chatbot.get_response(user_message)
        
        # 如果 response 是字典格式，提取 output_text
        if isinstance(response, dict) and 'output_text' in response:
            response = response['output_text']
            
        return jsonify({
            'answer': response,  # 直接返回文本内容
            'sources': sources
        })
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': f'发生错误: {str(e)}'})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        chatbot.clear_memory()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': f'清除历史记录时发生错误: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 