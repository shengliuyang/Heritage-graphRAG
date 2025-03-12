import streamlit as st
from chat_process_2 import ChatBot
import os


def initialize_session_state():
    """初始化 Session State"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        # persist_directory = "mnt/f/chroma_data"
        persist_directory = "./chroma_data"
        st.session_state.chatbot = ChatBot(persist_directory)


def clear_chat_history():
    """清除聊天历史记录"""
    st.session_state.messages = []
    if "chatbot" in st.session_state:
        st.session_state.chatbot.clear_memory()


def display_messages():
    """显示对话历史"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("参考来源"):
                    for source in message["sources"]:
                        st.write(f"- {source}")


def main():
    st.set_page_config(
        page_title="AI课堂小助手",
        page_icon="🤖",
        layout="wide"
    )

    st.header("🤖 AI 智能助手")

    # 添加清除按钮
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("清除历史记录", type="primary"):
            clear_chat_history()
            st.rerun()

    with col1:
        st.markdown("""
        欢迎使用 AI课堂小助手！我可以帮您回答各种问题。
        - 基于已有知识库进行回答
        - 支持多轮对话
        - 可查看参考来源
        """)

    initialize_session_state()

    # 显示对话历史
    display_messages()

    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 获取 AI 响应
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response, sources = st.session_state.chatbot.get_response(prompt)
                st.markdown(response)
                if sources:
                    with st.expander("参考来源"):
                        for source in sources:
                            st.write(f"- {source}")

        # 保存助手回复
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })


if __name__ == "__main__":
    main()
