let messageHistory = [];

function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    // 添加用户消息
    addMessage('user', message);
    input.value = '';
    
    // 发送请求到服务器
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            addMessage('bot', `错误: ${data.error}`);
        } else {
            addMessage('bot', data.answer);
            updateSources(data.sources);
        }
    })
    .catch(error => {
        addMessage('bot', `发生错误: ${error}`);
    });
}

function addMessage(type, content) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    if (type === 'bot') {
        const tempDiv = document.createElement('div');
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false
        });
        
        // 处理接收到的内容
        let cleanedContent = '';
        try {
            // 如果内容是对象，获取 answer 字段
            if (typeof content === 'object' && content.answer) {
                cleanedContent = content.answer;
            } else {
                cleanedContent = content;
            }
            
            // 处理换行和格式
            cleanedContent = cleanedContent
                .replace(/\n\n/g, '\n')  // 删除多余的空行
                .replace(/！\n/g, '！\n\n')  // 在感叹号后添加空行
                .replace(/。\n/g, '。\n\n')  // 在句号后添加空行
                .trim();
            
        } catch (e) {
            console.error('处理消息内容时出错:', e);
            cleanedContent = content;
        }
        
        tempDiv.innerHTML = marked.parse(cleanedContent);
        
        // 添加加载动画初始效果
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        
        // 先添加空的消息框
        messagesContainer.appendChild(messageDiv);
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0)';
        
        // 开始打字机效果
        let html = tempDiv.innerHTML;
        messageDiv.innerHTML = html;
        
        // 只对段落文本应用打字机效果
        const paragraphs = messageDiv.querySelectorAll('p, li');
        paragraphs.forEach(p => {
            const originalText = p.textContent;
            p.textContent = '';
            let index = 0;
            
            function typeWriter() {
                if (index < originalText.length) {
                    const chunkSize = 2;
                    const chunk = originalText.substr(index, chunkSize);
                    p.textContent += chunk;
                    index += chunkSize;
                    setTimeout(typeWriter, Math.random() * 20 + 15);
                }
            }
            
            setTimeout(typeWriter, Math.random() * 300);
        });
        
        // 如果有来源信息，更新来源抽屉
        if (content.sources) {
            updateSources(content.sources);
        }
        
    } else {
        // 用户消息自动调整大小
        messageDiv.textContent = content;
        messageDiv.style.maxWidth = content.length > 50 ? '85%' : 'fit-content';
        messagesContainer.appendChild(messageDiv);
    }
    
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    messageHistory.push({ type, content });
}

function updateSources(sources) {
    const sourcesContent = document.getElementById('sources-content');
    sourcesContent.innerHTML = sources.map(source => 
        `<div class="source-item">📚 ${source}</div>`
    ).join('');
}

function clearHistory() {
    fetch('/clear_history', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            document.getElementById('chat-messages').innerHTML = '';
            messageHistory = [];
            document.getElementById('sources-content').innerHTML = '';
        } else {
            alert('清除历史记录失败: ' + data.error);
        }
    })
    .catch(error => {
        alert('清除历史记录时发生错误: ' + error);
    });
}

function toggleSources() {
    const content = document.getElementById('sources-content');
    const arrow = document.querySelector('.arrow');
    content.classList.toggle('show');
    arrow.classList.toggle('up');
}

// 添加自动调整文本框高度的功能
function autoResizeTextarea() {
    const textarea = document.getElementById('user-input');
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

// 监听输入事件
document.getElementById('user-input').addEventListener('input', autoResizeTextarea);

// 修改回车发送功能，支持Shift+Enter换行
document.getElementById('user-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// 添加自动滚动功能
function autoScroll() {
    const messagesContainer = document.getElementById('chat-messages');
    if (messagesContainer.scrollHeight - messagesContainer.scrollTop <= messagesContainer.clientHeight + 100) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// 每100ms检查一次是否需要滚动
setInterval(autoScroll, 100); 