// Sistema de Histórico Avançado
class ChatHistory {
  constructor() {
    this.conversations = [];
    this.currentConversationId = null;
    this.init();
  }

  init() {
    this.loadConversations();
    this.setupEventListeners();
    
    if (this.conversations.length === 0) {
      this.createNewConversation();
    } else {
      this.loadConversation(this.conversations[0].id);
    }
  }

  loadConversations() {
    const saved = localStorage.getItem('chatHistory');
    if (saved) {
      this.conversations = JSON.parse(saved);
      this.sortConversations();
    }
  }

  saveConversations() {
    localStorage.setItem('chatHistory', JSON.stringify(this.conversations));
  }

  sortConversations() {
    this.conversations.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
  }

  createNewConversation() {
    const newConversation = {
      id: this.generateId(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      title: 'Nova conversa',
      messages: []
    };
    
    this.conversations.unshift(newConversation);
    this.currentConversationId = newConversation.id;
    this.saveConversations();
    this.renderHistory();
    
    return newConversation;
  }

  generateId() {
    return Date.now().toString() + Math.random().toString(36).substr(2, 9);
  }

  addMessage(conversationId, sender, text) {
    const conversation = this.conversations.find(c => c.id === conversationId);
    if (!conversation) return;

    const message = {
      sender,
      text,
      timestamp: new Date().toISOString()
    };
    
    conversation.messages.push(message);
    conversation.updatedAt = new Date().toISOString();
    
    // Atualiza título se for a primeira mensagem do usuário
    if (sender === 'user' && conversation.messages.length === 1) {
      conversation.title = text.length > 50 ? text.substring(0, 50) + '...' : text;
    }
    
    this.saveConversations();
    this.renderHistory();
  }

  loadConversation(conversationId) {
    const conversation = this.conversations.find(c => c.id === conversationId);
    if (!conversation) return;

    this.currentConversationId = conversationId;
    
    // Limpa e recria o chat
    const chatContainer = document.getElementById('chat');
    if (chatContainer) {
      chatContainer.innerHTML = '';
      conversation.messages.forEach(msg => {
        this.appendMessageToChat(msg.sender, msg.text);
      });
    }
    
    this.renderHistory();
  }

  deleteConversation(conversationId) {
    this.conversations = this.conversations.filter(c => c.id !== conversationId);
    this.saveConversations();
    
    if (this.currentConversationId === conversationId) {
      if (this.conversations.length > 0) {
        this.loadConversation(this.conversations[0].id);
      } else {
        this.createNewConversation();
      }
    }
    
    this.renderHistory();
  }

  renderHistory() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;

    // Agrupa conversas por data
    const grouped = this.groupConversationsByDate();
    historyList.innerHTML = '';

    for (const [date, conversations] of Object.entries(grouped)) {
      const dateGroup = document.createElement('div');
      dateGroup.className = 'history-date-group';
      
      const dateTitle = document.createElement('div');
      dateTitle.className = 'history-date-title';
      dateTitle.textContent = this.formatGroupDate(date);
      dateGroup.appendChild(dateTitle);

      conversations.forEach(conv => {
        const lastMessage = conv.messages[conv.messages.length - 1];
        const lastMessageTime = lastMessage ? this.formatTime(lastMessage.timestamp) : '';
        
        const item = document.createElement('div');
        item.className = `conversation-item ${conv.id === this.currentConversationId ? 'active' : ''}`;
        item.innerHTML = `
          <div class="conversation-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke="currentColor" stroke-width="2"/>
            </svg>
          </div>
          <div class="conversation-content">
            <div class="conversation-title">${conv.title}</div>
            <div class="conversation-time">${lastMessageTime}</div>
          </div>
        `;
        
        item.addEventListener('click', () => this.loadConversation(conv.id));
        
        // Adiciona menu de contexto para deletar
        item.addEventListener('contextmenu', (e) => {
          e.preventDefault();
          this.showContextMenu(e, conv.id);
        });
        
        dateGroup.appendChild(item);
      });
      
      historyList.appendChild(dateGroup);
    }
  }

  groupConversationsByDate() {
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    
    return this.conversations.reduce((groups, conv) => {
      const convDate = new Date(conv.updatedAt);
      let dateKey;
      
      if (this.isSameDay(convDate, today)) {
        dateKey = 'Hoje';
      } else if (this.isSameDay(convDate, yesterday)) {
        dateKey = 'Ontem';
      } else if (this.isThisWeek(convDate)) {
        dateKey = 'Esta semana';
      } else if (this.isThisMonth(convDate)) {
        dateKey = 'Este mês';
      } else {
        dateKey = convDate.toLocaleDateString('pt-PT', {
          month: 'long',
          year: 'numeric'
        });
      }
      
      if (!groups[dateKey]) {
        groups[dateKey] = [];
      }
      groups[dateKey].push(conv);
      return groups;
    }, {});
  }

  isSameDay(date1, date2) {
    return (
      date1.getFullYear() === date2.getFullYear() &&
      date1.getMonth() === date2.getMonth() &&
      date1.getDate() === date2.getDate()
    );
  }

  isThisWeek(date) {
    const today = new Date();
    const firstDayOfWeek = new Date(today.setDate(today.getDate() - today.getDay()));
    return date >= firstDayOfWeek;
  }

  isThisMonth(date) {
    const today = new Date();
    return (
      date.getFullYear() === today.getFullYear() &&
      date.getMonth() === today.getMonth()
    );
  }

  formatGroupDate(dateStr) {
    if (['Hoje', 'Ontem', 'Esta semana', 'Este mês'].includes(dateStr)) {
      return dateStr;
    }
    return dateStr.charAt(0).toUpperCase() + dateStr.slice(1);
  }

  formatTime(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleTimeString('pt-PT', {
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  showContextMenu(e, conversationId) {
    e.preventDefault();
    
    // Remove menu existente
    const existingMenu = document.querySelector('.context-menu');
    if (existingMenu) existingMenu.remove();
    
    const menu = document.createElement('div');
    menu.className = 'context-menu';
    menu.style.position = 'absolute';
    menu.style.left = `${e.clientX}px`;
    menu.style.top = `${e.clientY}px`;
    menu.style.backgroundColor = '#fff';
    menu.style.border = '1px solid #e5e5e6';
    menu.style.borderRadius = '6px';
    menu.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
    menu.style.zIndex = '1000';
    
    const deleteOption = document.createElement('div');
    deleteOption.className = 'menu-option';
    deleteOption.textContent = 'Apagar conversa';
    deleteOption.style.padding = '8px 16px';
    deleteOption.style.cursor = 'pointer';
    deleteOption.addEventListener('click', () => {
      this.deleteConversation(conversationId);
      menu.remove();
    });
    
    menu.appendChild(deleteOption);
    document.body.appendChild(menu);
    
    // Fecha o menu ao clicar em qualquer lugar
    const closeMenu = () => {
      menu.remove();
      document.removeEventListener('click', closeMenu);
    };
    
    setTimeout(() => {
      document.addEventListener('click', closeMenu);
    }, 100);
  }

  appendMessageToChat(sender, text) {
    // Implemente sua função existente de appendMessage aqui
    // Isso deve ser adaptado para trabalhar com seu sistema de mensagens atual
  }

  setupEventListeners() {
    // Botão nova conversa
    document.getElementById('newChatButton')?.addEventListener('click', () => {
      this.createNewConversation();
    });
    
    // Botão toggle histórico
    document.getElementById('toggleHistory')?.addEventListener('click', () => {
      document.getElementById('historyPanel').classList.toggle('active');
    });
    
    // Pesquisa no histórico
    document.getElementById('historySearch')?.addEventListener('input', (e) => {
      const term = e.target.value.toLowerCase();
      const items = document.querySelectorAll('.conversation-item');
      
      items.forEach(item => {
        const title = item.querySelector('.conversation-title').textContent.toLowerCase();
        item.style.display = title.includes(term) ? 'flex' : 'none';
      });
    });
  }
}

// Inicialize o histórico quando a página carregar
window.addEventListener('DOMContentLoaded', () => {
  const chatHistory = new ChatHistory();
  
  // Exponha a instância para uso global se necessário
  window.chatHistory = chatHistory;
  
  // Modifique sua função sendMessage para usar o histórico
  function sendMessage() {
    const message = input.value.trim();
    if (!message) return;
    
    // Adicione a mensagem ao histórico
    chatHistory.addMessage(chatHistory.currentConversationId, 'user', message);
    
    // ... resto da sua implementação existente
  }
  
  // Modifique sua função appendMessage para usar o histórico
  function appendMessage(sender, text) {
    // ... sua implementação existente
    
    // Adicione ao histórico se for mensagem do bot
    if (sender === 'bot') {
      chatHistory.addMessage(chatHistory.currentConversationId, 'bot', text);
    }
  }
});