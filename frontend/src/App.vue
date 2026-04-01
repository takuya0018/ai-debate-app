<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'

const topic = ref('')
const rounds = ref(2)
const messages = ref([])
const latestSources = ref([])
const loading = ref(false)
const statusText = ref('会話を開始してください。')
const error = ref('')
const provider = ref('auto')
const geminiConfigured = ref(false)
const chatWindowRef = ref(null)
const showTyping = ref(false)
const typingRole = ref('chatgpt')
const isComposing = ref(false)
const showSearchMessages = ref(true)

const isCompletedReviewMessage = (message) => {
  if (message?.role !== 'gemini') return false
  const content = String(message?.content || '')
  return content.includes('【COMPLETED】')
}

const shouldShowMessage = (message) => {
  if (isCompletedReviewMessage(message)) return false
  if (message?.role === 'search' && !showSearchMessages.value) return false
  if (message?.speaker === 'ChatGPT (再検討メモ)') return false
  return true
}

const providerLabel = computed(() => {
  const labelMap = {
    auto: 'auto',
    duckduckgo: 'DuckDuckGo',
    serpapi: 'SerpAPI / Google',
    bing: 'Bing Search',
    'open-meteo': 'Open-Meteo',
    legacy: 'legacy'
  }

  return labelMap[provider.value] || provider.value || 'auto'
})

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const formatTime = (timestamp) => {
  if (!timestamp) return '-'

  const parsed = new Date(String(timestamp).replace(' ', 'T'))
  if (Number.isNaN(parsed.getTime())) {
    return timestamp
  }

  return parsed.toLocaleTimeString('ja-JP', {
    hour: '2-digit',
    minute: '2-digit'
  })
}

const speakerLabel = (message) => {
  if (message?.speaker) return message.speaker
  if (message?.role === 'user') return 'You'
  if (message?.role === 'gemini') return 'Gemini'
  if (message?.role === 'search') return 'Search API'
  return 'ChatGPT'
}

const avatarLabel = (role) => {
  if (role === 'chatgpt') return '🤖'
  if (role === 'gemini') return '✨'
  if (role === 'search') return '🔎'
  return '🙂'
}

const phaseLabel = (message) => {
  if (typeof message?.round !== 'number') return ''

  if (message.role === 'chatgpt') {
    return message.round === 0
      ? '初稿'
      : `再検討 R${message.round}`
  }

  if (message.role === 'gemini') {
    return `レビュー R${message.round}`
  }

  if (message.role === 'search') {
    return `検索 R${message.round}`
  }

  return `R${message.round}`
}

const updateLatestSources = (items) => {
  const latestWithSources = [...items].reverse().find(
    (item) => Array.isArray(item?.sources) && item.sources.length > 0
  )
  latestSources.value = latestWithSources?.sources ?? []
}

const scrollToLatest = async () => {
  await nextTick()
  if (chatWindowRef.value) {
    chatWindowRef.value.scrollTop = chatWindowRef.value.scrollHeight
  }
}

const handleTextareaKeydown = (event) => {
  if (event.isComposing || isComposing.value || event.keyCode === 229) {
    return
  }

  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    runDebate()
  }
}

const loadHealth = async () => {
  try {
    const response = await fetch('/api/health')
    const data = await response.json()
    provider.value = data.search_provider || 'auto'
    geminiConfigured.value = Boolean(data.gemini_configured)
  } catch {
    provider.value = 'auto'
    geminiConfigured.value = false
  }
}

const loadHistory = async () => {
  try {
    const response = await fetch('/api/history')
    const data = await response.json()
    messages.value = Array.isArray(data.messages) ? data.messages : []
    updateLatestSources(messages.value)
    statusText.value = messages.value.length
      ? '保存済みの会話を読み込みました。'
      : '会話を開始してください。'
    await scrollToLatest()
  } catch {
    error.value = '履歴の読み込みに失敗しました。'
  }
}

const clearHistory = async () => {
  if (loading.value) return

  try {
    await fetch('/api/history/clear', { method: 'POST' })
    messages.value = []
    latestSources.value = []
    statusText.value = '履歴をクリアしました。'
    error.value = ''
  } catch {
    error.value = '履歴の削除に失敗しました。'
  }
}

const renderSequentialMessages = async (incomingMessages) => {
  for (const message of incomingMessages) {
    if (message?.speaker === 'ChatGPT (再検討メモ)') {
      continue
    }

    typingRole.value = message.role === 'gemini'
      ? 'gemini'
      : message.role === 'search'
        ? 'search'
        : 'chatgpt'

    const roundLabel = typeof message?.round === 'number' && message.round > 0 ? ` ラウンド${message.round}` : ''
    showTyping.value = true
    statusText.value = message.role === 'gemini'
      ? `✨ Gemini がレビューしています${roundLabel}...`
      : message.role === 'search'
        ? '🔍 検索中...'
        : message.round === 0
          ? '🤖 初稿を生成中...'
          : `🔄 改稿を生成中${roundLabel}...`

    await scrollToLatest()
    await wait(message.role === 'gemini' ? 650 : message.role === 'search' ? 500 : 900)

    showTyping.value = false
    messages.value.push(message)
    updateLatestSources(messages.value)
    await scrollToLatest()
    await wait(150)
  }
}

const runDebate = async () => {
  const trimmedTopic = topic.value.trim()
  if (!trimmedTopic || loading.value) return

  const userMessage = {
    role: 'user',
    speaker: 'You',
    content: trimmedTopic,
    timestamp: new Date().toISOString().replace('T', ' ').slice(0, 19)
  }

  messages.value.push(userMessage)
  topic.value = ''
  loading.value = true
  error.value = ''
  statusText.value = '🔍 検索中...'
  await scrollToLatest()

  try {
    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: trimmedTopic,
        rounds: Number(rounds.value) || 2
      })
    })
    const data = await response.json()

    if (!response.ok) {
      throw new Error(data.error || '返答の生成に失敗しました。')
    }

    provider.value = data.provider || provider.value
    updateLatestSources(data.new_messages || [])
    await renderSequentialMessages(data.new_messages || [])
    statusText.value = data.warning
      ? `⚠️ ${data.warning}`
      : '✅ 完成しました。'
  } catch (requestError) {
    error.value = requestError.message || '返答の生成に失敗しました。'
    showTyping.value = false
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  await loadHealth()
  await loadHistory()
})
</script>

<template>
  <div class="container">
    <header class="app-header">
      <div>
        <h1>💬 AI Debate Chat</h1>
        <p class="subtitle">
          ユーザー入力に対して ChatGPT が回答し、Gemini がレビューして、必要なら GPT が改善します。
        </p>
      </div>

      <div class="header-pills">
        <label
          class="pill checkbox-pill"
          :class="{ enabled: showSearchMessages, disabled: !showSearchMessages }"
        >
          <input v-model="showSearchMessages" type="checkbox">
          <span class="switch-track" aria-hidden="true">
            <span class="switch-thumb"></span>
          </span>
          <span class="toggle-label">
            Search API
            <strong class="toggle-state">{{ showSearchMessages ? '表示中' : '非表示' }}</strong>
          </span>
        </label>
        <span class="pill">Search: {{ providerLabel }}</span>
        <span class="pill" :class="{ active: geminiConfigured }">
          Gemini: {{ geminiConfigured ? 'ON' : 'OFF' }}
        </span>
      </div>
    </header>

    <div ref="chatWindowRef" class="chat-window">
      <template v-if="messages.length">
        <div
          v-for="(message, index) in messages"
          v-show="shouldShowMessage(message)"
          :key="`${message.timestamp}-${index}`"
          class="message-row"
          :class="message.role"
        >
          <div v-if="message.role !== 'user'" class="message-avatar" :class="message.role">
            {{ avatarLabel(message.role) }}
          </div>

          <div class="message-group">
            <div class="message-meta">
              <span class="message-name">{{ speakerLabel(message) }}</span>
              <span v-if="phaseLabel(message)" class="message-phase">{{ phaseLabel(message) }}</span>
              <span class="message-time">{{ formatTime(message.timestamp) }}</span>
            </div>

            <div class="bubble" :class="message.role">
              <pre :class="{ 'search-preview': message.role === 'search' }">{{ message.content }}</pre>
            </div>
          </div>
        </div>

        <div v-if="showTyping" class="message-row" :class="typingRole">
          <div class="message-avatar" :class="typingRole">
            {{ avatarLabel(typingRole) }}
          </div>
          <div class="message-group">
            <div class="message-meta">
              <span class="message-name">{{ typingRole === 'gemini' ? 'Gemini' : 'ChatGPT' }}</span>
            </div>
            <div class="bubble" :class="typingRole">
              <div class="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        </div>
      </template>

      <div v-else class="empty-chat">
        <h2>会話を始めましょう</h2>
        <p>例: 「今日の東京の天気は？」や「この仕様を3行で整理して」</p>
      </div>
    </div>

    <div class="composer">
      <label for="topic" class="input-label">メッセージ</label>

      <div class="message-compose-row">
        <textarea
          id="topic"
          v-model="topic"
          placeholder="例: RAGの仕組みを初心者向けに説明して"
          @keydown="handleTextareaKeydown"
          @compositionstart="isComposing = true"
          @compositionend="isComposing = false"
        ></textarea>

        <div class="action-buttons">
          <button class="send-button" :disabled="loading" @click="runDebate">
            {{ loading ? '処理中...' : '💌 送信' }}
          </button>
          <button class="secondary clear-button" :disabled="loading || !messages.length" @click="clearHistory">
            🧹 履歴クリア
          </button>
        </div>
      </div>

      <div class="controls">
        <label for="rounds">評価ラウンド (1-3)</label>
        <input id="rounds" v-model="rounds" type="number" min="1" max="3">
        <div class="input-hint">Enter で送信 / Shift + Enter で改行</div>
      </div>

      <div class="status" :class="{ error: error }">{{ error || statusText }}</div>
    </div>

  </div>
</template>

<style>
:root {
  --bg: #4a4f57;
  --panel: #f5f6f8;
  --user-bubble: #82dc63;
  --chatgpt-bubble: #6a7078;
  --gemini-bubble: #7b7387;
  --text: #2f3640;
  --muted: #6b7280;
  --border: #d9dde3;
  --chat-meta: rgba(249, 250, 251, 0.82);
}

* {
  box-sizing: border-box;
}

body {
  font-family: 'Inter', 'Hiragino Sans', 'Yu Gothic', sans-serif;
  background: linear-gradient(180deg, #6a7078 0%, #555b63 100%);
  margin: 0;
  padding: 18px;
  color: var(--text);
  min-height: 100vh;
}

button,
textarea,
input {
  font: inherit;
}

pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.75;
  background: transparent;
}

.search-preview {
  display: -webkit-box;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  overflow: hidden;
}

.message-phase {
  background: rgba(15, 23, 42, 0.12);
  color: #374151;
  border: 1px solid rgba(15, 23, 42, 0.2);
  border-radius: 999px;
  font-size: 11px;
  line-height: 1;
  padding: 4px 8px;
}
</style>

<style scoped>
.container {
  width: 98%;
  max-width: 1480px;
  margin: 0 auto;
}

.app-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 14px;
}

.app-header h1 {
  margin: 0 0 6px;
}

.subtitle {
  margin: 4px 0;
  color: #e5e7eb;
  max-width: 760px;
}

.header-pills {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-left: auto;
  justify-content: flex-end;
  align-items: center;
}

.pill {
  display: inline-flex;
  align-items: center;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  color: #f8fafc;
  font-size: 13px;
  font-weight: 700;
}

.checkbox-pill {
  gap: 10px;
  cursor: pointer;
  user-select: none;
  border: 1px solid rgba(255, 255, 255, 0.22);
  transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
}

.checkbox-pill.enabled {
  background: rgba(34, 197, 94, 0.26);
  color: #f0fdf4;
  border-color: rgba(134, 239, 172, 0.5);
}

.checkbox-pill.disabled {
  background: rgba(15, 23, 42, 0.38);
  color: #e2e8f0;
  border-color: rgba(148, 163, 184, 0.45);
}

.checkbox-pill:hover {
  transform: translateY(-1px);
}

.checkbox-pill input {
  display: none;
}

.switch-track {
  position: relative;
  width: 40px;
  height: 22px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.28);
  flex-shrink: 0;
}

.switch-thumb {
  position: absolute;
  top: 3px;
  left: 3px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #ffffff;
  transition: transform 0.2s ease;
}

.checkbox-pill.enabled .switch-thumb {
  transform: translateX(18px);
}

.toggle-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.toggle-state {
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.18);
  font-size: 12px;
}

.checkbox-pill.enabled .toggle-state {
  background: #14532d;
  color: #fca5a5;
}

.pill.active {
  background: rgba(134, 239, 172, 0.22);
  color: #dcfce7;
}

.chat-window {
  min-height: 420px;
  max-height: 58vh;
  overflow-y: auto;
  scrollbar-width: none;
  -ms-overflow-style: none;
  background: linear-gradient(180deg, #5a6068 0%, #4b5159 100%);
  border-radius: 28px;
  padding: 18px;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.05);
  margin-bottom: 14px;
}

.chat-window::-webkit-scrollbar {
  display: none;
}

.empty-chat {
  display: grid;
  place-items: center;
  min-height: 260px;
  text-align: center;
  color: #e5e7eb;
}

.empty-chat h2 {
  margin: 0 0 8px;
}

.composer,
.sources-panel {
  background: rgba(245, 246, 248, 0.96);
  padding: 18px 20px;
  border-radius: 24px;
  box-shadow: 0 14px 32px rgba(15, 23, 42, 0.12);
  border: 1px solid rgba(217, 221, 227, 0.9);
}

.composer {
  margin-bottom: 14px;
}

.input-label {
  display: block;
  font-weight: 700;
  margin-bottom: 8px;
}

.message-compose-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 170px;
  gap: 14px;
  margin-bottom: 12px;
}

textarea {
  width: 100%;
  min-height: 70px;
  max-height: 120px;
  padding: 13px 15px;
  border-radius: 18px;
  border: 1px solid #d9e6de;
  resize: vertical;
  background: #fcfffd;
}

.action-buttons {
  display: grid;
  grid-template-rows: 1fr 1fr;
  gap: 12px;
}

.send-button,
.clear-button {
  width: 100%;
  min-height: 56px;
}

.controls {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px 12px;
}

.input-hint {
  color: var(--muted);
  font-size: 13px;
}

input[type='number'] {
  width: 72px;
  padding: 10px;
  border-radius: 10px;
  border: 1px solid var(--border);
}

button {
  border: none;
  background: linear-gradient(135deg, #8fe0af 0%, #5dc488 55%, #43aa73 100%);
  color: white;
  border-radius: 18px;
  padding: 14px 18px;
  font-weight: 700;
  cursor: pointer;
}

button.secondary {
  background: linear-gradient(135deg, #f9fbfc 0%, #edf2f6 100%);
  color: #44515e;
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.status {
  margin-top: 10px;
  min-height: 24px;
  color: #166534;
  font-weight: 700;
}

.status.error {
  color: #b91c1c;
}

.message-row {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin: 12px 0;
}

.message-row.user {
  justify-content: flex-end;
}

.message-group {
  width: min(68%, 820px);
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.message-row.user .message-group {
  align-items: flex-end;
}

.message-avatar {
  width: 46px;
  height: 46px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 21px;
  flex-shrink: 0;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
}

.message-avatar.chatgpt {
  background: linear-gradient(135deg, #ffd56b 0%, #f4a340 100%);
}

.message-avatar.gemini {
  background: linear-gradient(135deg, #c6b7ff 0%, #f0a8d8 100%);
}

.message-avatar.search {
  background: linear-gradient(135deg, #93c5fd 0%, #2563eb 100%);
}

.message-meta {
  padding: 0 8px;
  font-size: 12px;
  color: var(--chat-meta);
  display: flex;
  align-items: center;
  gap: 8px;
}

.message-name {
  font-weight: 700;
}

.bubble {
  position: relative;
  padding: 13px 15px;
  border-radius: 20px;
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.12);
}

.bubble.user {
  background: linear-gradient(135deg, #90e26f 0%, #78d95c 100%);
  color: #223221;
  border-bottom-right-radius: 8px;
}

.bubble.chatgpt {
  background: var(--chatgpt-bubble);
  color: #f8fafc;
  border-bottom-left-radius: 8px;
}

.bubble.gemini {
  background: var(--gemini-bubble);
  color: #fdf7ff;
  border-bottom-left-radius: 8px;
}

.bubble.search {
  background: #415a77;
  color: #eff6ff;
  border-bottom-left-radius: 8px;
}

.typing-indicator {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  min-width: 42px;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #94a3b8;
  animation: blink 1.2s infinite ease-in-out;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes blink {
  0%, 80%, 100% {
    opacity: 0.3;
    transform: translateY(0);
  }
  40% {
    opacity: 1;
    transform: translateY(-2px);
  }
}

.sources-header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.sources-header h2 {
  margin: 0;
  font-size: 18px;
}

.source-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 10px;
}

.source-item {
  padding: 12px 14px;
  border-radius: 14px;
  background: white;
  border: 1px solid rgba(148, 163, 184, 0.2);
}

.source-item a {
  color: #2563eb;
  font-weight: 700;
  text-decoration: none;
}

.source-item p {
  margin: 8px 0 0;
  color: var(--muted);
}

@media (max-width: 900px) {
  .app-header {
    flex-direction: column;
  }

  .message-group {
    width: min(80%, 720px);
  }
}

@media (max-width: 640px) {
  body {
    padding: 10px;
  }

  .message-group {
    width: 88%;
  }

  .message-compose-row {
    grid-template-columns: 1fr;
  }

  .action-buttons,
  .controls {
    align-items: stretch;
    flex-direction: column;
  }

  button,
  .send-button,
  .clear-button {
    width: 100%;
  }
}
</style>
