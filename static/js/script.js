const { createApp } = Vue;

createApp({
    data() {
        return {
            question: '',
            history: [],
            latestItem: null,
            loading: false,
            statusText: '質問を入力してください。',
            error: '',
            provider: 'auto',
        };
    },
    computed: {
        latestSources() {
            return Array.isArray(this.latestItem?.sources) ? this.latestItem.sources : [];
        },
        providerLabel() {
            const labelMap = {
                auto: 'auto',
                duckduckgo: 'DuckDuckGo',
                serpapi: 'SerpAPI / Google',
                bing: 'Bing Search',
                legacy: 'legacy',
            };
            return labelMap[this.provider] || this.provider || 'auto';
        },
    },
    mounted() {
        this.loadHealth();
        this.loadHistory();
    },
    methods: {
        formatTime(timestamp) {
            if (!timestamp) {
                return '-';
            }

            const parsed = new Date(String(timestamp).replace(' ', 'T'));
            if (Number.isNaN(parsed.getTime())) {
                return timestamp;
            }

            return parsed.toLocaleString('ja-JP', {
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
            });
        },
        async loadHealth() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                this.provider = data.search_provider || 'auto';
            } catch (error) {
                this.provider = 'auto';
            }
        },
        async loadHistory() {
            try {
                const response = await fetch('/api/history');
                const data = await response.json();
                this.history = Array.isArray(data.items) ? data.items : [];
                this.latestItem = this.history[0] || null;
                this.statusText = this.history.length
                    ? '保存済みの履歴を読み込みました。'
                    : '質問を入力してください。';
            } catch (error) {
                this.error = '履歴の読み込みに失敗しました。';
            }
        },
        async clearHistory() {
            if (this.loading) {
                return;
            }

            try {
                await fetch('/api/history/clear', { method: 'POST' });
                this.history = [];
                this.latestItem = null;
                this.statusText = '履歴をクリアしました。';
                this.error = '';
            } catch (error) {
                this.error = '履歴の削除に失敗しました。';
            }
        },
        async ask() {
            const trimmedQuestion = this.question.trim();
            if (!trimmedQuestion || this.loading) {
                return;
            }

            this.loading = true;
            this.error = '';
            this.statusText = '検索して回答を生成しています...';

            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: trimmedQuestion }),
                });
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || '回答の生成に失敗しました。');
                }

                this.latestItem = data.item;
                this.provider = data.provider || this.provider;
                this.history = [data.item, ...this.history];
                this.question = '';
                this.statusText = `回答を生成しました (${data.log_file || 'logs/conversation_history.json'})`;
            } catch (error) {
                this.error = error.message || '回答生成に失敗しました。';
            } finally {
                this.loading = false;
            }
        },
    },
}).mount('#app');