import json
import os
import re
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
LOG_DIR = BASE_DIR / "logs"
HISTORY_PATH = LOG_DIR / "conversation_history.json"
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"

load_dotenv(BASE_DIR / ".env")

DEFAULT_CONFIG: Dict[str, Any] = {
    "models": {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash"
    },
    "search": {
        "provider": "auto",
        "max_results": 5,
        "timeout": 15
    },
    "settings": {
        "port": 5000,
        "history_limit": 150,
        "max_context_messages": 12,
        "max_rounds": 3,
        "temperature": 0.3,
        "finish_keyword": "【COMPLETED】"
    },
    "prompts": {
        "assistant_system": (
            "あなたは精度重視のAIアシスタントです。ユーザーの意図を正しく把握し、"
            "検索結果や確かな知識を根拠に回答してください。正式名称・一覧・個数を問われた場合は、"
            "誤りや重複がないかを確認し、推測で項目を補わないでください。"
            "根拠が不足する場合はその旨を明示してください。"
        ),
        "audit_instruction": (
            "あなたは品質監査役のGeminiです。ChatGPTの返答を確認し、"
            "事実誤認・漏れ・重複・不正確な表現を具体的に指摘してください。"
            "十分によければ 【COMPLETED】 とだけ返してください。"
        )
    }
}

app = Flask(__name__)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            loaded = json.load(file)
    except (json.JSONDecodeError, OSError):
        loaded = {}

    return deep_merge(DEFAULT_CONFIG, loaded if isinstance(loaded, dict) else {})


config = load_config()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def make_message(
    role: str,
    speaker: str,
    content: str,
    round_number: int | None = None,
    *,
    timestamp: str | None = None,
    sources: List[Dict[str, str]] | None = None,
    provider: str | None = None,
) -> Dict[str, Any]:
    message: Dict[str, Any] = {
        "role": role,
        "speaker": speaker,
        "content": str(content).strip(),
        "round": round_number,
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if sources:
        message["sources"] = sources
    if provider:
        message["provider"] = provider
    return message


def ensure_storage() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text("[]", encoding="utf-8")


def normalize_history(raw_history: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_history, list):
        return []

    normalized: List[Dict[str, Any]] = []

    for item in raw_history:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()

        if role in {"user", "chatgpt", "gemini", "search"} and content:
            normalized.append(make_message(
                role,
                str(item.get("speaker") or ("You" if role == "user" else "ChatGPT" if role == "chatgpt" else "Gemini")),
                content,
                round_number=item.get("round"),
                timestamp=str(item.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                sources=item.get("sources", []),
                provider=item.get("provider"),
            ))
            continue

        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        timestamp = str(item.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        provider = str(item.get("provider", "legacy")).strip() or "legacy"
        sources = item.get("sources", [])

        if question:
            normalized.append(make_message("user", "You", question, timestamp=timestamp))
        if answer:
            normalized.append(make_message(
                "chatgpt",
                "ChatGPT",
                answer,
                timestamp=timestamp,
                sources=sources,
                provider=provider,
            ))

    history_limit = int(config["settings"].get("history_limit", 150))
    return normalized[-history_limit:]


def load_history() -> List[Dict[str, Any]]:
    ensure_storage()
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        data = []
    return normalize_history(data)


def save_history(history: List[Dict[str, Any]]) -> None:
    ensure_storage()
    history_limit = int(config["settings"].get("history_limit", 150))
    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history[-history_limit:], file, ensure_ascii=False, indent=2)


WEATHER_QUERY_KEYWORDS = (
    "天気", "気温", "天候", "予報", "降水", "雨", "晴れ", "雪",
    "weather", "forecast", "temperature"
)

SPECIAL_SEARCH_VARIANTS = {
    "ストレングスファインダー": [
        "StrengthsFinder 34 themes official",
        "CliftonStrengths 34 themes official",
    ],
    "クリフトンストレングス": [
        "CliftonStrengths 34 themes official",
        "Gallup CliftonStrengths 34 themes",
    ],
}

WEATHER_CODE_LABELS = {
    0: "快晴",
    1: "おおむね晴れ",
    2: "一部くもり",
    3: "くもり",
    45: "霧",
    48: "着氷性の霧",
    51: "弱い霧雨",
    53: "霧雨",
    55: "強い霧雨",
    61: "弱い雨",
    63: "雨",
    65: "強い雨",
    71: "弱い雪",
    73: "雪",
    75: "強い雪",
    80: "弱いにわか雨",
    81: "にわか雨",
    82: "激しいにわか雨",
    95: "雷雨",
}


def is_weather_query(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in query for keyword in WEATHER_QUERY_KEYWORDS[:8]) or any(
        keyword in lowered for keyword in WEATHER_QUERY_KEYWORDS[8:]
    )


def weather_code_to_text(code: Any) -> str:
    try:
        numeric_code = int(code)
    except (TypeError, ValueError):
        return "不明"
    return WEATHER_CODE_LABELS.get(numeric_code, f"天気コード {numeric_code}")


def extract_weather_location(query: str) -> str:
    cleaned = re.sub(r"[?？!！。]", " ", query).strip()
    patterns = [
        r"(?:今日|明日|現在|いま|今週|週末)?(?:の)?(.+?)(?:の)?(?:天気|気温|天候|予報)",
        r"(.+?)\s*(?:weather|forecast|temperature)",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue

        candidate = re.sub(
            r"(今日|明日|現在|いま|今週|週末|天気|気温|天候|予報|教えて|知りたい|の|は)",
            " ",
            match.group(1),
        )
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate:
            return candidate

    tokens = re.findall(r"[一-龥ぁ-んァ-ンA-Za-z]+", cleaned)
    stop_words = {"今日", "明日", "現在", "天気", "気温", "予報", "教えて", "知りたい", "今"}
    filtered = [token for token in tokens if token not in stop_words]
    return filtered[0] if filtered else "東京"


def search_weather_with_open_meteo(query: str, timeout: int) -> List[Dict[str, str]]:
    location = extract_weather_location(query) or "東京"

    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "ja", "format": "json"},
        timeout=timeout,
    )
    geo_response.raise_for_status()
    geo_payload = geo_response.json()
    matches = geo_payload.get("results", [])
    if not matches:
        return []

    place = matches[0]
    latitude = place.get("latitude")
    longitude = place.get("longitude")
    timezone = place.get("timezone", "Asia/Tokyo")

    forecast_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": timezone,
            "forecast_days": 1,
        },
        timeout=timeout,
    )
    forecast_response.raise_for_status()
    forecast_payload = forecast_response.json()

    current = forecast_payload.get("current", {})
    daily = forecast_payload.get("daily", {})
    today = (daily.get("time") or [current.get("time", "")[:10]])[0]
    max_temp = (daily.get("temperature_2m_max") or [None])[0]
    min_temp = (daily.get("temperature_2m_min") or [None])[0]
    rain_probability = (daily.get("precipitation_probability_max") or [None])[0]

    display_name = str(place.get("name", location)).strip() or location
    admin1 = str(place.get("admin1", "")).strip()
    if admin1 and admin1 != display_name:
        display_name = f"{display_name} ({admin1})"

    snippet = (
        f"{today} の {display_name} の天気。"
        f"現在 {weather_code_to_text(current.get('weather_code'))}、"
        f"気温 {current.get('temperature_2m')}°C、風速 {current.get('wind_speed_10m')} km/h。"
        f"今日の最高 {max_temp}°C、最低 {min_temp}°C、降水確率 {rain_probability}%。"
    )

    return [{
        "title": f"{display_name} の天気 (Open-Meteo)",
        "url": f"https://open-meteo.com/en/docs?latitude={latitude}&longitude={longitude}",
        "snippet": snippet,
    }]


def build_search_queries(query: str) -> List[str]:
    base = re.sub(r"[?？!！。]+", " ", query).strip()
    variants: List[str] = []

    def add(candidate: str) -> None:
        cleaned = re.sub(r"\s+", " ", candidate).strip()
        if cleaned and cleaned not in variants:
            variants.append(cleaned)

    add(base)

    simplified = re.sub(r"(教えて|ください|とは|ですか|を|の|は)$", "", base).strip()
    add(simplified)

    if any(keyword in base for keyword in ("一覧", "公式", "正式", "34")):
        add(f"{simplified or base} 公式")
        add(f"{simplified or base} 一覧")

    for keyword, keyword_variants in SPECIAL_SEARCH_VARIANTS.items():
        if keyword in base:
            for variant in keyword_variants:
                add(variant)

    return variants[:6]


def search_with_wikipedia(query: str, max_results: int, timeout: int) -> List[Dict[str, str]]:
    endpoints = [
        ("ja", "https://ja.wikipedia.org/w/api.php"),
        ("en", "https://en.wikipedia.org/w/api.php"),
    ]

    results: List[Dict[str, str]] = []
    for language, endpoint in endpoints:
        response = requests.get(
            endpoint,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "utf8": 1,
                "format": "json",
                "srlimit": max_results,
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()

        for item in payload.get("query", {}).get("search", [])[:max_results]:
            title = str(item.get("title", "")).strip()
            if not title:
                continue

            extract_response = requests.get(
                endpoint,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exchars": 900,
                    "titles": title,
                    "format": "json",
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=timeout,
            )
            extract_response.raise_for_status()
            extract_payload = extract_response.json()
            pages = extract_payload.get("query", {}).get("pages", {})
            page_data = next(iter(pages.values()), {}) if pages else {}

            snippet = str(page_data.get("extract", "")).strip()
            if not snippet:
                snippet = unescape(re.sub(r"<[^>]+>", "", str(item.get("snippet", "")))).strip()

            url = f"https://{language}.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ', '_'))}"
            results.append({
                "title": f"{title} (Wikipedia)",
                "url": url,
                "snippet": snippet[:500],
            })

        if results:
            return results[:max_results]

    return []


def flatten_duckduckgo_topics(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    flattened: List[Dict[str, str]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue

        if "Topics" in entry:
            flattened.extend(flatten_duckduckgo_topics(entry.get("Topics", [])))
            continue

        text = str(entry.get("Text", "")).strip()
        url = str(entry.get("FirstURL", "")).strip()
        if text or url:
            flattened.append({
                "title": text[:80] or url,
                "url": url,
                "snippet": text
            })
    return flattened


def search_with_serpapi(query: str, max_results: int, timeout: int) -> List[Dict[str, str]]:
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        return []

    response = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "hl": "ja",
            "gl": "jp",
            "num": max_results
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    results: List[Dict[str, str]] = []
    for item in payload.get("organic_results", [])[:max_results]:
        results.append({
            "title": str(item.get("title", "")).strip(),
            "url": str(item.get("link", "")).strip(),
            "snippet": str(item.get("snippet", "")).strip()
        })
    return [item for item in results if item["title"] or item["url"]]


def search_with_bing(query: str, max_results: int, timeout: int) -> List[Dict[str, str]]:
    api_key = os.getenv("BING_SEARCH_API_KEY", "").strip()
    if not api_key:
        return []

    endpoint = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search").strip()
    response = requests.get(
        endpoint,
        headers={"Ocp-Apim-Subscription-Key": api_key},
        params={
            "q": query,
            "mkt": "ja-JP",
            "count": max_results,
            "textFormat": "Raw"
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    results: List[Dict[str, str]] = []
    for item in payload.get("webPages", {}).get("value", [])[:max_results]:
        results.append({
            "title": str(item.get("name", "")).strip(),
            "url": str(item.get("url", "")).strip(),
            "snippet": str(item.get("snippet", "")).strip()
        })
    return [item for item in results if item["title"] or item["url"]]


def search_with_duckduckgo(query: str, max_results: int, timeout: int) -> List[Dict[str, str]]:
    response = requests.get(
        "https://api.duckduckgo.com/",
        params={
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    results: List[Dict[str, str]] = []
    abstract_text = str(payload.get("AbstractText", "")).strip()
    abstract_url = str(payload.get("AbstractURL", "")).strip()
    heading = str(payload.get("Heading", "")).strip()

    if abstract_text or abstract_url:
        results.append({
            "title": heading or query,
            "url": abstract_url,
            "snippet": abstract_text
        })

    results.extend(flatten_duckduckgo_topics(payload.get("RelatedTopics", [])))
    return [item for item in results if item["title"] or item["url"]][:max_results]


def resolve_search_provider() -> str:
    preferred = str(os.getenv("SEARCH_PROVIDER", config["search"].get("provider", "auto"))).strip().lower()
    if preferred == "serpapi":
        return "serpapi"
    if preferred == "bing":
        return "bing"
    if os.getenv("SERPAPI_API_KEY", "").strip():
        return "serpapi"
    if os.getenv("BING_SEARCH_API_KEY", "").strip():
        return "bing"
    return "duckduckgo"


def search_web(query: str) -> Tuple[str, List[Dict[str, str]]]:
    max_results = int(config["search"].get("max_results", 5))
    timeout = int(config["search"].get("timeout", 15))

    if is_weather_query(query):
        try:
            weather_results = search_weather_with_open_meteo(query, timeout)
            if weather_results:
                return "open-meteo", weather_results
        except Exception:
            pass

    provider = resolve_search_provider()
    query_variants = build_search_queries(query)

    for candidate in query_variants:
        if provider == "serpapi":
            results = search_with_serpapi(candidate, max_results, timeout)
            if results:
                return provider, results
        elif provider == "bing":
            results = search_with_bing(candidate, max_results, timeout)
            if results:
                return provider, results

        ddg_results = search_with_duckduckgo(candidate, max_results, timeout)
        if ddg_results:
            return "duckduckgo", ddg_results

        wiki_results = search_with_wikipedia(candidate, max_results, timeout)
        if wiki_results:
            return "wikipedia", wiki_results

    return provider if provider in {"serpapi", "bing"} else "duckduckgo", []


def format_sources_for_prompt(results: List[Dict[str, str]]) -> str:
    if not results:
        return "検索結果は見つかりませんでした。"

    lines = []
    for index, item in enumerate(results, start=1):
        lines.append(
            f"[{index}] {item['title']}\nURL: {item['url']}\nSnippet: {item['snippet']}"
        )
    return "\n\n".join(lines)


def build_search_message(provider: str, search_results: List[Dict[str, str]]) -> Dict[str, Any]:
    if search_results:
        preview_lines = []
        for index, item in enumerate(search_results[:3], start=1):
            title = item.get("title") or item.get("url") or f"結果 {index}"
            snippet = str(item.get("snippet", "")).strip()
            preview_lines.append(f"{index}. {title}\n{snippet}" if snippet else f"{index}. {title}")

        content = (
            f"{provider or 'search'} で {len(search_results)} 件の関連情報を取得しました。\n\n"
            + "\n\n".join(preview_lines)
        )
    else:
        content = f"{provider or 'search'} で検索しましたが、関連情報を取得できませんでした。"

    return make_message(
        "search",
        "Search API",
        content,
        0,
        sources=search_results,
        provider=provider,
    )


def build_chatgpt_messages(
    history: List[Dict[str, Any]],
    latest_user_input: str,
    search_results: List[Dict[str, str]] | None = None,
    provider: str = "",
) -> List[Dict[str, str]]:
    max_context_messages = int(config["settings"].get("max_context_messages", 12))
    messages = [{
        "role": "system",
        "content": config["prompts"]["assistant_system"],
    }]

    for item in history[-max_context_messages:]:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if not content:
            continue

        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "chatgpt":
            messages.append({"role": "assistant", "content": content})
        elif role == "gemini":
            messages.append({"role": "user", "content": f"Gemini のレビュー:\n{content}"})

    context_note = ""
    if search_results is not None:
        provider_label = provider or "search"
        context_note = (
            f"\n\n参考検索結果 ({provider_label}):\n"
            f"{format_sources_for_prompt(search_results)}"
        )

    messages.append({
        "role": "user",
        "content": f"{latest_user_input}{context_note}",
    })
    return messages


def ask_chatgpt(messages: List[Dict[str, str]], *, temperature: float | None = None) -> str:
    if not client:
        raise RuntimeError("`OPENAI_API_KEY` が未設定です。.env に追加してください。")

    response = client.chat.completions.create(
        model=config["models"]["openai"],
        temperature=float(config["settings"].get("temperature", 0.3) if temperature is None else temperature),
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()


def ask_fallback_reviewer(content: str, topic: str, results: List[Dict[str, str]], provider: str, reason: str) -> Dict[str, Any]:
    try:
        review_text = ask_chatgpt([
            {
                "role": "system",
                "content": (
                    "あなたは建設的なレビュアーです。ChatGPT の回答を評価する際は次の形式を厳守してください:\n"
                    "問題がなければ 【COMPLETED】 とだけ返してください。\n\n"
                    "問題がある場合:\n"
                    "**いいところ:**\n"
                    "- 良い点1\n"
                    "- 良い点2\n\n"
                    "**改善点:**\n"
                    "- 改善すべき点1\n"
                    "- 改善すべき点2\n\n"
                    "**修正方針:**\n"
                    "- 対応1\n"
                    "- 対応2"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"ユーザー要望:\n{topic}\n\n"
                    f"参照情報 ({provider or 'search'}):\n{format_sources_for_prompt(results)}\n\n"
                    f"ChatGPT の回答:\n{content}"
                ),
            },
        ], temperature=0.1)
        return {
            "ok": True,
            "content": f"[Fallback review]\n{review_text}",
            "reason": f"fallback_openai:{reason}",
            "speaker": "AI Reviewer",
        }
    except Exception:
        return {
            "ok": False,
            "content": "Gemini から評価を取得できず、代替レビューも失敗しました。",
            "reason": reason,
            "speaker": "AI Reviewer",
        }


def ask_gemini(content: str, topic: str, results: List[Dict[str, str]], provider: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return ask_fallback_reviewer(content, topic, results, provider, "missing_api_key")

    fallback_models = []
    for model_name in [config["models"].get("gemini"), "gemini-2.5-flash", "gemini-2.0-flash"]:
        if model_name and model_name not in fallback_models:
            fallback_models.append(model_name)

    payload = {
        "contents": [{
            "parts": [{
                "text": (
                    "あなたは建設的なレビュアーです。ChatGPT の回答を評価する際は次の形式を厳守してください:\n"
                    "問題がなければ 【COMPLETED】 とだけ返してください。\n\n"
                    "問題がある場合:\n"
                    "**いいところ:**\n"
                    "- 良い点1\n"
                    "- 良い点2\n\n"
                    "**改善点:**\n"
                    "- 改善すべき点1\n"
                    "- 改善すべき点2\n\n"
                    "**修正方針:**\n"
                    "- 対応1\n"
                    "- 対応2\n\n"
                    f"ユーザー要望:\n{topic}\n\n"
                    f"参照情報 ({provider or 'search'}):\n{format_sources_for_prompt(results)}\n\n"
                    f"ChatGPT の回答:\n{content}"
                )
            }]
        }]
    }

    last_error = ""
    for model_id in fallback_models:
        endpoints = [
            f"https://generativelanguage.googleapis.com/v1/models/{model_id}:generateContent?key={GEMINI_API_KEY}",
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}",
        ]

        for url in endpoints:
            try:
                response = requests.post(url, json=payload, timeout=25)
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "ok": True,
                        "content": result["candidates"][0]["content"]["parts"][0]["text"],
                        "reason": None,
                        "speaker": "Gemini",
                    }

                body_text = response.text or ""
                lowered = body_text.lower()
                if response.status_code == 429 or "resource_exhausted" in lowered or "quota exceeded" in lowered:
                    return ask_fallback_reviewer(content, topic, results, provider, "quota_exceeded")

                last_error = f"Model '{model_id}' / Status {response.status_code}: {body_text}"
            except Exception as exc:
                last_error = f"Model '{model_id}' / {exc}"

    return ask_fallback_reviewer(content, topic, results, provider, last_error or "unknown_error")


def add_summary_section(text: str) -> str:
    """Add a summary section to the final answer if not already present."""
    content = str(text or "").strip()
    if not content:
        return ""

    if "## " in content or "【" in content or "まとめ" in content:
        return content

    lines = content.split("\n")
    if len(lines) < 3:
        return content

    summary = "\n\n## 📋 まとめ\n\n"
    summary += f"**{lines[-1]}**" if lines[-1] else ""
    return content + summary


def frontend_not_ready_response() -> Tuple[str, int, Dict[str, str]]:
    return (
        """
        <!doctype html>
        <html lang='ja'>
          <head>
            <meta charset='UTF-8' />
            <meta name='viewport' content='width=device-width, initial-scale=1.0' />
            <title>Frontend Not Built</title>
            <style>
              body { font-family: Inter, sans-serif; background: #0f172a; color: #e2e8f0; padding: 32px; }
              .card { max-width: 720px; margin: 40px auto; padding: 24px; border-radius: 16px; background: rgba(15,23,42,.92); border: 1px solid rgba(148,163,184,.25); }
              code { color: #93c5fd; }
            </style>
          </head>
          <body>
            <div class='card'>
              <h1>Vue フロントエンドをビルドしてください</h1>
              <p><code>frontend/</code> 側が未ビルドです。次を実行してください。</p>
              <pre><code>cd frontend
npm install
npm run build</code></pre>
            </div>
          </body>
        </html>
        """,
        503,
        {"Content-Type": "text/html; charset=utf-8"},
    )


def split_revision_output(text: str) -> Tuple[str, str]:
    content = str(text or "").strip()
    if not content:
        return "", ""

    marker = "[REVISED_ANSWER]"
    if marker not in content:
        return "", content

    before, after = content.split(marker, 1)
    plan = before.replace("[REVISION_PLAN]", "").strip()
    revised = after.strip()
    return plan, revised


@app.get("/")
def index() -> Any:
    built_index = FRONTEND_DIST / "index.html"
    if not built_index.exists():
        return frontend_not_ready_response()
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.get("/<path:resource>")
def frontend_routes(resource: str) -> Any:
    built_index = FRONTEND_DIST / "index.html"
    target = FRONTEND_DIST / resource

    if target.exists() and target.is_file():
        return send_from_directory(FRONTEND_DIST, resource)
    if built_index.exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return frontend_not_ready_response()


@app.get("/api/health")
def health() -> Any:
    return jsonify({
        "ok": True,
        "search_provider": resolve_search_provider(),
        "openai_configured": bool(OPENAI_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "log_file": "logs/conversation_history.json"
    })


@app.get("/api/history")
def get_history() -> Any:
    history = load_history()
    return jsonify({
        "messages": history,
        "log_file": "logs/conversation_history.json"
    })


@app.post("/api/history/clear")
def clear_history() -> Any:
    save_history([])
    return jsonify({"ok": True})


@app.post("/api/ask")
def ask() -> Any:
    data = request.get_json(silent=True) or {}
    topic = str(data.get("question", data.get("topic", ""))).strip()

    try:
        requested_rounds = int(data.get("rounds", 2))
    except (TypeError, ValueError):
        requested_rounds = 2

    max_rounds = int(config["settings"].get("max_rounds", 3))
    rounds = max(1, min(requested_rounds, max_rounds))

    if not topic:
        return jsonify({"error": "質問を入力してください。"}), 400

    history = load_history()
    user_message = make_message("user", "You", topic)
    history.append(user_message)
    save_history(history)

    new_messages: List[Dict[str, Any]] = []
    warning = None

    try:
        provider, search_results = search_web(topic)
        search_message = build_search_message(provider, search_results)
        history.append(search_message)
        new_messages.append(search_message)
        save_history(history)

        current_answer = ask_chatgpt(build_chatgpt_messages(
            history[:-1],
            (
                f"ユーザーの依頼:\n{topic}\n\n"
                "まずは質問に正面から答えてください。"
                "正式名称・一覧・個数を求められている場合は、誤りや重複がないように確認し、"
                "根拠が弱い項目は推測で追加しないでください。"
            ),
            search_results,
            provider,
        ), temperature=0.2)
    except Exception as exc:
        return jsonify({"error": f"回答生成に失敗しました: {exc}"}), 500

    chatgpt_message = make_message(
        "chatgpt",
        "ChatGPT",
        current_answer,
        0,
        sources=search_results,
        provider=provider,
    )
    history.append(chatgpt_message)
    new_messages.append(chatgpt_message)
    save_history(history)

    for round_index in range(rounds):
        gemini_result = ask_gemini(current_answer, topic, search_results, provider)
        feedback = gemini_result["content"]

        gemini_message = make_message(
            "gemini",
            str(gemini_result.get("speaker", "Gemini")),
            feedback,
            round_index + 1,
        )
        history.append(gemini_message)
        new_messages.append(gemini_message)
        save_history(history)

        if not gemini_result["ok"]:
            warning = feedback
            break

        if config["settings"]["finish_keyword"] in feedback:
            break

        revision_prompt = (
            f"元のユーザー要望:\n{topic}\n\n"
            f"前回のChatGPT回答:\n{current_answer}\n\n"
            f"Gemini の指摘:\n{feedback}\n\n"
            "次のルールで回答を全面修正してください。\n"
            "- Gemini の指摘を1つ残らず反映する\n"
            "- 事実誤認、重複、正式名称のズレを必ず直す\n"
            "- 正式な一覧や個数を求められている場合は、誤りがある項目は削除または修正する\n"
            "- 検索結果で裏づけられない推測は書かない\n"
            "- 出力形式を厳守する\n\n"
            "出力形式:\n"
            "[REVISION_PLAN]\n"
            "- 指摘と対応を1対1で短く列挙\n"
            "- 2-5項目\n\n"
            "[REVISED_ANSWER]\n"
            "(修正後の回答本文のみ)"
        )
        revision_raw = ask_chatgpt(
            build_chatgpt_messages(history, revision_prompt, search_results, provider),
            temperature=0.1,
        )
        revision_plan, revised_answer = split_revision_output(revision_raw)
        current_answer = revised_answer or revision_raw
        revised_message = make_message(
            "chatgpt",
            "ChatGPT",
            current_answer,
            round_index + 1,
            sources=search_results,
            provider=provider,
        )
        history.append(revised_message)
        new_messages.append(revised_message)
        save_history(history)

    final_with_summary = add_summary_section(current_answer)

    return jsonify({
        "ok": True,
        "new_messages": new_messages,
        "final": final_with_summary,
        "provider": provider,
        "sources": search_results,
        "warning": warning,
        "log_file": "logs/conversation_history.json"
    })


if __name__ == "__main__":
    app.run(debug=True, port=int(config["settings"].get("port", 5000)))
