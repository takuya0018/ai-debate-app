import json
import os
import re
import socket
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
        "port": 5001,
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


def find_available_port(preferred_port: int, max_attempts: int = 20) -> int:
    for offset in range(max_attempts):
        candidate_port = preferred_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", candidate_port))
                return candidate_port
            except OSError:
                continue

    raise RuntimeError(
        f"ポート {preferred_port} から {preferred_port + max_attempts - 1} まで使用中のため起動できません。"
    )


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

SEARCH_INTENT_KEYWORDS = (
    "とは", "意味", "教えて", "調べ", "検索", "最新", "ニュース", "公式", "比較",
    "一覧", "方法", "手順", "原因", "エラー", "how", "what", "why", "where", "when"
)

CASUAL_CHAT_NORMALIZED = {
    "おはよう", "こんにちは", "こんばんは", "やあ", "どうも", "ありがとう", "ありがと", "了解",
    "よろしく", "よろしくね", "元気", "おつかれ", "お疲れ", "はじめまして", "ただいま", "おやすみ"
}

SMALL_TALK_REPLIES = {
    "おはよう": "おはようございます。",
    "こんにちは": "こんにちは。",
    "こんばんは": "こんばんは。",
    "ありがとう": "どういたしまして。",
    "ありがと": "どういたしまして。",
    "おやすみ": "おやすみなさい。",
}

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


def should_use_web_search(query: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(query or "")).strip()
    if not cleaned:
        return False

    if is_weather_query(cleaned):
        return True

    normalized = re.sub(r"[\s!?！？。,.、]+", "", cleaned.lower())
    if normalized in CASUAL_CHAT_NORMALIZED:
        return False

    if any(keyword in cleaned.lower() for keyword in SEARCH_INTENT_KEYWORDS):
        return True

    if "?" in cleaned or "？" in cleaned:
        return True

    return False


def build_small_talk_reply(query: str) -> str | None:
    normalized = re.sub(r"[\s!?！？。,.、]+", "", str(query or "").lower())
    if normalized in SMALL_TALK_REPLIES:
        return SMALL_TALK_REPLIES[normalized]
    return None


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


def extract_weather_day_offset(query: str) -> int:
    if "明後日" in query:
        return 2
    if "明日" in query:
        return 1
    return 0


def needs_clothing_advice(query: str) -> bool:
    return any(keyword in query for keyword in ("服", "服装", "着る", "コーデ"))


def build_weather_answer(item: Dict[str, str], query: str) -> str:
    lines = [f"{item.get('day_label', '今日')}（{item.get('date', '')}）の{item.get('location', '東京')}の天気です。"]

    current_temp = item.get("current_temp", "")
    weather_text = item.get("weather", "不明")
    if current_temp:
        lines.append(f"現在の天気は{weather_text}、気温は{current_temp}°Cです。")
    else:
        lines.append(f"天気は{weather_text}です。")

    forecast_parts = []
    if item.get("max_temp"):
        forecast_parts.append(f"最高気温は{item['max_temp']}°C")
    if item.get("min_temp"):
        forecast_parts.append(f"最低気温は{item['min_temp']}°C")
    if item.get("rain_probability"):
        forecast_parts.append(f"降水確率は{item['rain_probability']}%")
    if item.get("wind_speed"):
        forecast_parts.append(f"風速は{item['wind_speed']} km/h")
    if forecast_parts:
        lines.append("、".join(forecast_parts) + "です。")

    if needs_clothing_advice(query):
        advice_parts = []
        try:
            max_temp_value = float(item.get("max_temp", ""))
        except (TypeError, ValueError):
            max_temp_value = None
        try:
            rain_probability_value = int(float(item.get("rain_probability", "")))
        except (TypeError, ValueError):
            rain_probability_value = None

        if max_temp_value is not None:
            if max_temp_value <= 12:
                advice_parts.append("厚手の上着やコートがあると安心です")
            elif max_temp_value <= 18:
                advice_parts.append("ライトアウターやカーディガンがあるとちょうどよいです")
            elif max_temp_value <= 24:
                advice_parts.append("長袖シャツや薄手の羽織りがおすすめです")
            else:
                advice_parts.append("半袖中心で過ごしやすい気温です")

        if rain_probability_value is not None and rain_probability_value >= 50:
            advice_parts.append("雨に備えて折りたたみ傘や撥水の上着があると便利です")

        if advice_parts:
            lines.append("服装の目安としては、" + "、".join(advice_parts) + "。")

    if item.get("url"):
        lines.append(f"詳細: {item['url']}")

    return "\n".join(lines)


def search_weather_with_open_meteo(query: str, timeout: int) -> List[Dict[str, str]]:
    location = extract_weather_location(query) or "東京"
    day_offset = extract_weather_day_offset(query)

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
            "forecast_days": max(2, day_offset + 1),
        },
        timeout=timeout,
    )
    forecast_response.raise_for_status()
    forecast_payload = forecast_response.json()

    current = forecast_payload.get("current", {})
    daily = forecast_payload.get("daily", {})
    daily_dates = daily.get("time") or [current.get("time", "")[:10]]
    selected_index = min(day_offset, max(0, len(daily_dates) - 1))
    target_date = daily_dates[selected_index]
    max_temp = (daily.get("temperature_2m_max") or [None])[selected_index]
    min_temp = (daily.get("temperature_2m_min") or [None])[selected_index]
    rain_probability = (daily.get("precipitation_probability_max") or [None])[selected_index]
    weather_code = (daily.get("weather_code") or [current.get("weather_code")])[selected_index]
    weather_text = weather_code_to_text(current.get("weather_code") if day_offset == 0 else weather_code)
    current_temp = "" if day_offset > 0 else str(current.get("temperature_2m", ""))
    wind_speed = "" if day_offset > 0 else str(current.get("wind_speed_10m", ""))
    day_label = "明後日" if day_offset == 2 else "明日" if day_offset == 1 else "今日"

    display_name = str(place.get("name", location)).strip() or location
    admin1 = str(place.get("admin1", "")).strip()
    if admin1 and admin1 != display_name:
        display_name = f"{display_name} ({admin1})"

    snippet = (
        f"{target_date} の {display_name} の天気。"
        f"{'現在 ' + weather_text + '、気温 ' + current_temp + '°C、風速 ' + wind_speed + ' km/h。' if day_offset == 0 else weather_text + '。'}"
        f"{day_label}の最高 {max_temp}°C、最低 {min_temp}°C、降水確率 {rain_probability}%。"
    )

    return [{
        "title": f"{display_name} の天気 (Open-Meteo)",
        "url": f"https://open-meteo.com/en/docs?latitude={latitude}&longitude={longitude}",
        "snippet": snippet,
        "location": display_name,
        "date": str(target_date),
        "day_label": day_label,
        "weather": weather_text,
        "current_temp": current_temp,
        "max_temp": str(max_temp),
        "min_temp": str(min_temp),
        "rain_probability": str(rain_probability),
        "wind_speed": wind_speed,
    }]


def build_search_queries(query: str) -> List[str]:
    # Keep search terms compact so GET-based search APIs do not exceed URL limits.
    source = re.sub(r"```[\s\S]*?```", " ", query)
    source = re.sub(r"`[^`]*`", " ", source)
    source = re.sub(r"https?://\S+", " ", source)
    source = re.sub(r"[\r\n\t]+", " ", source)

    token_pattern = r"[A-Za-z0-9_+\-\.]{2,}|[ぁ-んァ-ンー一-龥]{2,}"
    tokens = re.findall(token_pattern, source)
    compact = " ".join(tokens[:24]).strip()

    if not compact:
        compact = source.strip()

    max_query_length = int(config.get("search", {}).get("max_query_length", 220))
    base = re.sub(r"[?？!！。]+", " ", compact).strip()[:max_query_length]
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
    trimmed_query = query[:220].strip()
    if not trimmed_query:
        return []

    response = requests.get(
        "https://api.duckduckgo.com/",
        params={
            "q": trimmed_query,
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
            "title": heading or trimmed_query,
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
            try:
                results = search_with_serpapi(candidate, max_results, timeout)
                if results:
                    return provider, results
            except requests.RequestException:
                pass
        elif provider == "bing":
            try:
                results = search_with_bing(candidate, max_results, timeout)
                if results:
                    return provider, results
            except requests.RequestException:
                pass

        try:
            ddg_results = search_with_duckduckgo(candidate, max_results, timeout)
            if ddg_results:
                return "duckduckgo", ddg_results
        except requests.RequestException:
            pass

        try:
            wiki_results = search_with_wikipedia(candidate, max_results, timeout)
            if wiki_results:
                return "wikipedia", wiki_results
        except requests.RequestException:
            pass

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


def ask_chatgpt_review_of_gemini(answer: str, topic: str, results: List[Dict[str, str]], provider: str) -> str:
    return ask_chatgpt([
        {
            "role": "system",
            "content": (
                "あなたは建設的なレビュアーです。Geminiの回答を評価してください。"
                "問題がなければ【COMPLETED】とだけ返してください。"
                "問題がある場合は、良い点・改善点・修正方針を簡潔に示してください。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"ユーザー要望:\n{topic}\n\n"
                f"参照情報 ({provider or 'search'}):\n{format_sources_for_prompt(results)}\n\n"
                f"Gemini の回答:\n{answer}"
            ),
        },
    ], temperature=0.1)


def ask_fallback_reviewer(content: str, topic: str, results: List[Dict[str, str]], provider: str, reason: str) -> Dict[str, Any]:
    try:
        review_text = ask_chatgpt([
            {
                "role": "system",
                "content": (
                    "あなたは建設的なレビュアーです。ChatGPT の回答を評価する際は次の形式を厳守してください:\n"
                    "問題がなければ 【COMPLETED】 とだけ返してください。\n\n"
                    "問題がある場合:\n"
                    "[GEMINI_ISSUES]\n"
                    "- 問題点や改善すべき点を具体的に列挙（2〜5項目）\n\n"
                    "[GEMINI_ANSWER]\n"
                    "あなたとしての最善の回答（参照情報と知識を踏まえた完全な回答本文）"
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


def ask_fallback_initial_answer(topic: str, results: List[Dict[str, str]], provider: str, reason: str) -> Dict[str, Any]:
    try:
        answer_text = ask_chatgpt([
            {
                "role": "system",
                "content": config["prompts"]["assistant_system"],
            },
            {
                "role": "user",
                "content": (
                    f"ユーザーの依頼:\n{topic}\n\n"
                    f"参照情報 ({provider or 'search'}):\n{format_sources_for_prompt(results)}\n\n"
                    "上の参照情報を踏まえて回答してください。"
                ),
            },
        ], temperature=0.2)
        return {
            "ok": True,
            "content": answer_text,
            "reason": f"fallback_openai:{reason}",
            "speaker": "Gemini (fallback)",
        }
    except Exception:
        return {
            "ok": False,
            "content": "Gemini から回答を取得できず、代替回答も失敗しました。",
            "reason": reason,
            "speaker": "Gemini",
        }


def ask_gemini(content: str, topic: str, results: List[Dict[str, str]], provider: str, is_initial: bool = False, prev_gemini_answer: str = "") -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        if is_initial:
            return ask_fallback_initial_answer(topic, results, provider, "missing_api_key")
        return ask_fallback_reviewer(content, topic, results, provider, "missing_api_key")

    fallback_models = []
    for model_name in [config["models"].get("gemini"), "gemini-2.5-flash", "gemini-2.0-flash"]:
        if model_name and model_name not in fallback_models:
            fallback_models.append(model_name)

    if is_initial:
        prompt_text = (
            "あなたは精度重視のAIアシスタントです。ユーザーの意図を正しく把握し、"
            "検索結果や確かな知識を根拠に回答してください。正式名称・一覧・個数を問われた場合は、"
            "誤りや重複がないかを確認し、推測で項目を補わないでください。"
            "根拠が不足する場合はその旨を明示してください。\n\n"
            f"ユーザーの依頼:\n{topic}\n\n"
            f"参照情報 ({provider or 'search'}):\n{format_sources_for_prompt(results)}\n\n"
            "上の参照情報を踏まえて、ユーザーの依頼に回答してください。"
        )
    else:
        prev_note = (
            f"\nあなたの前回の回答（参考・進化の参照）:\n{prev_gemini_answer}\n\n"
            if prev_gemini_answer else ""
        )
        prompt_text = (
            "あなたはGeminiです。ChatGPTの回答をレビューし、以下の形式で回答してください。\n\n"
            "ChatGPTの回答に問題がなければ 【COMPLETED】 とだけ返してください。\n\n"
            "問題がある場合は必ず次の形式を厳守してください:\n\n"
            "[GEMINI_ISSUES]\n"
            "- 問題点や改善すべき点を具体的に列挙（2〜5項目）\n\n"
            "[GEMINI_ANSWER]\n"
            "Geminiとしての最善の回答（参照情報と知識を踏まえた完全な回答本文）\n\n"
            + prev_note
            + f"ユーザー要望:\n{topic}\n\n"
            f"参照情報 ({provider or 'search'}):\n{format_sources_for_prompt(results)}\n\n"
            f"ChatGPT の回答:\n{content}"
        )

    payload = {
        "contents": [{
            "parts": [{
                "text": prompt_text
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
                    if is_initial:
                        return ask_fallback_initial_answer(topic, results, provider, "quota_exceeded")
                    return ask_fallback_reviewer(content, topic, results, provider, "quota_exceeded")

                last_error = f"Model '{model_id}' / Status {response.status_code}: {body_text}"
            except Exception as exc:
                last_error = f"Model '{model_id}' / {exc}"

    if is_initial:
        return ask_fallback_initial_answer(topic, results, provider, last_error or "unknown_error")
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


def split_gemini_output(text: str) -> Tuple[str, str]:
    """Geminiのディベート出力を (issues, gemini_own_answer) に分割する。"""
    content = str(text or "").strip()
    if not content:
        return "", ""

    issues_marker = "[GEMINI_ISSUES]"
    answer_marker = "[GEMINI_ANSWER]"

    if issues_marker not in content or answer_marker not in content:
        # 旧フォーマットまたは単純な批評 → 全文をissuesとして扱う
        return content, ""

    after_issues = content.split(issues_marker, 1)[1]
    if answer_marker in after_issues:
        issues_part, answer_part = after_issues.split(answer_marker, 1)
        return issues_part.strip(), answer_part.strip()

    return after_issues.strip(), ""


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

    use_chatgpt = bool(data.get("use_chatgpt", True))
    use_gemini = bool(data.get("use_gemini", True))

    if not use_chatgpt and not use_gemini:
        return jsonify({"error": "少なくともChatGPTまたはGeminiを有効にしてください。"}), 400

    max_rounds = int(config["settings"].get("max_rounds", 3))
    rounds = max(1, min(requested_rounds, max_rounds)) if use_chatgpt and use_gemini else 1

    if not topic:
        return jsonify({"error": "質問を入力してください。"}), 400

    history = load_history()
    user_message = make_message("user", "You", topic)
    history.append(user_message)
    save_history(history)

    new_messages: List[Dict[str, Any]] = []
    warning = None
    provider = "none"
    search_results: List[Dict[str, str]] = []
    current_answer = ""

    try:
        small_talk_reply = build_small_talk_reply(topic)

        if small_talk_reply:
            current_answer = small_talk_reply
            if use_gemini and not use_chatgpt:
                initial_message = make_message("gemini", "Gemini", current_answer, 0)
            else:
                initial_message = make_message("chatgpt", "ChatGPT", current_answer, 0)
            history.append(initial_message)
            new_messages.append(initial_message)
            save_history(history)
        else:
            should_search = should_use_web_search(topic)

            if should_search:
                provider, search_results = search_web(topic)
                search_message = build_search_message(provider, search_results)
                history.append(search_message)
                new_messages.append(search_message)
                save_history(history)

            if provider == "open-meteo" and search_results:
                current_answer = build_weather_answer(search_results[0], topic)

            elif use_chatgpt and not use_gemini:
                current_answer = ask_chatgpt(build_chatgpt_messages(
                    history[:-1],
                    (
                        f"ユーザーの依頼:\n{topic}\n\n"
                        "まずは質問に正面から答えてください。"
                        "正式名称・一覧・個数を求められている場合は、誤りや重複がないように確認し、"
                        "根拠が弱い項目は推測で追加しないでください。"
                    ),
                    search_results if should_search else None,
                    provider if should_search else "",
                ), temperature=0.2)
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

            elif use_gemini and not use_chatgpt:
                gemini_result = ask_gemini("", topic, search_results, provider, is_initial=True)
                current_answer = gemini_result["content"]
                gemini_message = make_message(
                    "gemini",
                    str(gemini_result.get("speaker", "Gemini")),
                    current_answer,
                    0,
                    sources=search_results,
                    provider=provider,
                )
                history.append(gemini_message)
                new_messages.append(gemini_message)
                save_history(history)

                if OPENAI_API_KEY:
                    gpt_review = ask_chatgpt_review_of_gemini(current_answer, topic, search_results, provider)
                    gpt_review_message = make_message(
                        "chatgpt",
                        "ChatGPT (評価)",
                        gpt_review,
                        1,
                        sources=search_results,
                        provider=provider,
                    )
                    history.append(gpt_review_message)
                    new_messages.append(gpt_review_message)
                    save_history(history)
                else:
                    warning = "ChatGPT評価を有効化できませんでした。OPENAI_API_KEY を設定してください。"

            else:
                current_answer = ask_chatgpt(build_chatgpt_messages(
                    history[:-1],
                    (
                        f"ユーザーの依頼:\n{topic}\n\n"
                        "まずは質問に正面から答えてください。"
                        "正式名称・一覧・個数を求められている場合は、誤りや重複がないように確認し、"
                        "根拠が弱い項目は推測で追加しないでください。"
                    ),
                    search_results if should_search else None,
                    provider if should_search else "",
                ), temperature=0.2)
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

                prev_gemini_answer = ""
                for round_index in range(rounds):
                    gemini_result = ask_gemini(current_answer, topic, search_results, provider, prev_gemini_answer=prev_gemini_answer)
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

                    gemini_issues, gemini_own_answer = split_gemini_output(feedback)
                    prev_gemini_answer = gemini_own_answer or feedback

                    revision_prompt = (
                        f"元のユーザー要望:\n{topic}\n\n"
                        f"あなたの前回の回答:\n{current_answer}\n\n"
                        f"Gemini の指摘:\n{gemini_issues or feedback}\n\n"
                        + (f"Gemini 自身の回答（比較・取り込み対象）:\n{gemini_own_answer}\n\n" if gemini_own_answer else "")
                        + "次のルールで回答を改訂してください。\n"
                        "- Gemini の指摘をすべて反映する\n"
                        "- Gemini の独自回答に優れた観点や表現があれば積極的に取り入れる\n"
                        "- 事実誤認、重複、正式名称のズレを必ず直す\n"
                        "- 検索結果で裏づけられない推測は書かない\n"
                        "- 出力形式を厳守する\n\n"
                        "出力形式:\n"
                        "[REVISION_PLAN]\n"
                        "- Geminiの指摘への対応と、Geminiの回答から取り入れた点を列挙\n"
                        "- 2-5項目\n\n"
                        "[REVISED_ANSWER]\n"
                        "(修正後の回答本文のみ)"
                    )
                    revision_raw = ask_chatgpt(
                        build_chatgpt_messages(
                            history,
                            revision_prompt,
                            search_results if should_search else None,
                            provider if should_search else "",
                        ),
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

    except Exception as exc:
        return jsonify({"error": f"回答生成に失敗しました: {exc}"}), 500

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
    configured_port = int(config["settings"].get("port", 5000))
    existing_selected_port = os.getenv("APP_SELECTED_PORT", "").strip()
    selected_port = int(existing_selected_port) if existing_selected_port else find_available_port(configured_port)
    os.environ["APP_SELECTED_PORT"] = str(selected_port)
    if selected_port != configured_port:
        print(
            f"Port {configured_port} is already in use. "
            f"Starting on http://127.0.0.1:{selected_port} instead."
        )
    app.run(debug=True, host="127.0.0.1", port=selected_port)
