from __future__ import annotations

def build_prompt(topic: str, genre: str, style: str, summary: str) -> str:
    safe_topic = topic or "정보 없음"
    safe_genre = genre or "정보 없음"
    safe_style = style or "정보 없음"
    safe_summary = summary or "정보 없음"

    return f"""Write a gallery-ready "title" and "description" based on the following information.

Requirements:
- Write in Korean.
- Title: 20–28 characters, a single impactful line.
- Description: 1–2 sentences summarizing the webtoon's core emotion/theme/conflict.
- Do not copy dialogue verbatim.
- No emojis, hashtags, or quotation marks.
- No exaggerated marketing phrases.

Topic: {safe_topic}
Genre: {safe_genre}
Style: {safe_style}
Script summary: {safe_summary}

Return JSON only:
{{"title":"...","description":"..."}}"""
