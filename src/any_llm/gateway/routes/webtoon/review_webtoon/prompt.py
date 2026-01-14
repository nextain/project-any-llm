from __future__ import annotations

def build_panel_context(panels: list[str]) -> str:
    if not panels:
        return "없음"
    return "\n".join(panels)


def build_prompt(topic: str, genre: str, style: str, script_summary: str, panel_context: str) -> tuple[str, str]:
    system_prompt = """You are a senior webtoon editor and mentor.
Write a warm, constructive review in Korean that celebrates strengths and suggests improvement.
Return JSON only. No markdown."""

    user_prompt = f"""Webtoon Info:
Topic: {topic or "미정"}
Genre: {genre or "미정"}
Style: {style or "미정"}

Script Summary:
{script_summary or "없음"}

Panel Details:
{panel_context}

Return JSON only in this exact structure:
{{
  "headline": "작품을 칭찬하는 한 줄 헤드라인",
  "summary": "작품 전체를 2~3문장으로 따뜻하게 요약",
  "strengths": ["강점 1", "강점 2", "강점 3"],
  "improvements": ["개선 방향 1", "개선 방향 2"],
  "encouragement": "다음 작품으로 이어지는 격려 문장",
  "nextIdeas": [
    {{
      "title": "다음 작품 아이디어 제목",
      "topic": "주제/로그라인",
      "genre": "추천 장르",
      "style": "추천 스타일",
      "hook": "왜 추천하는지 한 줄 이유"
    }}
  ]
}}
All values must be in Korean."""

    return system_prompt, user_prompt
