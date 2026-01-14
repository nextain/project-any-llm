from __future__ import annotations

def build_user_prompt(topic: str, genre: str, script_summary: str) -> str:
    safe_topic = topic or "미정"
    safe_genre = genre or "General"
    safe_summary = script_summary or "Summary not provided."

    return f"""Comic Details:
Topic: {safe_topic}
Genre: {safe_genre}
Script Summary:
{safe_summary}

Platform requirements:

Facebook:
- 3 to 5 sentences with a storytelling, empathetic tone written in Korean
- Include a comment-inviting question to boost engagement
- Use 3 to 5 relevant hashtags

Instagram:
- 2 to 3 sentences; the first line must grab attention, deliver everything in Korean
- End with a short question that encourages comments
- Recommend 10 to 15 hashtags combining topic, emotion, and style

Threads:
- 1 to 2 sentences with an honest, diary-like voice in Korean
- Use a question or lingering thought at the end
- Use at most 2 hashtags

Respond with JSON in this structure:
{{
  "facebook": {{
    "caption": "...",
    "hashtags": ["...", "..."]
  }},
  "instagram": {{
    "caption": "...",
    "hashtags": ["...", "..."]
  }},
  "threads": {{
    "caption": "...",
    "hashtags": ["...", "..."]
  }}
}}"""
