from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from api.llm import run_llm_with_instructor
from api.settings import settings
from api.config import openai_plan_to_model_name


class Segment(BaseModel):
    line: int
    timestamp: str
    text: str


class Tip(BaseModel):
    text: str = Field(description="Actionable tip phrased concisely")
    line: int = Field(description="Transcript line number to anchor the tip")
    timestamp: str = Field(description="MM:SS timestamp from the transcript header")


class ContentStructureScores(BaseModel):
    content: float = Field(ge=0, le=5)
    structure: float = Field(ge=0, le=5)


class RubricAndTips(BaseModel):
    content: float
    structure: float
    clarity: float
    delivery: float
    tips: List[Tip]


def _extract_segments(whisper_result: Dict[str, Any]) -> List[Segment]:
    segments = whisper_result.get("segments", []) or []
    parsed: List[Segment] = []
    for i, seg in enumerate(segments, start=1):
        start_time = float(seg.get("start", 0.0))
        text = (seg.get("text", "") or "").strip()
        start_min = int(start_time // 60)
        start_sec = int(start_time % 60)
        timestamp = f"{start_min:02d}:{start_sec:02d}"
        if text:
            parsed.append(Segment(line=i, timestamp=timestamp, text=text))
    return parsed


def _segments_as_bulleted_text(segments: List[Segment]) -> str:
    return "\n".join([f"- Line {s.line} [{s.timestamp}]: {s.text}" for s in segments])


async def _score_content_and_structure_with_llm(transcript: str, segments: List[Segment]) -> Tuple[ContentStructureScores, List[Tip]]:
    class ScoresAndTips(BaseModel):
        content: float = Field(ge=0, le=5)
        structure: float = Field(ge=0, le=5)
        tips: List[Tip]

    system_prompt = (
        "You are an interviewing mentor. Score CONTENT and STRUCTURE (0-5 each). "
        "Be strict: 0=missing, 5=excellent. Then propose 2-3 concise, actionable tips. "
        "Tips MUST reference the provided transcript segments using exact line numbers and timestamps. "
        "Do not hallucinate lines; only use given segments."
    )

    segments_block = _segments_as_bulleted_text(segments)
    user_prompt = (
        f"Transcript (cleaned):\n````\n{transcript}\n````\n\n"
        f"Segments:\n````\n{segments_block}\n````\n\n"
        "Tasks:\n"
        "1) Score CONTENT and STRUCTURE from 0 to 5 (decimals allowed).\n"
        "2) Output 2-3 tips. Each tip must include the exact Line number and timestamp from the Segments."
    )

    response = await run_llm_with_instructor(
        api_key=settings.openai_api_key,
        model=openai_plan_to_model_name["text"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=ScoresAndTips,
        max_completion_tokens=1500,
    )

    return ContentStructureScores(content=response.content, structure=response.structure), response.tips


def _map_metrics_to_clarity_delivery(metrics_v2: Dict[str, Any]) -> Tuple[float, float]:
    clarity = float(metrics_v2.get("clarity_score", 0.0))
    delivery = float(metrics_v2.get("fluency_score", 0.0))
    # Already on 0..5 scale
    return round(clarity, 2), round(delivery, 2)


async def generate_rubric_and_tips(
    whisper_result: Dict[str, Any], metrics_v2: Dict[str, Any]
) -> RubricAndTips:
    segments = _extract_segments(whisper_result)
    transcript_text = (whisper_result.get("text", "") or "").strip()

    content_structure, tips = await _score_content_and_structure_with_llm(
        transcript_text, segments
    )
    clarity, delivery = _map_metrics_to_clarity_delivery(metrics_v2)

    return RubricAndTips(
        content=round(content_structure.content, 2),
        structure=round(content_structure.structure, 2),
        clarity=clarity,
        delivery=delivery,
        tips=tips[:3],
    )


def generate_rubric_and_tips_sync(
    whisper_result: Dict[str, Any], metrics_v2: Dict[str, Any]
) -> Optional[RubricAndTips]:
    try:
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # Caller should use the async variant in this case
            return None
        return asyncio.run(generate_rubric_and_tips(whisper_result, metrics_v2))
    except Exception:
        return None

