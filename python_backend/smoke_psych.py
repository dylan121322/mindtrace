from app.models import ChatMessage, PsychAnalyzeRequest
from app.services.psych_service import analyze_psych


def main() -> None:
    messages = [
        ChatMessage(seq=1, datetime="2026-05-01 09:00:00", sender="我", content="我最近有点累", is_mine=True, contact_key="smoke"),
        ChatMessage(seq=2, datetime="2026-05-08 09:00:00", sender="我", content="连续两周每天都起不来，什么都不想做，工作也做不下去", is_mine=True, contact_key="smoke"),
        ChatMessage(seq=3, datetime="2026-05-14 09:00:00", sender="我", content="还是好累，没力气，也不想见人", is_mine=True, contact_key="smoke"),
    ]
    request = PsychAnalyzeRequest(
        target_key="smoke",
        target_type="self",
        only_mine=True,
        options={
            "use_vector": False,
            "llm_screening": False,
            "llm_fact_extraction": False,
            "embedding_config": {"provider": "local"},
        },
        messages=messages,
    )
    result = analyze_psych(request)
    print(
        {
            "status": result.status,
            "steps": [step.key for step in result.process_steps],
            "score": result.score.depression_signal_score,
            "adjustments": result.score.scoring_adjustments,
            "dimension_scores": result.score.dimension_scores[:3],
            "has_disclaimer": "不构成医学诊断" in result.report_md,
        }
    )


if __name__ == "__main__":
    main()
