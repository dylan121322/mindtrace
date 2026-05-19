from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Contact(BaseModel):
    username: str = ""
    nickname: str = ""
    remark: str = ""
    avatar: str = ""
    is_group: bool = False


class ChatMessage(BaseModel):
    seq: int = 0
    datetime: str = ""
    sender: str = ""
    content: str = ""
    is_mine: bool = False
    contact_key: str = ""


class PsychAnalyzeRequest(BaseModel):
    target_key: Optional[str] = None
    target_type: Literal["self", "contact"] = "contact"
    time_from: Optional[Any] = None
    time_to: Optional[Any] = None
    only_mine: bool = True
    include_context: bool = False
    options: Dict[str, Any] = Field(default_factory=dict)
    messages: Optional[List[ChatMessage]] = None


class PsychFeature(BaseModel):
    group: str
    name: str
    value: float
    window_start: Optional[str] = None
    window_end: Optional[str] = None


class PsychEvidence(BaseModel):
    seq: int
    datetime: str
    sender: str
    content: str
    evidence_type: str
    severity: str
    reason: str
    labels: List[str] = Field(default_factory=list)
    risk_level: int = 0


class PsychFact(BaseModel):
    fact_type: str
    fact: str
    severity: str = "low"
    confidence: float = 0.0
    evidence: List[PsychEvidence] = Field(default_factory=list)
    source_from: Optional[int] = None
    source_to: Optional[int] = None


class PsychScore(BaseModel):
    depression_signal_score: int
    self_harm_risk: Literal["low", "medium", "high", "crisis"]
    overall_risk: Literal["low", "medium", "high", "crisis"]
    risk_level: int = 0
    risk_level_label: str = ""
    confidence: float
    summary: str
    main_signals: List[str] = Field(default_factory=list)
    symptom_labels: List[Dict[str, Any]] = Field(default_factory=list)
    dimension_scores: List[Dict[str, Any]] = Field(default_factory=list)
    scoring_adjustments: Dict[str, Any] = Field(default_factory=dict)


class PsychProcessStep(BaseModel):
    key: str
    name: str
    status: Literal["pending", "running", "completed", "failed"] = "completed"
    duration_ms: int = 0
    detail: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)


class PsychAnalyzeResponse(BaseModel):
    task_id: str
    status: str
    process_steps: List[PsychProcessStep] = Field(default_factory=list)
    features: List[PsychFeature] = Field(default_factory=list)
    evidences: List[PsychEvidence] = Field(default_factory=list)
    facts: List[PsychFact] = Field(default_factory=list)
    score: PsychScore
    report_md: str
    report_json: Dict[str, Any] = Field(default_factory=dict)


class TrainingReview(BaseModel):
    accurate: bool = False
    human_risk_level: int = 0
    human_score: Optional[int] = None
    missed_labels: List[str] = Field(default_factory=list)
    false_positive_labels: List[str] = Field(default_factory=list)
    suggested_keywords: List[str] = Field(default_factory=list)
    suggested_negative_keywords: List[str] = Field(default_factory=list)
    notes: str = ""


class TrainingSampleCreateRequest(BaseModel):
    name: str = ""
    target_key: str = ""
    target_type: str = ""
    analysis_task_id: str = ""
    analysis_result: Dict[str, Any] = Field(default_factory=dict)
    human_review: TrainingReview = Field(default_factory=TrainingReview)
    notes: str = ""


class TrainingSampleUpdateRequest(BaseModel):
    name: Optional[str] = None
    human_review: Optional[TrainingReview] = None
    notes: Optional[str] = None


class TrainingProposalRequest(BaseModel):
    sample_ids: Optional[List[str]] = None
    max_samples: Union[int, str] = 30
    use_llm: bool = True
    target_type: str = "all"
    target_key: str = ""


class TrainingProposalApplyRequest(BaseModel):
    proposal_id: str
    selected_suggestion_indexes: Optional[List[int]] = None
    human_review_confirmed: bool = False
    auto_repeat_after_apply: bool = False
    repeat_max_samples: Union[int, str] = "max"
    repeat_use_llm: bool = True
