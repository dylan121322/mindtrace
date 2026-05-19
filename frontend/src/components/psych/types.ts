export const LAST_PSYCH_RESULT_KEY = 'mindtrace:psych:last-result';

export type TargetType = 'self' | 'contact';
export type RiskLevel = 'low' | 'medium' | 'high' | 'crisis';

export interface PsychContact {
  username: string;
  nickname?: string;
  remark?: string;
  avatar?: string;
  is_group?: boolean;
  message_count?: number;
  last_message_time?: string;
  target_table?: string;
}

export interface PsychChatMessage {
  seq: number;
  datetime: string;
  sender: string;
  content: string;
  is_mine: boolean;
  contact_key: string;
}

export interface PsychFeature {
  group: string;
  name: string;
  value: number;
  window_start?: string | null;
  window_end?: string | null;
}

export interface PsychEvidence {
  seq: number;
  datetime: string;
  sender: string;
  content: string;
  evidence_type: string;
  severity: string;
  reason: string;
  labels?: string[];
  risk_level?: number;
}

export interface PsychFact {
  fact_type: string;
  fact: string;
  severity: string;
  confidence: number;
  evidence: PsychEvidence[];
  source_from?: number | null;
  source_to?: number | null;
}

export interface PsychScore {
  depression_signal_score: number;
  self_harm_risk: RiskLevel;
  overall_risk: RiskLevel;
  risk_level?: number;
  risk_level_label?: string;
  confidence: number;
  summary: string;
  main_signals: string[];
  symptom_labels?: Array<Record<string, unknown>>;
  dimension_scores?: Array<Record<string, unknown>>;
  scoring_adjustments?: Record<string, unknown>;
}

export interface PsychProcessStep {
  key: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  duration_ms: number;
  detail: string;
  metrics: Record<string, unknown>;
}

export interface PsychAnalyzeResponse {
  task_id: string;
  status: string;
  process_steps?: PsychProcessStep[];
  features: PsychFeature[];
  evidences: PsychEvidence[];
  facts: PsychFact[];
  score: PsychScore;
  report_md: string;
  report_json: Record<string, unknown>;
}
