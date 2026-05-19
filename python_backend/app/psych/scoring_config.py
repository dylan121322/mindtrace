import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.config import BASE_DIR


SCORING_CONFIG_PATH = BASE_DIR / "psych_scoring_config.json"
SCORING_CONFIG_VERSION = 5


def _level(score: int, description: str, rule: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {"score": score, "description": description, "rule": rule or {}}


DEFAULT_SCORING_CONFIG: Dict[str, Any] = {
    "version": SCORING_CONFIG_VERSION,
    "max_score": 100,
    "thresholds": {"medium": 35, "high": 65, "crisis": 85},
    "evidence_strength": {
        "enabled": True,
        "levels": [
            {"key": "weak", "label": "弱证据", "coefficient": 0.5, "description": "单次表达，或语境更像玩笑、口头禅、短暂抱怨。", "rule": {"max_messages": 1, "max_active_days": 1, "max_strong_hits": 0}},
            {"key": "medium", "label": "中证据", "coefficient": 1.0, "description": "多次出现，语境明确。", "rule": {"min_messages": 2}},
            {"key": "strong", "label": "强证据", "coefficient": 1.5, "description": "连续出现，或伴随工作学习、社交、生活功能变化。", "rule": {"min_active_days": 3, "min_messages": 3}},
            {"key": "extreme", "label": "极强证据", "coefficient": 2.0, "description": "明确红线风险表达，伴随计划、准备、告别、工具或时间地点等线索。", "rule": {"redline": True, "min_plan_hits": 1}},
        ],
    },
    "time_adjustment": {
        "enabled": True,
        "levels": [
            {"key": "short_1_2_days", "label": "1–2天", "min_days": 1, "max_days": 2, "level_shift": -1, "coefficient": 0.85, "score_delta": 0, "description": "降低一级判断，避免把短暂情绪波动放大。"},
            {"key": "observe_3_6_days", "label": "3–6天", "min_days": 3, "max_days": 6, "level_shift": 0, "coefficient": 1.0, "score_delta": 0, "description": "保持谨慎观察。"},
            {"key": "attention_7_13_days", "label": "7–13天", "min_days": 7, "max_days": 13, "level_shift": 0, "coefficient": 1.1, "score_delta": 0, "description": "中度关注。"},
            {"key": "persistent_14_days", "label": "≥14天", "min_days": 14, "max_days": 30, "level_shift": 0, "coefficient": 1.2, "score_delta": 0, "description": "明显提高权重。"},
            {"key": "persistent_1_month", "label": "≥1个月", "min_days": 31, "max_days": 9999, "level_shift": 0, "coefficient": 1.3, "score_delta": 0, "description": "考虑持续性风险。"},
        ],
        "worsening_bonus": {"enabled": True, "min_bonus": 5, "max_bonus": 10, "description": "同一维度信号在后半段明显增多时加分。"},
        "relief_reduction": {"enabled": True, "score_delta": -5, "keywords": ["好多了", "缓过来了", "已经解决", "没那么难受", "状态恢复"], "description": "有明显诱因后短期缓解时适当降低判断。"},
    },
    "protective_adjustment": {
        "enabled": True,
        "min_delta": -20,
        "max_delta": 10,
        "redline_blocks_reduction": True,
        "redline_bonus": {
            "enabled": True,
            "delta": 10,
            "description": "出现明确自伤/轻生意图、方法、时间、地点、工具、告别、遗书或近期自伤行为时，保护性信号不能自动抵消，并额外提示人工复核。",
        },
        "factors": [
            {
                "key": "positive_emotion",
                "label": "正面情绪词",
                "max_delta": -5,
                "label_keys": ["positive_emotion"],
                "description": "开心、期待、放松、舒服等正面情绪体验，提示仍保留愉快体验。",
                "levels": [
                    {"delta": 0, "description": "未观察到明显正面情绪"},
                    {"delta": -1, "description": "偶尔出现“开心、期待、放松、舒服”", "rule": {"min_messages": 1, "min_hits": 1}},
                    {"delta": -2, "description": "短期内出现明确正面体验，但持续性不足", "rule": {"min_messages": 2, "min_hits": 2}},
                    {"delta": -3, "description": "多次出现积极情绪", "rule": {"min_messages": 3, "min_hits": 3}},
                    {"delta": -5, "description": "积极情绪稳定、自然、与生活事件匹配", "rule": {"min_active_days": 3, "min_messages": 4, "min_hits": 4}},
                ],
            },
            {
                "key": "future_hope",
                "label": "希望感/未来感",
                "max_delta": -5,
                "label_keys": ["future_hope", "positive_plan"],
                "description": "短期计划、明确期待、未来目标或改善意愿，提示仍有自我调节能力。",
                "levels": [
                    {"delta": 0, "description": "未观察到明确计划或未来感"},
                    {"delta": -1, "description": "有短期计划", "rule": {"min_messages": 1}},
                    {"delta": -3, "description": "有明确期待", "rule": {"min_messages": 2}},
                    {"delta": -5, "description": "有未来目标、改善意愿", "rule": {"min_active_days": 2, "min_messages": 3}},
                ],
            },
            {
                "key": "social_connection",
                "label": "社会连接",
                "max_delta": -4,
                "label_keys": ["social_connection", "help_seeking"],
                "description": "愿意与人交流、主动找朋友/家人、接受帮助或愿意求助。",
                "levels": [
                    {"delta": 0, "description": "未观察到社会连接保护信号"},
                    {"delta": -1, "description": "愿意与人交流", "rule": {"min_messages": 1}},
                    {"delta": -2, "description": "主动找朋友/家人", "rule": {"min_messages": 2}},
                    {"delta": -4, "description": "接受帮助、愿意求助", "rule": {"min_messages": 3}},
                ],
            },
            {
                "key": "function_maintained",
                "label": "功能保持",
                "max_delta": -4,
                "label_keys": ["function_maintained"],
                "description": "仍能工作学习、完成基本生活事务、保持作息运动社交。",
                "levels": [
                    {"delta": 0, "description": "未观察到功能保持信号"},
                    {"delta": -1, "description": "能正常工作/学习", "rule": {"min_messages": 1}},
                    {"delta": -2, "description": "能完成基本生活事务", "rule": {"min_messages": 2}},
                    {"delta": -4, "description": "能保持规律作息、运动、社交", "rule": {"min_active_days": 2, "min_messages": 3}},
                ],
            },
            {
                "key": "humor_context",
                "label": "幽默、调侃、语境缓冲",
                "max_delta": -2,
                "label_keys": ["humor_context", "event_complaint_expression"],
                "description": "明显网络梗、夸张表达、幽默调侃或具体事件吐槽，用于降低误判。",
                "levels": [
                    {"delta": 0, "description": "未观察到明显语境缓冲"},
                    {"delta": -1, "description": "明显是网络梗/夸张表达", "rule": {"min_messages": 1}},
                    {"delta": -2, "description": "有幽默、调侃、夸张语境", "rule": {"min_messages": 2}},
                ],
            },
        ],
    },
    "symptom_labels": {
        "enabled": True,
        "labels": [
            {
                "key": "ordinary_complaint",
                "label": "普通抱怨",
                "category": "语境",
                "weight": "low",
                "weight_label": "低",
                "risk_level": 1,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "普通烦躁、吐槽、短期不满，不直接计入抑郁风险或仅低权重参考。",
                "keywords": ["烦死了", "真烦", "好烦", "压力有点大", "今天不爽", "有点累", "有点烦", "吐了", "无语"],
                "exclude_keywords": [],
                "dimension_keys": [],
                "vector_queries": ["普通抱怨，指短期吐槽、烦躁、事件性不满，不一定代表持续心理风险。"],
            },
            {
                "key": "event_complaint_expression",
                "label": "事件型抱怨",
                "category": "表达类型",
                "weight": "low",
                "weight_label": "低权重",
                "risk_level": 1,
                "enabled": True,
                "protective": False,
                "modifier": True,
                "description": "围绕堵车、考试、游戏、排队、临时工作等具体事件的吐槽，通常低权重，需要结合持续性判断。",
                "keywords": ["堵车烦死了", "堵车", "排队", "考试烦", "游戏输了", "今天烦死了", "今天好烦", "临时加活", "被催", "赶车", "迟到"],
                "exclude_keywords": ["一直", "每天", "没希望", "活着没意义", "不想活"],
                "dimension_keys": [],
                "vector_queries": [
                    "事件型抱怨，指今天堵车烦死了、排队很烦、考试太难、游戏输了、临时事件导致的短期吐槽，低权重。"
                ],
            },
            {
                "key": "pressure_expression",
                "label": "压力型表达",
                "category": "表达类型",
                "weight": "medium_low",
                "weight_label": "看持续时间",
                "risk_level": 2,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "工作、学习、项目、家庭责任等压力下的疲劳表达，重点看是否持续、是否伴随功能受损。",
                "keywords": ["项目太赶", "快累瘫了", "压力太大", "deadline", "赶项目", "赶进度", "工作太多", "作业太多", "被压垮", "累瘫"],
                "exclude_keywords": ["笑死", "开玩笑"],
                "dimension_keys": ["fatigue_low_energy", "concentration_decision_difficulty"],
                "vector_queries": [
                    "压力型表达，指项目太赶、工作学习压力很大、快累瘫了、被任务压垮，需要结合持续时间判断。"
                ],
            },
            {
                "key": "emotional_expression",
                "label": "情绪型表达",
                "category": "表达类型",
                "weight": "medium",
                "weight_label": "中权重",
                "risk_level": 3,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "直接表达最近持续低落、难受、压抑、想哭等情绪状态，中权重，若持续多天会提高关注。",
                "keywords": ["最近一直很低落", "一直很低落", "最近很低落", "情绪低沉", "心情一直不好", "一直很难受", "一直压抑", "总想哭", "最近很难受"],
                "exclude_keywords": [],
                "dimension_keys": ["low_mood_hopelessness"],
                "vector_queries": [
                    "情绪型表达，指最近一直很低落、情绪低沉、一直难受、压抑、想哭等持续情绪低落，中权重。"
                ],
            },
            {
                "key": "cognitive_expression",
                "label": "认知型表达",
                "category": "表达类型",
                "weight": "high",
                "weight_label": "高权重",
                "risk_level": 4,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "自我否定、失败感、无价值感、拖累感等认知表达，高权重，需要重点关注。",
                "keywords": ["我很失败", "我没用", "我太失败了", "什么都做不好", "都是我的错", "我不配", "我很差劲", "我是负担", "拖累别人"],
                "exclude_keywords": ["开玩笑", "哈哈"],
                "dimension_keys": ["self_blame_worthlessness"],
                "vector_queries": [
                    "认知型表达，指我很失败、我没用、什么都做不好、都是我的错、我不配、我是负担，高权重。"
                ],
            },
            {
                "key": "existential_death_expression",
                "label": "存在/死亡型表达",
                "category": "表达类型",
                "weight": "extreme",
                "weight_label": "红线",
                "risk_level": 5,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "活着没意义、不想活、死亡愿望、自伤或轻生相关表达，属于红线项；只做风险提示和求助建议，不做诊断。",
                "keywords": ["活着没意义", "不想活了", "不想活", "想死", "真想睡着不醒", "消失就好了", "结束生命", "离开这个世界", "遗书", "告别", "买了药"],
                "exclude_keywords": ["笑死了", "累死了", "社死了", "尴尬死了", "气死了"],
                "dimension_keys": ["self_harm_suicide_risk", "low_mood_hopelessness"],
                "vector_queries": [
                    "存在/死亡型表达，指活着没意义、不想活、想死、真想睡着不醒、消失就好了、自伤轻生相关表达，属于红线项。"
                ],
            },
            {
                "key": "stress_fatigue",
                "label": "压力疲劳",
                "category": "身心状态",
                "weight": "medium_low",
                "weight_label": "中低",
                "risk_level": 2,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "压力和疲劳表达，需要结合持续时间、功能影响和其他症状判断。",
                "keywords": ["压力大", "好累", "太累", "累", "疲惫", "没力气", "撑不住", "扛不住", "心累"],
                "exclude_keywords": ["笑死", "累死了"],
                "dimension_keys": ["fatigue_low_energy", "low_mood_hopelessness"],
                "vector_queries": ["压力疲劳，指压力很大、好累、心累、没力气、撑不住，需要结合持续时间判断。"],
            },
            {
                "key": "anxiety_worry",
                "label": "焦虑担忧",
                "category": "情绪",
                "weight": "medium",
                "weight_label": "中",
                "risk_level": 2,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "焦虑担忧可与抑郁相关信号共现，但不等同于抑郁。",
                "keywords": ["焦虑", "担心", "害怕", "紧张", "心慌", "不安", "慌", "睡不踏实", "脑子停不下来"],
                "exclude_keywords": [],
                "dimension_keys": ["sleep_disturbance", "psychomotor_change"],
                "vector_queries": ["焦虑担忧，指担心、害怕、紧张、心慌、不安，可能与抑郁信号共现。"],
            },
            {
                "key": "low_mood",
                "label": "情绪低落",
                "category": "情绪",
                "weight": "medium_high",
                "weight_label": "中高",
                "risk_level": 3,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "抑郁核心信号之一，关注低落、悲伤、压抑、难受等表达。",
                "keywords": ["难受", "低落", "心情不好", "压抑", "崩溃", "悲伤", "失落", "想哭", "委屈"],
                "exclude_keywords": [],
                "dimension_keys": ["low_mood_hopelessness"],
                "vector_queries": ["情绪低落，指心情低沉、悲伤、失落、压抑、难受、想哭。"],
            },
            {
                "key": "interest_loss",
                "label": "兴趣下降",
                "category": "动机",
                "weight": "high",
                "weight_label": "高",
                "risk_level": 3,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "抑郁核心信号之一，关注兴趣、动力、快感明显下降。",
                "keywords": ["没兴趣", "没意思", "提不起劲", "什么都不想做", "不想玩", "不想去", "没感觉", "以前喜欢现在没感觉"],
                "exclude_keywords": [],
                "dimension_keys": ["interest_loss_anhedonia"],
                "vector_queries": ["兴趣下降，指没兴趣、没意思、提不起劲、什么都不想做，对喜欢的事也没感觉。"],
            },
            {
                "key": "helpless_hopeless",
                "label": "无助绝望",
                "category": "认知",
                "weight": "high",
                "weight_label": "高",
                "risk_level": 4,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "风险明显上升的信号，关注没希望、无助、人生没意义等表达。",
                "keywords": ["没希望", "绝望", "无助", "没意义", "人生没意义", "怎么都好不了", "活着累", "看不到希望"],
                "exclude_keywords": [],
                "dimension_keys": ["low_mood_hopelessness"],
                "vector_queries": ["无助绝望，指没希望、绝望、无助、人生没意义、看不到希望、怎么都好不了。"],
            },
            {
                "key": "self_negation",
                "label": "自我否定",
                "category": "认知",
                "weight": "high",
                "weight_label": "高",
                "risk_level": 4,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "需重点关注的信号，包含自责、无价值感、拖累感。",
                "keywords": ["我没用", "我失败", "我太失败", "都是我的错", "我不配", "我拖累别人", "我很差劲", "没有我更好", "负担"],
                "exclude_keywords": [],
                "dimension_keys": ["self_blame_worthlessness"],
                "vector_queries": ["自我否定，指我没用、我失败、都是我的错、我不配、拖累别人、没有我更好。"],
            },
            {
                "key": "sleep_abnormal",
                "label": "睡眠异常",
                "category": "生理",
                "weight": "medium",
                "weight_label": "中",
                "risk_level": 2,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "辅助症状，需结合持续时间和日间状态判断。",
                "keywords": ["睡不着", "睡不好", "早醒", "三四点醒", "整晚没睡", "睡太多", "起不来", "昼夜颠倒", "失眠"],
                "exclude_keywords": [],
                "dimension_keys": ["sleep_disturbance"],
                "vector_queries": ["睡眠异常，指睡不着、早醒、整晚没睡、睡太多、起不来、昼夜颠倒。"],
            },
            {
                "key": "appetite_change",
                "label": "食欲变化",
                "category": "生理",
                "weight": "medium_low",
                "weight_label": "中低",
                "risk_level": 2,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "辅助症状，关注食欲与体重变化。",
                "keywords": ["吃不下", "没胃口", "不想吃饭", "暴食", "一直吃", "瘦了很多", "胖了很多"],
                "exclude_keywords": [],
                "dimension_keys": ["appetite_weight_change"],
                "vector_queries": ["食欲变化，指吃不下、没胃口、不想吃饭、暴食、一直吃、体重明显变化。"],
            },
            {
                "key": "concentration_problem",
                "label": "注意力下降",
                "category": "认知",
                "weight": "medium_low",
                "weight_label": "中低",
                "risk_level": 2,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "辅助症状，关注无法集中、脑子乱、决策困难。",
                "keywords": ["看不进去", "记不住", "脑子乱", "反应慢", "无法集中", "集中不了", "做不了决定", "想不明白"],
                "exclude_keywords": [],
                "dimension_keys": ["concentration_decision_difficulty"],
                "vector_queries": ["注意力下降，指无法集中、看不进去、记不住、脑子乱、反应慢、做不了决定。"],
            },
            {
                "key": "social_withdrawal",
                "label": "社交退缩",
                "category": "行为",
                "weight": "medium_high",
                "weight_label": "中高",
                "risk_level": 3,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "行为风险信号，关注不想见人、不回消息、回避社交。",
                "keywords": ["不想见人", "不想说话", "不想回消息", "谁也不想理", "一个人待着", "不想出门", "别烦我"],
                "exclude_keywords": [],
                "dimension_keys": ["social_withdrawal_function_impairment", "interest_loss_anhedonia"],
                "vector_queries": ["社交退缩，指不想见人、不想说话、不想回消息、回避邀约、谁也不想理。"],
            },
            {
                "key": "function_impairment",
                "label": "功能受损",
                "category": "功能",
                "weight": "high",
                "weight_label": "高",
                "risk_level": 4,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "严重程度关键指标，关注工作学习、家庭责任和基本生活受影响。",
                "keywords": ["请假", "无法完成", "做不下去", "不上学", "不工作", "不洗漱", "不吃饭", "起不来", "什么都干不动", "连洗澡都累"],
                "exclude_keywords": [],
                "dimension_keys": ["social_withdrawal_function_impairment", "fatigue_low_energy", "concentration_decision_difficulty"],
                "vector_queries": ["功能受损，指工作学习家庭责任受影响，无法完成任务，不上学不工作，基本生活困难。"],
            },
            {
                "key": "self_harm_suicide",
                "label": "自伤轻生",
                "category": "红线",
                "weight": "extreme",
                "weight_label": "极高",
                "risk_level": 5,
                "enabled": True,
                "protective": False,
                "modifier": False,
                "description": "红线项。只做风险提示与求助建议，不做诊断。",
                "keywords": ["睡着不醒", "消失就好了", "不想活", "想死", "结束生命", "自杀", "跳楼", "割腕", "吃药", "离开这个世界", "遗书", "告别", "最后一次", "买了药"],
                "exclude_keywords": ["笑死了", "累死了", "社死了", "尴尬死了", "气死了"],
                "dimension_keys": ["self_harm_suicide_risk"],
                "vector_queries": ["自伤轻生红线信号，包含不想活、想死、自杀、自伤方法、工具、时间地点、遗书或告别行为。"],
            },
            {
                "key": "positive_emotion",
                "label": "正面情绪",
                "category": "保护性",
                "weight": "protective",
                "weight_label": "保护性",
                "risk_level": 0,
                "enabled": True,
                "protective": True,
                "modifier": False,
                "description": "开心、期待、放松、舒服等自然正面体验，可作为保护性修正因子。",
                "keywords": ["开心", "挺开心", "有点开心", "很开心", "期待", "放松", "放松下来", "舒服", "舒服多了", "轻松", "还不错", "状态不错", "挺好", "高兴", "愉快", "满足", "安心", "踏实", "治愈", "被治愈", "感觉好多了", "心情好了"],
                "exclude_keywords": ["不开心", "没开心", "开心不起来", "开心不了", "并不开心", "一点也不开心", "开心个屁", "强颜欢笑", "假装开心", "装开心", "应该开心却", "没什么好期待", "不期待", "不舒服"],
                "dimension_keys": [],
                "vector_queries": ["正面情绪，指开心、期待、放松、舒服、轻松、还不错等自然愉快体验。"],
            },
            {
                "key": "future_hope",
                "label": "希望感/未来感",
                "category": "保护性",
                "weight": "protective",
                "weight_label": "保护性",
                "risk_level": 0,
                "enabled": True,
                "protective": True,
                "modifier": False,
                "description": "短期计划、明确期待、未来目标或改善意愿，提示仍有自我调节能力。",
                "keywords": ["明天", "下周", "计划", "准备", "期待", "想试试", "以后会好", "慢慢来", "再坚持一下", "重新调整", "周末休息", "会好起来"],
                "exclude_keywords": [],
                "dimension_keys": [],
                "vector_queries": ["希望感和未来感，指有明天、下周、计划、准备、期待、想试试、以后会好、慢慢来、重新调整等表达。"],
            },
            {
                "key": "positive_plan",
                "label": "积极计划",
                "category": "保护性",
                "weight": "protective",
                "weight_label": "保护性",
                "risk_level": 0,
                "enabled": True,
                "protective": True,
                "modifier": False,
                "description": "表达恢复计划、现实安排、行动计划，可降低误判。",
                "keywords": ["准备去", "打算去", "计划", "明天去", "我会去", "开始调整", "想办法", "先休息", "出去走走", "约朋友"],
                "exclude_keywords": [],
                "dimension_keys": [],
                "vector_queries": ["积极计划，指有恢复计划、现实安排、行动计划、准备求助或主动改善。"],
            },
            {
                "key": "social_connection",
                "label": "社会连接",
                "category": "保护性",
                "weight": "protective",
                "weight_label": "保护性",
                "risk_level": 0,
                "enabled": True,
                "protective": True,
                "modifier": False,
                "description": "愿意与人交流、主动联系朋友或家人、有人陪伴等保护性信号。",
                "keywords": ["找朋友聊聊", "找人聊聊", "和家人说", "约人吃饭", "有人陪", "朋友陪", "家人陪", "一起吃饭", "一起出去", "有人听我说"],
                "exclude_keywords": [],
                "dimension_keys": [],
                "vector_queries": ["社会连接保护信号，指愿意与人交流、主动找朋友家人、约人吃饭、有人陪伴或有人支持。"],
            },
            {
                "key": "help_seeking",
                "label": "求助意愿",
                "category": "保护性",
                "weight": "protective",
                "weight_label": "保护性",
                "risk_level": 0,
                "enabled": True,
                "protective": True,
                "modifier": False,
                "description": "表达愿意求助、倾诉、咨询或联系可信任的人，可降低风险判断。",
                "keywords": ["想找人聊", "找咨询", "看医生", "求助", "打热线", "告诉朋友", "和家人说", "去医院", "预约咨询"],
                "exclude_keywords": [],
                "dimension_keys": [],
                "vector_queries": ["求助意愿，指愿意找人聊、预约咨询、看医生、联系朋友家人或心理援助热线。"],
            },
            {
                "key": "function_maintained",
                "label": "功能保持",
                "category": "保护性",
                "weight": "protective",
                "weight_label": "保护性",
                "risk_level": 0,
                "enabled": True,
                "protective": True,
                "modifier": False,
                "description": "仍能工作学习、完成基本生活事务、保持规律作息、运动或社交。",
                "keywords": ["工作还在推进", "还在推进", "完成作业", "按时上班", "按时上课", "正常工作", "正常学习", "去跑步", "运动", "洗漱", "做饭", "规律作息", "早睡", "收拾房间"],
                "exclude_keywords": [],
                "dimension_keys": [],
                "vector_queries": ["功能保持，指能正常工作学习、完成基本生活事务、保持规律作息、运动、社交或日常安排。"],
            },
            {
                "key": "humor_context",
                "label": "幽默调侃",
                "category": "语境修正",
                "weight": "modifier",
                "weight_label": "语境修正",
                "risk_level": 0,
                "enabled": True,
                "protective": False,
                "modifier": True,
                "description": "玩笑、口头禅、网络梗等语境修正，降低误判。",
                "keywords": ["笑死", "笑死了", "累死了", "社死", "尴尬死", "气死了", "裂开", "绷不住了", "哈哈"],
                "exclude_keywords": [],
                "dimension_keys": [],
                "vector_queries": ["幽默调侃语境，指网络梗、口头禅、玩笑式表达，用于降低误判。"],
            },
        ],
        "risk_levels": [
            {"level": 0, "label": "无明显负面心理信号", "description": "没有观察到明显负面心理信号。"},
            {"level": 1, "label": "普通抱怨或短期烦躁", "description": "普通抱怨、短期烦躁、事件性不满。"},
            {"level": 2, "label": "明显压力或低落但有恢复线索", "description": "明显压力或情绪低落，但有诱因、有恢复计划或求助意愿。"},
            {"level": 3, "label": "多项抑郁相关信号", "description": "低落、兴趣下降、自责、睡眠问题等多项出现。"},
            {"level": 4, "label": "持续且功能受损或强烈无价值感", "description": "持续两周以上、功能明显受损，或强烈无价值感。"},
            {"level": 5, "label": "自伤/轻生红线信号", "description": "出现自伤/轻生表达、计划、工具、告别行为。"},
        ],
    },
    "dimensions": [
        {
            "key": "low_mood_hopelessness",
            "label": "情绪低落 / 绝望感",
            "max_points": 18,
            "enabled": True,
            "redline": False,
            "description": "识别低落、压抑、崩溃、撑不住、绝望和意义感下降等表达。",
            "keywords": ["难受", "低落", "心情不好", "压力大", "压抑", "崩溃", "撑不住", "没希望", "没意义", "绝望", "活着累", "人生没意义", "每天都很难受", "醒来就不想面对"],
            "strong_keywords": ["没希望", "没意义", "绝望", "人生没意义", "怎么都好不了", "活着累", "每天都很难受", "醒来就不想面对"],
            "exclude_keywords": [],
            "vector_queries": [
                "低落情绪，指心情持续低沉、悲伤、难受、压抑、崩溃和撑不住。",
                "绝望感，指反复觉得没希望、怎么都好不了、人生没意义、活着很累。",
                "连续多天醒来就不想面对，每天都很难受，情绪明显低落。",
            ],
            "levels": [
                _level(0, "很少出现低落、绝望、崩溃类表达"),
                _level(4, "偶尔说“烦、累、心情不好、压力大”", {"min_messages": 1}),
                _level(8, "多次出现“难受、撑不住、很低落、很压抑”", {"min_messages": 3}),
                _level(12, "连续多天表达明显低落，如“每天都很难受”“醒来就不想面对”", {"min_active_days": 3, "min_messages": 3}),
                _level(18, "反复出现绝望表达，如“没希望了”“怎么都好不了”“人生没意义”", {"min_strong_hits": 2}),
            ],
        },
        {
            "key": "interest_loss_anhedonia",
            "label": "兴趣下降 / 快感缺失",
            "max_points": 15,
            "enabled": True,
            "redline": False,
            "description": "识别兴趣下降、快感缺失、动力下降和回避活动。",
            "keywords": ["没兴趣", "没意思", "提不起劲", "不想玩", "不想去", "不想出门", "不想见人", "什么都不想做", "以前喜欢现在没感觉", "没感觉"],
            "strong_keywords": ["什么都不想做", "做什么都没意思", "以前喜欢现在没感觉", "连休息都没意义", "连娱乐都没意思", "见朋友都没意义"],
            "exclude_keywords": [],
            "vector_queries": [
                "兴趣下降，指不想玩、不想去、不太感兴趣，对原本喜欢的事也减少兴趣。",
                "快感缺失，指做什么都没意思，休息、娱乐、见朋友也觉得无意义。",
                "没有动力，什么都不想做，提不起劲，不想出门。",
            ],
            "levels": [
                _level(0, "仍能表达兴趣、期待、计划"),
                _level(3, "偶尔说“不想玩、不想去、不太感兴趣”", {"min_messages": 1}),
                _level(7, "对原本喜欢的事明显减少兴趣", {"min_messages": 2}),
                _level(11, "多次表达“什么都不想做”“做什么都没意思”", {"min_messages": 3, "min_strong_hits": 1}),
                _level(15, "持续丧失快感，连休息、娱乐、见朋友都觉得无意义", {"min_active_days": 3, "min_strong_hits": 2}),
            ],
        },
        {
            "key": "sleep_disturbance",
            "label": "睡眠异常",
            "max_points": 8,
            "enabled": True,
            "redline": False,
            "description": "识别失眠、早醒、睡太多、昼夜颠倒等表达。",
            "keywords": ["睡不着", "睡不好", "早醒", "三四点醒", "整晚没睡", "失眠", "睡太多", "起不来", "昼夜颠倒", "熬夜"],
            "strong_keywords": ["整晚没睡", "昼夜颠倒", "长期早醒", "连续失眠", "严重失眠"],
            "exclude_keywords": [],
            "vector_queries": [
                "睡眠异常，指睡不着、早醒、整晚没睡、睡不好或睡太多。",
                "昼夜颠倒，夜里睡不着，白天起不来，并影响白天状态。",
                "连续多天睡眠异常，并伴随情绪变差、疲惫或焦虑。",
            ],
            "levels": [
                _level(0, "没有睡眠异常表达"),
                _level(2, "偶尔说睡不好、熬夜", {"min_messages": 1}),
                _level(4, "多次出现失眠、早醒、睡太多", {"min_messages": 2}),
                _level(6, "连续多天睡眠异常，影响白天状态", {"min_active_days": 3, "min_messages": 3}),
                _level(8, "严重失眠/昼夜颠倒/长期早醒，并伴随情绪恶化", {"min_strong_hits": 1, "min_messages": 2}),
            ],
        },
        {
            "key": "fatigue_low_energy",
            "label": "精力下降 / 疲乏",
            "max_points": 8,
            "enabled": True,
            "redline": False,
            "description": "识别疲乏、没力气、起不来、生活动作困难等表达。",
            "keywords": ["好累", "太累了", "累", "没力气", "没劲", "疲惫", "起不来", "动不了", "什么都干不动", "连洗澡都累", "吃饭都累"],
            "strong_keywords": ["什么都干不动", "连洗澡都累", "吃饭都累", "起床都费劲", "动不了"],
            "exclude_keywords": ["笑死", "累死了"],
            "vector_queries": [
                "精力下降，指经常觉得累、没力气、疲惫、起不来。",
                "疲乏影响生活，工作学习家务做不动，基本生活动作也困难。",
                "连洗澡、吃饭、起床都很费劲，身体和心理都很累。",
            ],
            "levels": [
                _level(0, "无明显疲乏表达"),
                _level(2, "偶尔说累", {"min_messages": 1}),
                _level(4, "经常说累、没劲、疲惫", {"min_messages": 3}),
                _level(6, "明显影响工作、学习、家务", {"min_messages": 3, "min_strong_hits": 1}),
                _level(8, "连基本生活动作都困难，如洗澡、吃饭、起床都很费劲", {"min_strong_hits": 2}),
            ],
        },
        {
            "key": "appetite_weight_change",
            "label": "食欲 / 体重相关表达",
            "max_points": 5,
            "enabled": True,
            "redline": False,
            "description": "识别食欲变化、吃不下、暴食、体重明显变化等表达。",
            "keywords": ["吃不下", "没胃口", "不想吃饭", "暴食", "一直吃", "瘦了很多", "胖了很多", "胃口不好", "胃口变了"],
            "strong_keywords": ["瘦了很多", "胖了很多", "体重掉", "体重涨", "吃不下饭", "暴食"],
            "exclude_keywords": [],
            "vector_queries": [
                "食欲变化，指没胃口、吃不下、不想吃饭或明显吃多了。",
                "暴食或食量明显变化，并伴随身体不适或情绪变差。",
                "体重明显变化，瘦了很多或胖了很多。",
            ],
            "levels": [
                _level(0, "无明显食欲变化"),
                _level(1, "偶尔说没胃口或吃多了", {"min_messages": 1}),
                _level(3, "多次提到吃不下、暴食、胃口明显变化", {"min_messages": 2}),
                _level(5, "食欲变化持续，并伴随体重明显变化或身体不适", {"min_strong_hits": 1, "min_messages": 2}),
            ],
        },
        {
            "key": "self_blame_worthlessness",
            "label": "自责 / 无价值感 / 拖累感",
            "max_points": 14,
            "enabled": True,
            "redline": False,
            "description": "识别自责、自我否定、无价值感、拖累感等重要信号。",
            "keywords": ["我不行", "我做不好", "都是我的错", "我太失败了", "我失败", "我没用", "我不配", "我拖累别人", "我很差劲", "没有我会更好", "负担"],
            "strong_keywords": ["我没用", "我不配", "我拖累别人", "活着就是负担", "没有我更好", "没有我会更好"],
            "exclude_keywords": [],
            "vector_queries": [
                "自责和自我否定，指反复说都是我的错、我失败、我做不好。",
                "无价值感，指觉得自己没用、不配、很差劲、没有存在价值。",
                "拖累感，指觉得自己是负担、拖累别人、没有我更好。",
            ],
            "levels": [
                _level(0, "无明显自我否定"),
                _level(3, "偶尔说“我不行、我做不好”", {"min_messages": 1}),
                _level(7, "反复自责，如“都是我的错”“我太失败了”", {"min_messages": 2}),
                _level(10, "明显无价值感，如“我没用”“我不配”", {"min_strong_hits": 1}),
                _level(14, "出现拖累感或存在价值否定，如“我活着就是负担”“没有我更好”", {"min_strong_hits": 2}),
            ],
        },
        {
            "key": "concentration_decision_difficulty",
            "label": "注意力下降 / 决策困难",
            "max_points": 6,
            "enabled": True,
            "redline": False,
            "description": "识别注意力差、脑子乱、无法集中、难以做决定等表达。",
            "keywords": ["看不进去", "记不住", "脑子乱", "反应慢", "无法集中", "集中不了", "做不了决定", "想不明白", "注意力差"],
            "strong_keywords": ["做不了决定", "无法集中", "看不进去", "脑子乱"],
            "exclude_keywords": [],
            "vector_queries": [
                "注意力下降，指看不进去、记不住、无法集中、脑子乱。",
                "决策困难，指简单决定也做不了、想不明白、反应慢。",
                "注意力和决策困难影响工作、学习、沟通或生活安排。",
            ],
            "levels": [
                _level(0, "无明显异常"),
                _level(1, "偶尔说注意力差", {"min_messages": 1}),
                _level(3, "多次说无法集中、脑子乱", {"min_messages": 2}),
                _level(5, "影响工作、学习、沟通", {"min_messages": 3}),
                _level(6, "简单决定也困难，明显影响生活安排", {"min_strong_hits": 2}),
            ],
        },
        {
            "key": "psychomotor_change",
            "label": "行为迟滞或激越",
            "max_points": 6,
            "enabled": True,
            "redline": False,
            "description": "识别行动变慢、反应慢、长期躺着、坐立不安、烦躁等表达。",
            "keywords": ["动不了", "反应慢", "说话都累", "一直躺着", "坐不住", "心慌", "烦躁", "来回走", "拖延", "几乎不动", "坐立不安"],
            "strong_keywords": ["几乎不动", "一直躺着", "坐立不安", "说话都累", "动不了"],
            "exclude_keywords": [],
            "vector_queries": [
                "行为迟滞，指行动变慢、反应慢、说话都累、一直躺着、几乎不动。",
                "激越或烦躁，指坐不住、心慌、烦躁、坐立不安、来回走。",
                "迟滞或激越明显影响沟通、安排或日常生活。",
            ],
            "levels": [
                _level(0, "无明显表现"),
                _level(1, "偶尔说坐不住或行动慢", {"min_messages": 1}),
                _level(3, "多次出现“反应慢、拖延、烦躁坐不住”", {"min_messages": 2}),
                _level(5, "明显影响沟通或日常安排", {"min_messages": 3}),
                _level(6, "长期迟滞或激越，如几乎不动、长时间躺着、明显坐立不安", {"min_strong_hits": 2}),
            ],
        },
        {
            "key": "social_withdrawal_function_impairment",
            "label": "社交退缩 / 功能受损",
            "max_points": 10,
            "enabled": True,
            "redline": False,
            "description": "识别回避社交、长期不回消息、工作学习和家庭生活受影响。",
            "keywords": ["不想见人", "不想说话", "别烦我", "一个人待着", "谁也不想理", "不想回消息", "不想出门", "请假", "拖延", "无法完成", "不洗漱", "不吃饭", "不上学", "不工作"],
            "strong_keywords": ["长期不回消息", "不出门", "不工作", "不上学", "不洗漱", "无法完成", "谁也不想理"],
            "exclude_keywords": [],
            "vector_queries": [
                "社交退缩，指不想见人、不想说话、不想回消息、频繁拒绝社交邀约。",
                "功能受损，指工作学习家庭责任受影响，请假、拖延、无法完成任务。",
                "长期不回消息、不出门、不工作不上学、不洗漱不吃饭，生活功能明显下降。",
            ],
            "levels": [
                _level(0, "社交、工作、学习基本正常"),
                _level(2, "偶尔拒绝社交", {"min_messages": 1}),
                _level(5, "回复减少，明显不愿见人", {"min_messages": 2}),
                _level(8, "工作、学习、家庭责任受到影响", {"min_messages": 3, "min_strong_hits": 1}),
                _level(10, "长期不回消息、不出门、不工作/不上学，生活功能明显下降", {"min_strong_hits": 2}),
            ],
        },
        {
            "key": "self_harm_suicide_risk",
            "label": "自伤 / 轻生风险",
            "max_points": 10,
            "enabled": True,
            "redline": True,
            "description": "红线项：只做风险提示和求助建议，不输出医学诊断结论。",
            "keywords": ["睡着不醒", "消失就好了", "不想活", "想死", "结束生命", "自杀", "跳楼", "割腕", "吃药", "离开这个世界", "遗书", "告别", "最后一次", "买了药"],
            "exclude_keywords": ["笑死了", "累死了", "社死了", "尴尬死了", "气死了"],
            "strong_keywords": ["不想活", "想死", "结束生命", "自杀", "跳楼", "割腕", "吃药", "遗书", "告别", "最后一次", "买了药"],
            "subgroups": {
                "passive": ["睡着不醒", "消失就好了", "不如消失", "离开就好了"],
                "explicit": ["不想活", "想死", "结束生命", "自杀", "离开这个世界"],
                "method": ["跳楼", "割腕", "吃药", "上吊", "煤气"],
                "plan": ["今晚", "明天", "已经准备", "准备好了", "买了药", "遗书", "告别", "最后一次", "时间", "地点"],
            },
            "vector_queries": [
                "被动死亡愿望，指真想睡着不醒、消失就好了、不如不存在。",
                "明确轻生想法，指不想活了、想死、结束生命、自杀、离开这个世界。",
                "自伤或轻生计划线索，提到方法、时间、地点、工具、遗书、告别行为。",
            ],
            "levels": [
                _level(0, "无相关表达"),
                _level(3, "被动死亡愿望，如“真想睡着不醒”“消失就好了”", {"min_subgroup_hits": {"passive": 1}}),
                _level(6, "明确轻生想法，如“不想活了”“想死”", {"min_subgroup_hits": {"explicit": 1}}),
                _level(8, "提到方法但无明确时间/地点/准备", {"min_subgroup_hits": {"method": 1}}),
                _level(10, "有具体计划、时间、地点、工具、遗书、告别行为", {"min_subgroup_hits": {"plan": 1}, "min_strong_hits": 1}),
            ],
        },
    ],
    "confidence": {
        "base": 0.25,
        "message_cap": 80,
        "message_divisor": 120,
        "active_days_cap": 14,
        "active_days_divisor": 35,
        "small_sample_caps": [
            {"max_messages": 10, "confidence_cap": 0.45},
            {"max_messages": 30, "confidence_cap": 0.65},
        ],
    },
}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _string_list(value: Any, fallback: Iterable[str] | None = None) -> List[str]:
    source = value if isinstance(value, list) else list(fallback or [])
    out: List[str] = []
    seen = set()
    for item in source:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_levels(value: Any, fallback: List[Dict[str, Any]], max_points: int) -> List[Dict[str, Any]]:
    source = value if isinstance(value, list) and value else fallback
    levels: List[Dict[str, Any]] = []
    for item in source:
        if not isinstance(item, dict):
            continue
        score = max(0, min(max_points, _as_int(item.get("score"), 0)))
        levels.append(
            {
                "score": score,
                "description": str(item.get("description") or ""),
                "rule": item.get("rule") if isinstance(item.get("rule"), dict) else {},
            }
        )
    if not levels:
        levels = deepcopy(fallback)
    levels.sort(key=lambda item: int(item.get("score") or 0))
    return levels


def _normalize_evidence_strength(value: Any) -> Dict[str, Any]:
    fallback = DEFAULT_SCORING_CONFIG["evidence_strength"]
    raw = value if isinstance(value, dict) else {}
    raw_levels = raw.get("levels") if isinstance(raw.get("levels"), list) else fallback["levels"]
    fallback_by_key = {item["key"]: item for item in fallback["levels"]}
    levels: List[Dict[str, Any]] = []
    for item in raw_levels:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        base = fallback_by_key.get(key, {})
        levels.append(
            {
                "key": key,
                "label": str(item.get("label") or base.get("label") or key),
                "coefficient": max(0.0, _as_float(item.get("coefficient", base.get("coefficient", 1.0)), 1.0)),
                "description": str(item.get("description") or base.get("description") or ""),
                "rule": item.get("rule") if isinstance(item.get("rule"), dict) else deepcopy(base.get("rule", {})),
            }
        )
    if not levels:
        levels = deepcopy(fallback["levels"])
    levels.sort(key=lambda item: float(item.get("coefficient") or 0))
    return {"enabled": _as_bool(raw.get("enabled"), True), "levels": levels}


def _normalize_time_adjustment(value: Any) -> Dict[str, Any]:
    fallback = DEFAULT_SCORING_CONFIG["time_adjustment"]
    raw = value if isinstance(value, dict) else {}
    raw_levels = raw.get("levels") if isinstance(raw.get("levels"), list) else fallback["levels"]
    fallback_by_key = {item["key"]: item for item in fallback["levels"]}
    levels: List[Dict[str, Any]] = []
    for item in raw_levels:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        base = fallback_by_key.get(key, {})
        levels.append(
            {
                "key": key,
                "label": str(item.get("label") or base.get("label") or key),
                "min_days": max(0, _as_int(item.get("min_days", base.get("min_days", 0)), 0)),
                "max_days": max(0, _as_int(item.get("max_days", base.get("max_days", 9999)), 9999)),
                "level_shift": _as_int(item.get("level_shift", base.get("level_shift", 0)), 0),
                "coefficient": max(0.0, _as_float(item.get("coefficient", base.get("coefficient", 1.0)), 1.0)),
                "score_delta": _as_int(item.get("score_delta", base.get("score_delta", 0)), 0),
                "description": str(item.get("description") or base.get("description") or ""),
            }
        )
    if not levels:
        levels = deepcopy(fallback["levels"])
    levels.sort(key=lambda item: int(item.get("min_days") or 0))

    bonus_raw = raw.get("worsening_bonus") if isinstance(raw.get("worsening_bonus"), dict) else {}
    bonus_fallback = fallback["worsening_bonus"]
    relief_raw = raw.get("relief_reduction") if isinstance(raw.get("relief_reduction"), dict) else {}
    relief_fallback = fallback["relief_reduction"]
    return {
        "enabled": _as_bool(raw.get("enabled"), True),
        "levels": levels,
        "worsening_bonus": {
            "enabled": _as_bool(bonus_raw.get("enabled"), bool(bonus_fallback.get("enabled", True))),
            "min_bonus": _as_int(bonus_raw.get("min_bonus", bonus_fallback.get("min_bonus", 5)), 5),
            "max_bonus": _as_int(bonus_raw.get("max_bonus", bonus_fallback.get("max_bonus", 10)), 10),
            "description": str(bonus_raw.get("description") or bonus_fallback.get("description") or ""),
        },
        "relief_reduction": {
            "enabled": _as_bool(relief_raw.get("enabled"), bool(relief_fallback.get("enabled", True))),
            "score_delta": _as_int(relief_raw.get("score_delta", relief_fallback.get("score_delta", -5)), -5),
            "keywords": _string_list(relief_raw.get("keywords"), relief_fallback.get("keywords", [])),
            "description": str(relief_raw.get("description") or relief_fallback.get("description") or ""),
        },
    }


def _normalize_protective_adjustment(value: Any) -> Dict[str, Any]:
    fallback = DEFAULT_SCORING_CONFIG["protective_adjustment"]
    raw = value if isinstance(value, dict) else {}
    raw_factors = raw.get("factors") if isinstance(raw.get("factors"), list) else fallback["factors"]
    fallback_by_key = {item["key"]: item for item in fallback["factors"]}
    factors: List[Dict[str, Any]] = []
    for item in raw_factors:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        base = fallback_by_key.get(key, {})
        raw_levels = item.get("levels") if isinstance(item.get("levels"), list) else base.get("levels", [])
        levels: List[Dict[str, Any]] = []
        for level in raw_levels:
            if not isinstance(level, dict):
                continue
            levels.append(
                {
                    "delta": max(-50, min(50, _as_int(level.get("delta"), 0))),
                    "description": str(level.get("description") or ""),
                    "rule": level.get("rule") if isinstance(level.get("rule"), dict) else {},
                }
            )
        if key == "positive_emotion":
            default_rule_by_delta = {
                int(level.get("delta") or 0): level.get("rule", {})
                for level in base.get("levels", [])
                if isinstance(level, dict)
            }
            default_deltas = {int(level.get("delta") or 0) for level in levels}
            for default_level in base.get("levels", []):
                delta = int(default_level.get("delta") or 0)
                if delta not in default_deltas:
                    levels.append(deepcopy(default_level))
                    default_deltas.add(delta)
            for level in levels:
                if not isinstance(level.get("rule"), dict):
                    level["rule"] = {}
                delta = int(level.get("delta") or 0)
                default_rule = default_rule_by_delta.get(delta, {})
                if delta < 0:
                    for field in ("min_messages", "min_active_days", "min_hits"):
                        if field in default_rule:
                            level["rule"][field] = max(
                                int(level["rule"].get(field) or 0),
                                int(default_rule.get(field) or 0),
                            )
                    if "min_hits" not in level["rule"]:
                        level["rule"]["min_hits"] = max(1, int(level["rule"].get("min_messages") or 1))
        if not levels:
            levels = deepcopy(base.get("levels", []))
        levels.sort(key=lambda level: int(level.get("delta") or 0), reverse=True)
        factors.append(
            {
                "key": key,
                "label": str(item.get("label") or base.get("label") or key),
                "max_delta": max(-50, min(0, _as_int(item.get("max_delta", base.get("max_delta", 0)), 0))),
                "label_keys": _string_list(item.get("label_keys"), base.get("label_keys", [])),
                "description": str(item.get("description") or base.get("description") or ""),
                "levels": levels,
            }
        )
    if not factors:
        factors = deepcopy(fallback["factors"])
    else:
        seen = {str(item.get("key") or "") for item in factors}
        for item in fallback["factors"]:
            key = str(item.get("key") or "")
            if key and key not in seen:
                factors.append(deepcopy(item))
                seen.add(key)

    redline_raw = raw.get("redline_bonus") if isinstance(raw.get("redline_bonus"), dict) else {}
    redline_base = fallback.get("redline_bonus", {})
    return {
        "enabled": _as_bool(raw.get("enabled"), bool(fallback.get("enabled", True))),
        "min_delta": max(-50, min(0, _as_int(raw.get("min_delta", fallback.get("min_delta", -20)), -20))),
        "max_delta": max(0, min(50, _as_int(raw.get("max_delta", fallback.get("max_delta", 10)), 10))),
        "redline_blocks_reduction": _as_bool(
            raw.get("redline_blocks_reduction"),
            bool(fallback.get("redline_blocks_reduction", True)),
        ),
        "redline_bonus": {
            "enabled": _as_bool(redline_raw.get("enabled"), bool(redline_base.get("enabled", True))),
            "delta": max(0, min(50, _as_int(redline_raw.get("delta", redline_base.get("delta", 10)), 10))),
            "description": str(redline_raw.get("description") or redline_base.get("description") or ""),
        },
        "factors": factors,
    }


def _normalize_symptom_labels(value: Any) -> Dict[str, Any]:
    fallback = DEFAULT_SCORING_CONFIG["symptom_labels"]
    raw = value if isinstance(value, dict) else {}
    raw_labels = raw.get("labels") if isinstance(raw.get("labels"), list) else fallback["labels"]
    fallback_by_key = {item["key"]: item for item in fallback["labels"]}
    labels: List[Dict[str, Any]] = []
    for item in raw_labels:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        base = fallback_by_key.get(key, {})
        keywords = _string_list(item.get("keywords"), base.get("keywords", []))
        exclude_keywords = _string_list(item.get("exclude_keywords"), base.get("exclude_keywords", []))
        if key == "positive_emotion":
            for word in base.get("keywords", []):
                if word not in keywords:
                    keywords.append(word)
            for word in base.get("exclude_keywords", []):
                if word not in exclude_keywords:
                    exclude_keywords.append(word)
        labels.append(
            {
                "key": key,
                "label": str(item.get("label") or base.get("label") or key),
                "category": str(item.get("category") or base.get("category") or ""),
                "weight": str(item.get("weight") or base.get("weight") or ""),
                "weight_label": str(item.get("weight_label") or base.get("weight_label") or item.get("weight") or ""),
                "risk_level": max(0, min(5, _as_int(item.get("risk_level", base.get("risk_level", 0)), 0))),
                "enabled": _as_bool(item.get("enabled"), bool(base.get("enabled", True))),
                "protective": _as_bool(item.get("protective"), bool(base.get("protective", False))),
                "modifier": _as_bool(item.get("modifier"), bool(base.get("modifier", False))),
                "description": str(item.get("description") or base.get("description") or ""),
                "keywords": keywords,
                "exclude_keywords": exclude_keywords,
                "dimension_keys": _string_list(item.get("dimension_keys"), base.get("dimension_keys", [])),
                "vector_queries": _string_list(item.get("vector_queries"), base.get("vector_queries", [])),
            }
        )
    if not labels:
        labels = deepcopy(fallback["labels"])
    else:
        seen_keys = {str(item.get("key") or "") for item in labels}
        for item in fallback["labels"]:
            key = str(item.get("key") or "")
            if key and key not in seen_keys:
                labels.append(deepcopy(item))
                seen_keys.add(key)

    raw_levels = raw.get("risk_levels") if isinstance(raw.get("risk_levels"), list) else fallback["risk_levels"]
    fallback_levels = {int(item["level"]): item for item in fallback["risk_levels"]}
    risk_levels: List[Dict[str, Any]] = []
    for item in raw_levels:
        if not isinstance(item, dict):
            continue
        level = max(0, min(5, _as_int(item.get("level"), 0)))
        base = fallback_levels.get(level, {})
        risk_levels.append(
            {
                "level": level,
                "label": str(item.get("label") or base.get("label") or f"等级 {level}"),
                "description": str(item.get("description") or base.get("description") or ""),
            }
        )
    if not risk_levels:
        risk_levels = deepcopy(fallback["risk_levels"])
    risk_levels.sort(key=lambda item: int(item.get("level") or 0))
    return {"enabled": _as_bool(raw.get("enabled"), True), "labels": labels, "risk_levels": risk_levels}


def _incoming_is_current_version(incoming: Dict[str, Any]) -> bool:
    return _as_int(incoming.get("version"), 0) >= SCORING_CONFIG_VERSION and isinstance(incoming.get("dimensions"), list)


def normalize_scoring_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    base = deepcopy(DEFAULT_SCORING_CONFIG)
    incoming = config if isinstance(config, dict) and isinstance(config.get("dimensions"), list) else {}

    base["version"] = SCORING_CONFIG_VERSION
    base["max_score"] = max(1, _as_int(incoming.get("max_score", base["max_score"]), 100))

    thresholds = incoming.get("thresholds") if isinstance(incoming.get("thresholds"), dict) else {}
    medium = max(1, min(base["max_score"], _as_int(thresholds.get("medium", base["thresholds"]["medium"]), 35)))
    high = max(medium, min(base["max_score"], _as_int(thresholds.get("high", base["thresholds"]["high"]), 65)))
    crisis = max(high, min(base["max_score"], _as_int(thresholds.get("crisis", base["thresholds"]["crisis"]), 85)))
    base["thresholds"] = {"medium": medium, "high": high, "crisis": crisis}
    base["evidence_strength"] = _normalize_evidence_strength(incoming.get("evidence_strength"))
    base["time_adjustment"] = _normalize_time_adjustment(incoming.get("time_adjustment"))
    base["protective_adjustment"] = _normalize_protective_adjustment(incoming.get("protective_adjustment"))
    base["symptom_labels"] = _normalize_symptom_labels(incoming.get("symptom_labels"))

    incoming_dimensions = incoming.get("dimensions")
    if not isinstance(incoming_dimensions, list):
        incoming_dimensions = []
    incoming_by_key = {str(item.get("key") or ""): item for item in incoming_dimensions if isinstance(item, dict)}

    normalized_dimensions: List[Dict[str, Any]] = []
    for fallback in DEFAULT_SCORING_CONFIG["dimensions"]:
        raw = incoming_by_key.get(fallback["key"], {})
        max_points = max(0, _as_int(raw.get("max_points", fallback["max_points"]), fallback["max_points"]))
        dimension = {
            "key": fallback["key"],
            "label": str(raw.get("label") or fallback["label"]),
            "description": str(raw.get("description") or fallback.get("description") or ""),
            "max_points": max_points,
            "enabled": _as_bool(raw.get("enabled"), bool(fallback.get("enabled", True))),
            "redline": _as_bool(raw.get("redline"), bool(fallback.get("redline", False))),
            "keywords": _string_list(raw.get("keywords"), fallback.get("keywords", [])),
            "exclude_keywords": _string_list(raw.get("exclude_keywords"), fallback.get("exclude_keywords", [])),
            "strong_keywords": _string_list(raw.get("strong_keywords"), fallback.get("strong_keywords", [])),
            "vector_queries": _string_list(raw.get("vector_queries"), fallback.get("vector_queries", [])),
            "subgroups": raw.get("subgroups") if isinstance(raw.get("subgroups"), dict) else deepcopy(fallback.get("subgroups", {})),
            "levels": _normalize_levels(raw.get("levels"), fallback.get("levels", []), max_points),
        }
        normalized_dimensions.append(dimension)
    base["dimensions"] = normalized_dimensions

    confidence = incoming.get("confidence") if isinstance(incoming.get("confidence"), dict) else {}
    base["confidence"].update(
        {
            "base": _as_float(confidence.get("base", base["confidence"]["base"]), 0.25),
            "message_cap": max(0, _as_int(confidence.get("message_cap", base["confidence"]["message_cap"]), 80)),
            "message_divisor": max(1, _as_int(confidence.get("message_divisor", base["confidence"]["message_divisor"]), 120)),
            "active_days_cap": max(0, _as_int(confidence.get("active_days_cap", base["confidence"]["active_days_cap"]), 14)),
            "active_days_divisor": max(1, _as_int(confidence.get("active_days_divisor", base["confidence"]["active_days_divisor"]), 35)),
        }
    )
    return base


def enabled_dimensions(config: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    scoring_config = normalize_scoring_config(config) if config else load_scoring_config()
    return [item for item in scoring_config.get("dimensions", []) if item.get("enabled", True)]


def vector_queries_from_config(config: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for dimension in enabled_dimensions(config):
        for query in dimension.get("vector_queries", []):
            text = str(query or "").strip()
            if text:
                out.append({"dimension_key": dimension["key"], "dimension_label": dimension["label"], "query": text})
    scoring_config = normalize_scoring_config(config) if config else load_scoring_config()
    symptom_labels = scoring_config.get("symptom_labels", {})
    if symptom_labels.get("enabled", True):
        for label in symptom_labels.get("labels", []):
            if not label.get("enabled", True):
                continue
            for query in label.get("vector_queries", []):
                text = str(query or "").strip()
                if text:
                    out.append(
                        {
                            "dimension_key": f"label:{label.get('key')}",
                            "dimension_label": str(label.get("label") or label.get("key") or "症状标签"),
                            "query": text,
                        }
                    )
    return out


def load_scoring_config() -> Dict[str, Any]:
    if not SCORING_CONFIG_PATH.exists():
        config = normalize_scoring_config(None)
        save_scoring_config(config)
        return config
    try:
        raw = json.loads(SCORING_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    config = normalize_scoring_config(raw)
    if not _incoming_is_current_version(raw):
        save_scoring_config(config)
    return config


def save_scoring_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_scoring_config(config)
    SCORING_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORING_CONFIG_PATH.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def reset_scoring_config() -> Dict[str, Any]:
    return save_scoring_config(DEFAULT_SCORING_CONFIG)
