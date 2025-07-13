SEAHELM_FILE_IDENTIFIER = "seahelm"
OPENLLM_FILE_IDENTIFIER = "hf_leaderboard"
SEA_LANGUAGES_COLUMN = "Supported SEA Languages"
OPENLLM_LEADERBOARD_COLUMN = "Open LLM Leaderboard 2 (EN)"

PLOT_COLORMAP = "Mint"

RELEVANT_LANGS = ["ID", "VI", "TH", "TA", "TL", "SU", "JV", "MS", "MY", "KM", "ZH"]
LANGUAGE_NAMES = {
    "ID": "Indonesian",
    "VI": "Vietnamese",
    "TH": "Thai",
    "TA": "Tamil",
    "TL": "Tagalog",
    "SU": "Sundanese",
    "JV": "Javanese",
    "MS": "Malay",
    "MY": "Burmese",
    "KM": "Khmer",
    "ZH": "Chinese",
    "EN": "English",
    "sea_total": "SEA Avg",
    "total": "Aggregate"
}

SUPPORTED_LANGUAGES = {
    "Qwen2-": ["EN", "ZH", "VI", "TH", "ID", "MS", "LO", "MY", "KM", "TL"],
    "SeaLLMs-v3-": [
        "EN",
        "ZH",
        "ID",
        "VI",
        "TH",
        "TL",
        "MS",
        "MY",
        "KM",
        "LO",
        "TA",
        "JV",
    ],
    "aya-23-": ["EN", "ZH", "ID", "VI"],
    "aya-expanse-": ["EN", "ZH", "ID", "VI"],
    "sailor-": ["EN", "ID", "TH", "VI", "MS", "LO"],
    "sailor2-": [
        "EN",
        "ZH",
        "MY",
        "ID",
        "JV",
        "KM",
        "LO",
        "MS",
        "SU",
        "TL",
        "TH",
        "VI",
    ],  # English, Chinese, Burmese, Cebuano, Ilocano, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tagalog, Thai, Vietnamese, and Waray
    "Llama-3.1": ["EN", "TH"],
    "Llama-3.3": ["EN", "TH"],
    "sea-lionv2.1": ["EN", "ID", "TH", "VI", "TA"],
    "sea-lionv3": [
        "EN",
        "ZH",
        "VI",
        "ID",
        "TH",
        "TL",
        "TA",
        "MS",
        "KM",
        "LO",
        "MY",
        "JV",
        "SU",
    ],
}

class BhasaConfig:
    """Configuration class for Bhasa report generation."""

    LANGUAGES = ["id", "vi", "th", "ta", "en", "km", "ms", "my", "zh"]

    SKIP_LANGUAGES = ["ms", "my", "zh", "km", "en"]

    SCORE_NAMES = [
        "normalized_accuracy",
        "normalized_f1",
        "normalized_chrf_score",
        "weighted_win_rate",
        "overall_lang_normalized_acc",
    ]


class EnConfig:
    CHOICE_COUNTS = {
        "leaderboard_bbh": {
            "leaderboard_bbh_sports_understanding": 2,
            "leaderboard_bbh_tracking_shuffled_objects_three_objects": 3,
            "leaderboard_bbh_navigate": 2,
            "leaderboard_bbh_snarks": 2,
            "leaderboard_bbh_date_understanding": 6,
            "leaderboard_bbh_reasoning_about_colored_objects": 18,
            "leaderboard_bbh_object_counting": 19,
            "leaderboard_bbh_logical_deduction_seven_objects": 7,
            "leaderboard_bbh_geometric_shapes": 11,
            "leaderboard_bbh_web_of_lies": 2,
            "leaderboard_bbh_movie_recommendation": 6,
            "leaderboard_bbh_logical_deduction_five_objects": 5,
            "leaderboard_bbh_salient_translation_error_detection": 6,
            "leaderboard_bbh_disambiguation_qa": 3,
            "leaderboard_bbh_temporal_sequences": 4,
            "leaderboard_bbh_hyperbaton": 2,
            "leaderboard_bbh_logical_deduction_three_objects": 3,
            "leaderboard_bbh_causal_judgement": 2,
            "leaderboard_bbh_formal_fallacies": 2,
            "leaderboard_bbh_tracking_shuffled_objects_seven_objects": 7,
            "leaderboard_bbh_ruin_names": 6,
            "leaderboard_bbh_penguins_in_a_table": 5,
            "leaderboard_bbh_boolean_expressions": 2,
            "leaderboard_bbh_tracking_shuffled_objects_five_objects": 5,
        },
        "leaderboard_musr": {
            "leaderboard_musr_murder_mysteries": 2,
            "leaderboard_musr_object_placements": 5,
            "leaderboard_musr_team_allocation": 3,
        },
        "leaderboard_gpqa": {
            "leaderboard_gpqa_main": 4,
            "leaderboard_gpqa_diamond": 4,
            "leaderboard_gpqa_extended": 4,
        },
    }

    TASKS = (
        "leaderboard_mmlu_pro",
        "leaderboard_bbh",
        "leaderboard_gpqa",
        "leaderboard_math_hard",
        "leaderboard_ifeval",
        "leaderboard_musr",
    )

    OUTPUT_NAMES = {
        "leaderboard_mmlu_pro": "MMLU-PRO",
        "leaderboard_bbh": "BBH",
        "leaderboard_gpqa": "GPQA",
        "leaderboard_math_hard": "MATH Lvl 5",
        "leaderboard_ifeval": "IFEval",
        "leaderboard_musr": "MUSR",
    }
