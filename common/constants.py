import enum

DEFAULT_STYLE: str = "Neutral"
DEFAULT_STYLE_WEIGHT: float = 5.0


class Languages(str, enum.Enum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"


DEFAULT_SDP_RATIO: float = 0.2
DEFAULT_NOISE: float = 0.6
DEFAULT_NOISEW: float = 0.8
DEFAULT_LENGTH: float = 1.0
DEFAULT_LINE_SPLIT: bool = True
DEFAULT_SPLIT_INTERVAL: float = 0.5
DEFAULT_ASSIST_TEXT_WEIGHT: float = 0.7
DEFAULT_ASSIST_TEXT_WEIGHT: float = 1.0
