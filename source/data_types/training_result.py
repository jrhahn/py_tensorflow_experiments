from dataclasses import dataclass


@dataclass
class TrainingResult:
    performance: float
    validation_performance: float
