from src.processing.strategies.base import ChunkingStrategy
from src.processing.strategies.fixed import FixedSizeStrategy
from src.processing.strategies.sliding import SlidingWindowStrategy
from src.processing.strategies.structural import RecursiveStructureStrategy

class StrategyFactory:
    @staticmethod
    def get_strategy(strategy_type: str, **kwargs) -> ChunkingStrategy:
        if strategy_type == "fixed":
            return FixedSizeStrategy(**kwargs)
        elif strategy_type == "sliding":
            return SlidingWindowStrategy(**kwargs)
        elif strategy_type == "structural":
            return RecursiveStructureStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy_type}")
