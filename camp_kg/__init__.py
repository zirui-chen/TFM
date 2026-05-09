from .generator import SyntheticKGGenerator, KGSplit

def __getattr__(name):
    if name == "SyntheticCorpus":
        from .corpus import SyntheticCorpus
        return SyntheticCorpus
    raise AttributeError(f"module 'camp_kg' has no attribute {name!r}")

__all__ = ["SyntheticKGGenerator", "KGSplit", "SyntheticCorpus"]
