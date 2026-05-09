from setuptools import setup, find_packages

setup(
    name="camp-kg",
    version="1.0.0",
    description="CAMP-KG: Corpus-as-Model Pretraining for Zero-Shot Knowledge Graph Reasoning",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "torch>=2.0",
        "torch-geometric>=2.3",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
)
