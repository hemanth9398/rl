from setuptools import setup, find_packages

setup(
    name="rl-memory-agent",
    version="0.1.0",
    description="Self-evolving memory agent with NN policy, memory graph, and SymPy solver",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "sympy>=1.12",
        "networkx>=3.0",
        "numpy>=1.24",
        "rank_bm25>=0.2.2",
        "gymnasium>=0.29",
        "matplotlib>=3.7",
    ],
)
