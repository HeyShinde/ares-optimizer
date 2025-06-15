from setuptools import setup, find_packages

setup(
    name="ares-optimizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.3.0",
        "psutil>=7.0.0",
        "pytest>=8.4.0",
        "torch>=2.7.1",
        "matplotlib>=3.10.2",
        "seaborn>=0.13.2",
        "streamlit>=1.45.1",
    ],
    extras_require={
        "dev": [
            "black>=25.1.0",
            "ruff>=0.11.13"
        ],
    },
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="An autonomous reinforcement learning agent for algorithmic optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ares-optimizer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 