from setuptools import setup, find_packages

setup(
    name="visionflow",
    version="0.1.0",
    description="A modular pipeline for face verification dataset manipulation.",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "aiohttp",
        "pandas",
        "tqdm",
        "Pillow",
        "numpy"
    ],
)