# setup.py
# Makes Tark AI an installable package so `from config.config import ...` works
# from anywhere. Install once in editable mode from the project root:
#     pip install -e .

from pathlib import Path
from setuptools import setup, find_packages

# Read the dependency list from requirements.txt so we only maintain it once.
requirements_file = Path(__file__).parent / "requirements.txt"
install_requires = [
    line.strip()
    for line in requirements_file.read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="tark_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
)
