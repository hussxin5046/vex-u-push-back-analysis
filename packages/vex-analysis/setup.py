"""
VEX U Push Back Strategic Analysis Toolkit Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="vex-u-push-back-analysis",
    version="1.0.0",
    author="VEX U Analysis Team",
    description="Comprehensive Python system for analyzing VEX U Push Back game strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hussxin5046/vex-u-push-back-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Games/Entertainment :: Simulation",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'vex-analysis=vex_analysis.main:main',
        ],
    },
    package_data={
        'vex_analysis': ['**/*.py', '**/*.json', '**/*.pkl'],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="vex robotics strategy analysis simulation monte-carlo",
    project_urls={
        "Bug Reports": "https://github.com/hussxin5046/vex-u-push-back-analysis/issues",
        "Source": "https://github.com/hussxin5046/vex-u-push-back-analysis",
        "Documentation": "https://github.com/hussxin5046/vex-u-push-back-analysis/blob/main/README.md",
    }
)