from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sem-morphology-analyzer",
    version="1.0.0",
    author="Hafiz Asad Ullah Sajid",
    author_email="hafizasadullahsajid.iub@gmail.com",
    description="A comprehensive Python toolkit for quantitative SEM surface morphology analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asadullahsajid/SEM-Soil-Morphology-Analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sem-analyzer=src.sem_morphology_analyzer:main",
        ],
    },
    keywords="SEM, microscopy, image analysis, surface morphology, materials science, soil science",
    project_urls={
        "Bug Reports": "https://github.com/asadullahsajid/SEM-Soil-Morphology-Analyzer/issues",
        "Source": "https://github.com/asadullahsajid/SEM-Soil-Morphology-Analyzer",
        "Documentation": "https://github.com/asadullahsajid/SEM-Soil-Morphology-Analyzer/blob/main/docs/SEM_Analysis_Methodology.md",
    },
)
