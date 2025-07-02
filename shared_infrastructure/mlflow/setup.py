
from setuptools import setup, find_packages

setup(
    name="mlflow_setup",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mlflow",
        "psycopg2-binary",
        "boto3", # For S3 artifact storage
    ],
    entry_points={
        "console_scripts": [
            "mlflow_setup=mlflow_setup.main:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A setup script for MLflow dependencies",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/your_repo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)


