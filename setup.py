from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()



AUTHOR_NAME = "Durgesh Singh"
REPO_NAME = "End-to-End-SkillScan-AI-Resume-Analyzer"
AUTHOR_USER_NAME = "Durgeshsingh12712"
SRC_REPO = "skillScan"
AUTHOR_EMAIL = "durgeshsingh12712@gmail.com"
__version__ = "0.1.0"


setup(
    name = SRC_REPO,
    version = __version__,
    author=AUTHOR_NAME,
    author_email= AUTHOR_EMAIL,
    description= "A medium-complexity Python package",
    long_description = long_description,
    long_description_content_type= "text/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    license="MIT",
    keywords= ["python", "package"],
    project_urls = {
        "Homepage" : f"https://github.com/{AUTHOR_USER_NAME}",
        "Bug Reports" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
        "Source" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}" ,
        "Documentation" : f"https://{REPO_NAME}.readthedocs.io",
        "Changelog" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/blob/main/CHANGELOG.md",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    package_data= {
        SRC_REPO: [
            "artifacts/*.csv",
            "artifacts/model_evaluation/*.json",
            "config/*.yaml",
            "templates/*.html",
            "*.txt",
            "static/*.css",
            "static/*.js"
        ]
    },
    packages= find_packages(),
    install_requires = []
)