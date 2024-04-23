import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stabilized-ica",
    version="2.0.0",
    author="Nicolas Captier",
    author_email="nicolas.captier@curie.fr",
    description="Stabilized ICA algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ncaptier/stabilized-ica",
    packages=setuptools.find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "joblib>=1.1.0",
        "networkx>=2.7.1",
        "pandas>=1.4.2",
        "python-picard>=0.7",
        "scikit-learn>=1.2.0",
        "tqdm>=4.64.0",
        "umap-learn>=0.5.3",
    ],
    extras_require={
        "dev": ["pytest"],
        "docs": [
            "sphinx == 5.0.2",
            "sphinx-gallery == 0.10.0",
            "numpydoc == 1.2",
            "nbsphinx == 0.8.9",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
