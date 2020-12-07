import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sica",
    version="0.0.1",
    author="Nicolas Captier",
    author_email="nicolas.captier@curie.fr",
    description="Stabilized ICA algorithm and applications to single-cell data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ncaptier/Stabilized_ICA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
