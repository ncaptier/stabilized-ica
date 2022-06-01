import setuptools
import os

version = None
with open(os.path.join('sica', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stabilized-ica",
    version=version,
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
        "scikit-learn>=1.0.2",
        "tqdm>=4.64.0",
        "umap-learn>=0.5.3",
    ],
    extra_requires={"dev": ["pytest"],
                    "doc": ["sphinx >= 3.2.1", "sphinx-gallery >= 0.9.0", "numpydoc >= 1.1.0", "nbsphinx >= 0.8.7"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
