from setuptools import setup, find_packages

setup(
    name="mtl_causalte",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "joblib",
        "numpy",
        "scikit-learn",
        "pandas",
        # add other dependencies here
    ],
)
