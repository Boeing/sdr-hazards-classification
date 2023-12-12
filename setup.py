from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='sdr_classifier',
    version='0.0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["pandas>=1.1.5", "scikit-learn>=0.24.1", "xgboost"],
    #packages=setuptools.find_packages(),
    url='',
    description="Package to create SDR Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #license='Apache License 2.0',
    author='Hai Nguyen',
    author_email='hai.c.nguyen@boeing.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
