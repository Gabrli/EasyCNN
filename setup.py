from setuptools import setup, find_packages

setup(
    name="easycnn",
    version="1.0.0",
    description="Easy tool to create own cnn in 2 minutes with minmum code and a lot of functions.",
    packages=find_packages(),
    author="Gabriel Wiśniewski",
    author_email="gabrys.wisniewski@op.pl",
    classifiers=[  
        "Programming Language :: Python :: 3",  
        "License :: MIT License",  
        "Operating System :: OS Independent :: Windows", 
    ],
     install_requires=[
        "tensorflow",
        "numpy",
        "matplotlib",
    ],
    
)