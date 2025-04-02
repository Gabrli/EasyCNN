from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="easycnnpro",
    version="1.0.1",
    description="EasyCnnPro: Intuitive Python library for creating, training and visualizing convolutional neural networks. Features simplified CNN layer definition, automated training workflows, model visualization, and seamless Keras-to-ONNX conversion. Includes 15 pre-configured popular models for immediate use.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://github.com/Gabrli/EasyCNN---docs",
        "Source Code" :"https://github.com/Gabrli/easyCNN",
    },
    keywords="deep-learning, cnn, neural-networks, tensorflow, keras, machine-learning, ai, computer-vision, image-processing, model-training, visualization, easy-to-use, python, convolutional-networks",
    packages=find_packages(),
    author="Gabriel Wiśniewski",
    author_email="gabrys.wisniewski@op.pl",
    classifiers=[  
        "Programming Language :: Python :: 3",  
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent", 
    ],
     install_requires=[
        "tensorflow",
        "numpy",
        "matplotlib",
        "opencv-python",
        "tf2onnx",
        "onnx"
    ],
    entry_points={
        "console_scripts":[
            "easycnn-core = easycnn:EasyCNN",
            "easycnn-preset = easycnn:Preset",
            "easycnn-visualizer = easycnn:TrainingVisualizer",
            "easycnn-exporter = easycnn:EasyExporter",
            
        ]
    }
)