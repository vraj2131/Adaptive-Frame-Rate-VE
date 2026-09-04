from setuptools import setup, find_packages

setup(
    name="adaptive_frame_rate_engine",
    version="0.1.0",
    description="Adaptive Frame Rate Engine for vision model inference",
    author="AFRE Contributors",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "opencv-python",
        "psutil",
        "matplotlib",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "afre=adaptive_frame_rate_engine.main:main",
        ]
    },
    python_requires=">=3.9",
)
