from setuptools import setup, find_packages

setup(
    name="seisjax",
    version="0.1.0",
    author="Saleh Alatwah",
    author_email="atwahsz@gmail.com",
    description="A JAX-powered library for high-performance seismic attribute computation.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/atwahsz/SeisJAX",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.8',
) 