import setuptools


__packagename__ = 'ring-offload-memory-efficient'

setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version='0.1',
    python_requires='>=3.8',
    description='Ring Attention and Offload Attention with Memory Efficient Xformers',
    author='huseinzol05',
    url='https://github.com/mesolitica/ring-offload-memory-efficient-xformers',
)
