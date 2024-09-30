from setuptools import setup

setup(
    name="snake_env",
    version="0.0.1",
    install_requires=open("requirements.txt").read().splitlines(),
)
