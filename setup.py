from setuptools import find_packages,setup

# This is kind of identifier for setup.py in requirements.txt
HYPHEN_DOT_E="-e ."

def get_requirements(file_path):
    """This file will return a list"""
    
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPHEN_DOT_E in requirements:
            requirements.remove(HYPHEN_DOT_E)
        
    return requirements

setup(
    name="ML_Project",
    version="0.0.1",
    author="aswinvrpai",
    author_email="aswin.199.vr@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
) 