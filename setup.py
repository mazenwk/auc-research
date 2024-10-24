from setuptools import setup, find_packages

setup(
    name="LokiPedCrop",  # Adjust the name to your chosen one
    version="0.1.0",
    description="A library for LOKI dataset cropping pedestrians from point clouds and exporting to PLY files",
    author="Mazen W. Kamel",
    # author_email="your.email@example.com",
    packages=find_packages(),  # Automatically finds all Python packages in your project
    install_requires=[
        'numpy==2.1.1',
        'open3d==0.18.0',
        'pandas==2.2.3',
        'Pillow==10.4.0',
        'plyfile==1.1',
        'torch==2.4.1',
    ],
    include_package_data=True,  # Ensure non-code files are included if necessary
    # package_data={
    #     # Specify patterns to include any specific non-code files if necessary
    #     "": ["*.ply"],  # E.g., if you're including sample PLY files for testing
    # },
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",  # Or any other license you're using
    #     "Operating System :: OS Independent",
    # ],
    python_requires='>=3.10',  # Specify the Python versions you're supporting
)
