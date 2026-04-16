import setuptools


# ------ Call "python setup.py bdist_wheel" to create wheels. ------#


# Setup the package.
setuptools.setup(
    name="chemical_analysis_mobile",
    version="1.8.5",
    author="Prograf/UFF",
    author_email="prograf@ic.uff.br",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"chemical_analysis._resources": ["*.*"]},
    install_requires=[
        "opencv-python",
        "scipy",
        "torch",
        "torchvision",
        "fft_conv_pytorch",
        "tqdm"
    ],
    zip_safe=False,
)
