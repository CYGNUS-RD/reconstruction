from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "cython_cygno",
        ["cython_cygno.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions),
)
