from Cython.Build import cythonize
import numpy
from setuptools import Extension, setup

ext = Extension(
    "tandem_rng._tandem",
    ["src/tandem_rng/_tandem.pyx", "src/tandem_rng/c/tandem.c"],
    include_dirs=["src/tandem_rng/c", numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION")],
    extra_compile_args=["-O2"],
)

setup(ext_modules=cythonize([ext], language_level=3))
