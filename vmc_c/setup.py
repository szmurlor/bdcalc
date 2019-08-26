from distutils.core import setup
from Cython.Build import cythonize
import numpy
print("Adding inlude_dirs for numpy to cython compilation process: %s" % numpy.get_include())
setup(
    name="create_pareto_vmc_c",
    version="1.0.0",
    ext_modules = cythonize("create_pareto_vmc_c.pyx"), 
    include_dirs=[numpy.get_include()]
)
