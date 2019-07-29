from distutils.core import setup
from Cython.Build import cythonize
import numpy
print(numpy.get_include())
setup(
    ext_modules = cythonize("create_pareto_vmc_c.pyx"), 
    include_dirs=[numpy.get_include()]
)
