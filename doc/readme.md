Dodać do PATH folder ~/.local/bin - tam zainstalowany jest ray. Wtedy będzie można uruchomić serwer:

```bash
ray start --head --redis-port=59999
```


Instalacja vmc_c w katalogu: vmc_c:

```bash 
python3 setup.py install --user
```

Uwaga do standardowego setup.py musiałem dodać:

```python
import numpy
...

setup(
    ext_modules = cythonize("create_pareto_vmc_c.pyx"), 
    include_dirs=[numpy.get_include()]
)
...
```