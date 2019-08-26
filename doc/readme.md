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

Obsługa Dockera:

```bash
cd docker
docker build --tag=szmurlor:bdcalc-0.1 .

docker start first # first to jest nazwa kontenera można użyć też ID
docker exec -t -i first "/bin/bash"

```

Montowanie zasobu NFS:

```bash
$ docker volume create \
    --name mynfs \
    --opt type=nfs \
    --opt device=:<nfs export path> \
    --opt o=addr=<nfs host> \
    mynfs
$ docker run -it -v mynfs:/foo alpine sh
```

Na przykład tak:

```bash
docker volume create --driver local --opt type=nfs --opt o=addr=goope-nas-2 --opt device=:/nfs/doses doses-nf
```

Inny przykład (bliższy naszej konfiguracji):

```bash
docker run -ti -v doses-nfs:/doses-nfs bdcalc:0.1
```

Tworzymy teraz kontener:

```bash
docker create -ti  -v doses-nfs:/doses-nfs --name doses-ham-c1 bdcalc:0.1
```

Budowanie obrazu (najlepiej będąc w główny katalogu) - warto teraz dodać builddocker.sh

```bash
docker build -t bdcalc:0.1 -f docker/Dockerfile .
```

