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

## Podpowiedzi dla początkujących użytkowników ray:

1. Dzielenie na większe kawałki (nakład na komunikację między-procesową, przesyłanie danych itp.)
2. Czekanie na najkrótsze zadanie i automatyczna jego obsługa (```ray.wait()```)
3. ray ```autoscaler``` - narzędzie umożliwiające automatyczne uruchamianie kontenerów dockera na wskazanych węzłach

https://rise.cs.berkeley.edu/blog/ray-tips-for-first-time-users/



## Aktualizacja węzłów ray-x

```
   46  sudo hostnamectl set-hostname ray-3
   47  sudo vim /etc/cloud/cloud.cfg
   54  scp ham-10:.ssh/id_rsa.pub .ssh/authorized_keys
```

