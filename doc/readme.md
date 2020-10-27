## Przetwarzanie plików do uczenia sieci neuronowych

1. Zakładam że jest jakiś folder nadrzędny, a w nim znajdują się podfoldery,
które zawierają dane do analizy. 
2. Zakładam, że każdy podfolder zawiera strukturę: `input, processing, output`
3. W pliku `input/roi_mapping.json` może znajdować się mapowanie ROIów na
jakieś numery. Wtedy dla każdego przypadku spróbujemy uruchomić mapowanie
znajdujących się w tym plików ROIów na jakieś liczby i zostaną utworzone
obrazki PNG.


Możemy uruchamiać na przykład tak:

```
python3 bd4cnn_process.py /home/username/sims/root_folder_with_subfolders
```

Wtedy scenariusz może być taki:

1. Przechodzimy po wszystkich podkatalogach i sprawdzamy czy dane się przetworzone - czy utworzony jest określony zestaw plików w każdym `output` Jako wymagany zestaw plików przyjmujemy:
    
    ```bash
    approximated_ct.nparray
    total_doses.nparray
    roi_marks.nparray
    roi_mapping.txt
    ```

    Jeżeli będzie brakować, któregokolwiek pliku to zapuszczamy w tym katalogu `bd4cnn.py`.

2. Przejdziemy przez wszystkie podkatalogi i zbudujemy słownik z informacjami o wymiarach i parametrach danych - potrzebujemy zapisywać origin, spacing, dimensions?

3. Znajdziemy najmniejszy rozmiar (dimensions), który jesteśmy w stanie zbudować. (To wszystko w pamięci skryptu Python.)

4. Przeiterujemy przez wszystkie przypadki i dla każdego spróbujemy obciąć go do rozmiaru minimalnego. Wyniki zapiszemy do plików:

    ```bash
    approximated_ct_cropped.nparray
    total_doses_cropped.nparray
    roi_marks_cropped.nparray
    ```

5. W zależności od tego, czy użytkownik zażyczy sobie wygenerowanie obrazów PNG to wygenerujemy je w podfolderach:

    ```bash
    images_ct_cropped
    images_total_doses_cropped
    images_roi_marks_cropped
    ```

## Inne

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

