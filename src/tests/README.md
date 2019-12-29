Przykłady użycia

Wczytanie danych o wokselach i przynależności do roiów pozwala obejrzeć je w ParaView (http://paraview.org):

![demo_pv](demo-pv.gif)

Wczytywanie dawek.

uwaga sa dwie wersje:

1. operującą na plikach tekstowych `d_*.txt` (bardzo powoli!)

2. operująca na plikach binarnych `d_*.npbin` (wykorzystuje Pandas, oraz konieczne jest przekonwertowanie plików tekstowych `d_*.txt` do formatu binarnego `d_*.npbin`. Przygotowałem do tego narzędzie w języku c o nazwie doses_convert.c.

Kompilacja narzędzia do konwersji plików tekstwowych do formatu binarnego:

```bash
gcc convert_doses.c -o convert_doses
```

potem trzeba każdy plik przekonwertować tak:

```bash
for i in 1 2 3 4 5 6 7 8 9; do ./convert_doses /doses-nfs/sim/pacjent_5/output/PARETO_5/d_PARETO_5_$i.txt /doses-nfs/sim/pacjent_5/output/PARETO_5/d_PARETO_5_$i.npbin; done
```

Ważne aby pliki binarne były w tym samym katalogu co oryginalne.

Potem uruchamiamy skrypt do tworzenia VTI (uwaga wywołanie trochę się zmieniło):

```bash
CP=/doses-nfs/sim/pacjent_5/output/PARETO_5; python3 tests/recover_plan_ct.py $CP/m_PARETO_5.txt ala2
```

Przykładowo wczytane dawki:

![pv_doses](pv-doses.gif)