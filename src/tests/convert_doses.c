#include <stdio.h>
#include <stdlib.h>
/**
najlepiej skompilować:

gcc convert_doses.c -o convert_doses

potem przykładowe użycie: 

./convert_doses /doses-nfs/sim/pacjent_5/output/PARETO_5/d_PARETO_5_8.txt /doses-nfs/sim/pacjent_5/output/PARETO_5/d_PARETO_5_8.npbin

*/

int main (int argc, char **argv) {
    FILE * f = fopen(argv[1], "r");
    FILE * fout = fopen(argv[2], "w");
    int count,i;
    int *v, *b;
    float *d;

    fscanf(f, "%d", &count);
    v = (int*) malloc(sizeof(int) * count);
    b = (int*) malloc(sizeof(int) * count);
    d = (float*) malloc(sizeof(float) * count);

    for (i = 0; i < count; i++) {
        fscanf(f, "%d %d %f", &v[i], &b[i], &d[i]);
    }

    fwrite(&count, sizeof(count),1, fout);
    fwrite(v, sizeof(int), count, fout);
    fwrite(b, sizeof(int), count, fout);
    fwrite(d, sizeof(float), count, fout);
    fclose(fout);
    fclose(f);

    free(v);
    free(b);
    free(d);
    return 0;
}