#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_help() {
	printf("usage: convdoses NO_BEAMS D_PID_ [debug_head]\n\n"
	       "The program will read text files matching the pattern D_PID_{num}.txt and\n"
		   "save converted textual data to binary format file with name D_PID_{num}.nparray\n"
		   "The format of the binary files is:\n"
		   "   [4 bytes for integer with number of rows]\n"
		   "   [12 bytes for triples of integer numbers] * numer of rows\n"
		   "\n"
		   "The total length of each binary file is: 4+rows*4*3\n\n"
		   "Options are:\n"
		   "    NO_BEAMS - number of files with dose data\n"
		   "    D_PID_ - prefix of the file names which need to converted\n"
		   "    debug_head - optional number of first rows to be printed to the stdio\n"
		   "                 for each beam for the purpose of debuging\n"
		   );
}

int main(int argc, char **argv) {
	FILE *fin;
	FILE *fo;
	int i,j,n,c;
	char fname_in[255];
	char fname_out[255];
	int d[3];
	int debug_head = 0;

	if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
		print_help();
		return 0;
	}

	if (argc < 3) {
		print_help();
		exit(-3);
	}

	if (argc > 3) {
		debug_head = atoi(argv[3]);
	}	


	for (i = 0; i < atoi(argv[1]); i++) {
		sprintf(fname_in, "%s%d.txt", argv[2], i+1);
		sprintf(fname_out, "%s%d.nparray", argv[2], i+1);

		printf("Analysing file: %s\n", fname_in); 
		if ((fin = fopen(fname_in,"r")) == NULL) {
			printf("ERROR! Can not open input file with text data: %s\n", fname_in);
			exit(-1);
		}
		if ((fo = fopen(fname_out,"wb")) == NULL) {
			printf("ERROR! Unable to open output binary file for writing: %s\n", fname_out);
			exit(-1);
		}
		fscanf(fin, "%d",&n);
		fwrite(&n,sizeof(n),1,fo);

		for (j=0;j<n;j++) {
			if ((c = fscanf(fin, "%d %d %d", &d[0], &d[1], &d[2])) != 3) {
				printf("ERROR! In line %d. Expected 3 integer numbers got %d.\n", j+1, c);
				exit(-2);
			}
			fwrite(d, sizeof(d[0]), 3, fo);

			if (j < debug_head) {
				printf("%d %d %d\n", d[0], d[1], d[2]);
			}
		}
		fclose(fin);
		fclose(fo);		
	}

	return 0;
}
