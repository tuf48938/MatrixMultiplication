#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define min(x, y)((x) < (y) ? (x) : (y))

int main(int argc, char * argv[]) {
    int n;
    double * aa, * b, * c;
    double * buffer, ans;
    double * times;
    double total_times;
    int run_index;
    int nruns;
    int myid, master, numprocs;
    double starttime, endtime;
    MPI_Status status;
    int i, j, k, numsent, sender;
    int anstype, row;
    srand(time(0));
    MPI_Init( & argc, & argv);
    MPI_Comm_size(MPI_COMM_WORLD, & numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, & myid);

    if (argc > 1) {
        n = atoi(argv[1]);
        aa = (double * ) malloc(sizeof(double) * n * n);
        b = (double * ) malloc(sizeof(double) * n * n);
        c = (double * ) malloc(sizeof(double) * n * n);
        buffer = (double * ) malloc(sizeof(double) * n);
        master = 0;

        if (myid == master) {
            // Master Code goes here
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    aa[i * n + j] = (double) rand() / RAND_MAX + 1;
                    b[i * n + j] = (double) rand() / RAND_MAX + 1;
                }
            }

            // Printing first matrix
            printf("First Array\n");
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    printf("[%f]", aa[i * n + j]);
                }
                printf("\n");
            }

            // Printing second matrix
            printf("Second Array\n");
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    printf("[%f]", b[i * n + j]);
                }
                printf("\n");
            }

            starttime = MPI_Wtime();
            numsent = 0;

            MPI_Bcast(b, n, MPI_DOUBLE, master,
            MPI_COMM_WORLD);

            MPI_Bcast(aa, n, MPI_DOUBLE, master,
            MPI_COMM_WORLD);

            for (i = 0; i < min(numprocs - 1, n); i++) {
                for (j = 0; j < n; j++) {
                    buffer[j] = aa[i * n + j];
                }
                MPI_Send(buffer, n, MPI_DOUBLE, i + 1, i + 1,
                MPI_COMM_WORLD);
                numsent++;
            }

            for (i = 0; i < n; i++) {
                MPI_Recv( & ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE,
                MPI_ANY_TAG,
                MPI_COMM_WORLD, & status);
                sender = status.MPI_SOURCE;
                anstype = status.MPI_TAG;
                c[anstype - 1] = ans;
                if (numsent < n) {
                    for (j = 0; j < n; j++) {
                        buffer[j] = aa[numsent * n + j];
                    }
                    MPI_Send(buffer, n, MPI_DOUBLE, sender,
                numsent + 1,
                MPI_COMM_WORLD);
                numsent++;
                } else {
                    MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, sender, 0,
                    MPI_COMM_WORLD);
                }
            }
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    c[i*n + j] = 0;
                }
                for (k = 0; k < n; k++) {
                    for (j = 0; j < n; j++) {
                        c[i* n + j] += aa[i* n + k] * b[k* n + j];
                    }
                }
            }

            // Printing final matrix
            printf("First x Second Array\n");
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    printf("[%f]", c[i * n + j]);
                }
                printf("\n");
            }

            endtime = MPI_Wtime();
            printf("%f\n", (endtime - starttime));

        } else {
            // Slave Code goes here
            MPI_Bcast(b, n, MPI_DOUBLE, master,
            MPI_COMM_WORLD);
            MPI_Bcast(aa, n, MPI_DOUBLE, master,
            MPI_COMM_WORLD);

            if (myid <= n) {
                while (1) {
                    MPI_Recv(buffer, n, MPI_DOUBLE, master,
                    MPI_ANY_TAG,
                    MPI_COMM_WORLD, & status);
                    if (status.MPI_TAG == 0) {
                        break;
                    }
                    row = status.MPI_TAG;
                    ans = 0.0;
                    for (j = 0; j < n; j++) {
                        ans += buffer[j] * b[j];
                    }
                    MPI_Send( & ans, 1, MPI_DOUBLE, master, row,
                    MPI_COMM_WORLD);
                }
            }
        }
    } else {
        fprintf(stderr, "Usage matrix_times_vector <size>\n");
    }
    
    MPI_Finalize();
    return 0;
}