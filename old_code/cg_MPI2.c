#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*Assuming N = P*j*/
void ConjugateGradient(double** local_A, double* b, double* x, int N, int Iters, int rank, int P){
    /* Receive A part och p*/

    int n, rr, rrold, global_idx, local_r, local_p, Ap, pAp, alpha, beta;

    n = N/size;
    global_idx = rank * n;

    double* p = (double*)malloc(N * sizeof(double));
    double* local_r = (double*)malloc(n * sizeof(double));
    double* local_A = (double*)malloc(n * N * sizeof(double));
    double* local_Ap = (double*)malloc(n * sizeof(double));

    rr = 0.0;
    for (int i = 0; i < n; i++){
        rr += local_r[i] * local_r[i];
    }
    MPI_Allreduce(&rr, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < Iters; i++) {
        
        pAp = 0.0;
        for (int i = 0; i < n; i++) {
            local_Ap[i] = 0.0;
            for (int j = 0; j < N; j++) {
                local_Ap[i] += local_A[i*N+j] * p[j];
            }
            pAp += p[global_idx + i] * local_Ap[i];
        }
        MPI_Allreduce(&pAp, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        alpha = rr / pAp;
        rrold = rr;

        rr = 0.0;
        for (int i = 0; i < n; i++) {
            x[global_idx + i] += alpha * local_r[i];
            local_r[i] -= alpha * local_Ap[i];
            rr += local_r[i] * local_r[i];
        }

        MPI_Allreduce(&rr, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        beta = rr / rrold;

        for (int i = 0; i < n; i++) {
            p[global_idx + i] = local_r[i] + beta * p[global_idx + i];
        }

        MPI_Allreduce(p, p, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    }

    MPI_Allreduce(x, x, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    free(p);
    free(local_r);
    free(local_A);
    free(local_Ap);

}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size, n, N;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* x = (double*)malloc(n * sizeof(double));

    if (rank == 0) {
        /* Initialize A, b and x*/
    } else {
        /* Receive A, b and x*/
    }

    ConjugateGradient(A, b, x, N, Iters, rank, size);
    
    MPI_Finalize();

    free(x);
    free(A);
    free(b);

}