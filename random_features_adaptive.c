#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <complex.h>


double get_random(double sigma){
    return sigma * (double)rand() / (double)RAND_MAX;
}

void local_matmul(double complex* x, double complex* local_A, double complex* b, int n, int N, int rank) {
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        for (int j = 0; j < N; j++) {
            x[i] += local_A[i * N + j] * b[j];
        }
    }
}

void get_local_A(double complex* local_A, double* w, double* data_x, int n, int N, int Ndata, int rank){
    double temp;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < N; j++){
            local_A[i*N+j] = 0.0;
            for (int k = 0; k < Ndata; k++){
                temp = (w[j] - w[rank*n + i]) * data_x[k];
                local_A[i*N+j] += cos(temp) + I * sin(temp);
            }
        }
    }
}

void evaluate_fourier(double* y, double* x_eval, double complex* coef, double* w, int N, int Ndata){
    double tmp;
    for (int i = 0; i < Ndata; i++){
        y[i] = 0.0;
        for (int j = 0; j < N; j++){
            tmp = w[j] * x_eval[i];
            y[i] += creal(coef[j]) * cos(tmp) - cimag(coef[j]) * sin(tmp);
        }
    }
}

void get_local_b(double complex* local_b, double* w, double* data_x, double* data_y, int n, int N, int Ndata, int rank){
    double temp;
    for (int i = 0; i < n; i++){
        local_b[i] = 0.0;
        for (int k = 0; k < Ndata; k++){
            temp = w[rank*n + i] * data_x[k];
            local_b[i] += data_y[k] * (cos(temp) - I * sin(temp));
        }
    }
}

void complex_sum(void* in, void* inout, int* len, MPI_Datatype* dptr){
    double complex* in_data = in;
    double complex* inout_data = inout;
    for (int i = 0; i < *len; i++){
        inout_data[i] += in_data[i];
    }

}

MPI_Datatype get_mpi_complex_double(){
    MPI_Datatype MPI_COMPLEX_DOUBLE;
    MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_COMPLEX_DOUBLE);
    MPI_Type_commit(&MPI_COMPLEX_DOUBLE);
    return MPI_COMPLEX_DOUBLE;
}

void ConjugateGradient(double complex* local_x, double complex* local_A, double complex* b, int n, int N, int Iters, int rank, int size){
    double complex rrold, rrnew, alpha, beta, pAp;
    double complex* local_r = (double complex*)malloc(n * sizeof(double complex));
    double complex* p = (double complex*)malloc(N * sizeof(double complex));
    double complex* local_p = (double complex*)malloc(n * sizeof(double complex));
    double complex* local_Ap = (double complex*)malloc(n * sizeof(double complex));

    MPI_Datatype MPI_COMPLEX_DOUBLE = get_mpi_complex_double();
    MPI_Op MPI_COMPLEX_SUM;
    MPI_Op_create(complex_sum, 1, &MPI_COMPLEX_SUM);
    
    for (int i = 0; i < N; i++){
        p[i] = b[i];
    }

    rrnew = 0.0;
    for (int i = 0; i < n; i++) {
        local_p[i] = p[rank*n + i];
        local_r[i] = b[n*rank + i];
        rrnew += conj(local_r[i]) * local_r[i];
        local_x[i] = 0.0;
    }
    MPI_Allreduce(MPI_IN_PLACE, &rrnew, 1, MPI_COMPLEX_DOUBLE, MPI_COMPLEX_SUM, MPI_COMM_WORLD);

    // main loop
    for (int m = 0; m < Iters; m++) {
        pAp = 0.0;
        local_matmul(local_Ap, local_A, p, n, N, rank);
        for (int i = 0; i < n; i++){
            pAp += conj(local_p[i]) * local_Ap[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, &pAp, 1, MPI_COMPLEX_DOUBLE, MPI_COMPLEX_SUM, MPI_COMM_WORLD);

        alpha = rrnew / pAp;
        rrold = rrnew;

        rrnew = 0.0;
        for (int i = 0; i < n; i++) {
            local_x[i] += alpha * local_p[i];
            local_r[i] -= alpha * local_Ap[i];
            rrnew += conj(local_r[i]) * local_r[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, &rrnew, 1, MPI_COMPLEX_DOUBLE, MPI_COMPLEX_SUM, MPI_COMM_WORLD);
        // fprintf(stderr, "rank %d: rrnew = %f\n", rank, rrnew);
        beta = rrnew/rrold;
        for (int i = 0; i < n; i++) {
            local_p[i] = local_r[i] + beta * local_p[i];
        }
        MPI_Allgather(local_p, n, MPI_COMPLEX_DOUBLE, p, n, MPI_COMPLEX_DOUBLE, MPI_COMM_WORLD);
    }     
}

void get_weight_proposal(double* weights, double sigma, int N){
    for (int i = 0; i < N; i++){
        weights[i] += get_random(sigma);
    }
}

void load_data(char* path, double* data_x, double* data_y, int Ndata){
    FILE *inp;
    inp = fopen(path, "r");
    for (int i = 0; i < Ndata; i++)
    {
        fscanf(inp, "%lf %lf", &data_x[i], &data_y[i]);
    }
    fclose(inp);
}


int main(int argc, char** argv) {
    int rank, size, N, n, Ndata, Iters, N_metropolis;
    double gamma;
    int info;
    FILE *inp;

    MPI_Init(&argc, &argv);
    MPI_Datatype MPI_COMPLEX_DOUBLE = get_mpi_complex_double();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char data[] = "/cfs/klemming/home/e/emastr/Public/conjgrad-project/data.txt";
    char output[] = "/cfs/klemming/home/e/emastr/Public/conjgrad-project/run.txt";

    Ndata = 100;
    N = 16;
    n = N/size;
    Iters = 8;
    gamma = 2; 
    N_metropolis = 10;

    double complex* local_A = (double complex*)malloc(n * N * sizeof(double complex));
    double complex* local_b = (double complex*)malloc(n * sizeof(double complex));
    double complex* b = (double complex*)malloc(N * sizeof(double complex));
    double* weights = (double*)malloc(N*sizeof(double));
    double* data_x = (double*)malloc(Ndata * sizeof(double));
    double* data_y = (double*)malloc(Ndata * sizeof(double));
    double complex* local_coef = (double complex*)malloc(n * sizeof(double complex));
    double complex* local_coef_old = (double complex*)malloc(n * sizeof(double complex));
    double* y = (double*)malloc(Ndata * sizeof(double));

    for (int i = 0; i < N; i++){
        weights[i] = 0;
    }

    load_data(data, data_x, data_y, Ndata);


    
    //for (int n = 0; n < N_metropolis; n++){
    get_weight_proposal(weights, 1., N);

    get_local_A(local_A, weights, data_x, n, N, Ndata, rank);
    get_local_b(local_b, weights, data_x, data_y, n, N, Ndata, rank);
    MPI_Allgather(local_b, n, MPI_COMPLEX_DOUBLE, b, n, MPI_COMPLEX_DOUBLE, MPI_COMM_WORLD);
    ConjugateGradient(local_coef, local_A, b, n, N, Iters, rank, size);
    
    double complex* coef;
    if (rank == 0){
        coef = (double complex*)malloc(N * sizeof(double complex));
        MPI_Gather(local_coef, n, MPI_COMPLEX_DOUBLE, coef, n, MPI_COMPLEX_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else{
        MPI_Gather(local_coef, n, MPI_COMPLEX_DOUBLE, NULL, 0, MPI_COMPLEX_DOUBLE, 0, MPI_COMM_WORLD);
    }
    //}

    if (rank == 0){
        evaluate_fourier(y, data_x, coef, weights, N, Ndata);
    }


    if (rank == 0) {
        inp = fopen(output, "w");
        for (int i = 0; i < Ndata; i++) {
            fprintf(inp, "%f\n", y[i]);
        }
        fclose(inp);
        for (int i = 0; i < Ndata; i++) {
            printf("true %f, approx %f\n", data_y[i], y[i]);
        }

        for (int i = 0; i < N; i++) {
            printf("x[%d] = %f+%fi, b[%d]=%f+%fi\n", i, coef[i], i, b[i]);
        }
    }
    free(local_A);
    free(local_b);
    free(b);
    free(weights);
    free(data_x);
    free(data_y);
    free(local_coef);
    free(y);
    if (rank == 0){
        free(coef);
    }
    MPI_Finalize();
}