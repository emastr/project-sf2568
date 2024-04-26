#include <stdio.h>
#include <stdlib.h>

void conjgrad2(double *A, double *b, int m, int N, double *x) {
    double *r = (double*)malloc(N * sizeof(double));
    double *p = (double*)malloc(N * sizeof(double));
    double *Ap = (double*)malloc(N * sizeof(double));
    double rrold, rrnew, alpha, beta, pAp;

    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = r[i];
    }

    rrnew = 0.0;
    for (int i = 0; i < N; i++) {
        rrnew += r[i] * r[i];
    }

    for (int i = 0; i < m; i++) {
        rrold = rrnew;

        pAp = 0.0;
        for (int j = 0; j < N; j++) {
            Ap[j] = 0.0;
            for (int k = 0; k < N; k++) {
                Ap[j] += A[j * N + k] * p[k];
            }
            pAp += p[j] * Ap[j];
        }

        alpha = rrold / pAp;

        for (int j = 0; j < N; j++) {
            x[j] += alpha * p[j];
            r[j] -= alpha * Ap[j];
        }

        rrnew = 0.0;
        for (int j = 0; j < N; j++) {
            rrnew += r[j] * r[j];
        }

        beta = rrnew / rrold;

        for (int j = 0; j < N; j++) {
            p[j] = r[j] + beta * p[j];
        }
    }

    free(r);
    free(p);
    free(Ap);
}

int main() {
    int N = 8;
    int m = 8;

    double A[64] = {
    2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0};
    double b[8] = {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double x[8];

    conjgrad2(A, b, m, N, x);

    printf("Solution:\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", x[i]);
    }

    return 0;
}
