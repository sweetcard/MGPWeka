package weka.core.matrix;

/*
 * Matrix Multiplication 
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda <br />
 * http://www.jcuda.org
 */


/**
 * @authors 
 * {tengel,andrea}@inf.ufsm.br, 
 * Luiz-Angelo.Steffenel@univ-reims.fr, 
 * Manuele.Kirsch-Pinheiro@univ-paris1.fr
 */

import static jcuda.jcublas.JCublas2.*;
import static jcuda.runtime.JCuda.*;

import jcuda.*;
import jcuda.jcublas.cublasHandle;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;

/**
 * This class performs a BLAS 'dgemm' operation, i.e. for computing the matrix <br />
 * C = alpha * A * B + beta * C <br /> using double precision
 * 
 * if we set alpha= 1 and beta = 0, we have the matrix multiplication.
 */
public class JCublasMatrixMultiplication {

    /**
     * Implementation of dgemm using JCublas
     */
    public static Matrix times(Matrix MA, Matrix MB) {
        double alpha = 1.0f;
        double beta = 0.0f;

        /// C(m,n) = A(m,k) x B(k,n)
        int m = MA.getRowDimension(); 
        int k = MA.getColumnDimension();
        int n = MB.getColumnDimension(); 

        double[] A = MA.toArray();
        double[] B = MB.toArray();
        double[] C = new double[m * n];

        // Create a CUBLAS handle
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);

        // Allocate memory on the device
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();

        cudaMalloc(d_A, A.length * Sizeof.DOUBLE);
        cudaMalloc(d_B, B.length * Sizeof.DOUBLE);
        cudaMalloc(d_C, C.length * Sizeof.DOUBLE);

        // Copy the memory from the host to the device
        cublasSetVector(A.length, Sizeof.DOUBLE, Pointer.to(A), 1, d_A, 1);
        cublasSetVector(B.length, Sizeof.DOUBLE, Pointer.to(B), 1, d_B, 1);
        cublasSetVector(C.length, Sizeof.DOUBLE, Pointer.to(C), 1, d_C, 1);

        // Execute dgemm
        Pointer pAlpha = Pointer.to(new double[]{alpha});
        Pointer pBeta = Pointer.to(new double[]{beta});

        cublasDgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N, n, m, k, pAlpha, d_B, n, d_A, k, pBeta, d_C, n);

        // Copy the result from the device to the host
        cublasGetVector(m * n, Sizeof.DOUBLE, d_C, 1, Pointer.to(C), 1);

        // Clean up
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);

        return new Matrix(arrayToMatrix(C, m, n), m, n);
    }

    static double[][] arrayToMatrix(double[] v, int nlin, int ncol) {
        double[][] m = new double[nlin][ncol];
        for (int i = 0; i < nlin; i++) {
            System.arraycopy(v, i * ncol, m[i], 0, ncol);
        }
        return m;
    }
}