/**
 * Multicore Matrix Multiplication. Multiplies two matrices by dividing the problem
 * into subproblems, each core processes one part.
 */


/**
 * @author tengel
 * Contact: tengel@inf.ufsm.br
 */

package weka.core.matrix;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;



public class MulticoreMatrixMult {

    private static int OFFSET = 4;

    /**
     * Method who does the paralel multiplication of two Matrices.
     * Multiplication Logic: - Takes the number of avaliable processors and
     * divides it by the number of columns of the matrix B. - Each thread
     * multiplies separately a certain number of columns from the matrix B and
     * store the results at matrix X, this way, it isn't necessary to treat
     * problems with shared data.
     *
     * @param A
     * @param B
     * @return A * B
     */
    public static Matrix times(Matrix A, Matrix B) {
        if (A.getColumnDimension() != B.getRowDimension()) {
            throw new IllegalArgumentException("Matrix inner dimensions must agree.");
        }

        ArrayList<Future<Matrix>> threads = new ArrayList<Future<Matrix>>(); // Array of future 
        ExecutorService pool = null; // Thread's pool
        Matrix X = new Matrix(A.getRowDimension(), B.getColumnDimension()); // Matrix result
        final int N_THREADS = Runtime.getRuntime().availableProcessors(); // Number of threads
        // If # of columns of matrix B < Number of processors, executes sequencially.
        try {
            if (B.getColumnDimension() < N_THREADS * OFFSET) {
                pool = Executors.newFixedThreadPool(1);
                threads.add(pool.submit(new ParalelMatrixMultiplication(A, B, X, 0, B.getColumnDimension() - 1)));
            } // If # of columns of matix B >= Number of Processors
            else {
                int partitionSize = B.getColumnDimension() / N_THREADS - 1; // Number of columns for each thread
                // sets the thread's pool. 
                pool = Executors.newFixedThreadPool(N_THREADS);
                int aux = 0;
                for (int i = 0; aux < N_THREADS; i += partitionSize + 1) {                    
                    aux++;
                    if ((i + partitionSize) < B.getColumnDimension() && (aux+1) <= N_THREADS) {
                        threads.add(pool.submit(new ParalelMatrixMultiplication(A, B, X, i, i + partitionSize)));
                    } else {
                        threads.add(pool.submit(new ParalelMatrixMultiplication(A, B, X, i, B.getColumnDimension() - 1)));
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        //  Similar ao join(). Aguarda o termino das threads. BLOQUEANTE 
        for (Future f : threads) {
            try {
                f.get();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        pool.shutdown();
        return X;
    }

    /**
     * Callable thread class.
     */
    private static class ParalelMatrixMultiplication implements Callable<Matrix> {

        private Matrix A;
        private Matrix B;
        private Matrix X;
        private int startColumn;
        private int finalColumn;

        public ParalelMatrixMultiplication(Matrix A, Matrix B, Matrix X,
                int startColumn, int finalColumn) {
            this.A = A;
            this.B = B;
            this.X = X;
            this.startColumn = startColumn;
            this.finalColumn = finalColumn;
        }

        @Override
        public Matrix call() throws Exception {
            times();
            return X;
        }

        /**
         * Linear algebraic matrix multiplication, A * B
         *
         */
        private void times() {
            double[][] matA = A.getArray();
            double[][] matX = X.getArray();

            for (int k = startColumn; k <= finalColumn; k++) { // From and to determined columns
                double[] tmp = getColumn(B, k); // Tmp for not have to get each element each time.
                for (int i = 0; i < A.getRowDimension(); i++) { // A matrix lines
                    double sum = 0.0;
                    for (int j = 0; j < A.getColumnDimension(); j++) { // A matrix columns
                        sum += matA[i][j] * tmp[j];
                    }
                    matX[i][k] = sum;
                }
            }
        }

        /**
         * Gets the j column from the matrix M
         *
         * @param M The Matrix
         * @param j The Column
         * @return the vector that contains the elements of the column j
         */
        private double[] getColumn(Matrix M, int j) {
            if (j < 0 || j > M.getColumnDimension()) {
                throw new IllegalArgumentException("Index out of boundaries. ");
            }
            double[] vector = new double[M.getRowDimension()];
            for (int i = 0; i < M.getRowDimension(); i++) {
                vector[i] = M.get(i, j);
            }
            return vector;
        }
    }
}
