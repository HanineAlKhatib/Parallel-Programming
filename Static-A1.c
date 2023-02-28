#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#define MASTER_RANK 0

void calculate_mandelbrot(int width, int height, int start_row, int end_row, double left, double right, double lower, double upper, int *output)
{
    const int max_iterations = 1000;
    double x, y, x_new, y_new, x_squared, y_squared;
    int iteration;

    for (int i = start_row; i < end_row; i++)
    {
        for (int j = 0; j < width; j++)
        {
            x = left + (right - left) * j / width;
            y = lower + (upper - lower) * i / height;
            x_new = 0.0;
            y_new = 0.0;
            x_squared = 0.0;
            y_squared = 0.0;
            iteration = 0;

            while (x_squared + y_squared < 4.0 && iteration < max_iterations)
            {
                y_new = 2 * x_new * y + y;
                x_new = x_squared - y_squared + x;
                x_squared = x_new * x_new;
                y_squared = y_new * y_new;
                iteration++;
            }

            output[i * width + j] = iteration;
        }
    }
}

int main(int argc, char **argv)
{
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    int width = 800, height = 800;
    double left = -2.0, right = 1.0, lower = -1.5, upper = 1.5;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rows_per_process = height / size;
    const int start_row = rank * rows_per_process;
    const int end_row = (rank + 1) * rows_per_process + (rank == size - 1 ? height % size : 0);
    const int row_count = end_row - start_row;

    int *local_output = (int *)malloc(row_count * width * sizeof(int));
    calculate_mandelbrot(width, height, start_row, end_row, left, right, lower, upper, local_output);

    int *global_output = NULL;
    if (rank == MASTER_RANK)
    {
        global_output = (int *)malloc(width * height * sizeof(int));
    }

    MPI_Gather(local_output, row_count * width, MPI_INT, global_output, row_count * width, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);

    if (rank == MASTER_RANK)
    {
        FILE *fp = fopen("mandelbrot.pgm", "wb");
        fprintf(fp, "P2\n%d %d\n%d\n", width, height, max_iterations - 1);
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                fprintf(fp, "%d ", global_output[i * width + j]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        free(global_output);
    }

    free(local_output);
    MPI_Finalize();

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f seconds\n", cpu_time_used);
    return 0;
