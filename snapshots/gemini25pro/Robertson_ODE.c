/*
  This C file provides a complete implementation for solving the stiff Robertson
  ordinary differential equation (ODE) problem using the PETSc library.

  The Robertson ODE system is defined as:
  dy1/dt = -0.04*y1 + 1e4*y2*y3
  dy2/dt =  0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
  dy3/dt =  3e7*y2^2

  Initial conditions at t=0:
  y1(0) = 1
  y2(0) = 0
  y3(0) = 0

  The program solves this system over the time interval [0, 100] using the
  Crank-Nicolson (TSCN) method provided by PETSc's TS (Time Stepper) module.
*/

#include <petscts.h>

/*
  RHSFunction - Evaluates the Right-Hand Side of the ODE system.
  This function computes F(t, Y) where dY/dt = F(t, Y).

  Parameters:
  ts  - the time-stepper context
  t   - the current time
  Y   - the solution vector
  F   - the vector to store the function evaluation
  ctx - user-defined context (not used here)
*/
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec Y, Vec F, void *ctx)
{
  const PetscScalar *y;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Y, &y));
  PetscCall(VecGetArray(F, &f));

  f[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
  f[1] =  0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
  f[2] =  3.0e7 * y[1] * y[1];

  PetscCall(VecRestoreArrayRead(Y, &y));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  RHSJacobian - Evaluates the Jacobian of the ODE system.
  The Jacobian matrix J is defined as J_ij = dF_i/dY_j.

  Parameters:
  ts  - the time-stepper context
  t   - the current time
  Y   - the solution vector
  A   - the matrix to store the Jacobian
  P   - the matrix to be used for preconditioning (often the same as A)
  ctx - user-defined context (not used here)
*/
static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec Y, Mat A, Mat P, void *ctx)
{
  const PetscScalar *y;
  PetscScalar       J[3][3];
  PetscInt          row[3] = {0, 1, 2}, col[3] = {0, 1, 2};

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Y, &y));

  J[0][0] = -0.04;
  J[0][1] = 1.0e4 * y[2];
  J[0][2] = 1.0e4 * y[1];

  J[1][0] = 0.04;
  J[1][1] = -1.0e4 * y[2] - 6.0e7 * y[1];
  J[1][2] = -1.0e4 * y[1];

  J[2][0] = 0.0;
  J[2][1] = 6.0e7 * y[1];
  J[2][2] = 0.0;

  PetscCall(MatSetValues(P, 3, row, 3, col, &J[0][0], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (A != P) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }

  PetscCall(VecRestoreArrayRead(Y, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS           ts;   /* time-stepper */
  Vec          Y;    /* solution vector */
  Mat          A;    /* Jacobian matrix */
  PetscMPIInt  size;
  PetscScalar *y_ptr;

  /* Initialize PETSc and check for sequential execution */
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, "Robertson ODE Problem Solver\n"));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a sequential example and must be run with 1 process.");

  /* Create PETSc objects for the problem */
  /* Create the Jacobian matrix (3x3) */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 3, 3));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /* Create the solution vector (size 3) */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &Y));
  PetscCall(VecSetSizes(Y, PETSC_DECIDE, 3));
  PetscCall(VecSetFromOptions(Y));

  /* Set initial conditions: y = [1, 0, 0] */
  PetscCall(VecGetArray(Y, &y_ptr));
  y_ptr[0] = 1.0;
  y_ptr[1] = 0.0;
  y_ptr[2] = 0.0;
  PetscCall(VecRestoreArray(Y, &y_ptr));

  /* Create and configure the time-stepper (TS) context */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSCN)); /* Set method to Crank-Nicolson */
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, NULL));
  PetscCall(TSSetRHSJacobian(ts, A, A, RHSJacobian, NULL));

  /* Set time-stepping parameters */
  PetscCall(TSSetMaxTime(ts, 100.0));
  PetscCall(TSSetMaxSteps(ts, 1000));
  PetscCall(TSSetTimeStep(ts, 1.0e-4)); /* Initial time step, will be adapted */
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  
  /* Set tolerances. The Robertson problem is stiff and y2 becomes very small,
     so a small absolute tolerance is needed for an accurate solution. */
  PetscCall(TSSetTolerances(ts, 1.0e-8, NULL, 1.0e-8, NULL));

  /* Set the initial solution vector */
  PetscCall(TSSetSolution(ts, Y));

  /* Allow user to override settings from the command line (e.g., -ts_monitor) */
  PetscCall(TSSetFromOptions(ts));

  /* Solve the ODE system */
  PetscCall(TSSolve(ts, Y));

  /* View the final solution */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nFinal solution at t=100:\n"));
  PetscCall(VecView(Y, PETSC_VIEWER_STDOUT_WORLD));

  /* Free memory */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&Y));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}
