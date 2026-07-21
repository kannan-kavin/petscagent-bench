/* Robertson stiff ODE problem solved using PETSc TS with Crank-Nicolson */
#include <petscts.h>

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

static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec Y, Mat A, Mat B, void *ctx)
{
  const PetscScalar *y;
  PetscInt           idxm[3] = {0, 1, 2};
  PetscScalar        J[3][3];

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Y, &y));

  J[0][0] = -0.04;
  J[0][1] =  1.0e4 * y[2];
  J[0][2] =  1.0e4 * y[1];

  J[1][0] =  0.04;
  J[1][1] = -1.0e4 * y[2] - 6.0e7 * y[1];
  J[1][2] = -1.0e4 * y[1];

  J[2][0] = 0.0;
  J[2][1] = 6.0e7 * y[1];
  J[2][2] = 0.0;

  PetscCall(MatSetValues(B, 3, idxm, 3, idxm, &J[0][0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(Y, &y));

  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS           ts;
  Vec          Y;
  Mat          J;
  PetscScalar *y;
  PetscInt     steps;
  PetscReal    ftime;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  /* Create solution vector */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &Y));
  PetscCall(VecSetSizes(Y, PETSC_DECIDE, 3));
  PetscCall(VecSetFromOptions(Y));

  PetscCall(VecGetArray(Y, &y));
  y[0] = 1.0;
  y[1] = 0.0;
  y[2] = 0.0;
  PetscCall(VecRestoreArray(Y, &y));

  /* Create Jacobian matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 3, 3));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  /* Create TS */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSCN));  /* Crank-Nicolson */

  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, NULL));
  PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, NULL));

  PetscCall(TSSetMaxTime(ts, 100.0));
  PetscCall(TSSetMaxSteps(ts, 999));
  PetscCall(TSSetTimeStep(ts, 1.0e-4));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  /* Carefully chosen tolerances for the stiff Robertson problem.
     y_2 is very small (~1e-5) so absolute tolerance must be tight. */
  PetscCall(TSSetTolerances(ts, 1.0e-8, NULL, 1.0e-6, NULL));

  PetscCall(TSSetSolution(ts, Y));
  PetscCall(TSSetFromOptions(ts));

  /* Solve */
  PetscCall(TSSolve(ts, Y));
  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &steps));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
             "Robertson ODE: final time = %g, steps = %" PetscInt_FMT "\n",
             (double)ftime, steps));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final solution:\n"));
  PetscCall(VecView(Y, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&Y));
  PetscCall(MatDestroy(&J));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}
