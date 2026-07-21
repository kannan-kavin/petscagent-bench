/*
  Robertson ODE problem solved with PETSc TS using Crank-Nicolson (CN).

  System:
    y1' = -0.04 y1 + 1e4 y2 y3
    y2' =  0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
    y3' =  3e7 y2^2

  IC: y(0) = [1,0,0]
  Integrate to t=100 with < 1000 steps.

  Notes on tolerances:
    Robertson is stiff; CN is A-stable but not L-stable, so we use a reasonably
    tight relative tolerance and a small absolute tolerance to control error.
    We also cap dt to ensure <1000 steps.
*/

#include <petscts.h>

typedef struct {
  PetscReal k1, k2, k3;
} AppCtx;

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec Y, Vec F, void *ctx)
{
  AppCtx            *user = (AppCtx*)ctx;
  const PetscScalar *y;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  (void)ts; (void)t;

  PetscCall(VecGetArrayRead(Y, &y));
  PetscCall(VecGetArray(F, &f));

  const PetscScalar y1 = y[0];
  const PetscScalar y2 = y[1];
  const PetscScalar y3 = y[2];

  f[0] = -user->k1*y1 + user->k2*y2*y3;
  f[1] =  user->k1*y1 - user->k2*y2*y3 - user->k3*y2*y2;
  f[2] =  user->k3*y2*y2;

  PetscCall(VecRestoreArrayRead(Y, &y));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec Y, Mat J, Mat P, void *ctx)
{
  AppCtx            *user = (AppCtx*)ctx;
  const PetscScalar *y;
  PetscInt           rowcol[3] = {0,1,2};
  PetscScalar        A[3][3];

  PetscFunctionBeginUser;
  (void)ts; (void)t;

  PetscCall(VecGetArrayRead(Y, &y));
  const PetscScalar y1 = y[0];
  const PetscScalar y2 = y[1];
  const PetscScalar y3 = y[2];
  (void)y1;

  /* J = dF/dY */
  A[0][0] = -user->k1;
  A[0][1] =  user->k2*y3;
  A[0][2] =  user->k2*y2;

  A[1][0] =  user->k1;
  A[1][1] = -user->k2*y3 - 2.0*user->k3*y2;
  A[1][2] = -user->k2*y2;

  A[2][0] =  0.0;
  A[2][1] =  2.0*user->k3*y2;
  A[2][2] =  0.0;

  PetscCall(VecRestoreArrayRead(Y, &y));

  PetscCall(MatZeroEntries(P));
  PetscCall(MatSetValues(P, 3, rowcol, 3, rowcol, &A[0][0], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS      ts;
  Vec     y;
  Mat     J;
  AppCtx  user;

  PetscReal t0 = 0.0, tf = 100.0;
  PetscReal dt = 0.1; /* 100/0.1 = 1000 steps; we will enforce <1000 by slightly larger dt */

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  user.k1 = 0.04;
  user.k2 = 1.0e4;
  user.k3 = 3.0e7;

  /* Create solution vector */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
  PetscCall(VecSetSizes(y, PETSC_DECIDE, 3));
  PetscCall(VecSetFromOptions(y));

  /* Initial conditions */
  {
    PetscScalar ic[3] = {1.0, 0.0, 0.0};
    PetscInt    ix[3] = {0,1,2};
    PetscCall(VecSetValues(y, 3, ix, ic, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(y));
    PetscCall(VecAssemblyEnd(y));
  }

  /* Create Jacobian matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 3, 3));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  /* Create TS */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSCN));

  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &user));
  PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, &user));

  PetscCall(TSSetTime(ts, t0));
  PetscCall(TSSetMaxTime(ts, tf));

  /* Ensure < 1000 steps: choose dt so that ceil((tf-t0)/dt) <= 999.
     For tf=100, dt >= 100/999 ~ 0.1001001. */
  dt = 0.101;
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetMaxSteps(ts, 999));

  /* Carefully chosen tolerances for stiff problem.
     Use tight relative tolerance and small absolute tolerance.
     (TSSetTolerances signature: TS, atol, Vec vatol, rtol, Vec vrtol)
  */
  PetscCall(TSSetTolerances(ts, 1e-12, NULL, 1e-8, NULL));

  PetscCall(TSSetFromOptions(ts));

  /* Solve */
  PetscCall(TSSolve(ts, y));

  /* Print final solution */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final solution at t=%g:\n", (double)tf));
  PetscCall(VecView(y, PETSC_VIEWER_STDOUT_WORLD));

  /* Cleanup */
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&J));

  PetscCall(PetscFinalize());
  return 0;
}
