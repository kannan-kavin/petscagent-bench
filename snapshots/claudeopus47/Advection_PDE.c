/* 1D Linear Advection PDE: du/dt + c du/dx = 0 on [0,1] with periodic BCs
 * Initial condition: u(x,0) = sin(2*pi*x)
 * First-order upwind finite difference (assuming c > 0)
 * Solved using PETSc TS from t=0 to t=1
 */
#include <petscts.h>
#include <petscdmda.h>
#include <math.h>

typedef struct {
  PetscReal c;  /* advection speed */
  PetscReal dx;
  PetscInt  N;
} AppCtx;

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  AppCtx            *user = (AppCtx*)ctx;
  DM                 da;
  PetscInt           i, xs, xm, Mx;
  const PetscScalar *u;
  PetscScalar       *f;
  Vec                Ulocal;
  PetscReal          c = user->c, dx = user->dx;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMGetLocalVector(da, &Ulocal));
  PetscCall(DMGlobalToLocalBegin(da, U, INSERT_VALUES, Ulocal));
  PetscCall(DMGlobalToLocalEnd(da, U, INSERT_VALUES, Ulocal));

  PetscCall(DMDAGetInfo(da, NULL, &Mx, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));

  PetscCall(DMDAVecGetArrayRead(da, Ulocal, (void*)&u));
  PetscCall(DMDAVecGetArray(da, F, &f));

  /* Upwind for c > 0: du/dx ~ (u[i] - u[i-1]) / dx */
  for (i = xs; i < xs + xm; i++) {
    f[i] = -c * (u[i] - u[i-1]) / dx;
  }

  PetscCall(DMDAVecRestoreArrayRead(da, Ulocal, (void*)&u));
  PetscCall(DMDAVecRestoreArray(da, F, &f));
  PetscCall(DMRestoreLocalVector(da, &Ulocal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetInitialCondition(DM da, Vec U, AppCtx *user)
{
  PetscInt     i, xs, xm;
  PetscScalar *u;
  PetscReal    x;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));
  PetscCall(DMDAVecGetArray(da, U, &u));
  for (i = xs; i < xs + xm; i++) {
    x = i * user->dx;
    u[i] = PetscSinReal(2.0 * PETSC_PI * x);
  }
  PetscCall(DMDAVecRestoreArray(da, U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS        ts;
  Vec       U;
  DM        da;
  AppCtx    user;
  PetscReal dt, tfinal = 1.0;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  user.N  = 100;
  user.c  = 1.0;
  user.dx = 1.0 / user.N;

  /* Periodic 1D DMDA with N points, dof=1, stencil width 1 */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, user.N, 1, 1, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateGlobalVector(da, &U));
  PetscCall(SetInitialCondition(da, U, &user));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, da));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &user));
  PetscCall(TSSetType(ts, TSRK));

  /* CFL: dt < dx/c = 0.01. Use dt = 0.005 -> 200 steps for tfinal=1 */
  dt = 0.005;
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetMaxTime(ts, tfinal));
  PetscCall(TSSetMaxSteps(ts, 999));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  /* Tolerances for adaptive control */
  PetscCall(TSSetTolerances(ts, 1e-6, NULL, 1e-6, NULL));

  PetscCall(TSSetSolution(ts, U));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSolve(ts, U));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final solution at t=%g:\n", (double)tfinal));
  PetscCall(VecView(U, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
