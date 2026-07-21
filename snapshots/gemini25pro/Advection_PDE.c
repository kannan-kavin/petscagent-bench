/*
  1D linear advection equation: u_t + c * u_x = 0
  Domain: [0, 1] with periodic boundary conditions
  Initial condition: u(x,0) = sin(2*pi*x)
  Discretization: First-order upwind finite difference

  This example demonstrates the use of PETSc's TS (Time Stepper) module
  to solve a time-dependent PDE. It uses a DMDA (Distributed Mesh Data Array)
  to manage the grid and data layout, which simplifies parallelization and
  handling of ghost points for boundary conditions.
*/

#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>

/* Application context to hold problem parameters */
typedef struct {
  PetscReal c;  /* Advection speed */
  PetscReal dx; /* Grid spacing */
  DM        dm; /* DMDA object managing the grid */
} AppCtx;

/* Function prototypes */
static PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx);
static PetscErrorCode SetInitialCondition(Vec U, void *ctx);

int main(int argc, char **argv)
{
  TS             ts;   /* Time-integrator context */
  Vec            U;    /* Solution vector */
  AppCtx         appctx;
  PetscInt       N = 100; /* Number of grid points */
  PetscReal      t_final = 1.0;

  /* Initialize PETSc */
  PetscCall(PetscInitialize(&argc, &argv, (char*)0, NULL));

  /* Set up problem parameters in the application context */
  appctx.c = 1.0;

  /* Create a 1D DMDA for the grid. 
     - DM_BOUNDARY_PERIODIC handles the periodic boundary conditions automatically.
     - N: global number of grid points
     - 1: degrees of freedom per node
     - 1: stencil width (for upwind, we need the i-1 point)
  */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, N, 1, 1, NULL, &appctx.dm));
  PetscCall(DMSetFromOptions(appctx.dm));
  PetscCall(DMSetUp(appctx.dm));

  /* Calculate grid spacing based on the domain [0,1] and N points */
  appctx.dx = 1.0 / (PetscReal)N;

  /* Create a global vector U to store the solution */
  PetscCall(DMCreateGlobalVector(appctx.dm, &U));

  /* Create the TS (Time Stepper) solver context */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  /* Associate the DMDA with the TS solver. This allows TS to handle grid-related tasks. */
  PetscCall(TSSetDM(ts, appctx.dm));
  /* Set the ODE to be solved: dU/dt = F(t,U) */
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR)); /* General form for explicit methods */
  PetscCall(TSSetRHSFunction(ts, NULL, FormRHSFunction, &appctx));

  /* Set time-stepping parameters */
  PetscCall(TSSetMaxTime(ts, t_final));
  PetscCall(TSSetMaxSteps(ts, 999)); /* Ensure less than 1000 steps */
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));

  /* Set solver type, e.g., a 4th-order Runge-Kutta method (TSRK) */
  PetscCall(TSSetType(ts, TSRK));

  /* Set tolerances for the time-stepper. 
     These are important for adaptive time-stepping methods, but it's good practice to set them. */
  PetscCall(TSSetTolerances(ts, 1.0e-7, NULL, 1.0e-7, NULL));

  /* Allow user to override settings from the command line */
  PetscCall(TSSetFromOptions(ts));

  /* Set the initial condition for the solution vector U */
  PetscCall(SetInitialCondition(U, &appctx));

  /* Solve the PDE system from t=0 to t=t_final */
  PetscCall(TSSolve(ts, U));

  /* 
    With c=1.0 and t_final=1.0, the wave travels exactly one domain length.
    Due to periodic boundary conditions, the final solution should be identical
    to the initial condition. We can visually inspect the output to verify this.
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nFinal solution vector at t=1.0:\n"));
  PetscCall(VecView(U, PETSC_VIEWER_STDOUT_WORLD));

  /* Clean up allocated PETSc objects */
  PetscCall(VecDestroy(&U));
  PetscCall(DMDestroy(&appctx.dm));
  PetscCall(TSDestroy(&ts));

  /* Finalize PETSc */
  PetscCall(PetscFinalize());
  return 0;
}

/**
 * @brief Sets the initial condition u(x,0) = sin(2*pi*x).
 * 
 * @param U Solution vector to be filled.
 * @param ctx Application context with problem parameters.
 * @return PetscErrorCode 
 */
static PetscErrorCode SetInitialCondition(Vec U, void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  DM             dm = appctx->dm;
  PetscScalar    *u_array;
  PetscInt       i, xs, xm;
  PetscReal      dx = appctx->dx;
  PetscReal      x;

  PetscFunctionBeginUser;
  /* Get a pointer to the local data of the vector */
  PetscCall(DMDAVecGetArray(dm, U, &u_array));
  /* Get the range of local grid points owned by this process */
  PetscCall(DMDAGetCorners(dm, &xs, NULL, NULL, &xm, NULL, NULL));

  /* Loop over the local grid points and set the initial value */
  for (i = xs; i < xs + xm; i++) {
    x = i * dx;
    u_array[i] = sin(2.0 * PETSC_PI * x);
  }

  /* Restore the vector, making the changes visible */
  PetscCall(DMDAVecRestoreArray(dm, U, &u_array));
  PetscFunctionReturn(0);
}

/**
 * @brief Computes the right-hand-side (RHS) of the semi-discretized ODE system.
 *        For u_t + c*u_x = 0, the ODE is dU/dt = -c * (D*U), where D is the
 *        spatial discretization operator (first-order upwind).
 * 
 * @param ts Time-stepper context.
 * @param t Current time.
 * @param U Current solution vector.
 * @param F RHS vector to be computed.
 * @param ctx Application context.
 * @return PetscErrorCode 
 */
static PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  DM             dm;
  Vec            Ulocal;
  const PetscScalar *u_array;
  PetscScalar    *f_array;
  PetscInt       i, xs, xm;
  PetscReal      c = appctx->c;
  PetscReal      dx = appctx->dx;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  /* Get a local vector that includes ghost points */
  PetscCall(DMGetLocalVector(dm, &Ulocal));

  /* Scatter global solution U to local vector Ulocal to fill ghost points */
  PetscCall(DMGlobalToLocalBegin(dm, U, INSERT_VALUES, Ulocal));
  PetscCall(DMGlobalToLocalEnd(dm, U, INSERT_VALUES, Ulocal));

  /* Get read-only access to the local solution data (including ghost points) */
  PetscCall(DMDAVecGetArrayRead(dm, Ulocal, &u_array));
  /* Get write access to the RHS vector data */
  PetscCall(DMDAVecGetArray(dm, F, &f_array));

  /* Get the range of local grid points (excluding ghost points) */
  PetscCall(DMDAGetCorners(dm, &xs, NULL, NULL, &xm, NULL, NULL));

  /* Loop over local grid points to compute the RHS using first-order upwind scheme.
     Assuming c > 0, the scheme is (u_i - u_{i-1})/dx.
     The periodic boundary condition is handled by the ghost point communication. 
     For i=0, u_array[i-1] correctly accesses the value from the last point on the grid.
  */
  for (i = xs; i < xs + xm; i++) {
    f_array[i] = -c * (u_array[i] - u_array[i-1]) / dx;
  }

  /* Restore access to the vectors */
  PetscCall(DMDAVecRestoreArrayRead(dm, Ulocal, &u_array));
  PetscCall(DMDAVecRestoreArray(dm, F, &f_array));
  PetscCall(DMRestoreLocalVector(dm, &Ulocal));

  PetscFunctionReturn(0);
}
