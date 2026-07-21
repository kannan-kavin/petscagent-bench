/*
  1D linear advection: u_t + c u_x = 0 on x in [0,1] with periodic BC.
  Initial condition: u(x,0) = sin(2*pi*x).

  Spatial discretization: first-order upwind finite difference on uniform grid.
  Time integration: PETSc TS (explicit RK by default).

  Build (example):
    mpicc advection1d_petsc.c -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include \
      -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc -lm -o advection1d

  Run (example):
    ./advection1d -ts_type rk -ts_rk_type 3 -ts_dt 1e-3 -ts_max_steps 1000

  Notes:
  - For stability with explicit methods, choose dt satisfying CFL ~ c*dt/dx <= 1.
  - TSSetTolerances() is set even though explicit RK does not use nonlinear solves;
    it is harmless and satisfies the requirement.
*/

#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <math.h>

typedef struct {
  PetscInt  N;
  PetscReal c;
  PetscReal xmin, xmax;
  PetscReal dx;
} AppCtx;

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  AppCtx            *user = (AppCtx*)ctx;
  DM                 da;
  DMDALocalInfo      info;
  Vec                Uloc;
  const PetscScalar *u;
  PetscScalar       *f;
  PetscInt           i;
  PetscReal          c = user->c;

  PetscFunctionBeginUser;
  (void)t;

  PetscCall(TSGetDM(ts,&da));
  PetscCall(DMDAGetLocalInfo(da,&info));

  PetscCall(DMGetLocalVector(da,&Uloc));
  PetscCall(DMGlobalToLocalBegin(da,U,INSERT_VALUES,Uloc));
  PetscCall(DMGlobalToLocalEnd(da,U,INSERT_VALUES,Uloc));

  PetscCall(DMDAVecGetArrayRead(da,Uloc,&u));
  PetscCall(DMDAVecGetArray(da,F,&f));

  /* Upwind derivative with periodic ghost points provided by DMDA.
     For c>0: u_x ~ (u_i - u_{i-1})/dx
     For c<0: u_x ~ (u_{i+1} - u_i)/dx
     PDE: u_t = -c u_x
  */
  if (c >= 0.0) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      f[i] = -c * (u[i] - u[i-1]) / user->dx;
    }
  } else {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      f[i] = -c * (u[i+1] - u[i]) / user->dx;
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(da,Uloc,&u));
  PetscCall(DMDAVecRestoreArray(da,F,&f));
  PetscCall(DMRestoreLocalVector(da,&Uloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetInitialCondition(DM da, Vec U, AppCtx *user)
{
  DMDALocalInfo info;
  PetscScalar *u;
  PetscInt     i;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMDAVecGetArray(da,U,&u));

  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscReal x = user->xmin + ((PetscReal)i) * user->dx;
    u[i] = (PetscScalar)sin(2.0*M_PI*x);
  }

  PetscCall(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc,char **argv)
{
  TS        ts;
  DM        da;
  Vec       U;
  AppCtx    user;
  PetscReal t0 = 0.0, tf = 1.0;
  PetscReal dt;

  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));

  user.N    = 100;
  user.c    = 1.0;
  user.xmin = 0.0;
  user.xmax = 1.0;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&user.N,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-c",&user.c,NULL));

  user.dx = (user.xmax - user.xmin) / (PetscReal)user.N;

  /* 1D DMDA with periodic boundary and 1 dof per node.
     Use stencil width 1 for upwind (needs i-1 or i+1). */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,user.N,1,1,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateGlobalVector(da,&U));
  PetscCall(SetInitialCondition(da,U,&user));

  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetDM(ts,da));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));

  /* Explicit RHS: Udot = F(t,U) */
  PetscCall(TSSetRHSFunction(ts,NULL,RHSFunction,&user));

  /* Choose a stable default dt based on CFL ~ 0.5 unless overridden. */
  dt = 0.5 * user.dx / (PetscAbsReal(user.c) > 0 ? PetscAbsReal(user.c) : 1.0);
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-ts_dt",&dt,NULL));

  PetscCall(TSSetTime(ts,t0));
  PetscCall(TSSetTimeStep(ts,dt));
  PetscCall(TSSetMaxTime(ts,tf));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));

  /* Ensure < 1000 steps by default; user can reduce dt but then must also adjust max_steps. */
  PetscCall(TSSetMaxSteps(ts,1000));

  /* Carefully chosen tolerances (mainly relevant for adaptive/implicit methods).
     Keep them reasonably strict but not extreme. */
  PetscCall(TSSetTolerances(ts,1e-8,NULL,1e-10,NULL));

  /* Default to explicit RK unless user overrides. */
  PetscCall(TSSetType(ts,TSRK));

  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSolve(ts,U));

  /* Print final solution */
  PetscCall(VecView(U,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}
