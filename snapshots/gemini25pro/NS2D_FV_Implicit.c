/*
 * 2D Incompressible Navier-Stokes Solver (Lid-Driven Cavity)
 * Method: Fully implicit, Finite Volume Method on a staggered MAC grid.
 * Solver: PETSc TS (Time Stepping) module.
 *
 * Equations (nondimensional form):
 *   Continuity:   div(u) = 0
 *   x-momentum:   du/dt + div(u*u) = -dp/dx + (1/Re) * laplacian(u)
 *   y-momentum:   dv/dt + div(v*u) = -dp/dy + (1/Re) * laplacian(v)
 *
 * Discretization:
 * - A staggered grid (Marker-and-Cell, MAC) is used.
 *   - Pressure (p) is at cell centers (i, j).
 *   - Horizontal velocity (u) is on vertical faces (i, j+1/2).
 *   - Vertical velocity (v) is on horizontal faces (i+1/2, j).
 * - The equations are integrated over control volumes (CVs) specific to each variable:
 *   - Continuity: Integrated over the main p-CV (the grid cell).
 *   - x-momentum: Integrated over a u-CV, staggered to be centered on the u-velocity component.
 *   - y-momentum: Integrated over a v-CV, staggered to be centered on the v-velocity component.
 * - All unknowns (u, v, p) are packed into a single PETSc vector.
 * - Boundary conditions are enforced by replacing the corresponding FVM equation with a simple Dirichlet condition (e.g., u_boundary - value = 0).
 * - Pressure is pinned at one cell (p_0,0 = 0) to ensure a unique solution.
 */

#include <petscts.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscsys.h>

/* Application context */
typedef struct {
  PetscInt    nx, ny;    /* Number of cells in x and y directions */
  PetscReal   dx, dy;    /* Grid spacing */
  PetscReal   re;        /* Reynolds number */
  PetscInt    p_offset, u_offset, v_offset; /* Offsets for fields in the global vector */
  PetscInt    n_p, n_u, n_v; /* Number of variables for each field */
} AppCtx;

/* Function prototypes */
PetscErrorCode FormIFunction(TS, PetscReal, Vec, Vec, Vec, void*);
PetscErrorCode FormIJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void*);
PetscErrorCode SetInitialConditions(Vec, AppCtx*);
PetscErrorCode ComputeMaxDivergence(Vec, AppCtx*, PetscReal*);

/* Helper macros for indexing into the 1D global vector from 2D grid coordinates */
#define P_IDX(i, j, nx) ((j)*(nx) + (i))
#define U_IDX(i, j, nx) ((j)*(nx+1) + (i))
#define V_IDX(i, j, nx) ((j)*(nx) + (i))

int main(int argc, char **argv)
{
  AppCtx         app;                 /* Application context */
  TS             ts;                  /* PETSc time stepper */
  Vec            X;                   /* Solution vector (u, v, p) */
  Mat            J;                   /* Jacobian matrix */
  PetscInt       total_dof;
  PetscReal      max_div;

  PetscCall(PetscInitialize(&argc, &argv, (char*)0, NULL));

  /* --- Get user-defined parameters --- */
  app.nx = 16;
  app.ny = 16;
  app.re = 100.0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &app.nx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ny", &app.ny, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-re", &app.re, NULL));

  /* --- Setup grid and data structure layout --- */
  app.dx = 1.0 / app.nx;
  app.dy = 1.0 / app.ny;

  /* Number of variables for each field */
  app.n_p = app.nx * app.ny;         /* p is at cell centers */
  app.n_u = (app.nx + 1) * app.ny;   /* u is on vertical faces */
  app.n_v = app.nx * (app.ny + 1);   /* v is on horizontal faces */

  /* Offsets for each field in the global vector */
  app.p_offset = 0;
  app.u_offset = app.p_offset + app.n_p;
  app.v_offset = app.u_offset + app.n_u;
  total_dof = app.n_p + app.n_u + app.n_v;

  /* --- Create PETSc objects --- */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &X));
  PetscCall(VecSetSizes(X, PETSC_DECIDE, total_dof));
  PetscCall(VecSetFromOptions(X));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, total_dof, total_dof));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  /* --- Configure the Time Stepper (TS) --- */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSBEULER)); /* Backward Euler */
  PetscCall(TSSetIFunction(ts, NULL, FormIFunction, &app));
  PetscCall(TSSetIJacobian(ts, J, J, FormIJacobian, &app));
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetTimeStep(ts, 0.01));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));

  /* --- Set initial conditions and solve --- */
  PetscCall(SetInitialConditions(X, &app));
  PetscCall(TSSolve(ts, X));

  /* --- Post-processing --- */
  PetscCall(ComputeMaxDivergence(X, &app, &max_div));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%e\n", max_div));

  /* --- Cleanup --- */
  PetscCall(VecDestroy(&X));
  PetscCall(MatDestroy(&J));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode SetInitialConditions(Vec X, AppCtx *app)
{
  PetscScalar *x_ptr;
  PetscInt    i, j;

  PetscCall(VecGetArray(X, &x_ptr));
  PetscCall(VecZeroEntries(X));

  /* Lid velocity u=1 at y=1. This is handled by the u-momentum equation's boundary flux term.
     However, we can set the initial guess for the boundary velocities for better solver start.
     The actual enforcement is done in FormIFunction/FormIJacobian.
     Here we just set the initial state of the fluid (at rest).*/

  PetscCall(VecRestoreArray(X, &x_ptr));
  return 0;
}

PetscErrorCode FormIFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ctx)
{
  AppCtx*            app = (AppCtx*)ctx;
  const PetscScalar *x_ptr, *xdot_ptr;
  PetscScalar       *f_ptr;
  PetscInt           i, j, p_idx, u_idx, v_idx;
  PetscReal          dx = app->dx, dy = app->dy, re = app->re;
  PetscInt           nx = app->nx, ny = app.ny;

  PetscCall(VecGetArrayRead(X, &x_ptr));
  PetscCall(VecGetArrayRead(Xdot, &xdot_ptr));
  PetscCall(VecGetArray(F, &f_ptr));
  PetscCall(VecZeroEntries(F));

  const PetscScalar *p = x_ptr + app->p_offset;
  const PetscScalar *u = x_ptr + app->u_offset;
  const PetscScalar *v = x_ptr + app->v_offset;

  const PetscScalar *pdot = xdot_ptr + app->p_offset;
  const PetscScalar *udot = xdot_ptr + app->u_offset;
  const PetscScalar *vdot = xdot_ptr + app->v_offset;

  PetscScalar *fp = f_ptr + app->p_offset;
  PetscScalar *fu = f_ptr + app->u_offset;
  PetscScalar *fv = f_ptr + app->v_offset;

  /* --- Continuity equations (at p-locations) --- */
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      p_idx = P_IDX(i, j, nx);
      if (i == 0 && j == 0) { /* Pressure pinning */
        fp[p_idx] = p[p_idx] - 0.0;
      } else {
        PetscScalar u_e = u[U_IDX(i + 1, j, nx)];
        PetscScalar u_w = u[U_IDX(i, j, nx)];
        PetscScalar v_n = v[V_IDX(i, j + 1, nx)];
        PetscScalar v_s = v[V_IDX(i, j, nx)];
        fp[p_idx] = (u_e - u_w) / dx + (v_n - v_s) / dy;
      }
    }
  }

  /* --- X-Momentum equations (at u-locations) --- */
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx + 1; i++) {
      u_idx = U_IDX(i, j, nx);
      /* Boundary conditions */
      if (i == 0 || i == nx) { /* Left/Right walls u=0 */
        fu[u_idx] = u[u_idx] - 0.0;
        continue;
      }

      /* Interior u-nodes */
      PetscScalar u_P = u[u_idx];
      PetscScalar u_E = u[U_IDX(i + 1, j, nx)];
      PetscScalar u_W = u[U_IDX(i - 1, j, nx)];
      PetscScalar u_N = (j == ny - 1) ? 2.0 * 1.0 - u_P : u[U_IDX(i, j + 1, nx)]; /* u_top=1 */
      PetscScalar u_S = (j == 0)     ? 2.0 * 0.0 - u_P : u[U_IDX(i, j - 1, nx)]; /* u_bot=0 */

      PetscScalar v_NE = v[V_IDX(i, j + 1, nx)];
      PetscScalar v_NW = v[V_IDX(i - 1, j + 1, nx)];
      PetscScalar v_SE = v[V_IDX(i, j, nx)];
      PetscScalar v_SW = v[V_IDX(i - 1, j, nx)];

      /* Convective fluxes */
      PetscScalar conv_u_e = 0.5 * (u_P + u_E);
      PetscScalar conv_u_w = 0.5 * (u_P + u_W);
      PetscScalar conv_v_n = 0.5 * (v_NE + v_NW);
      PetscScalar conv_v_s = 0.5 * (v_SE + v_SW);

      PetscScalar flux_ue = conv_u_e * conv_u_e;
      PetscScalar flux_uw = conv_u_w * conv_u_w;
      PetscScalar flux_un = conv_v_n * 0.5 * (u_P + u_N);
      PetscScalar flux_us = conv_v_s * 0.5 * (u_P + u_S);

      PetscScalar convection = ((flux_ue - flux_uw) / dx) + ((flux_un - flux_us) / dy);

      /* Diffusive fluxes */
      PetscScalar diffusion = (1.0 / re) * ((u_E - 2.0 * u_P + u_W) / (dx * dx) + (u_N - 2.0 * u_P + u_S) / (dy * dy));

      /* Pressure gradient */
      PetscScalar p_E = p[P_IDX(i, j, nx)];
      PetscScalar p_W = p[P_IDX(i - 1, j, nx)];
      PetscScalar pressure_grad = (p_E - p_W) / dx;

      fu[u_idx] = udot[u_idx] + convection + pressure_grad - diffusion;
    }
  }

  /* --- Y-Momentum equations (at v-locations) --- */
  for (j = 0; j < ny + 1; j++) {
    for (i = 0; i < nx; i++) {
      v_idx = V_IDX(i, j, nx);
      /* Boundary conditions */
      if (j == 0 || j == ny) { /* Bottom/Top walls v=0 */
        fv[v_idx] = v[v_idx] - 0.0;
        continue;
      }

      /* Interior v-nodes */
      PetscScalar v_P = v[v_idx];
      PetscScalar v_N = v[V_IDX(i, j + 1, nx)];
      PetscScalar v_S = v[V_IDX(i, j - 1, nx)];
      PetscScalar v_E = (i == nx - 1) ? 2.0 * 0.0 - v_P : v[V_IDX(i + 1, j, nx)]; /* v_right=0 */
      PetscScalar v_W = (i == 0)     ? 2.0 * 0.0 - v_P : v[V_IDX(i - 1, j, nx)]; /* v_left=0 */

      PetscScalar u_NE = u[U_IDX(i + 1, j, nx)];
      PetscScalar u_NW = u[U_IDX(i, j, nx)];
      PetscScalar u_SE = u[U_IDX(i + 1, j - 1, nx)];
      PetscScalar u_SW = u[U_IDX(i, j - 1, nx)];

      /* Convective fluxes */
      PetscScalar conv_u_e = 0.5 * (u_NE + u_SE);
      PetscScalar conv_u_w = 0.5 * (u_NW + u_SW);
      PetscScalar conv_v_n = 0.5 * (v_P + v_N);
      PetscScalar conv_v_s = 0.5 * (v_P + v_S);

      PetscScalar flux_ve = conv_u_e * 0.5 * (v_P + v_E);
      PetscScalar flux_vw = conv_u_w * 0.5 * (v_P + v_W);
      PetscScalar flux_vn = conv_v_n * conv_v_n;
      PetscScalar flux_vs = conv_v_s * conv_v_s;

      PetscScalar convection = ((flux_ve - flux_vw) / dx) + ((flux_vn - flux_vs) / dy);

      /* Diffusive fluxes */
      PetscScalar diffusion = (1.0 / re) * ((v_E - 2.0 * v_P + v_W) / (dx * dx) + (v_N - 2.0 * v_P + v_S) / (dy * dy));

      /* Pressure gradient */
      PetscScalar p_N = p[P_IDX(i, j, nx)];
      PetscScalar p_S = p[P_IDX(i, j - 1, nx)];
      PetscScalar pressure_grad = (p_N - p_S) / dy;

      fv[v_idx] = vdot[v_idx] + convection + pressure_grad - diffusion;
    }
  }

  PetscCall(VecRestoreArrayRead(X, &x_ptr));
  PetscCall(VecRestoreArrayRead(Xdot, &xdot_ptr));
  PetscCall(VecRestoreArray(F, &f_ptr));
  return 0;
}

PetscErrorCode FormIJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat J, Mat P, void *ctx)
{
  AppCtx*            app = (AppCtx*)ctx;
  const PetscScalar *x_ptr;
  PetscInt           i, j, row;
  PetscReal          dx = app->dx, dy = app->dy, re = app->re;
  PetscInt           nx = app->nx, ny = app->ny;
  PetscScalar        v[13];
  PetscInt           c[13];

  PetscCall(VecGetArrayRead(X, &x_ptr));
  PetscCall(MatZeroEntries(J));

  const PetscScalar *p = x_ptr + app->p_offset;
  const PetscScalar *u = x_ptr + app->u_offset;
  const PetscScalar *v = x_ptr + app->v_offset;

  /* --- Continuity equations --- */
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      row = app->p_offset + P_IDX(i, j, nx);
      if (i == 0 && j == 0) { /* Pressure pinning */
        v[0] = 1.0;
        c[0] = row;
        PetscCall(MatSetValues(J, 1, &row, 1, c, v, INSERT_VALUES));
      } else {
        c[0] = app->u_offset + U_IDX(i + 1, j, nx); v[0] = 1.0 / dx;
        c[1] = app->u_offset + U_IDX(i, j, nx);     v[1] = -1.0 / dx;
        c[2] = app->v_offset + V_IDX(i, j + 1, nx); v[2] = 1.0 / dy;
        c[3] = app->v_offset + V_IDX(i, j, nx);     v[3] = -1.0 / dy;
        PetscCall(MatSetValues(J, 1, &row, 4, c, v, INSERT_VALUES));
      }
    }
  }

  /* --- X-Momentum equations --- */
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx + 1; i++) {
      row = app->u_offset + U_IDX(i, j, nx);
      if (i == 0 || i == nx) { /* Boundary condition rows */
        v[0] = 1.0;
        c[0] = row;
        PetscCall(MatSetValues(J, 1, &row, 1, c, v, INSERT_VALUES));
        continue;
      }

      /* Get stencil values */
      PetscScalar u_P = u[U_IDX(i, j, nx)];
      PetscScalar u_E = u[U_IDX(i + 1, j, nx)];
      PetscScalar u_W = u[U_IDX(i - 1, j, nx)];
      PetscScalar u_N = (j == ny - 1) ? 2.0 * 1.0 - u_P : u[U_IDX(i, j + 1, nx)];
      PetscScalar u_S = (j == 0)     ? 2.0 * 0.0 - u_P : u[U_IDX(i, j - 1, nx)];
      PetscScalar v_NE = v[V_IDX(i, j + 1, nx)];
      PetscScalar v_NW = v[V_IDX(i - 1, j + 1, nx)];
      PetscScalar v_SE = v[V_IDX(i, j, nx)];
      PetscScalar v_SW = v[V_IDX(i - 1, j, nx)];

      PetscInt k = 0;
      /* Time derivative term: a * d(u)/d(u) */
      c[k] = row; v[k] = a;

      /* Convection terms (derivatives) */
      /* d/du_P */ v[k] += (0.5*(u_P+u_E) - 0.5*(u_P+u_W))/dx + (0.5*0.5*(v_NE+v_NW) + 0.5*0.5*(v_SE+v_SW))/dy;
      /* d/du_E */ c[++k] = app->u_offset + U_IDX(i+1,j,nx); v[k] = (0.5*(u_P+u_E))/dx;
      /* d/du_W */ c[++k] = app->u_offset + U_IDX(i-1,j,nx); v[k] = (-0.5*(u_P+u_W))/dx;
      /* d/du_N */ if (j < ny-1) { c[++k] = app->u_offset + U_IDX(i,j+1,nx); v[k] = (0.5*0.5*(v_NE+v_NW))/dy; }
      /* d/du_S */ if (j > 0)    { c[++k] = app->u_offset + U_IDX(i,j-1,nx); v[k] = (0.5*0.5*(v_SE+v_SW))/dy; }
      /* d/dv_NE */ c[++k] = app->v_offset + V_IDX(i,j+1,nx);   v[k] = (0.5*0.5*(u_P+u_N))/dy;
      /* d/dv_NW */ c[++k] = app->v_offset + V_IDX(i-1,j+1,nx); v[k] = (0.5*0.5*(u_P+u_N))/dy;
      /* d/dv_SE */ c[++k] = app->v_offset + V_IDX(i,j,nx);     v[k] = (-0.5*0.5*(u_P+u_S))/dy;
      /* d/dv_SW */ c[++k] = app->v_offset + V_IDX(i-1,j,nx);   v[k] = (-0.5*0.5*(u_P+u_S))/dy;

      /* Diffusion terms (derivatives) */
      PetscScalar dxx = 1.0/(re*dx*dx), dyy = 1.0/(re*dy*dy);
      v[0] += -(-2.0*dxx - 2.0*dyy);
      if (j == ny-1 || j == 0) v[0] += -( -dyy ); /* Boundary modification */
      c[++k] = app->u_offset + U_IDX(i+1,j,nx); v[k] = -dxx;
      c[++k] = app->u_offset + U_IDX(i-1,j,nx); v[k] = -dxx;
      if (j < ny-1) { c[++k] = app->u_offset + U_IDX(i,j+1,nx); v[k] = -dyy; }
      if (j > 0)    { c[++k] = app->u_offset + U_IDX(i,j-1,nx); v[k] = -dyy; }

      /* Pressure gradient terms */
      c[++k] = app->p_offset + P_IDX(i,j,nx);     v[k] = 1.0/dx;
      c[++k] = app->p_offset + P_IDX(i-1,j,nx);   v[k] = -1.0/dx;

      PetscCall(MatSetValues(J, 1, &row, k+1, c, v, INSERT_VALUES));
    }
  }

  /* --- Y-Momentum equations --- */
  for (j = 0; j < ny + 1; j++) {
    for (i = 0; i < nx; i++) {
      row = app->v_offset + V_IDX(i, j, nx);
      if (j == 0 || j == ny) { /* Boundary condition rows */
        v[0] = 1.0;
        c[0] = row;
        PetscCall(MatSetValues(J, 1, &row, 1, c, v, INSERT_VALUES));
        continue;
      }

      PetscScalar v_P = v[V_IDX(i, j, nx)];
      PetscScalar v_N = v[V_IDX(i, j + 1, nx)];
      PetscScalar v_S = v[V_IDX(i, j - 1, nx)];
      PetscScalar v_E = (i == nx - 1) ? 2.0 * 0.0 - v_P : v[V_IDX(i + 1, j, nx)];
      PetscScalar v_W = (i == 0)     ? 2.0 * 0.0 - v_P : v[V_IDX(i - 1, j, nx)];
      PetscScalar u_NE = u[U_IDX(i + 1, j, nx)];
      PetscScalar u_NW = u[U_IDX(i, j, nx)];
      PetscScalar u_SE = u[U_IDX(i + 1, j - 1, nx)];
      PetscScalar u_SW = u[U_IDX(i, j - 1, nx)];

      PetscInt k = 0;
      c[k] = row; v[k] = a;

      /* Convection */
      v[k] += (0.5*0.5*(u_NE+u_SE) - 0.5*0.5*(u_NW+u_SW))/dx + (0.5*(v_P+v_N) - 0.5*(v_P+v_S))/dy;
      if (i < nx-1) { c[++k] = app->v_offset + V_IDX(i+1,j,nx); v[k] = (0.5*0.5*(u_NE+u_SE))/dx; }
      if (i > 0)    { c[++k] = app->v_offset + V_IDX(i-1,j,nx); v[k] = (-0.5*0.5*(u_NW+u_SW))/dx; }
      c[++k] = app->v_offset + V_IDX(i,j+1,nx); v[k] = (0.5*(v_P+v_N))/dy;
      c[++k] = app->v_offset + V_IDX(i,j-1,nx); v[k] = (-0.5*(v_P+v_S))/dy;
      c[++k] = app->u_offset + U_IDX(i+1,j,nx); v[k] = (0.5*0.5*(v_P+v_E))/dx;
      c[++k] = app->u_offset + U_IDX(i,j,nx);   v[k] = (-0.5*0.5*(v_P+v_W))/dx;
      if (j > 0) {
        c[++k] = app->u_offset + U_IDX(i+1,j-1,nx); v[k] = (0.5*0.5*(v_P+v_E))/dx;
        c[++k] = app->u_offset + U_IDX(i,j-1,nx);   v[k] = (-0.5*0.5*(v_P+v_W))/dx;
      }

      /* Diffusion */
      PetscScalar dxx = 1.0/(re*dx*dx), dyy = 1.0/(re*dy*dy);
      v[0] += -(-2.0*dxx - 2.0*dyy);
      if (i == nx-1 || i == 0) v[0] += -(-dxx);
      if (i < nx-1) { c[++k] = app->v_offset + V_IDX(i+1,j,nx); v[k] = -dxx; }
      if (i > 0)    { c[++k] = app->v_offset + V_IDX(i-1,j,nx); v[k] = -dxx; }
      c[++k] = app->v_offset + V_IDX(i,j+1,nx); v[k] = -dyy;
      c[++k] = app->v_offset + V_IDX(i,j-1,nx); v[k] = -dyy;

      /* Pressure */
      c[++k] = app->p_offset + P_IDX(i,j,nx);   v[k] = 1.0/dy;
      c[++k] = app->p_offset + P_IDX(i,j-1,nx); v[k] = -1.0/dy;

      PetscCall(MatSetValues(J, 1, &row, k+1, c, v, INSERT_VALUES));
    }
  }

  PetscCall(VecRestoreArrayRead(X, &x_ptr));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  }
  return 0;
}

PetscErrorCode ComputeMaxDivergence(Vec X, AppCtx *app, PetscReal *max_div)
{
  const PetscScalar *x_ptr;
  PetscInt           i, j;
  PetscReal          dx = app->dx, dy = app->dy;
  PetscInt           nx = app->nx, ny = app->ny;
  PetscReal          local_max_div = 0.0;

  PetscCall(VecGetArrayRead(X, &x_ptr));
  const PetscScalar *u = x_ptr + app->u_offset;
  const PetscScalar *v = x_ptr + app->v_offset;

  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      PetscScalar u_e = u[U_IDX(i + 1, j, nx)];
      PetscScalar u_w = u[U_IDX(i, j, nx)];
      PetscScalar v_n = v[V_IDX(i, j + 1, nx)];
      PetscScalar v_s = v[V_IDX(i, j, nx)];
      PetscReal   div = PetscRealPart((u_e - u_w) / dx + (v_n - v_s) / dy);
      if (PetscAbsReal(div) > local_max_div) {
        local_max_div = PetscAbsReal(div);
      }
    }
  }

  PetscCall(VecRestoreArrayRead(X, &x_ptr));
  PetscCall(MPI_Allreduce(&local_max_div, max_div, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD));

  return 0;
}
