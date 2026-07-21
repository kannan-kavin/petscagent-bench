/*
  2D Incompressible Navier-Stokes on unit square using FVM on a staggered (MAC) grid.
  Driven cavity: top wall u=1, others 0. Re = 1/nu, here nu=0.01 (Re=100).
  Implicit time stepping via PETSc TS (TSSetIFunction / TSSetIJacobian by coloring).

  Layout (Nx x Ny pressure cells):
    - p(i,j): cell center, i=0..Nx-1, j=0..Ny-1               -> Np = Nx*Ny
    - u(i,j): vertical faces, i=0..Nx,  j=0..Ny-1             -> Nu = (Nx+1)*Ny
        u(0,j)=0 (left wall), u(Nx,j)=0 (right wall) enforced as Dirichlet eqs
    - v(i,j): horizontal faces, i=0..Nx-1, j=0..Ny             -> Nv = Nx*(Ny+1)
        v(i,0)=0 (bottom), v(i,Ny)=0 (top) enforced as Dirichlet eqs

  Global vector ordering: [u(0..Nu-1), v(0..Nv-1), p(0..Np-1)]

  Equations assembled as F(t,X,Xdot) = 0:
    - For interior u faces: d(u*vol)/dt + sum fluxes + (p_E - p_W)*dy = 0
      vol_u = dx*dy (control volume size for u)
      Convective fluxes computed on u-CV faces using central differencing of u with
      mass-flux face velocities interpolated from neighbors.
      Diffusive fluxes: nu * du/dn * face_length.
    - For interior v faces: analogous.
    - For boundary u/v faces: Dirichlet (u=u_bc) directly.
    - For pressure cells: continuity (u_e - u_w)*dy + (v_n - v_s)*dx = 0,
      with one cell pinned to fix pressure level (p(0,0)=0).

  Output: a single line with the maximum |divergence| over pressure cells.
*/

#include <petscts.h>
#include <petscsnes.h>
#include <petscdmshell.h>
#include <math.h>
#include <stdio.h>

typedef struct {
  PetscInt  Nx, Ny;
  PetscInt  Nu, Nv, Np, N;
  PetscReal Lx, Ly;
  PetscReal dx, dy;
  PetscReal nu;
} AppCtx;

static inline PetscInt UIDX(AppCtx *c, PetscInt i, PetscInt j){ return i*c->Ny + j; }
static inline PetscInt VIDX(AppCtx *c, PetscInt i, PetscInt j){ return c->Nu + i*(c->Ny+1) + j; }
static inline PetscInt PIDX(AppCtx *c, PetscInt i, PetscInt j){ return c->Nu + c->Nv + i*c->Ny + j; }

/* Helpers to fetch field values with array */
static inline PetscReal Uval(const PetscScalar *x, AppCtx *c, PetscInt i, PetscInt j){ return PetscRealPart(x[UIDX(c,i,j)]); }
static inline PetscReal Vval(const PetscScalar *x, AppCtx *c, PetscInt i, PetscInt j){ return PetscRealPart(x[VIDX(c,i,j)]); }
static inline PetscReal Pval(const PetscScalar *x, AppCtx *c, PetscInt i, PetscInt j){ return PetscRealPart(x[PIDX(c,i,j)]); }

static PetscErrorCode FormIFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ctx)
{
  AppCtx *c = (AppCtx*)ctx;
  const PetscScalar *x, *xdot;
  PetscScalar *f;
  PetscReal dx = c->dx, dy = c->dy, nu = c->nu;
  PetscInt Nx = c->Nx, Ny = c->Ny;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  /* ---------------- U momentum ---------------- */
  for (PetscInt i = 0; i <= Nx; ++i) {
    for (PetscInt j = 0; j < Ny; ++j) {
      PetscInt row = UIDX(c,i,j);
      if (i == 0 || i == Nx) {
        /* Left/right wall: u = 0 */
        f[row] = x[row] - 0.0;
        continue;
      }
      PetscReal uC = Uval(x,c,i,j);
      PetscReal uE = Uval(x,c,i+1,j);
      PetscReal uW = Uval(x,c,i-1,j);
      /* North/south u values with ghost for walls */
      PetscReal uN, uS;
      PetscReal u_top_bc = 1.0, u_bot_bc = 0.0;
      if (j == Ny-1) {
        /* u above is ghost; wall at y=1 with u_bc=1: face value = u_bc => ghost = 2*u_bc - uC */
        uN = 2.0*u_top_bc - uC;
      } else {
        uN = Uval(x,c,i,j+1);
      }
      if (j == 0) {
        uS = 2.0*u_bot_bc - uC;
      } else {
        uS = Uval(x,c,i,j-1);
      }

      /* Face mass-flux velocities on u-CV faces.
         u-CV centered at (i, j+0.5) in (x,y) face coords; spans [x_{i-1/2}, x_{i+1/2}] in x? 
         Actually staggered: u stored at x=i*dx, y=(j+0.5)*dy. u-CV spans x in [(i-0.5)dx,(i+0.5)dx], y in [j*dy,(j+1)*dy].
         East face of u-CV at x=(i+0.5)dx -> pressure cell center i,j (in pressure index). Face u = 0.5*(uC+uE).
         West face at x=(i-0.5)dx -> pressure cell i-1,j. Face u = 0.5*(uW+uC).
         North face at y=(j+1)*dy -> v faces at this y line: v(i-1,j+1) and v(i,j+1). Face v = 0.5*(v(i-1,j+1)+v(i,j+1)).
         South face at y=j*dy -> v(i-1,j) and v(i,j). Face v = 0.5.
         u value at north face = 0.5*(uC+uN); south = 0.5*(uC+uS). */
      PetscReal ue = 0.5*(uC + uE);
      PetscReal uw = 0.5*(uW + uC);
      PetscReal vn = 0.5*(Vval(x,c,i-1,j+1) + Vval(x,c,i,j+1));
      PetscReal vs = 0.5*(Vval(x,c,i-1,j)   + Vval(x,c,i,j));
      PetscReal u_face_n = 0.5*(uC + uN);
      PetscReal u_face_s = 0.5*(uC + uS);

      /* Convective fluxes (mass flux * u) across faces, with face length */
      PetscReal Fconv_e = (ue) * (ue) * dy;
      PetscReal Fconv_w = (uw) * (uw) * dy;
      PetscReal Fconv_n = (vn) * (u_face_n) * dx;
      PetscReal Fconv_s = (vs) * (u_face_s) * dx;

      /* Diffusive fluxes: nu * du/dn * face_length */
      PetscReal Fdiff_e = nu * (uE - uC)/dx * dy;
      PetscReal Fdiff_w = nu * (uC - uW)/dx * dy;
      PetscReal dudy_n = (j == Ny-1) ? (uN - uC)/dy : (uN - uC)/dy; /* uN already ghost-adjusted */
      PetscReal dudy_s = (j == 0)    ? (uC - uS)/dy : (uC - uS)/dy;
      PetscReal Fdiff_n = nu * dudy_n * dx;
      PetscReal Fdiff_s = nu * dudy_s * dx;

      /* Pressure gradient: (p_E - p_W) * dy, where p_E = p(i,j), p_W = p(i-1,j) */
      PetscReal pE = Pval(x,c,i,j);
      PetscReal pW = Pval(x,c,i-1,j);
      PetscReal Fp = (pE - pW) * dy;

      PetscReal vol = dx*dy;
      f[row] = vol * PetscRealPart(xdot[row])
             + (Fconv_e - Fconv_w + Fconv_n - Fconv_s)
             - (Fdiff_e - Fdiff_w + Fdiff_n - Fdiff_s)
             + Fp;
    }
  }

  /* ---------------- V momentum ---------------- */
  for (PetscInt i = 0; i < Nx; ++i) {
    for (PetscInt j = 0; j <= Ny; ++j) {
      PetscInt row = VIDX(c,i,j);
      if (j == 0 || j == Ny) {
        f[row] = x[row] - 0.0;
        continue;
      }
      PetscReal vC = Vval(x,c,i,j);
      PetscReal vN = Vval(x,c,i,j+1);
      PetscReal vS = Vval(x,c,i,j-1);
      PetscReal vE, vW;
      PetscReal v_left_bc = 0.0, v_right_bc = 0.0;
      if (i == Nx-1) vE = 2.0*v_right_bc - vC; else vE = Vval(x,c,i+1,j);
      if (i == 0)    vW = 2.0*v_left_bc  - vC; else vW = Vval(x,c,i-1,j);

      /* v-CV centered at (i+0.5, j) -> spans x in [i*dx,(i+1)*dx], y in [(j-0.5)dy,(j+0.5)dy].
         North face y=(j+0.5)dy at pressure cell (i,j); v_face = 0.5*(vC+vN).
         South face y=(j-0.5)dy at pressure cell (i,j-1); v_face = 0.5*(vS+vC).
         East face x=(i+1)*dx: u(i+1,j-1) and u(i+1,j) -> u_face = 0.5; v_face value 0.5*(vC+vE).
         West face x=i*dx:    u(i,j-1) and u(i,j)     -> u_face = 0.5; v_face value 0.5*(vW+vC). */
      PetscReal vn = 0.5*(vC + vN);
      PetscReal vs = 0.5*(vS + vC);
      PetscReal ue = 0.5*(Uval(x,c,i+1,j-1) + Uval(x,c,i+1,j));
      PetscReal uw = 0.5*(Uval(x,c,i,  j-1) + Uval(x,c,i,  j));
      PetscReal v_face_e = 0.5*(vC + vE);
      PetscReal v_face_w = 0.5*(vW + vC);

      PetscReal Fconv_n = (vn) * (vn) * dx;
      PetscReal Fconv_s = (vs) * (vs) * dx;
      PetscReal Fconv_e = (ue) * (v_face_e) * dy;
      PetscReal Fconv_w = (uw) * (v_face_w) * dy;

      PetscReal Fdiff_n = nu * (vN - vC)/dy * dx;
      PetscReal Fdiff_s = nu * (vC - vS)/dy * dx;
      PetscReal Fdiff_e = nu * (vE - vC)/dx * dy;
      PetscReal Fdiff_w = nu * (vC - vW)/dx * dy;

      PetscReal pN = Pval(x,c,i,j);
      PetscReal pS = Pval(x,c,i,j-1);
      PetscReal Fp = (pN - pS) * dx;

      PetscReal vol = dx*dy;
      f[row] = vol * PetscRealPart(xdot[row])
             + (Fconv_e - Fconv_w + Fconv_n - Fconv_s)
             - (Fdiff_e - Fdiff_w + Fdiff_n - Fdiff_s)
             + Fp;
    }
  }

  /* ---------------- Continuity ---------------- */
  for (PetscInt i = 0; i < Nx; ++i) {
    for (PetscInt j = 0; j < Ny; ++j) {
      PetscInt row = PIDX(c,i,j);
      if (i == 0 && j == 0) {
        /* Pin pressure */
        f[row] = x[row] - 0.0;
        continue;
      }
      PetscReal ue = Uval(x,c,i+1,j);
      PetscReal uw = Uval(x,c,i,  j);
      PetscReal vn = Vval(x,c,i,  j+1);
      PetscReal vs = Vval(x,c,i,  j);
      f[row] = (ue - uw)*dy + (vn - vs)*dx;
    }
  }

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  AppCtx ctx;
  TS ts;
  Vec X, R;
  Mat J;
  PetscInt Nx = 16, Ny = 16;
  PetscReal Tfinal = 1.0;
  PetscReal dt = 0.05;

  ierr = PetscInitialize(&argc,&argv,NULL,NULL);if (ierr) return ierr;

  ierr = PetscOptionsGetInt(NULL,NULL,"-Nx",&Nx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Ny",&Ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-tf",&Tfinal,NULL);CHKERRQ(ierr);

  ctx.Nx = Nx; ctx.Ny = Ny;
  ctx.Lx = 1.0; ctx.Ly = 1.0;
  ctx.dx = ctx.Lx/Nx; ctx.dy = ctx.Ly/Ny;
  ctx.nu = 0.01;
  ctx.Nu = (Nx+1)*Ny;
  ctx.Nv = Nx*(Ny+1);
  ctx.Np = Nx*Ny;
  ctx.N  = ctx.Nu + ctx.Nv + ctx.Np;

  ierr = VecCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = VecSetSizes(X,PETSC_DECIDE,ctx.N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&R);CHKERRQ(ierr);
  ierr = VecSet(X,0.0);CHKERRQ(ierr);

  /* Build a dense-ish matrix using MatAIJ with generous preallocation (small problem). */
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,ctx.N,ctx.N);CHKERRQ(ierr);
  ierr = MatSetType(J,MATAIJ);CHKERRQ(ierr);
  /* Up to ~15 nonzeros per row should suffice */
  ierr = MatSeqAIJSetPreallocation(J,25,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(J,25,NULL,25,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,R,FormIFunction,&ctx);CHKERRQ(ierr);
  /* Use finite-difference coloring? Simpler: use SNES FD with default. We'll request matrix-free + FD Jacobian via TSComputeIJacobianDefault */
  ierr = TSSetIJacobian(ts,J,J,TSComputeIJacobianDefault,&ctx);CHKERRQ(ierr);

  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,Tfinal);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  SNES snes;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetTolerances(snes,1e-8,1e-10,1e-10,50,1000);CHKERRQ(ierr);
  KSP ksp; PC pc;
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);

  ierr = TSSolve(ts,X);CHKERRQ(ierr);

  /* Compute max divergence over pressure cells */
  const PetscScalar *x;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  PetscReal maxdiv = 0.0;
  for (PetscInt i = 0; i < ctx.Nx; ++i) {
    for (PetscInt j = 0; j < ctx.Ny; ++j) {
      PetscReal ue = Uval(x,&ctx,i+1,j);
      PetscReal uw = Uval(x,&ctx,i,  j);
      PetscReal vn = Vval(x,&ctx,i,  j+1);
      PetscReal vs = Vval(x,&ctx,i,  j);
      PetscReal div = (ue - uw)*ctx.dy + (vn - vs)*ctx.dx;
      PetscReal ad = PetscAbsReal(div);
      if (ad > maxdiv) maxdiv = ad;
    }
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"%.12e\n",(double)maxdiv);CHKERRQ(ierr);

  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&R);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
