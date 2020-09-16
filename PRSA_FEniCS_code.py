from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Tested with dolfin version 2017.2.0

# Geometric parameters
eps = 1.0
Lx = 3*eps
t = Constant(1.e-2)
G0 = 1.0

# Constitutive parameters
Y = Constant(1.)
nu = Constant(0.)
D = Y*t**3/(12.*(1 - nu**2))

# Clamping steps
steps = 10

# Mesh
mesh_divs = 500
mesh = IntervalMesh(mesh_divs, 0., Lx)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2.0

# Solver tolerance
abs_tol = 1E-12

# Discrete space
DG = FiniteElement("DG", interval, degree = 0)
P1 = FiniteElement("Lagrange", interval, degree = 1)
P2 = FiniteElement("Lagrange", interval, degree = 2)
P3 = FiniteElement("Lagrange", interval, degree = 3)
element = MixedElement([P2, P2, P2])

# Functions
Q = FunctionSpace(mesh, element)
q_, q, q_t = Function(Q), TrialFunction(Q), TestFunction(Q)
v_, k_, f_ = split(q_)
v, k, f = split(q)
v_t, k_t, f_t = split(q_t)

# Stress-free shape (natural)
stressfree_shape = Expression(('0.0', 'G0', '0.0'), eps=eps, G0=G0, degree = 0)
q0 = interpolate(stressfree_shape, Q)
v0, k0, f0 = q0.split()

# Energy densities
delk_ = k_ - k0
psi_b = 0.5*((1/eps**2)*(delk_**2) + (1/eps)*(2*nu*delk_*v_.dx(0).dx(0)) \
    + ((1./6.)*(1. - nu)*(delk_.dx(0))**2 + (v_.dx(0).dx(0))**2) + (eps**2)*(k_.dx(0).dx(0))**2/720.)

psi_m = 0.5*((1/eps**4)*(720*f_**2) + (1/eps**2)*(240./7.)*((1. + nu)*(f_.dx(0))**2 \
    + nu*f_*f_.dx(0).dx(0)) + (10./7.)*(f_.dx(0).dx(0))**2)

psi_c = (1/eps)*(v_.dx(0)*(k_*f_).dx(0) - v0.dx(0)*(k0*f_).dx(0)) + (1./84.)*((k_.dx(0)*(k_*f_).dx(0) \
    - k0.dx(0)*(k0*f_).dx(0)) + (3./2.)*(k_**2 - k0**2)*f_.dx(0).dx(0))

# Continuous-Discontinuous Galerkin formulation penalisation weights
Cv = D
Ck = (eps**2)*D/720.
Cf = (10./7.)*(1./(t*Y))
alpha_v = Constant(t*Cv)
alpha_k = Constant(t*Ck)
alpha_f = Constant(Cf/t)

v_CDG = -avg(Cv*v_.dx(0).dx(0))*jump(v_t.dx(0))*dS  \
    -jump(Cv*v_.dx(0))*avg(v_t.dx(0).dx(0))*dS \
    +alpha_v('+')/h_avg*jump(v_.dx(0))*jump(v_t.dx(0))*dS

k_CDG = -avg(Ck*k_.dx(0).dx(0))*jump(k_t.dx(0))*dS  \
    -jump(Ck*k_.dx(0))*avg(k_t.dx(0).dx(0))*dS \
    +alpha_k('+')/h_avg*jump(k_.dx(0))*jump(k_t.dx(0))*dS

f_CDG = +avg(Cf*f_.dx(0).dx(0))*jump(f_t.dx(0))*dS  \
    +jump(Cf*f_.dx(0))*avg(f_t.dx(0).dx(0))*dS \
    -alpha_f('+')/h_avg*jump(f_.dx(0))*jump(f_t.dx(0))*dS

# Weak boundary conditions
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] <= DOLFIN_EPS

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - Lx) <= DOLFIN_EPS

Gamma_L = Left()
Gamma_R = Right()

exterior_facet_domains = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
exterior_facet_domains.set_all(0)
Gamma_L.mark(exterior_facet_domains, 1)
Gamma_R.mark(exterior_facet_domains, 2)
ds = ds(subdomain_data = exterior_facet_domains)

alpha_vBC, alpha_kBC, alpha_fBC = 1E5, 1E5, 1E5

v_BC = -Cv*v_.dx(0).dx(0)*v_t.dx(0)*ds(1)  \
    -Cv*v_.dx(0)*v_t.dx(0).dx(0)*ds(1) \
    +alpha_vBC*v_.dx(0)*v_t.dx(0)*ds(1)

k_BC = -Ck*k_.dx(0).dx(0)*k_t.dx(0)*ds(1)  \
    -Ck*k_.dx(0)*k_t.dx(0).dx(0)*ds(1) \
    +alpha_kBC*k_.dx(0)*k_t.dx(0)*ds(1)

f_BCL = +Cf*f_.dx(0).dx(0)*f_t.dx(0)*ds(1)  \
    +Cf*f_.dx(0)*f_t.dx(0).dx(0)*ds(1) \
    -alpha_fBC*f_.dx(0)*f_t.dx(0)*ds(1)

f_BCR = +Cf*f_.dx(0).dx(0)*f_t.dx(0)*ds(2)  \
    +Cf*f_.dx(0)*f_t.dx(0).dx(0)*ds(2) \
    -alpha_fBC*f_.dx(0)*f_t.dx(0)*ds(2)

# Strong boundary conditions
vbc = Constant(0.0)
kbc = Expression('(1. - c)*G0', c=0., G0=G0, degree=0, domain=mesh)
fbc = Constant(0.0)
bcv = DirichletBC(Q.sub(0), project(vbc, Q.sub(0).collapse()), Gamma_L)
bcf1 = DirichletBC(Q.sub(2), fbc, Gamma_L)
bcf2 = DirichletBC(Q.sub(2), fbc, Gamma_R)

# Problem
L = D*psi_b*dx - 1./(t*Y)*psi_m*dx + psi_c*dx
F = derivative(L, q_, q_t) + v_CDG + k_CDG + f_CDG + v_BC + k_BC + f_BCL + f_BCR
dF = derivative(F, q_, q)

# Initial guess
q_.assign(stressfree_shape)

# Steps
cs = np.linspace(0.0, 1., steps)

# Output directory
out_dir = "output-enhancedrod/"

# Solve
for count, i in enumerate(cs):

    # Update boundary conditions
    kbc.c = i
    bck = DirichletBC(Q.sub(1), project(kbc, Q.sub(0).collapse()), Gamma_L)
    bcs = [bcv, bck, bcf1, bcf2]

    # Solution
    problem = NonlinearVariationalProblem(F, q_, bcs, J = dF)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters.newton_solver.absolute_tolerance = abs_tol

    solver.solve()
    v_h, k_h, f_h = q_.split(deepcopy=True)

# Post processing
xls, vls, kls, fls = [], [], [], []
coordinates = sorted(mesh.coordinates())
xtol = 0.0
coordinates = np.linspace(0, Lx, 2000)
xlim = 2.0*eps
xf = xlim

for i in coordinates:
    xls.append(i)
    vls.append(v_h(i))
    kls.append(k_h(i))
    fls.append(f_h(i))

# Figure 1
plt.figure(1)
plt.plot(xls, vls, 'b', linewidth=2.)
plt.xlabel(r"$x/\varepsilon$", fontsize=14)
plt.ylabel(r"$v$", fontsize=14)
plt.xticks(np.arange(0, xf, step=xf/10))
plt.xlim(0.0, xlim)
plt.grid(True)
plt.savefig(out_dir + "v.png")

# Figure 2
plt.figure(2)
plt.plot(xls, kls, 'g', linewidth=2.)
plt.xlabel(r"$x/\varepsilon$", fontsize=14)
plt.ylabel(r"$k/k_0$", fontsize=14)
plt.xticks(np.arange(0, xf, step=xf/10))
plt.xlim(0.0, xlim)
plt.grid(True)
plt.savefig(out_dir + "k.png")

# Figure 3
plt.figure(3)
plt.plot(xls, fls, 'r', linewidth=2.)
plt.xlabel(r"$x/\varepsilon$", fontsize=14)
plt.ylabel(r"$f$", fontsize=14)
plt.xticks(np.arange(0, xf, step=xf/10))
plt.xlim(0.0, xlim)
plt.grid(True)
plt.savefig(out_dir + "f.png")