# Schrödinger

### About
Project for simulating the Schrödinger equation with configurable boundary conditions using Python 3.11. Work in progress.

### Preview

Harmonic oscillator, $n=3$
![alt text](images/ho3.gif)

Particle in a growing box, $n=3$
![alt text](images/pib3.gif)

### Instructions

1. Clone the repository and open its folder from the CLI.

2. (Optional) Create a virtual environment on which to install dependencies with the command `python -m venv venv` followed by `venv/Scripts/activate`.

3. Run the command `pip install -r requirements.txt` to install dependencies.

The modules in `src` should now work, and you should be able to use the `examples.ipynb` notebook.

### Theory 

#### Trotterisation

Our goal is to solve the Schrödinger equation for a 1D Hamiltonian

$$
\hat{H} = \hat{T} + \hat{V}(t, x),\quad
\hat{T}=-\frac{1}{2}\partial_x^2
$$

with a potential $\hat{V}$ which depends on time $t$, position $x\in[x_1,x_2]$.

A formal solution may be given by time ordered exponential, but for numerical integration, it is most common to work with Suzuki-Trotter decomposition:

$$
e^{-i\delta t(\hat{T} + \hat{V})} = e^{-i\delta t \hat{T}}e^{-i\delta t\hat{V}}+O(\delta t^2)
$$

$$
U(t_0,t)=\mathcal{T} \{e^{-i\int_{t_0}^t \hat{H}(t')\mathrm{d}t'} \}=\lim_{\delta t\rightarrow0}\prod_{t'=t_0}^t [e^{-i\delta t \hat{T}}e^{-i\delta t\hat{V}(t')}]
$$

We can improve convergence with arbitrarily high-order Suzuki-Trotter decompositions [1], but it typically suffices to stay at second order: 

$$
e^{-i\delta t(\hat{T} + \hat{V})} = e^{-i\frac{\delta t}{2} \hat{V}}e^{-i\delta t \hat{T}}e^{-i\frac{\delta t}{2} \hat{V}}+O(\delta t^3)
$$

Our calculations now become massively simplified if we notice that $\hat{V}$ is diagonal in the position basis and $\hat{T}$ is diagonal in the momentum basis, which means that these operator exponentials just become the exponentials of numbers under an appropriate change of basis

$$
e^{-i\delta t(\hat{T} + \hat{V})} = e^{-i\frac{\delta t}{2} \hat{V}}\mathcal{F}^\dagger e^{-i\frac{\delta t}{2}k^2}\mathcal{F} e^{-i\frac{\delta t}{2} \hat{V}}+O(\delta t^3)
$$

with $\mathcal{F}$ the 1D Fourier transform and $k$ the corresponding momenta. This integration scheme is the well known split operator method.

#### Boundary Conditions
The main boundary conditions (BCs) of interest in quantum mechanics are periodic BCs and Dirichlet BCs, implementable via ordinary Fourier transform and sine Fourier transform, respectively. We can even assume moving boundaries if we wish to study the effect of diabatic/adiabatic changes on our system.

Additionally, for Dirichlet boundary conditions, we can calculate solutions more general than simply vanishing at the boundary:

$$
\begin{cases}
\psi\bigr\vert_{x=x_1(t)}=f_1(t) \\
\psi\bigr\vert_{x=x_2(t)}=f_2(t)
\end{cases}
$$

If we introduce an ansatz $\psi=u_h+u_p$, where $u_h$ vanishes at the boundary and $u_p$ satisfies the boundary conditions, we obtain the equation

$$
i\partial_t u_h = \hat{H}u_h + (\hat{H} - i\partial_t) u_p
$$

which can be solved via integrating factor:

$$
i\partial_t(e^{it\hat{H}}u_h)=e^{it\hat{H}}[(\hat{H}-i\partial_t)u_p]
$$

We can then see it is convenient to define $u_p$ as a cubic polynomial, so that it satisfies the boundary conditions and so that $(\hat{H} - i\partial_t) u_p$ vanishes at the boundary.

$$
u_p = Ax^3+Bx^2+Cx+D
$$

$$
\begin{pmatrix}
A\\ 
B
\end{pmatrix}=
\frac{1}{6(x_2-x_1)}
\begin{pmatrix}
-1 & 1\\
3x_2 & -3x_1
\end{pmatrix}
\begin{pmatrix}
(V\bigr\vert_{x_1}-i\partial_t)f_1\\
(V\bigr\vert_{x_2}-i\partial_t)f_2
\end{pmatrix}
$$

$$
\begin{pmatrix}
C\\ 
D
\end{pmatrix}=
\frac{1}{x_2-x_1}
\begin{pmatrix}
-1 & 1\\
x_2 & -x_1
\end{pmatrix}
\begin{pmatrix}
f_1-Ax_1^3-Bx_1^2\\
f_2-Ax_2^3-Bx_2^2
\end{pmatrix}
$$

It is convenient to handle the derivatives of $f_1$, $f_2$ and $u_p$ via automatic differentiation [2].

### References

[1] Hatano, N., & Suzuki, M. (2005). *Finding Exponential Product Formulas of Higher Orders*. Springer Berlin Heidelberg. https://arxiv.org/abs/math-ph/0506007

[2] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, & Qiao Zhang. (2018). *JAX: composable transformations of Python+NumPy programs*. https://jax.readthedocs.io/en/latest/quickstart.html