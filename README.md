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

The modules in `schrodinger/` should now work, and you should be able to use the notebooks in the `examples/` folder.

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
U(t,t_0)=\mathcal{T} \{e^{-i\int_{t_0}^t \hat{H}(t')\mathrm{d}t'} \}=\lim_{\delta t\rightarrow0}\prod_{t'=t_0}^t [e^{-i\delta t \hat{T}}e^{-i\delta t\hat{V}(t')}]
$$

We can improve convergence with arbitrarily high-order Suzuki-Trotter decompositions [1], but it typically suffices to stay at second order: 

$$
e^{-i\delta t(\hat{T} + \hat{V})} = e^{-i\frac{\delta t}{2} \hat{V}}e^{-i\delta t \hat{T}}e^{-i\frac{\delta t}{2} \hat{V}}+O(\delta t^3)
$$

Our calculations now become massively simplified if we notice that $\hat{V}$ is diagonal in the position basis and $\hat{T}$ is diagonal in the momentum basis, which means that these operator exponentials just become the exponentials of numbers under an appropriate change of basis

$$
e^{-i\delta t(\hat{T} + \hat{V})} = e^{-i\frac{\delta t}{2} \hat{V}}{\mathcal{F}_x}^{-1} e^{-i\frac{\delta t}{2}k^2}\mathcal{F}_x e^{-i\frac{\delta t}{2} \hat{V}}+O(\delta t^3)
$$

with $\mathcal{F}_x$ the 1D Fourier transform in space and $k$ the corresponding momenta. This integration scheme is the well known split operator method, also known as Strang splitting.

#### Moving Boundaries

It is interesting to allow our quantum system to have moving boundaries $[x_1(t), x_2(t)]$, if we wish to study the effect of diabatic/adiabatic changes. Solving on a moving domain, we find 

$$
i\dot{\psi}=\hat{H}\psi+i\dot{x}\partial_x\psi
$$

where $\dot{}=\frac{\mathrm{d}}{\mathrm{d}t}$, and it is convenient to calculate $\dot{x}$ via automatic differentiation [3].

Let's introduce the transformation $\Psi=e^{-ix\dot x}\psi$, the PDE for our new wavefunction now reads

$$
i\dot\Psi=-\frac{1}{2}\partial_x^2\Psi+V_\mathrm{eff}\Psi
$$

$$
V_\mathrm{eff} = V + x\ddot x + \frac{1}{2}\dot x^2
$$

#### Boundary Conditions
The main boundary conditions (BCs) of interest in quantum mechanics are periodic BCs and Dirichlet BCs, implementable via ordinary Fourier transform and sine Fourier transform, respectively.

To study infinite domains, we can simulate their effect by implementing an absorbing boundary. This is typically done by adding non-reflective, absorbing layers (perfectly matched layers) to the boundary, which can be derived via complex coordinate stretching [2]. Formally, we replace

$$
x\rightarrow F(x)=x+e^{i\gamma}\Sigma(x), \qquad
\Sigma(x)=\Sigma'(x)=0\quad \forall\quad x\in[x_1,x_2]
$$

If we wish to incorporate moving boundaries, the change of variables is now $\Psi=e^{-iF(x)\dot x}\psi$, with the corresponding equation:

$$
i\dot\Psi=-\frac{1}{2}\partial_F^2\Psi+V_\mathrm{eff}\Psi
$$

$$
V_\mathrm{eff} = V + F(x)\ddot x + \left(F'(x)-\frac{1}{2}\right)\dot x^2
$$

It is not immediately obvious how to calculate derivatives wrt $F$, since we must compute

$$
-\frac{1}{2}\partial_F^2\Psi={\mathcal{F}_{F}}^{-1}\left(\frac{k^2}{2}\mathcal{F}_F(\Psi)\right)
$$

We may notice however, that

$$
\mathcal{F}_F[\cdot]=\int\mathrm{d}F\ e^{-ikF}[\cdot]=
\int\mathrm{d}x\ e^{-ikF(x)}F'(x)[\cdot]
$$

$$
{\mathcal{F}_F}^{-1}[\cdot]=\frac{1}{2\pi}\int\mathrm{d}k\ e^{ikF}[\cdot]=\frac{1}{2\pi}
\int\mathrm{d}k\ e^{ikF(x)}[\cdot]
$$

which can be computed via nonuniform FFT [4].

### References

[1] Hatano, N., & Suzuki, M. (2005). *Finding Exponential Product Formulas of Higher Orders*. Springer Berlin Heidelberg. https://arxiv.org/abs/math-ph/0506007

[2] Nissen, A., & Kreiss, G. (2010). An Optimized Perfectly Matched Layer for the Schrodinger Equation. *Communications in Computational Physics - COMMUN COMPUT PHYS, 9*. https://www.researchgate.net/publication/241245803_An_Optimized_Perfectly_Matched_Layer_for_the_Schrodinger_Equation

[3] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, & Qiao Zhang. (2018). *JAX: composable transformations of Python+NumPy programs*. https://jax.readthedocs.io/en/latest/quickstart.html

[4] Barnett, A., & others. (2017–2025). *FINUFFT: Fast and accurate nonuniform fast Fourier transform library*. Simons Foundation, Inc. https://finufft.readthedocs.io/en/latest/