This note explains the parameters in `baroclinic_qg_vortex3D_fastwaves.py`. It accompanies the complete `BaroclinicQGVortex` class.

**Some parameters are prescribed, some are calculated, and some are numerical choices.** There is no unique mathematical optimum without a specified error tolerance, cost model and a convergence study. The defaults define a small experiment designed to show a clean gravity-wave signal. The actual `atmpy` time integrator has not been run here.

Coordinates are `(x, y, z)`, with **y vertical**. Velocities are `(u, v, w)`, so **v is vertical velocity**. In the dimensional equations below, lengths are in metres and times in seconds unless stated otherwise. A star denotes a nondimensional code quantity.

**The constructor parameters have the following meanings and origins.** The calculations behind them follow the table.

| Constructor parameter | Default | Origin |
|---|---:|---|
| `config_override` | `None` | Let this class configure the framework. A supplied configuration remains caller-owned and must agree with the case. |
| `mode` | `"pseudo_incompressible"` | Choose wave-permitting, sound-filtered dynamics. |
| `target_wind_ms` | 0.6 m/s | Preserve the original Rossby number: `0.06 * abs(f_s) * radius_m`. |
| `f_s` | 0.0001 s⁻¹ | Prescribed rotation, retained from the original case. |
| `N_s` | 0.01 s⁻¹ | Prescribed stable stratification, retained from the original case. |
| `duration_hours` | 1.5 h | Round the estimated 87.59-minute wave period up to 90 minutes. |
| `resolution` | `(48, 16, 48)` | Approximately 12 points per horizontal wave, 16 cells across the depth, and 6 cells per vortex radius in both horizontal directions. |
| `initial_projection` | `False` | Retained initial-projection policy. This does not disable the subsequent pressure solve. |
| `add_unbalanced_anomaly` | `True` | Enable the thermal disturbance; `False` gives the matching vortex control. |
| `thermal_anomaly_K` | 0.5 K | A visible but small thermodynamic disturbance; checked against velocity and stability estimates below. |
| `anomaly_kind` | `"mode"` | One resolved horizontal Fourier mode; the least complicated default signal. |
| `anomaly_horizontal_mode` | 4 | Four wavelengths in the periodic 800 km domain. |
| `anomaly_vertical_mode` | 1 | The broadest vertical sine mode compatible with zero wave vertical velocity at the walls. |
| `radius_m` | 100,000 m | A compact vortex radius chosen to permit a small domain. |
| `half_depth_m` | 5,000 m | Retained vertical half-depth of the pressure vortex. |
| `center_height_m` | 6,000 m | Centre of the 12 km deep domain. |
| `domain_width_m` | 800,000 m | Eight vortex radii; also four 200 km wavelengths. Applies to both x and z. |
| `domain_height_m` | 12,000 m | Retained depth; accommodates the vortex and its boundary buffers. |
| `dt_seconds` | `None` → 10 s | A conservative soundproof timestep; `None` selects 0.2 s in compressible mode. |
| `output_interval_seconds` | 60 s | About 88 saved samples per estimated wave period. |

The constructor defaults are explicit inputs, not coupled setters. For example, changing `radius_m` alone does not automatically change `target_wind_ms` or `domain_width_m`. The equations below show which other inputs to update.

**Rotation and stratification are inputs to the experiment.** We retain

\[
f=10^{-4}\ \mathrm{s}^{-1},\qquad N=10^{-2}\ \mathrm{s}^{-1}.
\]

These are angular frequencies. Their corresponding periods are

\[
T_f=\frac{2\pi}{|f|}=62\,831.85\ \mathrm{s}=17.45\ \mathrm{h},
\qquad
T_N=\frac{2\pi}{N}=628.32\ \mathrm{s}=10.47\ \mathrm{min}.
\]

An internal gravity wave's period also depends on its spatial structure; it is not generally equal to either of these periods.

**The vortex wind is calculated from the desired Rossby number.** The original case had

\[
Ro_0=\frac{3}{10^{-4}(500\,000)}=0.06.
\]

Choose the compact radius \(R=100\,000\ \mathrm m\). To retain the same characteristic Rossby number,

\[
U=Ro_0|f|R=0.06(10^{-4})(100\,000)=0.6\ \mathrm{m\,s^{-1}}.
\]

At this wind the neglected curvature acceleration is of relative size

\[
\frac{U^2/R}{|f|U}=Ro_0=0.06.
\]

Consequently, the control is geostrophically and hydrostatically balanced but is not an exact steady gradient-wind vortex of the full momentum equations.

With vertical half-depth \(H_v=5\,000\ \mathrm m\), the vortex Burger number becomes

\[
Bu_v=\left(\frac{NH_v}{|f|R}\right)^2
=\left(\frac{50}{10}\right)^2=25.
\]

The original value was 1. We preserve the small Rossby number, not complete dynamical similarity. If both horizontal and vertical scales were reduced by the same factor, their aspect ratio would remain unchanged and a geometrically similar thermal disturbance would not acquire a shorter wave period merely by being smaller.

**The domain contains the vortex with a small periodic-seam tail.** Choose

\[
L_x=L_z=8R=800\,000\ \mathrm m.
\]

The boundary is four radii from the centre. For the pressure Gaussian,

\[
\frac{F(4R)}{F(0)}=\exp\!\left[-\frac{(4R)^2}{2R^2}\right]
=e^{-8}=3.3546\times10^{-4}.
\]

This is a practical isolation criterion, not an exact compact horizontal support. The nearest-image Gaussian remains continuous across periodic seams but has a small derivative mismatch there. The input guard `MIN_DOMAIN_RADII=6` is a looser design guard; at a boundary three radii away the tail would be \(e^{-4.5}\approx0.0111\).

Vertically, choose \(D=12\,000\ \mathrm m\) and \(y_c=D/2=6\,000\ \mathrm m\). The vortex occupies

\[
y_c-H_v=1\,000\ \mathrm m
\quad\text{to}\quad
y_c+H_v=11\,000\ \mathrm m,
\]

leaving a 1 km buffer at each wall. The horizontal centre is \(x_c=z_c=0\), and all three background velocities are zero. There is no imposed translation.

**The thermal anomaly uses different scales from the pressure vortex.** The default nodal shape is

\[
E(x,y)=\cos(k_h(x-x_c))\sin(m(y-y_{\min})),
\qquad
k_h=\frac{2\pi n}{L_x},\quad m=\frac{j\pi}{D}.
\]

The integer \(n\) enforces horizontal periodicity. The positive integer \(j\) selects the vertical sine mode. The disturbance is independent of horizontal z, which keeps its structure easy to resolve and interpret.

Choose \(j=1\). Then

\[
\lambda_v=\frac{2\pi}{m}=\frac{2D}{j}=24\,000\ \mathrm m.
\]

Only half of this vertical wavelength fits in the 12 km domain. That is intentional: the sine is zero at both vertical walls and has one broad maximum between them. This is a vertical standing-mode seed, not an upward-radiating source.

**The desired wave period motivates the horizontal wavelength.** In a uniform rotating Boussinesq reference problem,

\[
\omega^2=\frac{N^2 k_h^2+f^2m^2}{k_h^2+m^2}.
\]

This is used as a local design estimate for the stratified pseudo-incompressible case. Density variation, the vortex, the grid and the pressure scheme change the actual modal response. The relation and the distinction between balanced and wave modes are discussed by [Wirth (2025)](https://npg.copernicus.org/articles/32/261/2025/).

A reproducible way to choose the wavelength is to target a period near 90 minutes:

\[
\omega_*=\frac{2\pi}{5400\ \mathrm s}.
\]

Solving the dispersion relation for the aspect ratio gives

\[
\left(\frac{k_h}{m}\right)^2
=\frac{\omega_*^2-f^2}{N^2-\omega_*^2},
\]

and therefore

\[
\lambda_{h,*}=\lambda_v
\sqrt{\frac{N^2-\omega_*^2}{\omega_*^2-f^2}}
=205\,624.6\ \mathrm m.
\]

The nearest convenient periodic choice in the 800 km domain is \(n=4\), giving

\[
\lambda_h=\frac{L_x}{n}=200\,000\ \mathrm m,
\quad
k_h=3.14159\times10^{-5}\ \mathrm{m^{-1}},
\quad
m=2.61799\times10^{-4}\ \mathrm{m^{-1}}.
\]

Thus \(k_h/m=0.12\), and

\[
\omega=1.195582\times10^{-3}\ \mathrm{s^{-1}},
\quad
T_w=\frac{2\pi}{\omega}=5255.34\ \mathrm s=87.59\ \mathrm{min}.
\]

This is why the broad vertical structure matters. For fixed \(N\) and \(f\), increasing \(k_h/m\) raises the frequency toward \(N\). Shrinking a positive bubble in both directions with its aspect ratio fixed does not have this effect.

The wave-scale ratio \((Nk_h/(|f|m))^2=144\) is distinct from the vortex Burger number \(Bu_v=25\); they use different vertical and horizontal scales.

**The 0.5 K amplitude is chosen and then checked, rather than fixed by the dispersion relation.** The reference potential temperature satisfies

\[
N^2=g\frac{d\log\theta_0}{dy},
\]

so constant \(N\) and surface value \(T_{\mathrm{ref}}=300\ \mathrm K\) give

\[
\theta_0(y)=300\exp\!\left(\frac{N^2y}{g}\right)\ \mathrm K.
\]

At 6 km, \(\theta_0=318.921\ \mathrm K\). The chosen amplitude \(A=0.5\ \mathrm K\) is only

\[
\frac{A}{\theta_0}=1.568\times10^{-3}
\]

of the background potential temperature, and its buoyancy scale is

\[
b_0=\frac{gA}{\theta_0}=0.015380\ \mathrm{m\,s^{-2}}.
\]

For a single linear mode with initially zero wave velocity, pressure projects the initial buoyancy force into an admissible acceleration. Define

\[
Q=\frac{k_h^2}{k_h^2+m^2}=0.0141956.
\]

The projected vertical-velocity oscillator obeys

\[
\ddot v+\omega^2v=0,\qquad v(0)=0,\qquad\dot v(0)=Qb_0,
\]

and hence

\[
v(t)=\frac{Qb_0}{\omega}\sin(\omega t),
\qquad
v_{\mathrm{amp}}=0.18261\ \mathrm{m\,s^{-1}}.
\]

The first maximum occurs at

\[
t_{\mathrm{peak}}=\frac{\pi}{2\omega}=\frac{T_w}{4}
=1313.83\ \mathrm s=21.90\ \mathrm{min}.
\]

The raw buoyancy \(b_0\) must not be integrated as a constant vertical acceleration: pressure removes most of it for this geometry. The estimate applies locally about a resting uniform reference state; it is not a prediction of the exact maximum in `atmpy`.

An oscillatory displacement scale is \(v_{\mathrm{amp}}/\omega=152.74\ \mathrm m\), with steepness \(m v_{\mathrm{amp}}/\omega\approx0.040\). A first-order stability estimate gives

\[
|\delta N^2|\lesssim
\frac{gA}{\theta_0}\left(m+\frac{N^2}{g}\right)
\approx4.18\times10^{-6}\ \mathrm{s^{-2}},
\]

about 4% of the background \(N^2=10^{-4}\ \mathrm{s^{-2}}\). The code makes the decisive check on the full initialized field, including the vortex:

\[
\min_{\mathrm{interior}}\left[g\,\partial_y\log\theta\right]>0.
\]

The 0.5 K choice makes the wave signal prominent. It is not a tiny dynamical disturbance relative to the weak 0.6 m/s vortex: the corresponding along-wave horizontal velocity scale is approximately \((m/k_h)v_{\mathrm{amp}}=1.52\ \mathrm{m\,s^{-1}}\). Reduce the amplitude if preserving a visually dominant vortex matters more than making the waves conspicuous.

**Cell averaging is included in the amplitude definition.** Let \(\mathcal A_8\) be the average of the eight nodal corners of a cell, and let

\[
E_c=\mathcal A_8E,\qquad
E_{\max}=\max_{\mathrm{interior}}|E_c|.
\]

The actual addition is

\[
\delta\theta^*_c=\frac{A}{T_{\mathrm{ref}}}\frac{E_c}{E_{\max}}.
\]

Therefore the maximum cell perturbation is exactly \(|A|=0.5\ \mathrm K\), even when the grid misses a nodal maximum. A negative `thermal_anomaly_K` reverses the phase. Zero amplitude or `add_unbalanced_anomaly=False` disables the perturbation.

The class retains pressure, P and velocity during this addition, then recomputes

\[
\chi=\frac1{\theta^*},\qquad
\chi'=\chi-S_0,\qquad
\rho=P\chi.
\]

This keeps \(P=\rho\theta^*\) consistent while deliberately breaking the original force balance. The pseudo-incompressible pressure subsequently adjusts diagnostically.

**The default grid follows from resolving both the wave and the vortex.** Choose approximately 12 cells per horizontal wave:

\[
\Delta x=\frac{\lambda_h}{12}=16\,666.7\ \mathrm m,
\qquad
n_x=\frac{L_x}{\Delta x}=48.
\]

Using the same horizontal spacing for the vortex gives \(n_z=48\), and

\[
\frac{R}{\Delta x}=\frac{R}{\Delta z}=6.
\]

Choose \(n_y=16\), giving nominal \(\Delta y=D/16=750\ \mathrm m\). This yields

\[
\frac{\lambda_v}{\Delta y}=32,
\qquad
\frac{2H_v}{\Delta y}=13.33.
\]

Thus there are 16 cells across the vertical half-wave and about 13 across the full vortex support. These are modest visualization choices; 12 samples per horizontal wavelength is not a theorem about numerical accuracy.

These counts assume `nx`, `ny`, `nz` describe interior cells. If the framework instead counts nodes or includes halos, the actual spacings differ. The runtime checks use `grid.dxyz`, and the printed diagnostics give the actual samples per wavelength.

The guards `MIN_HORIZONTAL_WAVE_POINTS=10` and `MIN_VERTICAL_WAVE_POINTS=12` reject severely under-resolved seeds. They are numerical policies, not convergence guarantees. The separate `MIN_GRID_COUNT=9` check is retained from the original case.

**The packet option adds two horizontal modes and requires more x resolution.** With `anomaly_kind="packet"`, multiply the horizontal carrier by

\[
W(x)=\frac{1+\cos(2\pi(x-x_c)/L_x)}{2}.
\]

Writing \(\phi=2\pi(x-x_c)/L_x\),

\[
W\cos(n\phi)=\tfrac12\cos(n\phi)
+\tfrac14\cos((n-1)\phi)
+\tfrac14\cos((n+1)\phi).
\]

For \(n=4\), the modes are 3, 4 and 5. The shortest wavelength is \(L_x/5=160\ \mathrm{km}\). To retain roughly 12 samples, \(n_x\gtrsim5\times12=60\); the suggested convenient choice is 64. Then \(\Delta x=12.5\ \mathrm{km}\), and the shortest wavelength has 12.8 samples. The vertical and z counts can remain 16 and 48.

Requiring packet carrier number \(n\ge2\) excludes a zero-horizontal-wavenumber component. The packet has several frequencies, so 87.59 minutes describes its carrier, not the entire packet. Initially zero wave velocity excites both propagation directions; this broad periodic envelope is not a one-way radiation condition.

**The run duration follows the wave estimate, while the timestep includes a margin.** Round \(T_w=5255.34\ \mathrm s\) to

\[
t_{\mathrm{end}}=5400\ \mathrm s=90\ \mathrm{min}=1.5\ \mathrm h.
\]

The first strong vertical-velocity signal is expected on a roughly 22-minute scale. A 90-minute record captures about one estimated cycle, including a sign change. This is a demonstration duration, not a long statistical sample.

For the soundproof run choose \(\Delta t=10\ \mathrm s\). Then

\[
N\Delta t=0.1,\qquad
|f|\Delta t=0.001,\qquad
\omega\Delta t=0.011956.
\]

There are about 63 steps per background buoyancy period and 526 per target wave period. Even including the estimated 1.52 m/s wave horizontal velocity, advective Courant numbers are small: \((0.6+1.52)\Delta t/\Delta x\approx0.00127\) and \(0.183\Delta t/\Delta y\approx0.00244\). These estimates do not establish stability of the full numerical algorithm.

For optional compressible mode, `dt_seconds=None` selects 0.2 s. Using reference sound speed

\[
c_{\mathrm{ref}}=\sqrt{\gamma R_{\mathrm{gas}}T_{\mathrm{ref}}}
=347.43\ \mathrm{m\,s^{-1}},
\]

the nominal vertical acoustic Courant number is \(c_{\mathrm{ref}}(0.2)/750\approx0.093\). This is below the retained `CFL=0.45` setting. Acoustic CFL checking is enabled in compressible mode; a refined grid or changed atmosphere requires reconsidering the timestep. Pseudo-incompressible mode filters acoustic waves, as discussed by [Hindman and Julien (2024)](https://arxiv.org/abs/2309.10079), while retaining internal gravity-wave dynamics.

**Reference scales convert dimensional inputs into the original code convention.** These are inherited reference choices, not vortex scales:

| Quantity | Value | Role |
|---|---:|---|
| `h_ref` | 10,000 m | Reference length \(L_*\) |
| `t_ref` | 100 s | Reference time \(t_*\) |
| `T_ref` | 300 K | Reference potential temperature |
| `p_ref` | 100,000 Pa | Reference pressure |
| `R_gas` | 287.4 J kg⁻¹ K⁻¹ | Retained dry-air gas constant |
| `gamma` | 1.4 | Retained ideal-gas heat-capacity ratio |
| `g_ms2` / `grav` | 9.81 m s⁻² | Prescribed physical gravity |

The velocity scale is

\[
U_*=\frac{L_*}{t_*}=100\ \mathrm{m\,s^{-1}},
\]

so the vortex peak is \(U^*=0.006\), and the estimated wave vertical amplitude is \(v^*\approx0.001826\). The code-space geometry is

\[
R_0=10,\quad H_{\mathrm{scale}}=0.5,\quad y_c^*=0.6,
\]

\[
x^*,z^*\in[-40,40],\qquad y^*\in[0,1.2].
\]

Other conversions are

\[
f^*=ft_*=0.01,\qquad N^*=Nt_*=1,
\]

\[
\delta\theta^*_{\max}=0.5/300=0.0016667,
\quad
\Delta t^*=10/100=0.1,
\quad
t_{\max}^*=5400/100=54.
\]

`Nsq_ref` is passed to the framework as the dimensional input \(N_s^2=10^{-4}\ \mathrm{s^{-2}}\); it is not replaced by \((N^*)^2=1\).

For standard ideal-gas notation,

\[
c_p=\frac{\gamma R_{\mathrm{gas}}}{\gamma-1}=1005.9\ \mathrm{J\,kg^{-1}\,K^{-1}},
\qquad \Gamma=\frac{\gamma-1}{\gamma}=\frac27.
\]

The implementation reads `Gamma` from `Thermodynamics` and `Msq` from the framework. The definition and actual numerical value of `Msq` are outside the supplied class, so no numerical value is invented here. It is not calculated from the 0.6 m/s vortex wind by this class.

The gravity coefficient is chosen to satisfy the physical acceleration conversion:

\[
g^*_{\mathrm{coefficient}}
=\mathrm{Msq}\,g\frac{t_*^2}{L_*}
=9.81\,\mathrm{Msq}.
\]

Thus

\[
\frac{g^*_{\mathrm{coefficient}}}{\mathrm{Msq}}
\frac{L_*}{t_*^2}=9.81\ \mathrm{m\,s^{-2}}.
\]

This `Msq` factor must be retained. Similarly, the stored nodal pressure perturbation is the unscaled Exner perturbation divided by `Msq`.

**The pressure-vortex amplitude is solved on the actual grid.** Define the unit nodal shape

\[
F=\exp\!\left[-\frac{d_x^2+d_z^2}{2R_0^2}\right]V(\eta),
\qquad
\eta=\frac{y^*-y_c^*}{H_{\mathrm{scale}}},
\]

\[
V(\eta)=
\begin{cases}
(1-\eta^2)^3,&|\eta|<1,\\
0,&|\eta|\ge1.
\end{cases}
\]

Distances \(d_x,d_z\) are nearest-image periodic distances. The exponent 3 is inherited from the supplied vortex: it makes the profile and its first two vertical derivatives continuous where the compact support ends. It is a smoothness choice, not a result of wave theory.

Let \(a>0\) be the pressure amplitude magnitude, so the unscaled Exner perturbation is \(-aF\) and `p_amp=-a`. Let \(F_x,F_y,F_z\) be the framework's actual nodal-to-cell discrete gradients. Put

\[
D_i=\frac{(F_y)_i}{\Gamma g^*_{\mathrm{coefficient}}},
\qquad
C_i=\frac{U_*\sqrt{(F_x)_i^2+(F_z)_i^2}}
{|f^*|\Gamma\,\mathrm{Msq}}.
\]

With background inverse potential temperature \(S_{0i}=1/\theta^*_{0i}\), hydrostatic balance gives

\[
\chi_i=S_{0i}+aD_i,\qquad \theta^*_i=1/\chi_i.
\]

The dimensional geostrophic speed in a cell is

\[
V_i(a)=\frac{aC_i}{S_{0i}+aD_i}.
\]

Solving \(V_i(a)=U\) gives

\[
a_i=\frac{US_{0i}}{C_i-UD_i}.
\]

The initializer selects

\[
a=\min_{i\ \mathrm{interior},\ C_i-UD_i>0}a_i.
\]

As long as the resulting inverse temperatures stay positive, this makes the largest discrete initial swirl equal the requested \(U=0.6\ \mathrm{m\,s^{-1}}\). The amplitude is therefore a runtime result depending on the grid and reference thermodynamics; it is not an independently prescribed constant. `p_amp=None` before initialization is just a placeholder.

Let \(G_i=\partial_i\pi_2\), where \(\pi_2=-aF/\mathrm{Msq}\). The initial velocities and thermodynamic anomaly satisfy

\[
u^*=\frac{\theta^*G_z}{f^*\Gamma},\qquad
w^*=-\frac{\theta^*G_x}{f^*\Gamma},\qquad v^*=0,
\qquad
\chi'=-\frac{\mathrm{Msq}}{\Gamma g^*_{\mathrm{coefficient}}}G_y.
\]

Consequently, with \(\rho=P\chi\),

\[
\frac P\Gamma G_x+f^*\rho w^*=0,\qquad
\frac P\Gamma G_y+\frac{g^*_{\mathrm{coefficient}}}{\mathrm{Msq}}P\chi'=0,\qquad
\frac P\Gamma G_z-f^*\rho u^*=0.
\]

These are the force residuals reported by the class. This cancellation includes neither the vortex curvature acceleration nor the discrete divergence constraint of the full time integrator.

In soundproof modes \(P=P_0\). In compressible mode the initialized equation of state is

\[
P=P_0\left(1+\frac{\delta\pi_c}{\pi_0}\right)^{1/(\gamma-1)},
\qquad \pi_0=P_0^{\gamma-1},
\]

where \(\delta\pi_c\) is the corner-averaged unscaled Exner anomaly. The code evaluates this with `log1p` for numerical accuracy. Both cell and nodal total Exner pressure must remain positive. The reference hydrostate \(P_0,S_0\) is generated by `mpv.state`, not independently reconstructed by this class.

**The remaining switches and numerical values are policies or inherited solver choices.** They cannot be deduced uniquely from the dispersion relation.

| Setting | Value and explanation |
|---|---|
| `qg` switches | `(0, 0, 0)` for nongeostrophic, nonhydrostatic, compressible. This regime does not retain free inertia–gravity waves. |
| `pseudo_incompressible` switches | `(1, 1, 0)`: prognostic momentum with sound filtered. This is the default experiment. |
| `compressible` switches | `(1, 1, 1)`: additionally retain compressible/acoustic dynamics. |
| `baroclinic`, `stratified_atmosphere` | `True`; the vortex varies vertically and the reference state has positive stratification. |
| Horizontal boundaries | Periodic for both state and pressure; compatible with integer horizontal mode numbers. |
| Vertical boundaries | `REFLECTIVE_GRAVITY` for state, `WALL` for pressure; retained from the original case. They reflect waves. |
| Sponge | Disabled to avoid deliberately damping the demonstration's standing mode. |
| `initial_projection` | `False` by default. This concerns the optional initialization operation, not ongoing diagnostic pressure enforcement. A divergence projection is distinct from QG wave filtering. |
| `CFL` | 0.45, retained. A stability/control setting, not derived from this initial condition. |
| `use_acoustic_cfl` | `True` exactly when compressible mode is selected. |
| `dtfixed0` | `None`, retaining the framework's startup-step convention. |
| `GHOST_CELLS` | 2 in each direction, retained for the existing stencil and boundary interface. |
| Limiter | `AVERAGE`, retained. |
| Riemann solver | `MODIFIED_HLL`, retained. |
| Reconstruction | `MODIFIED_MUSCL`, retained. |
| First-order advection routine | `FIRST_ORDER_RK`, retained. |
| Second-order advection routine | `STRANG_SPLIT`, retained. |
| Linear solver | `BICGSTAB`, retained. |
| Preconditioner | `DIAGONAL`, retained. |
| Gradient used in balance | `_gradient_3d_numba`, the supplied pressure-gradient operator. |
| Stability/plot derivative `edge_order=2` | Second-order one-sided treatment at array ends in NumPy's gradient. It does not set the solver's spatial order. |
| Corner-average weight | `0.125 = 1/8`, because a 3D cell has eight nodal corners. |
| `rtol=1e-8` checks | Tight unit/parameter consistency checks for f and gravity, not physical accuracy targets. Other `np.isclose/allclose` checks retain NumPy defaults. |
| Positivity/finiteness checks | Reject invalid theta, density-related reference quantities, Exner pressure, or parameters. |
| Output/diagnostic names | Bookkeeping strings including mode, anomaly type and on/off state; they have no physical derivation. |

The temporal bookkeeping follows directly from the chosen duration and step:

\[
n_{\mathrm{nominal}}=\left\lceil\frac{54}{0.1}\right\rceil=540,
\]

\[
\texttt{stepmax}=\max(10\,000,\ 10n_{\mathrm{nominal}})=10\,000.
\]

The factor 10 and floor 10,000 are headroom policies if the driver reduces steps. They do not request a longer physical run: `tmax` still specifies 90 minutes. Monitor the attained end time if the timestep changes substantially.

For output,

\[
\Delta t_{\mathrm{output}}^*=60/100=0.6,
\qquad
\texttt{output_freq}=\operatorname{round}(60/10)=6.
\]

The requested time list runs from 0 to 90 minutes inclusive, nominally 91 times. `output_timesteps=True` also retains step-based output; if the driver varies the timestep, a six-step interval is no longer exactly one minute. The precise interaction of these two output mechanisms belongs to the framework.

Checkpoint spacing is chosen as 30 minutes:

\[
\texttt{CHECKPOINT_INTERVAL_SECONDS}=1800,
\qquad
\texttt{checkpoint_freq}=\operatorname{round}(1800/10)=180.
\]

Both step-based frequencies are bounded below by one to avoid an invalid zero frequency.

**The diagnostic values separate measured initialization properties from estimates.** The actual peak swirl, minimum total \(N^2\), maximum added temperature, actual grid spacings and force residuals are computed from the initialized arrays. Wavelengths follow from the configured box and mode numbers. The period, quarter-period and wave velocity scale use the local linear formulas above. The raw vertical force divided by density is reported before the pressure responds and must not be interpreted as a sustained acceleration.

`dimensional_fields` converts momentum to velocity with

\[
(u,v,w)=U_*\frac{(\rho u^*,\rho v^*,\rho w^*)}{\rho},
\qquad
\theta_{\mathrm K}=T_{\mathrm{ref}}\frac P\rho.
\]

It also estimates horizontal divergence as \(\partial_xu+\partial_zw\), in s⁻¹. Plot vertical `v_m_s` in an x–y slice; a colour range around ±0.2 m/s follows from the 0.183 m/s estimate. Compare anomaly-on and anomaly-off runs in the same dynamics mode.

**The refactor preserves the previous initialization.** Checks using explicit stand-ins for the unavailable framework reproduced the previous state arrays, pressure, diagnostics, plotting fields and configuration payloads exactly for the default mode, anomaly-off control, packet, compressible initialization and QG control. These checks verify the refactor and initialization arithmetic; they do not verify `atmpy` time evolution, damping, stability or a minimum usable grid.
