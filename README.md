# relative_state_estimator

State estimator for the CAMLS three-drone cable-suspended load formation, built with only UWB peer-to-peer ranging and onboard IMUs. Cable-direction can be sensed from IMU measurements. The filter is a standard EKF with attitude parameterised by euler angles to maintain observability. The system definition, dynamics derivation, and observability analysis are in [CAMLS_Range_Based_Observability_Analysis.pdf](docs/CAMLS_Range_Based_Observability_Analysis.pdf).

## States

Each of the four bodies (payload + 3 drones) carries a full 6-DOF state expressed in the world frame:

$$
x_L = \begin{bmatrix} p_L^\top & v_L^\top & \Theta_L^\top & \omega_L^\top \end{bmatrix}^\top, \qquad
x_i = \begin{bmatrix} p_i^\top & v_i^\top & \Theta_i^\top & \omega_i^\top \end{bmatrix}^\top,\ i \in \mathcal{D}
$$

So the full state of the system is: 
$$
x = \begin{bmatrix} x_L^\top & x_1^\top & x_2^\top & x_3^\top \end{bmatrix}^\top \in \mathbb{R}^{48}
$$

| Symbol | Meaning | Dim |
|---|---|---|
| $p$ | position | 3 |
| $v$ | linear velocity | 3 |
| $\Theta = [\phi, \theta, \psi]$ | Euler angles (roll, pitch, yaw) | 3 |
| $\omega$ | body angular rate | 3 |

Cable states are not part of the filter state, as they can be computed using the states and the payload/drone geometry. 

## Inputs
For each drone, the body-frame collective force $F_i$ and moment $M_i$ produced by the rotors. These can be computed from motor RPM given a motor/airframe model.

## Measurements

The measurement vector $y \in \mathbb{R}^{36}$:

| # | Measurement | Model | Rows |
|---|---|---|---|
| 1 | Drone 1 MoCap position | $\hat p_1 = p_1$ | 3 |
| 2 | Drone 1 MoCap orientation | $\hat\Theta_1 = \Theta_1$ | 3 |
| 3 | UWB inter-drone ranges | $\hat r_{ij} = \lVert a_j - a_i \rVert,\ a_i = p_i + R_i u^\text{ant}_i$, pairs $(1,2),(1,3),(2,3)$ | 3 |
| 4 | Body-frame cable directions | $\hat s_i = R_i^\top q_i$ | 9 |
| 5 | Accelerometer specific force | $\hat f_i = R_i^\top(\dot v_i - g) = (F_i + \tau_i s_i)/m_i$ | 9 |
| 6 | Gyroscope | $\hat\omega_i = \omega_i$ | 9 |

Drone 1's MoCap pose anchors the formation in the world frame. Every other body is observed only relative to it, through ranges, cable directions, and IMU data. The accelerometer is treated as a measurement rather than an input because it senses cable tension.

## Filter

A standard EKF over $x \in \mathbb{R}^{48}$:

- **Predict:** integrate $\dot x = f(x, u)$ with the dynamics above. Propagate the covariance with $A = \partial f / \partial x$, discretised as $\Phi \approx I + A\,\Delta t$.
- **Update:** $h(x)$ from the measurement table and $C = \partial h / \partial x \in \mathbb{R}^{36 \times 48}$.

Section 9 of the PDF derives $A$ and $C$ analytically. It uses six per-cable Jacobians ($De_i,\ D\dot e_i,\ Dl_i,\ Dq_i,\ D\tau_i,\ Ds_i/Du_i$), which are computed once per cable per step and reused.

## Observability

The local observability matrix $\mathcal{O} = [C;\ CA;\ \dots;\ CA^{47}]$ is full rank (48) at every tested equilibrium where the cables are splayed. When all cables are vertical, the thrust axis is parallel to the cable and drone yaw cannot be observed. The rank then drops to 40, or to 44 when the hook and UWB antenna are offset from the drone's vertical axis. Keep the cables splayed. This is already the normal CAMLS operating condition.

| Equilibrium | $\angle(F_i, q_i)$ | rank $\mathcal{O}$ |
|---|---|---|
| Vertical cables, equal $l_i$ | $0^\circ$ | 40 |
| $20^\circ$ splay, equal / unequal $l_i$ | $16.2^\circ$ | 48 |
| Generic equilibrium | $16.9$–$18.1^\circ$ | 48 |
| Vertical, with hook/antenna offsets | $0^\circ$ | 44 |
| $20^\circ$ splay, unequal $l_i$, with offsets | $16.2^\circ$ | 48 |

