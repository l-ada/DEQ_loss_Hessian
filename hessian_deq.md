# Derivation of the Hessian for Deep Equilibrium Models

## 1. Problem Statement

We consider a recurrent system that converges to a fixed point $h_*$.
The fixed point is defined implicitly by the equation:
$$ h_* = f(x, h_*, \theta) $$

We wish to minimize a loss function that depends on this fixed point:
$$ \mathcal{L} = L(h_*) $$

Our goal is to derive the exact Hessian of the loss with respect to the parameters $\theta$:
$$ H = \frac{d^2 \mathcal{L}}{d \theta^2} $$

---

## 2. First Derivative (The Gradient)

First, we find the gradient $\nabla_\theta \mathcal{L}$.
By the chain rule:
$$ \frac{d \mathcal{L}}{d \theta} = \left( \frac{d h_*}{d \theta} \right)^T \nabla_{h_*} L $$

To find the total sensitivity $Z = \frac{d h_*}{d \theta}$, we differentiate the fixed point equation:
$$ h_* = f(x, h_*, \theta) $$
$$ \frac{d h_*}{d \theta} = \frac{\partial f}{\partial \theta} + \frac{\partial f}{\partial h_*} \frac{d h_*}{d \theta} $$

Rearranging terms to solve for $\frac{d h_*}{d \theta}$:
$$ \left( I - \frac{\partial f}{\partial h_*} \right) \frac{d h_*}{d \theta} = \frac{\partial f}{\partial \theta} $$

Let $J = \frac{\partial f}{\partial h_*}$ be the Jacobian of the dynamics.
$$ Z = \frac{d h_*}{d \theta} = (I - J)^{-1} \frac{\partial f}{\partial \theta} $$

Thus, the gradient is:
$$ \nabla_\theta \mathcal{L} = Z^T \nabla_{h_*} L $$

---

## 3. Second Derivative (The Hessian)

To find the Hessian, we differentiate the gradient vector $\nabla_\theta \mathcal{L}$ with respect to $\theta$.
$$ H = \frac{d}{d \theta} \left( Z^T \nabla_{h_*} L \right) $$

Applying the product rule, we get two terms:
$$ H = \underbrace{Z^T \frac{d}{d \theta}(\nabla_{h_*} L)}_{\text{Term A}} + \underbrace{\left[ \frac{d}{d \theta} Z^T \right] \nabla_{h_*} L}_{\text{Term B}} $$

### Term A: Loss Curvature (Gauss-Newton Term)
Differentiating $\nabla_{h_*} L$ with respect to $\theta$ (via $h_*$):
$$ \frac{d}{d \theta} (\nabla_{h_*} L) = \nabla^2_{h_*} L \cdot \frac{d h_*}{d \theta} = \nabla^2_{h_*} L \cdot Z $$

Substituting this back:
$$ \text{Term A} = Z^T (\nabla^2_{h_*} L) Z $$

### Term B: Dynamics Curvature
This term involves the second derivative of the state $\frac{d^2 h_*}{d \theta^2}$.
$$ \text{Term B} = \text{Contract}_{state}\left( \frac{d^2 h_*}{d \theta^2}, \nabla_{h_*} L \right) $$

To find $\frac{d^2 h_*}{d \theta^2}$, we differentiate the defining equation for $Z$:
$$ (I - J) Z = \frac{\partial f}{\partial \theta} $$

Differentiating w.r.t $\theta$:
$$ (I - J) \frac{d Z}{d \theta} + \left[ \frac{d}{d \theta}(I - J) \right] Z = \frac{d}{d \theta} \left( \frac{\partial f}{\partial \theta} \right) $$

Using $\frac{d}{d \theta}(I - J) = - \frac{d J}{d \theta}$:
$$ (I - J) \frac{d^2 h_*}{d \theta^2} = \frac{d}{d \theta} \left( \frac{\partial f}{\partial \theta} \right) + \frac{d J}{d \theta} Z $$

Expanding the total derivatives (chain rule):
1.  **RHS Term:** $\frac{d}{d \theta} (\frac{\partial f}{\partial \theta}) = \frac{\partial^2 f}{\partial \theta^2} + \text{Contract}(\frac{\partial^2 f}{\partial h \partial \theta}, Z)$
2.  **LHS Term:** $\frac{d J}{d \theta} Z = \text{Contract}(\frac{\partial^2 f}{\partial \theta \partial h}, Z) + \text{DoubleContract}(\frac{\partial^2 f}{\partial h^2}, Z, Z)$

Solving for the second derivative:
$$ \frac{d^2 h_*}{d \theta^2} = (I - J)^{-1} \left[ \frac{\partial^2 f}{\partial \theta^2} + \left( Z^T \frac{\partial^2 f}{\partial h \partial \theta} + \left( \frac{\partial^2 f}{\partial h \partial \theta} \right)^T Z \right) + Z^T \frac{\partial^2 f}{\partial h^2} Z \right] $$

### The Adjoint Method
Substituting $\frac{d^2 h_*}{d \theta^2}$ back into Term B:
$$ \text{Term B} = \nabla_{h_*} L^T (I - J)^{-1} \left[ \dots \right] $$

We define the **Adjoint Vector** $\lambda$ as:
$$ \lambda^T = \nabla_{h_*} L^T (I - J)^{-1} \quad \iff \quad (I - J)^T \lambda = \nabla_{h_*} L $$

Now, Term B becomes the contraction of $\lambda$ with the total curvature tensor of $f$.
If we define a scalar function $S = \lambda^T f$ (treating $\lambda$ as fixed), then Term B is simply the Hessian of $S$ projected by the augmented Jacobian.

---

## 4. The Master Formula

Combining everything, the exact Hessian is:

$$ H = \underbrace{Z^T \left( \nabla^2_{h_*} L \right) Z}_{\text{Loss Curvature}} + \underbrace{\begin{pmatrix} I \\ Z \end{pmatrix}^T \left[ \nabla^2_{(\theta, h)} (\lambda_{\text{fixed}}^T f) \right] \begin{pmatrix} I \\ Z \end{pmatrix}}_{\text{Dynamics Curvature}} $$

## 5. Alternative Interpretation (Simplification)

If we substitute the definition of $\lambda$ back into the gradient formula, we get a very simple expression for the first derivative:
$$ \nabla_\theta \mathcal{L} = \nabla_{h_*} L^T \underbrace{(I - J)^{-1} \frac{\partial f}{\partial \theta}}_{Z} = \underbrace{\nabla_{h_*} L^T (I - J)^{-1}}_{\lambda^T} \frac{\partial f}{\partial \theta} = \lambda^T \frac{\partial f}{\partial \theta} $$

Thus, the Hessian is simply the total derivative of this product:
$$ H = \frac{d}{d \theta} \left( \lambda^T \frac{\partial f}{\partial \theta} \right) = \left( \frac{d \lambda}{d \theta} \right)^T \frac{\partial f}{\partial \theta} + \lambda^T \frac{d}{d \theta} \left( \frac{\partial f}{\partial \theta} \right) $$

*   The term $\frac{d \lambda}{d \theta}$ generates the **Loss Curvature** (Term A) and the mixed parts of Term B.
*   The term $\frac{d}{d \theta} (\frac{\partial f}{\partial \theta})$ generates the explicit parameter curvature in Term B.

### Explicit Computation Graph

1.  **Solve Forward:** Find $h_*$ such that $h_* = f(x, h_*, \theta)$.
2.  **Compute Jacobian:** $J = \frac{\partial f}{\partial h_*}$.
3.  **Solve Adjoint:** Find $\lambda$ such that $(I - J)^T \lambda = \nabla_{h_*} L$.
4.  **Compute Sensitivity:** Find $Z$ such that $(I - J) Z = \frac{\partial f}{\partial \theta}$.
5.  **Assemble Hessian:**
    Substituting $Z = (I - J)^{-1} \frac{\partial f}{\partial \theta}$:
    $$ H = \left( (I - J)^{-1} \frac{\partial f}{\partial \theta} \right)^T (\nabla^2_{h_*} L) \left( (I - J)^{-1} \frac{\partial f}{\partial \theta} \right) $$
    $$ + \sum_{i=1}^N \lambda_i \left[ \frac{\partial^2 f_i}{\partial \theta^2} + \left( \left( (I - J)^{-1} \frac{\partial f}{\partial \theta} \right)^T \frac{\partial^2 f_i}{\partial h \partial \theta} + \left( \frac{\partial^2 f_i}{\partial h \partial \theta} \right)^T \left( (I - J)^{-1} \frac{\partial f}{\partial \theta} \right) \right) + \left( (I - J)^{-1} \frac{\partial f}{\partial \theta} \right)^T \frac{\partial^2 f_i}{\partial h^2} \left( (I - J)^{-1} \frac{\partial f}{\partial \theta} \right) \right] $$
