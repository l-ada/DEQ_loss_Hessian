Hello, I am GitHub Copilot.

This derivation relies on the **Implicit Function Theorem**. Since the hidden state $h_*$ is defined implicitly by the fixed point equation rather than an explicit unrolled computation graph, we differentiate the fixed point condition directly.

### Derivation

1.  **Fixed Point Condition**
    At convergence, the hidden state satisfies:
    $$ h_* = f_\theta(x, h_*) $$

2.  **Implicit Differentiation**
    We differentiate both sides with respect to the parameters $\theta$. Note that $h_*$ depends on $\theta$.
    $$ \frac{d h_*}{d \theta} = \frac{\partial f}{\partial \theta} + \frac{\partial f}{\partial h_*} \frac{d h_*}{d \theta} $$

3.  **Solve for the Jacobian of the Fixed Point**
    Rearrange the terms to isolate $\frac{d h_*}{d \theta}$:
    $$ \frac{d h_*}{d \theta} - \frac{\partial f}{\partial h_*} \frac{d h_*}{d \theta} = \frac{\partial f}{\partial \theta} $$
    $$ \left( I - \frac{\partial f}{\partial h_*} \right) \frac{d h_*}{d \theta} = \frac{\partial f}{\partial \theta} $$
    $$ \frac{d h_*}{d \theta} = \left( I - \frac{\partial f}{\partial h_*} \right)^{-1} \frac{\partial f}{\partial \theta} $$

4.  **Loss Gradient**
    Apply the chain rule to the loss function $L(h_*)$:
    $$ \frac{d L}{d \theta} = \frac{\partial L}{\partial h_*} \frac{d h_*}{d \theta} $$

5.  **Final Formula**
    Substitute the result from step 3 into step 4:
    $$ \frac{d L}{d \theta} = \frac{\partial L}{\partial h_*} \left( I - \frac{\partial f}{\partial h_*} \right)^{-1} \frac{\partial f}{\partial \theta} $$

### Practical Implementation Note
In practice (e.g., in Deep Equilibrium Models), computing the inverse $(I - J)^{-1}$ explicitly is too expensive ($O(N^3)$). Instead, one usually solves the linear system:
$$ v^T = \frac{\partial L}{\partial h_*} \left( I - \frac{\partial f}{\partial h_*} \right)^{-1} $$
which is equivalent to solving for $v^T$:
$$ v^T \left( I - \frac{\partial f}{\partial h_*} \right) = \frac{\partial L}{\partial h_*} $$
using a fixed-point iteration or a linear solver, and then computing $v^T \frac{\partial f}{\partial \theta}$.

Understood. We will proceed one step at a time.

Here is the first line, where we set up the Hessian as the total derivative of the gradient formula we derived previously:

$$ H = \frac{d}{d \theta} \left[ \left( \frac{d h_*}{d \theta} \right)^T \nabla_{h_*} L \right] $$



The change comes from the difference between a **Jacobian** (row vector) and a **Gradient** (column vector), and the dimensions required to produce a Hessian matrix.

### 1. Dimensions in the First Derivation
In the first derivation, we calculated the derivative of a scalar $L$ with respect to a vector $\theta$. In matrix calculus, this is a **row vector** (shape $1 \times P$):

$$ \frac{d L}{d \theta} = \underbrace{\frac{\partial L}{\partial h_*}}_{1 \times H} \cdot \underbrace{\frac{d h_*}{d \theta}}_{H \times P} $$

### 2. The Goal for the Hessian
The Hessian is a square matrix of shape $P \times P$.
To get a $P \times P$ matrix, we need to differentiate a **column vector** of shape $P \times 1$ with respect to the parameter vector $\theta$ (size $P$).

### 3. The Transposition
Therefore, before we start the second derivation, we must transpose the result from the first derivation to turn it into a column vector:

$$ \nabla_\theta L = \left( \frac{d L}{d \theta} \right)^T = \left( \frac{\partial L}{\partial h_*} \frac{d h_*}{d \theta} \right)^T $$

Using the linear algebra rule $(AB)^T = B^T A^T$:

$$ \nabla_\theta L = \underbrace{\left( \frac{d h_*}{d \theta} \right)^T}_{P \times H} \cdot \underbrace{\left( \frac{\partial L}{\partial h_*} \right)^T}_{H \times 1} $$

This is why $\frac{d h_*}{d \theta}$ appears first and transposed in the Hessian derivation.

Shall we proceed to the next line (applying the product rule)?



You are absolutely right to call that out. It is inconsistent in many textbooks, and it is "sneaky" if not explicitly defined.

In strict matrix calculus:
1.  **The Derivative (Jacobian) of a scalar** $L$ is a **row vector** ($1 \times P$).
    $$ J_L = \frac{d L}{d \theta} $$
2.  **The Gradient** is defined as the transpose of the Jacobian, a **column vector** ($P \times 1$).
    $$ \nabla L = \left( \frac{d L}{d \theta} \right)^T $$
3.  **The Hessian** is the derivative of the **Gradient** (not the Jacobian).

So, the "sneakiness" you detected is that we silently switched from differentiating the scalar $L$ (which gives a row) to differentiating the gradient vector $\nabla L$ (which is a column).

Let's make it explicit and rigorous.

**Corrected Line 1:**
We define the Hessian $H$ as the derivative of the gradient vector $\nabla_\theta L$.
$$ H = \frac{d}{d \theta} \left( \nabla_\theta L \right) = \frac{d}{d \theta} \left[ \left( \frac{d h_*}{d \theta} \right)^T \nabla_{h_*} L \right] $$

*   $\nabla_{h_*} L$ is the gradient w.r.t state (Column, size $N \times 1$).
*   $\frac{d h_*}{d \theta}$ is the Jacobian of state w.r.t params (Matrix, size $N \times P$).
*   The term inside the brackets is now explicitly a **Column Vector** of size $P \times 1$.

Ready for the next line?



Great. Now we apply the product rule. Since we are differentiating a product of a matrix (the transposed Jacobian) and a vector (the gradient), the result splits into two distinct terms:

$$ H = \underbrace{\left( \frac{d h_*}{d \theta} \right)^T \frac{d}{d \theta} (\nabla_{h_*} L)}_{\text{Term A: Loss Curvature}} + \underbrace{\left[ \frac{d}{d \theta} \left( \frac{d h_*}{d \theta} \right)^T \right] \nabla_{h_*} L}_{\text{Term B: Dynamics Curvature}} $$

Ready for the next line (expanding Term A)?



Here is the expansion of **Term A** (Loss Curvature).

We need to evaluate $\frac{d}{d \theta} (\nabla_{h_*} L)$. Since the gradient $\nabla_{h_*} L$ depends on $\theta$ only through the state $h_*$, we apply the chain rule:

$$ \frac{d}{d \theta} (\nabla_{h_*} L) = \frac{\partial (\nabla_{h_*} L)}{\partial h_*} \frac{d h_*}{d \theta} = \nabla^2_{h_*} L \cdot \frac{d h_*}{d \theta} $$

Substituting this back into Term A gives us the standard Gauss-Newton curvature term:

$$ \text{Term A} = \left( \frac{d h_*}{d \theta} \right)^T \nabla^2_{h_*} L \left( \frac{d h_*}{d \theta} \right) $$

Ready for the next line (expanding Term B)?



This is a crucial question because standard matrix notation starts to break down here. We are now dealing with **Rank-3 Tensors** (3D arrays of numbers), not just 2D matrices.

Let $N$ be the size of the hidden state $h$, and $P$ be the number of parameters $\theta$.

The entire equation represents an equality between tensors of shape **$N \times P \times P$**.

Here is the breakdown of each object:

1.  **$\frac{d^2 h_*}{d \theta^2}$** (The Unknown)
    *   This is the second derivative of a vector ($N$) with respect to a vector ($P$).
    *   **Dimension:** **$N \times P \times P$** (Rank-3 Tensor).
    *   *Meaning:* Entry $(i, j, k)$ is $\frac{\partial^2 (h_*)_i}{\partial \theta_j \partial \theta_k}$.

2.  **$\left( I - \frac{\partial f}{\partial h} \right)$**
    *   **Dimension:** **$N \times N$** (Matrix).
    *   *Operation:* It multiplies the first dimension ($N$) of the tensor above.

3.  **$\frac{d h_*}{d \theta}$**
    *   **Dimension:** **$N \times P$** (Matrix).

4.  **$\left[ \frac{d}{d \theta} \left( I - \frac{\partial f}{\partial h} \right) \right]$**
    *   This is the derivative of a matrix ($N \times N$) with respect to a vector ($P$).
    *   **Dimension:** **$N \times N \times P$** (Rank-3 Tensor).
    *   *Operation:* This tensor contracts with the matrix $\frac{d h_*}{d \theta}$ along the inner dimension $N$.
    *   *Result:* A tensor of shape $N \times P \times P$.

5.  **RHS: $\frac{d}{d \theta} \left( \frac{\partial f}{\partial \theta} \right)$**
    *   This is the derivative of the Jacobian matrix ($N \times P$) with respect to parameters ($P$).
    *   **Dimension:** **$N \times P \times P$** (Rank-3 Tensor).

Shall we proceed to the next line (expanding the derivatives)?

Here is the start of the expansion for **Term B** (Dynamics Curvature).

To evaluate this term, we need the second derivative of the state, $\frac{d^2 h_*}{d \theta^2}$. We find this by differentiating the fixed point gradient equation $(I - \frac{\partial f}{\partial h}) \frac{d h_*}{d \theta} = \frac{\partial f}{\partial \theta}$ with respect to $\theta$.

Applying the product rule to the left side and the total derivative to the right side gives:

$$ (I - \frac{\partial f}{\partial h}) \frac{d^2 h_*}{d \theta^2} + \left[ \frac{d}{d \theta} \left( I - \frac{\partial f}{\partial h} \right) \right] \frac{d h_*}{d \theta} = \frac{d}{d \theta} \left( \frac{\partial f}{\partial \theta} \right) $$

Ready for the next line (expanding the derivatives)?

You are asking the right questions. Standard matrix notation is notoriously bad at representing these Rank-3 operations (derivatives of matrices).

The operations are **Tensor Contractions**.

Since we cannot easily write this with standard matrix multiplication symbols ($\cdot$), let's define it precisely using **Index Notation** (Einstein Summation).

Let indices be:
*   $i, j, m$: State indices ($1 \dots N$)
*   $k, l$: Parameter indices ($1 \dots P$)

### 1. The Object Shapes
*   **$z' = \frac{d h_*}{d \theta}$**: Matrix ($N \times P$). Indices $(j, l)$.
*   **$M = (I - \frac{\partial f}{\partial h})$**: Matrix ($N \times N$). Indices $(i, j)$.
*   **$\frac{d M}{d \theta}$**: Rank-3 Tensor ($N \times N \times P$). Indices $(i, j, k)$.
    *   This represents $\frac{\partial M_{ij}}{\partial \theta_k}$.

### 2. The Operation in Question
You asked about this term:
$$ \mathcal{T} = \left[ \frac{d}{d \theta} \left( I - \frac{\partial f}{\partial h} \right) \right] \frac{d h_*}{d \theta} $$

This is a product of a **Rank-3 Tensor** and a **Matrix**, resulting in a **Rank-3 Tensor** ($N \times P \times P$).

In index notation, the operation is:
$$ \mathcal{T}_{ilk} = \sum_{j=1}^{N} \left( \frac{\partial M_{ij}}{\partial \theta_k} \right) (z')_{jl} $$

**Interpretation:**
For every parameter $k$ (the "slice" of the tensor), you perform a standard **Matrix-Matrix Multiplication** between the derivative of the Jacobian matrix slice and the state Jacobian matrix $z'$.

### 3. The "Hidden" Chain Rule Expansion
Inside that term, there is a deeper contraction.
Recall $\frac{d M}{d \theta}$ involves $\frac{\partial^2 f}{\partial h^2}$.
$$ \frac{\partial^2 f}{\partial h^2} \text{ is a Rank-3 Tensor } (N \times N \times N) \text{ with indices } (i, j, m) $$

The term $\frac{\partial^2 f}{\partial h^2} \left( \frac{d h_*}{d \theta} \right)^2$ effectively becomes:
$$ \mathcal{R}_{ilk} = \sum_{j=1}^N \sum_{m=1}^N \frac{\partial^2 f_i}{\partial h_j \partial h_m} (z')_{jl} (z')_{mk} $$

**This is a "Double Contraction"**:
You are contracting the Hessian tensor of $f$ with the Jacobian matrix $z'$ along **two** different modes simultaneously.

### Summary
*   It is **not** a Kronecker product.
*   It is **not** a simple Matrix product.
*   It is a **Tensor Contraction** (or Batched Matrix Multiplication).

In code (like `einsum`), it would look like: `einsum('ijk,jl->ilk', dM_dtheta, z_prime)`.

Shall we proceed to the final assembly of the formula with this understanding?



Understood. Let's step back to the equation we obtained from the product rule and expand the derivatives fully, without using any shorthand notation like $z$ or $M$.

We are differentiating this equation with respect to $\theta$:
$$ \left( I - \frac{\partial f}{\partial h_*} \right) \frac{d h_*}{d \theta} = \frac{\partial f}{\partial \theta} $$

Applying the product rule gives:
$$ \left( I - \frac{\partial f}{\partial h_*} \right) \frac{d^2 h_*}{d \theta^2} + \left[ \frac{d}{d \theta} \left( I - \frac{\partial f}{\partial h_*} \right) \right] \frac{d h_*}{d \theta} = \frac{d}{d \theta} \left( \frac{\partial f}{\partial \theta} \right) $$

Now, let's expand the two **total derivative** terms (the terms with $\frac{d}{d \theta}$).

### 1. Expand the Right Hand Side (RHS)
The term is $\frac{d}{d \theta} \left( \frac{\partial f}{\partial \theta} \right)$.
Since $f$ depends on $\theta$ directly AND via $h_*$, we apply the chain rule:

$$ \text{RHS} = \frac{\partial^2 f}{\partial \theta^2} + \text{Contract}\left( \frac{\partial^2 f}{\partial h_* \partial \theta}, \frac{d h_*}{d \theta} \right) $$

*(Note: The "Contract" here means we multiply the tensor $\frac{\partial^2 f}{\partial h_* \partial \theta}$ by the matrix $\frac{d h_*}{d \theta}$ along the $h_*$ dimension).*

### 2. Expand the Bracketed Term on the Left
The term is $\frac{d}{d \theta} \left( I - \frac{\partial f}{\partial h_*} \right)$.
The Identity matrix $I$ is constant, so its derivative is 0. We are left with the derivative of the Jacobian $-\frac{\partial f}{\partial h_*}$. Again, we apply the chain rule:

$$ \left[ \dots \right] = - \left( \frac{\partial^2 f}{\partial \theta \partial h_*} + \text{Contract}\left( \frac{\partial^2 f}{\partial h_*^2}, \frac{d h_*}{d \theta} \right) \right) $$

Shall we substitute these expansions back into the main equation to solve for $\frac{d^2 h_*}{d \theta^2}$?



Here is the expansion of the Right Hand Side (RHS) term $\frac{d}{d \theta} \left( \frac{\partial f}{\partial \theta} \right)$.

Since $f$ depends on $\theta$ both directly and through $h_*$, we apply the chain rule (total derivative):

$$ \text{RHS} = \frac{\partial^2 f}{\partial \theta^2} + \text{Contract}\left( \frac{\partial^2 f}{\partial h_* \partial \theta}, \frac{d h_*}{d \theta} \right) $$

*   **First term:** The direct second derivative w.r.t. parameters ($N \times P \times P$).
*   **Second term:** The mixed derivative ($N \times N \times P$) contracted with the state Jacobian ($N \times P$) along the state dimension $N$.

Ready for the next line (expanding the LHS bracket)?



Here is the equation we started with (the result of the product rule):

$$ \left( I - \frac{\partial f}{\partial h_*} \right) \frac{d^2 h_*}{d \theta^2} + \underbrace{\left[ \frac{d}{d \theta} \left( I - \frac{\partial f}{\partial h_*} \right) \right] \frac{d h_*}{d \theta}}_{\mathcal{B}} = \underbrace{\frac{d}{d \theta} \left( \frac{\partial f}{\partial \theta} \right)}_{\text{RHS}} $$

Now, substituting the expansions for $\mathcal{B}$ and RHS:

$$ \left( I - \frac{\partial f}{\partial h_*} \right) \frac{d^2 h_*}{d \theta^2} - \left( \text{Contract}\left( \frac{\partial^2 f}{\partial h_* \partial \theta}, \frac{d h_*}{d \theta} \right) + \text{DoubleContract}\left( \frac{\partial^2 f}{\partial h_*^2}, \frac{d h_*}{d \theta}, \frac{d h_*}{d \theta} \right) \right) = \frac{\partial^2 f}{\partial \theta^2} + \text{Contract}\left( \frac{\partial^2 f}{\partial h_* \partial \theta}, \frac{d h_*}{d \theta} \right) $$

Notice that the mixed derivative term $\text{Contract}\left( \frac{\partial^2 f}{\partial h_* \partial \theta}, \frac{d h_*}{d \theta} \right)$ appears on both sides, but with opposite signs (due to the minus sign on the LHS). When we move the LHS term to the right, they will add up.

Ready for the next line (solving for $\frac{d^2 h_*}{d \theta^2}$)?



Yes, that is a perfect way to visualize it.

If you project everything onto a specific coordinate basis vector $e_i$ (effectively looking at just the $i$-th component of the state vector $h_*$), the Rank-3 tensors collapse into standard **Matrices** ($P \times P$).

### 1. The Objects become Matrices
*   **The Unknown:** $\frac{d^2 (h_*)_i}{d \theta^2}$ is just the Hessian matrix of the $i$-th scalar component of the fixed point.
*   **The RHS:** The term involving $f$ becomes the total Hessian of the scalar function $f_i(x, h_*)$ with respect to $\theta$.

### 2. The Equation becomes a Linear Combination
However, you cannot solve for these matrices one by one independently. The term $\left( I - \frac{\partial f}{\partial h_*} \right)$ **couples** them all together.

If we look at the $i$-th row of our tensor equation, it says:

$$ \sum_{j=1}^N \left( \delta_{ij} - \frac{\partial f_i}{\partial h_j} \right) \underbrace{\left[ \frac{d^2 (h_*)_j}{d \theta^2} \right]}_{\text{Matrix } P \times P} = \underbrace{\left[ \text{Total Hessian of } f_i \right]}_{\text{Matrix } P \times P} $$

So, the Hessian of the $i$-th component depends on the Hessians of **all other components** $j$.

### Why this matters for the final formula
This is exactly why the **Adjoint Vector** $\lambda$ is so powerful.
Instead of solving for all $N$ matrices (which would be huge), we compute the weighted sum of the RHS matrices:

$$ \sum_{i=1}^N \lambda_i \left[ \text{Total Hessian of } f_i \right] $$

This collapses the stack of $N$ matrices into a single $P \times P$ matrix directly.

Shall we proceed to the final line where we write the solved equation for $\frac{d^2 h_*}{d \theta^2}$?



Here is the final rearrangement and solution for the second derivative of the state.

First, we move the negative term from the Left Hand Side to the Right Hand Side. Notice that we now have two identical mixed derivative terms (one from the RHS expansion, one from the LHS expansion), so they sum up to a factor of 2.

$$ \left( I - \frac{\partial f}{\partial h_*} \right) \frac{d^2 h_*}{d \theta^2} = \frac{\partial^2 f}{\partial \theta^2} + 2 \cdot \text{Contract}\left( \frac{\partial^2 f}{\partial h_* \partial \theta}, \frac{d h_*}{d \theta} \right) + \text{DoubleContract}\left( \frac{\partial^2 f}{\partial h_*^2}, \frac{d h_*}{d \theta}, \frac{d h_*}{d \theta} \right) $$

Finally, we multiply by the inverse of the Jacobian term to isolate $\frac{d^2 h_*}{d \theta^2}$:

$$ \frac{d^2 h_*}{d \theta^2} = \left( I - \frac{\partial f}{\partial h_*} \right)^{-1} \left[ \frac{\partial^2 f}{\partial \theta^2} + 2 \frac{\partial^2 f}{\partial h_* \partial \theta} \frac{d h_*}{d \theta} + \frac{\partial^2 f}{\partial h_*^2} \left( \frac{d h_*}{d \theta} \right)^2 \right] $$

The term in the square brackets represents the **Total Hessian** of the function $f$ with respect to $\theta$ (accounting for both explicit dependence and implicit dependence via the state).

This completes the derivation of the unknown term needed for the Hessian of the Loss.



Yes, exactly! You have the right intuition.

If you fix the last dimension (let's say the $l$-th parameter, corresponding to your `[,,3]`), you are taking a "slice" of the tensor.

### 1. The Slice is a Matrix
If $\mathcal{T}_{total}$ has shape **$N \times P \times P$**, and you fix the last index $l$, you get a matrix $\mathcal{T}_{slice}$ of shape **$N \times P$**.
*   Rows: The $N$ state components.
*   Columns: The $P$ parameters (for the first derivative dimension).

### 2. The Operation is Matrix Multiplication
The equation for that specific slice becomes standard matrix multiplication:

$$ \left( \frac{d^2 h_*}{d \theta^2} \right)_{[:, :, l]} = \underbrace{(I - J)^{-1}}_{N \times N} \cdot \underbrace{\mathcal{T}_{[:, :, l]}}_{N \times P} $$

So, you can think of the Rank-3 tensor operation as performing this matrix multiplication $P$ times—once for each slice of the last dimension.

Does this confirm your visualization?

Understood. Let's rewrite the final formula for the Hessian of the state, $\frac{d^2 h_*}{d \theta^2}$, using explicit `Contract` notation to show exactly how the dimensions combine.

We are solving for a Rank-3 Tensor of shape **$N \times P \times P$**.

### 1. The "Total Hessian" Tensor (Inside the Brackets)
First, let's define the huge term inside the square brackets. This represents the total curvature of $f$ with respect to $\theta$. Let's call it $\mathcal{T}_{total}$.

$$ \mathcal{T}_{total} = \underbrace{\frac{\partial^2 f}{\partial \theta^2}}_{\text{Direct}} + 2 \cdot \underbrace{\text{Contract}\left( \frac{\partial^2 f}{\partial h_* \partial \theta}, \frac{d h_*}{d \theta} \right)}_{\text{Mixed}} + \underbrace{\text{DoubleContract}\left( \frac{\partial^2 f}{\partial h_*^2}, \frac{d h_*}{d \theta}, \frac{d h_*}{d \theta} \right)}_{\text{Implicit}} $$

*   **Shape:** All three terms result in a tensor of shape **$N \times P \times P$**.
*   **Indices:** $(i, k, l)$ where $i$ is the state component, and $k, l$ are parameter components.

### 2. The Inverse Jacobian Matrix
Now consider the inverse term:
$$ A^{-1} = \left( I - \frac{\partial f}{\partial h_*} \right)^{-1} $$
*   **Shape:** Matrix **$N \times N$**.
*   **Indices:** $(r, i)$.

### 3. The Final Contraction
To get the Hessian of the state, we contract the inverse matrix $A^{-1}$ with the tensor $\mathcal{T}_{total}$ along the state dimension.

$$ \frac{d^2 h_*}{d \theta^2} = \text{Contract}_{state}\left( A^{-1}, \mathcal{T}_{total} \right) $$

**In Index Notation:**
$$ \left( \frac{d^2 h_*}{d \theta^2} \right)_{rkl} = \sum_{i=1}^N (A^{-1})_{ri} \cdot (\mathcal{T}_{total})_{ikl} $$

**Interpretation:**
For every pair of parameters $(k, l)$, the curvature vector $(\mathcal{T}_{total})_{\cdot kl}$ is multiplied by the inverse Jacobian matrix to "propagate" the curvature through the fixed point equilibrium.

Does this explicit contraction view clarify the structure of $\frac{d^2 h_*}{d \theta^2}$?



Here is the expansion of **Term B** (Dynamics Curvature) by substituting our result for the second derivative of the state.

Recall that Term B is the contraction of the state Hessian tensor with the loss gradient vector:

$$ \text{Term B} = \text{Contract}_{state}\left( \frac{d^2 h_*}{d \theta^2}, \nabla_{h_*} L \right) $$

Substituting the formula we just derived for $\frac{d^2 h_*}{d \theta^2}$:

$$ \text{Term B} = \text{Contract}_{state}\left( \underbrace{\text{Contract}_{state}\left( (I - J)^{-1}, \mathcal{T}_{total} \right)}_{\text{Rank-3 Tensor } (N \times P \times P)}, \underbrace{\nabla_{h_*} L}_{\text{Vector } (N)} \right) $$

This is a nested contraction. We are taking the huge "Total Hessian" tensor $\mathcal{T}_{total}$, transforming it by the inverse Jacobian $(I-J)^{-1}$, and then contracting the result with the gradient $\nabla_{h_*} L$.

Ready to rearrange this to define the Adjoint Vector?



Here is the full expansion of **Term B** as a weighted sum of $P \times P$ matrices.

$$ \text{Term B} = \sum_{r=1}^{N} \lambda_r \underbrace{\left[ \frac{\partial^2 f_r}{\partial \theta^2} + 2 \left( \frac{d h_*}{d \theta} \right)^T \frac{\partial^2 f_r}{\partial h_* \partial \theta} + \left( \frac{d h_*}{d \theta} \right)^T \frac{\partial^2 f_r}{\partial h_*^2} \left( \frac{d h_*}{d \theta} \right) \right]}_{\text{Total Hessian of scalar } f_r \text{ w.r.t. } \theta \text{ (Matrix } P \times P \text{)}} $$

### Breakdown of the Terms inside the Bracket
For a single state component $r$:

1.  **Direct Parameter Hessian** ($P \times P$):
    $$ \frac{\partial^2 f_r}{\partial \theta^2} $$
    *Curvature of $f_r$ with respect to parameters, holding state fixed.*

2.  **Mixed Hessian Term** ($P \times P$):
    $$ 2 \left( \frac{d h_*}{d \theta} \right)^T \frac{\partial^2 f_r}{\partial h_* \partial \theta} $$
    *Interaction between state sensitivity and parameter sensitivity.*

3.  **Implicit State Hessian Term** ($P \times P$):
    $$ \left( \frac{d h_*}{d \theta} \right)^T \frac{\partial^2 f_r}{\partial h_*^2} \left( \frac{d h_*}{d \theta} \right) $$
    *Curvature of $f_r$ with respect to state, projected onto the parameter manifold via the Jacobian $\frac{d h_*}{d \theta}$.*

This form is extremely useful for implementation because you can compute the Hessian-vector products for each $f_r$ efficiently.

Shall we combine Term A and Term B for the final Hessian formula?

Here is the crucial optimization step: **The Adjoint Method**.

We have a contraction of three objects:
$$ \text{Term B} = \nabla_{h_*} L \cdot \left[ (I - J)^{-1} \cdot \mathcal{T}_{total} \right] $$

Because matrix multiplication is associative, we can group the gradient vector and the inverse Jacobian together first. This avoids computing the expensive $N \times P \times P$ tensor explicitly.

### 1. Define the Adjoint Vector $\lambda$
We define $\lambda$ as the solution to the linear system involving the transposed Jacobian:
$$ \lambda^T = (\nabla_{h_*} L)^T (I - J)^{-1} $$
Or equivalently:
$$ (I - J)^T \lambda = \nabla_{h_*} L $$

### 2. The Final Compact Formula
Now, **Term B** becomes a simple weighted sum of the "Total Hessian" tensor slices, weighted by $\lambda$:

$$ \text{Term B} = \text{Contract}_{state}\left( \lambda, \mathcal{T}_{total} \right) $$

$$ \text{Term B} = \text{Contract}_{state}\left( \lambda, \left[ \frac{\partial^2 f}{\partial \theta^2} + 2 \frac{\partial^2 f}{\partial h_* \partial \theta} \frac{d h_*}{d \theta} + \frac{\partial^2 f}{\partial h_*^2} \left( \frac{d h_*}{d \theta} \right)^2 \right] \right) $$

This is the final, computationally efficient formula for the second part of the Hessian.

You are absolutely right. I simplified too aggressively by writing $2 (\dots)$.

In general, the matrix product $Z^T M$ is **not symmetric**, so we cannot just calculate one and multiply by 2. The correct term is the **symmetrized sum**:

$$ Z^T M + M^T Z $$

### Where do the two parts come from?

1.  **From the RHS** ($\frac{d}{d \theta} \frac{\partial f}{\partial \theta}$):
    This generates the term **$Z^T M$**.
    $$ \left( \frac{d h_*}{d \theta} \right)^T \frac{\partial^2 f_r}{\partial h_* \partial \theta} $$

2.  **From the LHS** ($\frac{d}{d \theta} \frac{\partial f}{\partial h_*}$):
    This generates the term **$M^T Z$**.
    $$ \left( \frac{\partial^2 f_r}{\partial h_* \partial \theta} \right)^T \frac{d h_*}{d \theta} $$

### The Corrected Full Formula

So, the term inside the brackets (the Total Hessian of $f_r$) should explicitly be:

$$ \left[ \frac{\partial^2 f_r}{\partial \theta^2} + \underbrace{\left( \frac{d h_*}{d \theta} \right)^T \frac{\partial^2 f_r}{\partial h_* \partial \theta} + \left( \frac{\partial^2 f_r}{\partial h_* \partial \theta} \right)^T \frac{d h_*}{d \theta}}_{\text{Symmetrized Mixed Term}} + \left( \frac{d h_*}{d \theta} \right)^T \frac{\partial^2 f_r}{\partial h_*^2} \left( \frac{d h_*}{d \theta} \right) \right] $$

This ensures the resulting Hessian matrix is perfectly symmetric, as it must be. Thank you for catching that



You are absolutely right. I apologize. I was trying to "fix" the formula to be symmetric prematurely, which obscured the direct translation of the derivation we just did.

If we strictly translate the abstract term $2 \cdot \text{Contract}\left( \frac{\partial^2 f}{\partial h_* \partial \theta}, \frac{d h_*}{d \theta} \right)$ that we derived earlier, preserving the order of the terms (Curvature first, then Jacobian), we get this **non-symmetrized** matrix expression:

$$ \text{Term B} = \sum_{r=1}^{N} \lambda_r \left[ \frac{\partial^2 f_r}{\partial \theta^2} + 2 \underbrace{\left( \frac{\partial^2 f_r}{\partial h_* \partial \theta} \right)^T \frac{d h_*}{d \theta}}_{M^T Z} + \left( \frac{d h_*}{d \theta} \right)^T \frac{\partial^2 f_r}{\partial h_*^2} \left( \frac{d h_*}{d \theta} \right) \right] $$

### Why this looks "wrong" (Non-Symmetric)
As you correctly suspected, the term $M^T Z$ is **not** generally symmetric.
*   $M^T Z$ represents the curvature contribution from the Right Hand Side of the derivation.
*   $Z^T M$ represents the curvature contribution from the Left Hand Side.

By writing $2 M^T Z$, we are effectively assuming $M^T Z \approx Z^T M$. While this is often computationally "close enough" in some optimization settings (Gauss-Newton approximations), the **exact** Hessian requires the sum of the two distinct terms ($M^T Z + Z^T M$) as we discussed.

But this is the strict mathematical translation of the "factor of 2" line I wrote earlier.


```python

```

Here is the final, complete formula for the Hessian of the Loss with respect to parameters $\theta$.

It combines the **Loss Curvature** (Term A) and the **Dynamics Curvature** (Term B).

$$ H = \underbrace{\left( \frac{d h_*}{d \theta} \right)^T \nabla^2_{h_*} L \left( \frac{d h_*}{d \theta} \right)}_{\text{Term A: Gauss-Newton}} + \underbrace{\sum_{r=1}^{N} \lambda_r \left[ \frac{\partial^2 f_r}{\partial \theta^2} + \left( \frac{d h_*}{d \theta} \right)^T \frac{\partial^2 f_r}{\partial h_* \partial \theta} + \left( \frac{\partial^2 f_r}{\partial h_* \partial \theta} \right)^T \frac{d h_*}{d \theta} + \left( \frac{d h_*}{d \theta} \right)^T \frac{\partial^2 f_r}{\partial h_*^2} \left( \frac{d h_*}{d \theta} \right) \right]}_{\text{Term B: Dynamics Correction}} $$

### Definitions
*   **$h_*$**: The equilibrium state ($N \times 1$).
*   **$\frac{d h_*}{d \theta}$**: The Jacobian of the state w.r.t parameters ($N \times P$).
*   **$\lambda$**: The Adjoint Vector ($N \times 1$), solution to $(I - J)^T \lambda = \nabla_{h_*} L$.
*   **$f_r$**: The $r$-th scalar component of the fixed point function.

This formula allows you to compute the exact Hessian efficiently by summing $P \times P$ matrices, without ever instantiating the massive $N \times P \times P$ tensors.

Shall we move on to implementing this in code?



This is a great idea. Substituting the explicit Jacobian reveals a beautiful structure: **Term B is actually a projection of the Hessian of the "energy" $\lambda^T f$**.

Let's define $P = \frac{\partial f}{\partial \theta}$ (Direct parameter Jacobian) and $A = (I - J)^{-1}$ (Inverse state Jacobian).
Then our state sensitivity is $Z = \frac{d h_*}{d \theta} = A P$.

### 1. Define the "Adjoint-Weighted Hessians"
Instead of carrying the sum $\sum \lambda_r (\dots)$ everywhere, let's define the Hessians of the scalar function $S = \lambda^T f(x, h_*)$:

1.  **$H_{\theta \theta}^{(\lambda)} = \sum_{r} \lambda_r \frac{\partial^2 f_r}{\partial \theta^2}$**  (Size $P \times P$)
2.  **$H_{h \theta}^{(\lambda)} = \sum_{r} \lambda_r \frac{\partial^2 f_r}{\partial h_* \partial \theta}$** (Size $N \times P$)
3.  **$H_{hh}^{(\lambda)} = \sum_{r} \lambda_r \frac{\partial^2 f_r}{\partial h_*^2}$** (Size $N \times N$)

### 2. Substitute into Term B
Now substitute $Z$ and these definitions into Term B:

$$ \text{Term B} = H_{\theta \theta}^{(\lambda)} + Z^T H_{h \theta}^{(\lambda)} + (H_{h \theta}^{(\lambda)})^T Z + Z^T H_{hh}^{(\lambda)} Z $$

### 3. The "Block Matrix" Form
This expression is exactly the quadratic form of a block matrix!

$$ \text{Term B} = \begin{pmatrix} I \\ Z \end{pmatrix}^T \begin{pmatrix} H_{\theta \theta}^{(\lambda)} & (H_{h \theta}^{(\lambda)})^T \\ H_{h \theta}^{(\lambda)} & H_{hh}^{(\lambda)} \end{pmatrix} \begin{pmatrix} I \\ Z \end{pmatrix} $$

Substituting $Z = A P$:

$$ \text{Term B} = \begin{pmatrix} I \\ (I-J)^{-1} \frac{\partial f}{\partial \theta} \end{pmatrix}^T \underbrace{\nabla^2 (\lambda^T f)}_{\text{Hessian of scalar } \lambda^T f} \begin{pmatrix} I \\ (I-J)^{-1} \frac{\partial f}{\partial \theta} \end{pmatrix} $$

### Conclusion
There is no algebraic "cancellation" to zero, but there is a massive **conceptual simplification**:

1.  **Term A** is the curvature of the Loss $L$, projected by the total Jacobian $Z$.
2.  **Term B** is the curvature of the Dynamics $\lambda^T f$, projected by the augmented Jacobian $\begin{pmatrix} I \\ Z \end{pmatrix}$.

This is the most compact and elegant way to write (and implement) the formula.

You are spot on. This is a subtle but crucial distinction.

**Yes, you are correct:** In the full optimization landscape, the adjoint vector $\lambda$ **does** depend on $\theta$ (because the equilibrium point $h_*$ changes, and thus $\nabla_{h_*} L$ and the Jacobian $J$ change).

**However, for "Term B", we treat $\lambda$ as a constant vector.**

### Why do we treat $\lambda$ as constant here?

Recall where Term B comes from:
$$ \text{Term B} = \nabla_{h_*} L \cdot \frac{d^2 h_*}{d \theta^2} $$

We used the adjoint equation to substitute the first part:
$$ \lambda^T = \nabla_{h_*} L^T (I - J)^{-1} $$

So Term B becomes:
$$ \text{Term B} = \lambda^T \left[ \text{Total Curvature of } f \right] $$

In this expression, $\lambda$ is just the **value** of the adjoint vector at the current equilibrium point. It acts purely as a "weight vector" that aggregates the curvatures of the vector function $f$. We are not differentiating $\lambda$ here; we are differentiating $f$ and $h$.

### The Precise Interpretation

When I say "Hessian of $\lambda^T f$", I mean:

1.  Take the scalar function $S(\theta, h) = \sum_{i} \lambda_i f_i(\theta, h)$.
2.  **Freeze $\lambda$** (treat it as `stop_gradient` or a fixed constant).
3.  Compute the Hessian of $S$ with respect to the inputs $\theta$ and $h$.

$$ \nabla^2_{(\theta, h)} (\lambda_{\text{fixed}}^T f) = \begin{pmatrix} \sum \lambda_i \frac{\partial^2 f_i}{\partial \theta^2} & \sum \lambda_i \frac{\partial^2 f_i}{\partial \theta \partial h} \\ \sum \lambda_i \frac{\partial^2 f_i}{\partial h \partial \theta} & \sum \lambda_i \frac{\partial^2 f_i}{\partial h^2} \end{pmatrix} $$

This is exactly the matrix that appears in the block form.

### Connection to Lagrangian Methods
This structure is identical to the **Hessian of the Lagrangian** in constrained optimization.
If you define a Lagrangian $\mathcal{L} = L(h) + \lambda^T (f(h, \theta) - h)$, the total Hessian involves terms where $\lambda$ is treated as a fixed Lagrange multiplier.

So, your intuition is perfect: $\lambda$ varies globally, but in the local curvature calculation for Term B, it acts as a fixed projection vector.



Here is the final, fully expanded expression for the Hessian of the Loss, incorporating the Adjoint method and the block-matrix structure we derived.

This formula represents the exact Hessian $H = \frac{d^2 L}{d \theta^2}$.

### 1. Definitions
*   **State Jacobian:** $J = \frac{\partial f}{\partial h_*}$
*   **Total Sensitivity:** $Z = \frac{d h_*}{d \theta} = (I - J)^{-1} \frac{\partial f}{\partial \theta}$
*   **Adjoint Vector:** $\lambda = (I - J)^{-T} \nabla_{h_*} L$
*   **Dynamics Energy:** $S(\theta, h) = \lambda^T f(\theta, h)$ (treating $\lambda$ as fixed)

### 2. The Full Hessian Formula

$$ H = \underbrace{Z^T \left( \nabla^2_{h_*} L \right) Z}_{\text{Term A: Loss Curvature}} + \underbrace{\begin{pmatrix} I \\ Z \end{pmatrix}^T \left[ \nabla^2_{(\theta, h)} S \right] \begin{pmatrix} I \\ Z \end{pmatrix}}_{\text{Term B: Dynamics Curvature}} $$

### 3. Expanded Form (for Implementation)

Substituting the definitions back in to show the explicit computation graph:

$$ H = \underbrace{\left( \frac{\partial f}{\partial \theta} \right)^T (I - J)^{-T} \left[ \nabla^2_{h_*} L \right] (I - J)^{-1} \left( \frac{\partial f}{\partial \theta} \right)}_{\text{Propagated Loss Hessian}} $$
$$ + \underbrace{\begin{pmatrix} I \\ (I - J)^{-1} \frac{\partial f}{\partial \theta} \end{pmatrix}^T \begin{pmatrix} \sum \lambda_i \frac{\partial^2 f_i}{\partial \theta^2} & \sum \lambda_i \frac{\partial^2 f_i}{\partial \theta \partial h} \\ \sum \lambda_i \frac{\partial^2 f_i}{\partial h \partial \theta} & \sum \lambda_i \frac{\partial^2 f_i}{\partial h^2} \end{pmatrix} \begin{pmatrix} I \\ (I - J)^{-1} \frac{\partial f}{\partial \theta} \end{pmatrix}}_{\text{Projected Dynamics Hessian}} $$

This is the "Master Formula" for Second-Order Optimization in Deep Equilibrium Models (DEQ). It separates the curvature of the cost function from the curvature of the physical constraint, coupled only by the sensitivity matrix $Z$.
