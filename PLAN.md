# Prompt

I am writting this book book.tex I covered neural networks of dense, fully connected layers. Now I would like to introduce advanced topics, of networks of different topologies. What topics do you recomend, in order of complexity? Point out any theorems relevant at each topic, as well as literature papers.

# PLAN.md — Advanced Architectures: A Roadmap for *Machine Learning for Physicists*

**Status:** planning document. Nothing here has been written into `book.tex`.
**Audience of the book:** PhD-level physicists who know calculus, linear algebra,
statistical mechanics and PDEs, and who are learning ML from scratch.
**Prerequisite in the book:** Chapters 1–4 (single neuron → two neurons → dense
multi-neuron network in index notation → concrete Keras examples, softmax +
cross-entropy, error signals, loss/activation tables).

---

## 0. The unifying thesis (state this explicitly in the book)

> **Every network topology is a constraint imposed on the dense weight tensor.**

Chapter 3 built the fully general linear map between two layers,

$$u_\mu = w_{\mu\nu}\, x_\nu + b_\mu ,$$

with **no structure whatsoever** on $w_{\mu\nu}$: every input coordinate talks to
every neuron, and all $\mu\times\nu$ entries are free parameters. That is the
*most expressive* and the *least informed* object you can write down. Every
architecture in the rest of the book is obtained by restricting it:

| Architecture | Constraint on $w_{\mu\nu}$ | Symmetry / prior encoded |
|---|---|---|
| Dense (Chap. 3) | none | none |
| Residual | $w \to 1 + \epsilon\,W$ | near-identity flow; depth as time |
| Autoencoder | shape: $\dim(\text{latent}) \ll \dim(\text{input})$ | low intrinsic dimension |
| Convolutional | banded **and** Toeplitz/circulant: $w_{\mu\nu}=k_{\mu-\nu}$ | translation equivariance + locality |
| Recurrent | same $w$ reused at every time step | time-translation invariance |
| Hopfield / Boltzmann | $w_{\mu\nu}=w_{\nu\mu}$, $w_{\mu\mu}=0$ | detailed balance; an energy exists |
| Graph net | $w_{\mu\nu}\neq 0$ only if $(\mu,\nu)\in E$ | permutation equivariance; given topology |
| Attention | $w_{\mu\nu} \to A_{\mu\nu}(x)$, **input-dependent** | dynamic all-to-all coupling |
| Equivariant / geometric | $w$ commutes with a group action $\rho(g)$ | an exact physical symmetry (E(3), SO(3), Lorentz, gauge) |

This is a framing a physicist will recognise instantly: *you are gauging in a
symmetry, and every symmetry you impose removes parameters and buys you data
efficiency.* Make this table a figure in the opening of the advanced part, and
refer back to it at the head of every chapter. It is the single strongest
pedagogical device available to this book, and no standard ML textbook uses it.

**A second thesis worth stating once, early:** the reason constrained
architectures beat dense ones is not that they can represent *more* functions —
they can represent strictly *fewer*. They win because the functions they can
represent are the ones physics actually produces (local, causal, symmetric,
hierarchical). The architecture is a prior, in the Bayesian sense. Cite
*Lin, Tegmark & Rolnick, "Why does deep and cheap learning work so well?",
J. Stat. Phys. 168, 1223 (2017)* — a paper written for exactly your reader.

---

## 1. Chapter template (use the same four beats every time)

Repetition of *form* is what will make the book feel short even after twelve
more chapters. For each architecture:

1. **The constraint.** What structure is imposed on $w_{\mu\nu}$, written in the
   book's existing index notation. One equation.
2. **The forward pass.** Full index-notation derivation, dimensions of every
   index stated explicitly, as in Eqs. (multi-input)–(multi-output).
3. **The backward pass, and its one new twist.** Derive the error signal
   $\delta_\mu$ and show what the constraint does to it. Every architecture has
   exactly one interesting new phenomenon in the backward pass — name it:
   - residual → an additive identity term in the Jacobian;
   - conv → the gradient of a convolution is a correlation, and weight sharing
     means gradients get **summed** over all positions;
   - RNN → the gradient is a *product* of $T$ Jacobians, hence a spectral-radius
     problem;
   - attention → the gradient flows through the weights themselves, since the
     weights are functions of the data.
4. **Concrete example.** A short Keras snippet in the style of Chapter 4, plus
   one physics dataset or physics problem. Then one or two exercises.

**A fifth optional beat, highly recommended for this audience:** a short
"*Physicist's aside*" box making the analogy explicit (transfer matrix,
Green's function, RG coarse-graining, Lyapunov exponent, free energy,
Fokker–Planck). Keep it to a paragraph so it reads as insight, not digression.

---

## 2. Notation you will need to extend

Decide these **before** writing, or you will pay for it later:

- **Layer index.** Promote $^{(1)},^{(2)}$ to a general $^{(\ell)}$, $\ell=1..L$,
  now that networks get deep.
- **Time/sequence index.** Reserve one letter (suggestion: $t$) for sequence
  position, kept distinct from the data-point index $i$ and the layer index
  $\ell$. RNNs need all three simultaneously: $z^{(\ell),i}_{\mu,t}$. This gets
  ugly; consider dropping the explicit $i$ once mini-batching is established
  ("from here on the data index is implicit").
- **Spatial index.** For convolutions you need position separate from channel:
  $z_{c,\,\alpha}$ with $c$ the channel and $\alpha$ the spatial site. Physicists
  read this immediately as "internal index + lattice site", which is a gift —
  say so.
- **Token/node index.** For attention and graphs, $\mu$ becomes a *set* index
  with no natural order. Emphasise that the loss of ordering is the whole point,
  and permutation equivariance is what replaces it.
- **Einstein summation.** You are already implicitly summing repeated indices.
  State the convention once, formally, and note that `numpy.einsum` /
  `torch.einsum` is literally this convention as an API. Physicists find this
  delightful and it makes the code read like the equations.

---

# TIER 1 — Still an MLP, one structural twist

These three cost almost no new mathematics but change the reader's mental model.

---

## Chapter 5 — Depth, and Why It Hurts

*This is a bridge chapter, not a topology. Nothing that follows is believable
without it.*

### The one idea
Composing $L$ layers means the gradient is a **product of $L$ Jacobians**. Products
of matrices are not benign: they either shrink geometrically or blow up
geometrically, governed by the spectrum. Deep learning is, at the level of
gradients, a stability problem in a discrete dynamical system.

### Derivation to include
From the chain rule already established in Chapter 3, write for the error signal
at layer $\ell$

$$\delta^{(\ell)}_\mu = \delta^{(L)}_\eta
\left[\prod_{k=\ell+1}^{L} w^{(k)}_{\cdot\cdot}\,
\mathrm{diag}\big(\sigma^{(k)\prime}\big)\right]_{\eta\mu},$$

then take norms. If each factor has typical singular value $s$, the gradient
scales like $s^{L-\ell}$. Sigmoid has $\sigma' \le 1/4$, so with $s<1$ you get
exponential starvation — this is exactly the calculation to do explicitly,
because it retroactively justifies the activation-function table in Appendix A
(which currently asserts "starves deep networks of gradient" without proof).

Then: **initialization as variance propagation.** Demand
$\mathrm{Var}(z^{(\ell+1)}) = \mathrm{Var}(z^{(\ell)})$ under random weights;
this is a one-line moment computation that yields $\mathrm{Var}(w) = 2/n_{\rm in}$
for ReLU (He) and $2/(n_{\rm in}+n_{\rm out})$ for tanh (Xavier). A physicist will
recognise it as a fixed-point condition for a variance map, and the "edge of
chaos" language then comes for free.

### Theorems to name
- **Universal approximation theorem.** Cybenko (1989); Hornik (1991); Leshno et
  al. (1993) — a single hidden layer of *arbitrary width* with any
  non-polynomial activation is dense in $C(K)$. Pinkus, *Acta Numerica* 8, 143
  (1999) is the definitive review. **Pedagogical point to make loudly:** this
  theorem says depth is never *necessary*, and is therefore almost useless as
  guidance. It is an existence result with no statement about the number of
  neurons, learnability, or generalization. Physicists tend to over-weight it;
  disarm it early.
- **Depth-separation theorems.** Telgarsky, *Benefits of depth in neural
  networks* (COLT 2016) — there are functions computable by a depth-$k$ network
  of polynomial width that require exponential width at depth $O(k^{1/3})$.
  Eldan & Shamir (COLT 2016) give a depth-2-vs-3 separation. **This** is the real
  argument for depth: expressivity per parameter is exponential in depth.
- **Counting linear regions.** Montúfar, Pascanu, Cho & Bengio (NeurIPS 2014) —
  a ReLU network's number of linear regions grows exponentially with depth. A
  very concrete, geometric way to *see* the previous item.
- **Fixed points of the variance map / order–chaos transition.** Poole et al.,
  *Exponential expressivity in deep neural networks through transient chaos*
  (NeurIPS 2016) and Schoenholz et al., *Deep information propagation* (ICLR
  2017). This is mean-field theory applied to signal propagation, with a genuine
  phase transition and a divergent depth scale at criticality. **Highest-value
  citation in this chapter for your audience** — it is statistical mechanics,
  it explains initialization, and it predicts trainable depth.
- **Dynamical isometry.** Pennington, Schoenholz & Ganguli, *Resurrecting the
  sigmoid in deep learning through dynamical isometry* (NeurIPS 2017) — uses
  free probability to compute the full singular-value spectrum of the
  input–output Jacobian. Also Saxe, McClelland & Ganguli (ICLR 2014) for exact
  learning dynamics in deep *linear* networks (a solvable model — physicists
  love a solvable model, and it explains why orthogonal initialization works).

### Papers for the practical machinery
- Hochreiter, *Untersuchungen zu dynamischen neuronalen Netzen* (Diploma thesis,
  TU Munich, 1991) — the original diagnosis of vanishing gradients.
- Bengio, Simard & Frasconi, *Learning long-term dependencies with gradient
  descent is difficult*, IEEE TNN 5, 157 (1994) — the citable version.
- Glorot & Bengio (AISTATS 2010) — Xavier init.
- He, Zhang, Ren & Sun, *Delving deep into rectifiers* (ICCV 2015) — He init and
  PReLU (already in your Appendix A; cite it there).
- Ioffe & Szegedy, *Batch normalization* (ICML 2015), **and** Santurkar et al.,
  *How does batch normalization help optimization?* (NeurIPS 2018), which
  demonstrates that the original "internal covariate shift" explanation is
  wrong and that the real effect is a smoother loss landscape. Teaching both
  together is a good lesson in how empirical ML actually progresses.
- Ba, Kiros & Hinton, *Layer normalization* (2016) — needed later for
  transformers.
- Srivastava et al., *Dropout*, JMLR 15, 1929 (2014). Physicist framing:
  training an exponentially large ensemble with shared parameters, and at test
  time taking a geometric-mean approximation.
- Li et al., *Visualizing the loss landscape of neural nets* (NeurIPS 2018) —
  the pictures that make "smoother landscape" concrete.

### Physics hooks
Variance propagation as an RG flow with a fixed point; order/chaos transition;
the Jacobian spectrum as a Lyapunov spectrum; dropout as an ensemble average.

### Exercises
(i) Compute $\max_x \sigma'(x)$ for sigmoid and tanh and deduce the maximum depth
at which the gradient retains 1% of its magnitude, assuming unit-norm weights.
(ii) Numerically measure $\|\partial L/\partial w^{(\ell)}\|$ versus $\ell$ for a
20-layer sigmoid MLP and again for ReLU + He init. This one experiment teaches
the entire chapter.

---

## Chapter 6 — Skip and Residual Connections

### The one idea
Instead of learning a map $F$, learn the *deviation from doing nothing*:

$$z^{(\ell+1)}_\mu = z^{(\ell)}_\mu + F_\mu\big(z^{(\ell)}\big).$$

In the language of §0, $w \to 1 + W$. The network is now a perturbation
around the identity, and — crucially — the backward Jacobian becomes

$$\frac{\partial z^{(\ell+1)}_\mu}{\partial z^{(\ell)}_\nu}
= \delta_{\mu\nu} + \frac{\partial F_\mu}{\partial z^{(\ell)}_\nu}.$$

The Kronecker delta is a gradient highway: the product over layers now contains
a term that is *exactly* 1 regardless of depth, so Chapter 5's exponential
starvation is defeated by construction. Derive this in two lines; it is the
single most economical "aha" in the entire advanced section.

### The physics reframing (do not skip this)
Rewrite the residual update as

$$\frac{z^{(\ell+1)} - z^{(\ell)}}{\Delta \ell} = F\big(z^{(\ell)}\big),
\qquad \Delta\ell = 1,$$

i.e. **a forward-Euler step of the ODE $\dot z = F(z)$, with depth playing the
role of time.** A very deep residual network is a discretised flow. This
observation (a) explains why residual networks are stable, (b) lets you import
the entire theory of numerical integration as architecture design, and (c) sets
up Neural ODEs in Tier 5 for free. State it here and cash it in later.

### Theorems to name
- **Universality with narrow residual nets.** Lin & Jegelka, *ResNet with
  one-neuron hidden layers is a universal approximator* (NeurIPS 2018) — width
  can be traded entirely for depth. A nice complement to Chapter 5's
  width-based universality theorem.
- **Stability of the discretised flow.** Haber & Ruthotto, *Stable architectures
  for deep neural networks*, Inverse Problems 34, 014004 (2018) — designs the
  residual block so the underlying ODE is well-posed (eigenvalues of the
  Jacobian in the left half-plane). This is CFL-condition reasoning applied to
  a neural network, and it will feel completely natural to your reader.

### Papers
- He, Zhang, Ren & Sun, *Deep residual learning for image recognition*
  (CVPR 2016) — the original, and the paper that made >100-layer nets trainable.
- He et al., *Identity mappings in deep residual networks* (ECCV 2016) — the
  pre-activation ordering, and the clean derivation of the additive gradient.
- Srivastava, Greff & Schmidhuber, *Highway networks* (2015) — the gated
  predecessor; shows the idea was in the air, and connects to LSTM gating.
- Veit, Wilber & Belongie, *Residual networks behave like ensembles of
  relatively shallow networks* (NeurIPS 2016) — an alternative and useful
  interpretation: the residual net is a sum over $2^L$ paths.
- E, Weinan, *A proposal on machine learning via dynamical systems*, Commun.
  Math. Stat. 5, 1 (2017) — the continuous-time viewpoint, from a
  mathematician working in the physicist's idiom.

### Exercise
Train a 50-layer plain MLP and a 50-layer residual MLP on the same toy
regression. Plot training loss versus epoch and gradient norm versus layer.

---

## Chapter 7 — Autoencoders: Topology by Shape

### The one idea
No weight sharing, no new symmetry — just a **bottleneck**. Force the network to
reproduce its own input, $\hat x \approx x$, through a latent layer of dimension
$d \ll N$:

$$z_\alpha = \sigma\big(w^{\rm enc}_{\alpha\nu} x_\nu + b^{\rm enc}_\alpha\big),
\qquad
\hat x_\nu = \sigma\big(w^{\rm dec}_{\nu\alpha} z_\alpha + b^{\rm dec}_\nu\big),
\qquad \alpha = 1..d .$$

The task is trivial (copy the input), so all the content is in the constraint:
the network must discover a $d$-dimensional coordinate system in which the data
is describable. This is the reader's first **unsupervised** model — no labels
$y_i$ anywhere — which is a conceptual milestone worth flagging.

### The theorem that makes this chapter land
**A linear autoencoder trained with squared loss recovers the principal
subspace.** If both maps are linear and the loss is MSE, the optimum spans the
same subspace as the top $d$ principal components of the data (not necessarily
the individual PCs — the solution is defined up to an invertible $d\times d$
mixing, a gauge freedom worth pointing out).

- Bourlard & Kamp, *Auto-association by multilayer perceptrons and singular
  value decomposition*, Biol. Cybern. 59, 291 (1988).
- Baldi & Hornik, *Neural networks and principal component analysis: learning
  from examples without local minima*, Neural Networks 2, 53 (1989) — and note
  the second half of that title: the landscape has **no spurious local minima**,
  only saddles. A rare and reassuring global-optimality result.
- Underneath both sits the **Eckart–Young–Mirsky theorem** (1936/1960): the best
  rank-$d$ approximation of a matrix in Frobenius (or spectral) norm is its
  truncated SVD. Your reader almost certainly knows this from a different
  context; connecting it to a neural network is a strong moment.

So: *the nonlinear autoencoder is nonlinear PCA* (Kramer, AIChE J. 37, 233
(1991)). Everything the reader knows about PCA, normal modes and collective
coordinates transfers directly, and the latent variables are candidate **order
parameters**.

### Papers
- Hinton & Salakhutdinov, *Reducing the dimensionality of data with neural
  networks*, Science 313, 504 (2006) — the paper that revived deep autoencoders.
- Vincent, Larochelle, Bengio & Manzagol, *Extracting and composing robust
  features with denoising autoencoders* (ICML 2008) — corrupt the input, demand
  the clean output; the model learns the data manifold's tangent structure.
- Rifai et al., *Contractive auto-encoders* (ICML 2011) — penalise the Jacobian
  norm explicitly. Direct manifold-geometry language.
- Wetzel, *Unsupervised learning of phase transitions: from principal component
  analysis to variational autoencoders*, Phys. Rev. E 96, 022140 (2017) —
  autoencoders on the 2D Ising model, latent variable ≈ magnetisation. **This is
  the concrete example to build the chapter around.** It is a physics paper, the
  data is generatable with a 30-line Metropolis sampler your reader can write in
  an afternoon, and the result (the bottleneck discovers the order parameter
  unsupervised) is genuinely striking.

### Physics hooks
PCA/normal modes; order parameters; collective variables and reaction
coordinates; intrinsic dimension of a phase-space trajectory; lossy compression
as coarse-graining.

### Exercise
Sample 2D Ising configurations at a range of temperatures, train a
2-unit-bottleneck autoencoder, and colour the latent plane by $T$ and by
magnetisation.

---

# TIER 2 — Genuine weight sharing

Now the weight tensor really changes shape.

---

## Chapter 8 — Convolutional Networks

*The first true topology, and the template for everything after it.*

### The constraint
Two constraints at once, and it is worth separating them cleanly:

1. **Locality (sparsity):** $w_{\mu\nu} = 0$ unless $|\mu-\nu| \le r$.
2. **Translation invariance (sharing):** the surviving entries depend only on
   the *displacement*, $w_{\mu\nu} = k_{\mu-\nu}$ — a Toeplitz matrix (circulant,
   with periodic boundary conditions).

Hence

$$u_{c,\alpha} = \sum_{c'}\sum_{\beta} k_{c c',\,\beta}\; x_{c',\,\alpha-\beta} + b_c ,$$

with $c$ a channel index and $\alpha$ a lattice site. Parameter count collapses
from $O(N^2)$ to $O(r)$ per channel pair. **Say the magic word: this is a
stencil.** Your reader has been writing $\nabla^2 \to (1,-2,1)/h^2$ since their
first numerical methods course. A convolutional layer is a learned stencil, and
a deep CNN is a learned composition of stencils.

### Derivations to include
- **The backward pass is a correlation.** $\partial u/\partial x$ gives a
  convolution with the *flipped* kernel — one clean index manipulation, and a
  satisfying structural symmetry.
- **Weight sharing sums gradients.** Because $k_\beta$ appears at every site,
  $\partial L/\partial k_\beta = \sum_\alpha \delta_{\alpha}\, x_{\alpha-\beta}$.
  Emphasise: *shared parameters accumulate gradient from every position where
  they act.* This is the general lesson for all weight sharing (it recurs
  verbatim for RNNs), so derive it carefully once here.
- **Receptive field growth.** After $L$ layers of kernel radius $r$ the output at
  a site depends on inputs within $Lr$. This is a light cone, and stacking
  layers is how information propagates at finite speed. Dilated convolutions
  (Yu & Koltun, ICLR 2016) buy an exponentially growing cone.
- **Equivariance, stated precisely.** $\text{Conv}(T_a x) = T_a \text{Conv}(x)$
  for a translation $T_a$: the layer *commutes* with the symmetry, it is not
  invariant. Invariance only arrives with global pooling at the end. This
  distinction (equivariant intermediate layers, invariant readout) is the whole
  design principle of geometric deep learning, so plant it firmly here.

### Theorems to name
- **Convolution theorem.** Convolution is diagonal in Fourier space. Both an
  efficiency statement and a conceptual one: a CNN's linear part is a learned
  multiplication of the spectrum, i.e. a learned filter bank. Sets up Fourier
  Neural Operators in Tier 5.
- **Equivariance $\Rightarrow$ convolution.** Kondor & Trivedi, *On the
  generalization of equivariance and convolution in neural networks to the
  action of compact groups* (ICML 2018): a linear map equivariant to the action
  of a compact group **must** be a group convolution. This is the deep theorem
  of the chapter — *the architecture is not a heuristic, it is forced by the
  symmetry*. See also Cohen, Geiger & Weiler, *A general theory of equivariant
  CNNs on homogeneous spaces* (NeurIPS 2019). Compare to how demanding Lorentz
  invariance forces the form of a Lagrangian; your reader will get it at once.
- **Nyquist–Shannon sampling theorem.** Strided convolution and pooling
  subsample, and subsampling below Nyquist aliases — which is why CNNs are *not*
  as shift-invariant as advertised. Zhang, *Making convolutional networks
  shift-invariant again* (ICML 2019) fixes it with a low-pass filter before
  downsampling. A rare case where classical signal-processing theory directly
  corrects a modern architecture.
- **Universality of deep CNNs.** Zhou, *Universality of deep convolutional
  neural networks*, Appl. Comput. Harmon. Anal. 48, 787 (2020).

### Papers
- Fukushima, *Neocognitron*, Biol. Cybern. 36, 193 (1980) — the origin.
- LeCun et al., *Backpropagation applied to handwritten zip code recognition*,
  Neural Comput. 1, 541 (1989); LeCun, Bottou, Bengio & Haffner, *Gradient-based
  learning applied to document recognition*, Proc. IEEE 86, 2278 (1998).
- Krizhevsky, Sutskever & Hinton, *ImageNet classification with deep
  convolutional neural networks* (NeurIPS 2012) — the paper that started the
  modern era.
- Simonyan & Zisserman, *Very deep convolutional networks* (ICLR 2015) — the
  $3\times3$-everywhere design rule, worth explaining as receptive-field algebra.
- Ronneberger, Fischer & Brox, *U-Net* (MICCAI 2015) — encoder–decoder with skip
  connections; the workhorse for anything field-to-field, and it composes
  Chapters 6, 7 and 8. Also the backbone of every diffusion model in Tier 4.

### Physics literature (this is where the chapter earns its keep)
- Mehta & Schwab, *An exact mapping between the variational renormalization
  group and deep learning* (2014, arXiv:1410.3831) — RBM stacking ↔ Kadanoff
  block-spin decimation. Speculative and contested, but irresistible framing:
  **pooling is coarse-graining**.
- Carrasquilla & Melko, *Machine learning phases of matter*, Nature Physics 13,
  431 (2017) — a CNN classifies Ising/spin-ice phases directly from
  configurations.
- Carleo & Troyer, *Solving the quantum many-body problem with artificial neural
  networks*, Science 355, 602 (2017) — neural-network quantum states.
- Baldi, Sadowski & Whiteson, *Searching for exotic particles in high-energy
  physics with deep learning*, Nature Comm. 5, 4308 (2014).
- Review to point the reader at: Carleo et al., *Machine learning and the
  physical sciences*, Rev. Mod. Phys. 91, 045002 (2019).

### Exercise
Initialise a single $3\times3$ convolution to the discrete Laplacian stencil and
verify it reproduces `scipy.ndimage.laplace`. Then *train* a one-layer CNN to
map a field to its Laplacian and inspect the learned kernel. Nothing demystifies
convolution faster.

---

## Chapter 9 — Recurrent Networks

### The constraint
The same weight matrix, reused at every step of a sequence:

$$h_{\mu,t} = \sigma\big(W_{\mu\nu}\, h_{\nu,t-1} + U_{\mu\nu}\, x_{\nu,t} + b_\mu\big).$$

Sharing along **time** rather than space. The physicist's reading is immediate:
this is a **discrete-time dynamical system driven by an input**, and $W$ is
essentially a transfer matrix. Unrolling in time turns it into a very deep
feedforward network with tied weights — so Chapter 5's problem returns, in its
most acute form.

### The central derivation: BPTT and the spectral radius
The gradient through $T$ steps involves

$$\frac{\partial h_T}{\partial h_0}
= \prod_{t=1}^{T} \mathrm{diag}\big(\sigma'(\cdot)\big)\, W ,$$

so its magnitude behaves like $\rho(W)^T$ where $\rho$ is the spectral radius (up
to the activation derivative). Therefore:

- $\rho < 1$: gradients vanish exponentially → the network cannot learn
  long-range dependence;
- $\rho > 1$: gradients explode → training diverges;
- the useful regime is a knife-edge.

State it as a **Lyapunov exponent**: $\lambda \simeq \frac{1}{T}\log\|\partial h_T/\partial h_0\|$,
and "vanishing gradient" is just $\lambda<0$. Your reader has computed Lyapunov
exponents before; this reframing turns an ML folk problem into a familiar
stability calculation. It also explains the crude but effective fix (gradient
clipping) and motivates the real one (gating).

### Gating (LSTM/GRU) — the point to make
The LSTM's cell state has an *additive* update, $c_t = f_t \odot c_{t-1} + i_t \odot g_t$.
When the forget gate $f_t \approx 1$ this is precisely the residual/identity
trick of Chapter 6 applied along time: the Jacobian gets a term $\approx 1$
and the memory persists. Presenting LSTM as "Chapter 6, but in the time
direction, with learned gates controlling the effective spectral radius" makes
an architecture that is usually memorised into something derived.

### Theorems to name
- **Turing completeness of RNNs.** Siegelmann & Sontag, *On the computational
  power of neural nets*, J. Comput. Syst. Sci. 50, 132 (1995) — a finite RNN with
  rational weights and unbounded time can simulate a Turing machine. A statement
  about *idealised* computation; contrast honestly with what trains in practice.
- **Universal approximation for dynamical systems.** Schäfer & Zimmermann,
  *Recurrent neural networks are universal approximators* (ICANN 2006) — RNNs are
  dense in the space of open dynamical systems.
- **Echo state property.** Jaeger (2001): the reservoir forgets its initial
  condition iff (roughly) $\rho(W)<1$. The condition that makes reservoir
  computing work is exactly the vanishing-gradient condition, viewed from the
  other side — a lovely duality to point out.

### Papers
- Elman, *Finding structure in time*, Cognitive Science 14, 179 (1990).
- Werbos, *Backpropagation through time*, Proc. IEEE 78, 1550 (1990).
- Bengio, Simard & Frasconi (1994) — as in Chapter 5.
- Hochreiter & Schmidhuber, *Long short-term memory*, Neural Comput. 9, 1735
  (1997).
- Cho et al., *Learning phrase representations using RNN encoder–decoder*
  (EMNLP 2014) — GRU, and the encoder–decoder pattern that leads to attention.
- Pascanu, Mikolov & Bengio, *On the difficulty of training recurrent neural
  networks* (ICML 2013) — the spectral-radius analysis and gradient clipping.
  **The one to derive from.**
- Jaeger, *The echo state approach* (GMD Report 148, 2001); Maass, Natschläger &
  Markram, *Real-time computing without stable states*, Neural Comput. 14, 2531
  (2002) — reservoir computing / liquid state machines.

### Physics literature
- Pathak, Hunt, Girvan, Lu & Ott, *Model-free prediction of large
  spatiotemporally chaotic systems from data: a reservoir computing approach*,
  Phys. Rev. Lett. 120, 024102 (2018) — predicting Kuramoto–Sivashinsky
  dynamics past several Lyapunov times, with an untrained random reservoir.
  **This is the concrete example for the chapter**: it is a PRL, it is chaos, and
  it makes the Lyapunov framing above pay off immediately.
- Vlachas et al., *Data-driven forecasting of high-dimensional chaotic systems
  with LSTM networks*, Proc. R. Soc. A 474, 20170844 (2018).

### Exercise
Train an RNN to integrate the Lorenz system one step ahead, then run it
autonomously and measure how many Lyapunov times the trajectory tracks the
truth before decorrelating.

---

## Chapter 10 — *(Optional aside)* Hopfield Networks and Boltzmann Machines

*Short chapter, enormous return on investment for this specific audience. Place
it as a boxed interlude if you don't want to break the main line.*

### Why include it
Because it is the Ising model wearing a lab coat, because it teaches that
**learning is not synonymous with backpropagation**, and because the 2024 Nobel
Prize in Physics went to Hopfield and Hinton for exactly this material. Your
reader will feel, correctly, that they were already halfway to inventing it.

### The constraint
Symmetric weights with no self-coupling, $w_{\mu\nu}=w_{\nu\mu}$, $w_{\mu\mu}=0$,
recurrently connected. Symmetry is precisely the condition for an **energy
function** to exist:

$$E(s) = -\tfrac{1}{2} w_{\mu\nu}\, s_\mu s_\nu - b_\mu s_\mu .$$

Dynamics is asynchronous spin flipping; learning is Hebbian,
$w_{\mu\nu} \propto \sum_p \xi^p_\mu \xi^p_\nu$, for stored patterns $\xi^p$.

### Theorems to name
- **Convergence of Hopfield dynamics.** $E$ is a Lyapunov function: asynchronous
  updates never increase it, so the dynamics converges to a fixed point.
  Memories are attractors; retrieval is relaxation.
- **Storage capacity.** Amit, Gutfreund & Sompolinsky, *Storing infinite numbers
  of patterns in a spin-glass model of neural networks*, Phys. Rev. Lett. 55,
  1530 (1985) and Ann. Phys. 173, 30 (1987): the critical load is
  $\alpha_c = p/N \approx 0.138$; beyond it the retrieval states are destabilised
  by spin-glass states. This is a **replica-symmetric calculation** — one of the
  very few places in ML where a physicist's native technique gives the exact
  answer, and a wonderful thing to show them.
- **Dense associative memory.** Krotov & Hopfield (NeurIPS 2016): higher-order
  interactions raise capacity superlinearly (polynomial or exponential in $N$,
  depending on the energy).

### Papers
- Hopfield, *Neural networks and physical systems with emergent collective
  computational abilities*, PNAS 79, 2554 (1982).
- Ackley, Hinton & Sejnowski, *A learning algorithm for Boltzmann machines*,
  Cognitive Science 9, 147 (1985) — learning rule = difference of correlations
  between "clamped" and "free" phases, i.e. matching data and model moments.
  Pure statistical mechanics.
- Smolensky (1986) — Harmonium, i.e. the restricted Boltzmann machine.
- Hinton, *Training products of experts by minimizing contrastive divergence*,
  Neural Comput. 14, 1771 (2002) — CD-$k$ as truncated Gibbs sampling.
- Ramsauer et al., *Hopfield networks is all you need* (ICLR 2021) — **modern
  continuous Hopfield update = one attention layer.** Use this as the bridge
  into Chapter 12; it retro-justifies the whole aside.
- Torlai & Melko, *Learning thermodynamics with Boltzmann machines*, Phys. Rev. B
  94, 165134 (2016) — and the RBM branch of neural-network quantum states.

---

# TIER 3 — The graph generalises everything

---

## Chapter 11 — Graph Neural Networks

### The constraint
An arbitrary adjacency: $w_{\mu\nu} \neq 0$ only if $(\mu,\nu) \in E$. Message
passing on nodes $\mu$ with neighbourhoods $\mathcal{N}(\mu)$:

$$m_{\mu} = \bigoplus_{\nu \in \mathcal{N}(\mu)} \phi\big(h_\mu, h_\nu, e_{\mu\nu}\big),
\qquad h'_\mu = \psi\big(h_\mu, m_\mu\big),$$

with $\bigoplus$ a **permutation-invariant** aggregator (sum, mean, max).

### Why it belongs here pedagogically
Because it *contains the previous chapters as special cases*, and saying so
consolidates everything:

- complete graph → dense layer (Chapter 3);
- regular lattice with shared kernels → convolution (Chapter 8);
- path graph traversed in order → recurrence (Chapter 9);
- no edges → the same MLP applied to each node independently.

The new symmetry is **permutation equivariance**: the nodes carry no intrinsic
order, so the output must transform by the same permutation as the input,
$f(P h) = P f(h)$. That requirement alone forces the sum-over-neighbours form.

### Theorems to name
- **Deep Sets / permutation-invariance representation theorem.** Zaheer et al.
  (NeurIPS 2017): a function on sets is invariant iff it can be written
  $\rho\big(\sum_\mu \phi(x_\mu)\big)$. The sum is not a heuristic; it is the
  general solution. (Caveats on countability/latent dimension: Wagstaff et al.,
  ICML 2019.)
- **GNN expressivity = 1-Weisfeiler–Lehman.** Xu, Hu, Leskovec & Jegelka, *How
  powerful are graph neural networks?* (ICLR 2019) and Morris et al. (AAAI
  2019): message-passing GNNs cannot distinguish two graphs that 1-WL colour
  refinement cannot. Concrete consequence a physicist will appreciate: a
  standard GNN **cannot count triangles** or tell a 6-cycle from two 3-cycles.
  Architecture ⇒ hard expressivity ceiling; a genuinely useful negative result.
- **Over-smoothing.** Oono & Suzuki (ICLR 2020): repeated message passing
  contracts node features exponentially toward a low-dimensional invariant
  subspace — diffusion reaches its stationary distribution and all nodes become
  identical. Depth in a GNN is not free. This is a heat-equation statement, so
  say it that way.
- **Over-squashing and curvature.** Alon & Yahav (ICLR 2021); Topping et al.,
  *Understanding over-squashing and bottlenecks on graphs via curvature*
  (ICLR 2022) — bottlenecks diagnosed with **Ricci curvature** of the graph. If
  ever a result was designed to appeal to a physicist, this is it.

### Papers
- Scarselli, Gori, Tsoi, Hagenbuchner & Monfardini, *The graph neural network
  model*, IEEE TNN 20, 61 (2009) — the original.
- Bruna, Zaremba, Szlam & LeCun, *Spectral networks and locally connected
  networks on graphs* (ICLR 2014) — convolution via the graph Laplacian
  eigenbasis. **Start here for physicists:** it is spectral theory of the
  Laplacian, which they already own.
- Defferrard, Bresson & Vandergheynst (NeurIPS 2016) — ChebNet, Chebyshev
  polynomial approximation of the spectral filter, giving locality back.
- Kipf & Welling, *Semi-supervised classification with graph convolutional
  networks* (ICLR 2017) — the first-order simplification everyone uses.
- Gilmer, Schoenholz, Riley, Vinyals & Dahl, *Neural message passing for quantum
  chemistry* (ICML 2017) — the unifying MPNN framework, on QM9 molecular
  properties.
- Battaglia, Pascanu, Lai, Rezende & Kavukcuoglu, *Interaction networks for
  learning about objects, relations and physics* (NeurIPS 2016) — **learns
  N-body dynamics with edges as pairwise forces.** The most physics-native entry
  point available.
- Battaglia et al., *Relational inductive biases, deep learning, and graph
  networks* (2018, arXiv:1806.01261) — the conceptual synthesis.

### Physics literature
- Sanchez-Gonzalez et al., *Learning to simulate complex physics with graph
  networks* (ICML 2020) — particle-based fluids and granular media; the GNN
  learns a local interaction law and generalises across particle counts.
- Cranmer et al., *Discovering symbolic models from deep learning with inductive
  biases* (NeurIPS 2020) — train a GNN on an N-body system, then symbolically
  regress the learned edge function and **recover Newton's law of gravitation**.
  A superb capstone example: the network's message is the force.
- Shlomi, Battaglia & Vlimant, *Graph neural networks in particle physics*,
  Mach. Learn.: Sci. Technol. 2, 021001 (2021) — jets and tracking as graphs.
- Batzner et al., *NequIP*, Nature Comm. 13, 2453 (2022) — interatomic
  potentials (forward-reference to Tier 5 equivariance).

---

## Chapter 12 — Attention and Transformers

### The qualitative jump
Every architecture so far had **static** weights: $w_{\mu\nu}$ was a parameter,
fixed after training. Attention makes the coupling a **function of the data**:

$$q_{\mu a} = W^Q_{a\nu} h_{\mu\nu},\quad
k_{\mu a} = W^K_{a\nu} h_{\mu\nu},\quad
v_{\mu a} = W^V_{a\nu} h_{\mu\nu},$$

$$A_{\mu\nu} = \mathrm{softmax}_\nu\!\left(\frac{q_{\mu a} k_{\nu a}}{\sqrt{d}}\right),
\qquad h'_{\mu a} = A_{\mu\nu}\, v_{\nu a}.$$

Physicist's reading: **a self-consistent, configuration-dependent pairwise
interaction kernel.** The coupling matrix is computed from the current state,
like a mean-field or self-consistent-field problem, rather than fixed in
advance. Note also that a transformer is a GNN on the complete graph with
learned, data-dependent edge weights — which is why Chapter 11 comes first.

### Things to derive explicitly
- **Why $\sqrt{d}$.** If $q$ and $k$ have i.i.d. unit-variance components, then
  $q\cdot k$ has variance $d$; without the $1/\sqrt{d}$ the softmax saturates and
  its gradient dies. A three-line variance argument that connects straight back
  to Chapter 5 and to the softmax Jacobian your Chapter 4 already computed.
- **The softmax Jacobian with its cross-terms** — you have already derived this
  (Eq. multi-error-signals and the loss table). Reuse it verbatim; attention is
  where that algebra earns a second dividend.
- **Permutation equivariance, and hence positional encoding.** Self-attention
  is exactly permutation-equivariant, so order information must be injected by
  hand. Sinusoidal encodings are a Fourier basis over position; RoPE (Su et al.,
  2021/2024) implements relative position as a **rotation** in each 2D subspace,
  which is a clean group-theoretic statement.
- **Cost.** $O(n^2 d)$ in sequence length — all-to-all coupling is expensive,
  exactly as in an N-body problem with no cutoff. The efficient-attention
  literature is, structurally, the fast-multipole/tree-code problem again.

### Theorems to name
- **Universality.** Yun, Bhojanapalli, Rawat, Reddi & Kumar, *Are transformers
  universal approximators of sequence-to-sequence functions?* (ICLR 2020) — yes,
  with positional encodings; note the resource cost.
- **Turing completeness.** Pérez, Marinković & Barceló (ICLR 2019); Pérez,
  Barceló & Marinković, JMLR 22 (2021) — with unbounded precision/steps.
- **Attention as kernel smoothing.** Tsai, Bai, Yamada, Morency & Salakhutdinov,
  *Transformer dissection: a unified understanding of transformer's attention via
  the lens of kernel* (EMNLP 2019) — attention is Nadaraya–Watson regression
  with a learned kernel. The single most clarifying reframing for a
  mathematically trained reader.
- **Modern Hopfield equivalence.** Ramsauer et al. (ICLR 2021) — one attention
  step = one update of a continuous Hopfield energy. Attention *is* associative
  retrieval, and Chapter 10 pays off.
- **Interacting-particle view.** Geshkovski, Letrouit, Polyanskiy & Rigollet,
  *A mathematical perspective on Transformers* (2023/2024, arXiv:2312.10794) —
  tokens as interacting particles on a sphere, with clustering as $t\to\infty$.
  A mean-field/kinetic-theory treatment; recommend it as further reading for
  exactly this audience.

### Papers
- Bahdanau, Cho & Bengio, *Neural machine translation by jointly learning to
  align and translate* (ICLR 2015) — attention invented as a fix for the RNN
  encoder–decoder bottleneck. Teach it in this order; attention makes far more
  sense as an answer to a problem the reader has already felt in Chapter 9.
- Vaswani et al., *Attention is all you need* (NeurIPS 2017).
- Dosovitskiy et al., *An image is worth 16x16 words* (ICLR 2021) — ViT; makes
  the CNN-vs-transformer inductive-bias trade-off concrete (weaker prior, more
  data).
- Katharopoulos et al., *Transformers are RNNs* (ICML 2020) and Choromanski et
  al., *Rethinking attention with Performers* (ICLR 2021) — linear-attention
  approximations; kernel feature maps.
- Elhage et al., *A mathematical framework for transformer circuits* (Anthropic,
  2021) — residual stream as a shared communication channel; the cleanest
  mechanistic account of what the layers actually do.

### Physics literature
- Jumper et al., *Highly accurate protein structure prediction with AlphaFold*,
  Nature 596, 583 (2021) — attention over residue pairs, with an explicit
  triangle-inequality-like geometric consistency structure.
- Qu, Li & Qian, *Particle Transformer for jet tagging* (ICML 2022) — attention
  over particle-cloud data with physics-motivated pairwise features.

### Exercise
Implement single-head attention in ~15 lines of numpy from the index equations
above and verify it against `keras.layers.MultiHeadAttention` with one head.
Then plot $A_{\mu\nu}$ for a periodic input and observe what it attends to.

---

# TIER 4 — Different objectives, not just different wiring

A deliberate shift: the *topology* is now secondary and the **objective
function** is the new content. Flag this transition explicitly — the reader
should know the axis of variation has changed.

---

## Chapter 13 — Variational Autoencoders

### The one idea
Chapter 7's bottleneck, made **probabilistic**: the encoder outputs a
distribution $q_\phi(z|x)$ rather than a point, and the decoder is a likelihood
$p_\theta(x|z)$. Maximise the evidence lower bound

$$\log p_\theta(x) \ \ge\ \mathcal{L}_{\rm ELBO}
= \mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x|z)\big] - D_{\rm KL}\big(q_\phi(z|x)\,\|\,p(z)\big).$$

### The physics translation (make this the spine of the chapter)
$-\mathcal{L}_{\rm ELBO}$ **is a variational free energy.** The first term is an
energy (reconstruction), the second is (minus) an entropy-like term pulling
$q$ toward the prior, and the bound is exactly the statement
$F_{\rm var} \ge F_{\rm true}$ that your reader knows as the
**Gibbs–Bogoliubov–Feynman inequality** — the same inequality behind mean-field
theory. "Variational inference" is variational free energy minimisation with a
neural-network ansatz for the trial distribution. State this and the chapter
essentially teaches itself.

- Underlying facts to name: **Jensen's inequality** (whence the bound) and
  **Gibbs' inequality** $D_{\rm KL}\ge 0$ with equality iff the distributions
  agree; optionally the **Donsker–Varadhan** variational representation of KL.
- **The reparametrisation trick.** Write $z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon$
  with $\epsilon\sim\mathcal{N}(0,1)$, so that the randomness carries no
  parameters and the gradient can pass through. Derive why the naive
  score-function estimator has higher variance — this is the one genuinely new
  piece of calculus in the chapter, and it is worth doing slowly.

### Papers
- Kingma & Welling, *Auto-encoding variational Bayes* (ICLR 2014).
- Rezende, Mohamed & Wierstra, *Stochastic backpropagation and approximate
  inference in deep generative models* (ICML 2014) — the simultaneous
  independent derivation.
- Jordan, Ghahramani, Jaakkola & Saul, *An introduction to variational methods
  for graphical models*, Machine Learning 37, 183 (1999) — the pre-neural
  variational framework, in language close to statistical physics.
- Higgins et al., *β-VAE* (ICLR 2017) — tempering the KL term; note that $\beta$
  is a Lagrange multiplier / inverse temperature, and there is a
  rate–distortion trade-off behind it (Alemi et al., *Fixing a broken ELBO*,
  ICML 2018).
- Bowman et al. (CoNLL 2016) — posterior collapse, the characteristic failure.

### Physics literature
- Wetzel, Phys. Rev. E 96, 022140 (2017) — as in Chapter 7; carries the Ising
  example forward from deterministic to variational autoencoder, which makes a
  very clean before/after.
- Rocchetto, Grant, Strelchuk, Carleo & Severini, *Learning hard quantum
  distributions with variational autoencoders*, npj Quantum Inf. 4, 28 (2018).

---

## Chapter 14 — Generative Adversarial Networks

*Include for completeness and contrast; be honest that the field has largely
moved on to diffusion for scientific applications.*

### The one idea
Two networks, one objective, opposite signs: a generator $G$ and a discriminator
$D$ playing

$$\min_G \max_D \ \mathbb{E}_{x\sim p_{\rm data}}[\log D(x)]
+ \mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))].$$

The novelty is **the topology of the training procedure**, not of a network: no
loss is being minimised jointly, so the fixed point is a saddle, not a minimum.
The distinction between "descend a potential" and "find a Nash equilibrium of a
two-player game" is the lesson, and it explains all the instability.

### Theorems to name
- **Optimal discriminator and the JS divergence.** For fixed $G$,
  $D^*(x) = p_{\rm data}(x)/(p_{\rm data}(x)+p_G(x))$, and substituting it makes
  the objective $-\log 4 + 2\,D_{\rm JS}(p_{\rm data}\|p_G)$ — so the ideal game
  minimises a symmetrised KL. Goodfellow et al. (2014), Prop. 1 & Thm. 1. Derive
  it; it is short and it makes the whole scheme principled.
- **Why that fails in practice.** Arjovsky & Bottou, *Towards principled methods
  for training GANs* (ICLR 2017): if the supports are disjoint (as they are, for
  a low-dimensional manifold in high-dimensional space), the JS divergence is
  constant and the gradient vanishes identically.
- **Kantorovich–Rubinstein duality.** Arjovsky, Chintala & Bottou, *Wasserstein
  GAN* (ICML 2017): replace JS by $W_1$, whose dual is a supremum over
  1-Lipschitz functions — hence weight clipping, and then the gradient penalty
  of Gulrajani et al. (NeurIPS 2017). Optimal transport is a subject your reader
  can be pointed to (Villani) with confidence.
- **Local convergence.** Mescheder, Geiger & Nowozin, *Which training methods for
  GANs do actually converge?* (ICML 2018) — a linear-stability analysis of the
  training dynamics near equilibrium. Exactly the analysis a physicist would
  reach for.

### Physics literature
- Paganini, de Oliveira & Nachman, *CaloGAN*, Phys. Rev. D 97, 014021 (2018) —
  GANs as fast surrogates for calorimeter shower simulation, i.e. replacing
  expensive Monte Carlo. The canonical HEP application, and a good illustration
  of *why* a physicist wants a generative model at all: sampling speed.

---

## Chapter 15 — Normalizing Flows

*The generative model that will feel most natural to a physicist, and the one
with the most immediate research payoff.*

### The one idea
Build an **invertible** map $f_\theta: z \mapsto x$ with tractable Jacobian, so
the density transforms exactly:

$$p_X(x) = p_Z\big(f^{-1}(x)\big)\,
\left|\det \frac{\partial f^{-1}}{\partial x}\right| ,
\qquad\text{i.e.}\qquad
\log p_X(x) = \log p_Z(z) - \log\left|\det \frac{\partial f}{\partial z}\right| .$$

No bound, no adversary: the exact log-likelihood is computable, so you can just
maximise it. The entire art is designing layers that are invertible *and* whose
Jacobian determinant is cheap (triangular structure → determinant is a product
of diagonal entries).

### Theorems / classical results to name
- **Change of variables formula** — the whole chapter is one theorem your reader
  has known since undergraduate mechanics.
- **Liouville's theorem** as the volume-preserving special case
  ($|\det J| = 1$), which is precisely what the NICE architecture imposes.
  Symplectic integrators are volume-preserving flows; say so.
- **Instantaneous change of variables.** Chen et al. (2018): for a continuous
  flow $\dot z = g(z,t)$,
  $\frac{d\log p}{dt} = -\nabla\!\cdot g$ — the **continuity equation**. The
  determinant becomes a trace, i.e. a divergence, and the cost drops from
  $O(d^3)$ to $O(d)$ with a Hutchinson trace estimator (Grathwohl et al.,
  *FFJORD*, ICLR 2019).
- **Universality.** Huang, Krueger, Lacoste & Courville, *Neural autoregressive
  flows* (ICML 2018) — sufficiently flexible autoregressive flows are universal
  density approximators.

### Papers
- Tabak & Vanden-Eijnden, *Density estimation by dual ascent of the
  log-likelihood*, Commun. Math. Sci. 8, 217 (2010); Tabak & Turner (2013) — the
  origin of the idea, from applied mathematicians. Worth noting that flows were
  invented outside the ML mainstream.
- Rezende & Mohamed, *Variational inference with normalizing flows* (ICML 2015)
  — the name and the ML framing.
- Dinh, Krueger & Bengio, *NICE* (ICLR 2015 workshop) and Dinh,
  Sohl-Dickstein & Bengio, *Density estimation using Real NVP* (ICLR 2017) —
  coupling layers; the triangular-Jacobian trick.
- Kingma & Dhariwal, *Glow* (NeurIPS 2018); Papamakarios, Pavlakou & Murray,
  *Masked autoregressive flow* (NeurIPS 2017); Durkan et al., *Neural spline
  flows* (NeurIPS 2019).
- Papamakarios et al., *Normalizing flows for probabilistic modeling and
  inference*, JMLR 22, 57 (2021) — the review to cite.

### Physics literature (the strongest in the book — lead with it)
- Noé, Olsson, Köhler & Wu, *Boltzmann generators: sampling equilibrium states of
  many-body systems with deep learning*, Science 365, eaaw1147 (2019) — a flow
  that maps a Gaussian to the Boltzmann distribution, with **exact reweighting**
  because the likelihood is exact. This is the paper that will convince your
  reader the whole book was worth reading.
- Albergo, Kanwar & Shanahan, *Flow-based generative models for Markov chain
  Monte Carlo in lattice field theory*, Phys. Rev. D 100, 034515 (2019);
  Kanwar et al., *Equivariant flow-based sampling for lattice gauge theory*,
  Phys. Rev. Lett. 125, 121601 (2020) — flows that respect gauge symmetry, and
  which attack critical slowing down.
- Nicoli et al., *Asymptotically unbiased estimation of physical observables with
  neural samplers*, Phys. Rev. E 101, 023304 (2020).
- Wirnsberger et al., *Targeted free energy estimation via learned mappings*,
  J. Chem. Phys. 153, 144112 (2020) — learned maps inside a **Jarzynski**-style
  free-energy estimator.

---

## Chapter 16 — Diffusion Models *(the natural climax)*

### Why this is the right ending
Because the foundational paper is *literally* a nonequilibrium statistical
mechanics paper, and every ingredient is something your reader already owns:
Langevin dynamics, Fokker–Planck, Ornstein–Uhlenbeck processes, detailed
balance, annealed importance sampling.

### The construction
**Forward** (fixed, no learning): progressively destroy the data with noise,
$q(x_t|x_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t 1\big)$,
which in the continuum is an Ornstein–Uhlenbeck SDE
$dx = -\tfrac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dW$, driving any
distribution to $\mathcal{N}(0,1)$ — thermalisation.

**Reverse** (learned): run the diffusion backwards. The remarkable fact is that
the time-reversed process is again a diffusion, with drift corrected by the
**score** $\nabla_x \log p_t(x)$:

$$dx = \left[-\tfrac{1}{2}\beta(t)x - \beta(t)\nabla_x \log p_t(x)\right]dt
+ \sqrt{\beta(t)}\,d\bar W .$$

So the network's only job is to learn the score — equivalently, to denoise. The
network architecture is just a U-Net (Chapters 6+8) or a transformer (Chapter
12); *all* the novelty is in this objective.

### Theorems to name
- **Anderson's reverse-time SDE theorem.** Anderson, *Reverse-time diffusion
  equation models*, Stochastic Processes Appl. 12, 313 (1982) — the pre-ML
  probability result that makes everything work. Cite the original; it dignifies
  the subject and shows the physics was there first.
- **Fokker–Planck equation** for $p_t$, and the **probability-flow ODE**: a
  deterministic ODE with the same time-marginals as the SDE, which turns a
  diffusion model into a continuous normalizing flow and hence gives exact
  likelihoods. This unifies Chapters 15 and 16 — an excellent closing argument.
- **Hyvärinen's score-matching identity.** Hyvärinen, JMLR 6, 695 (2005): the
  intractable $\mathbb{E}\|s_\theta - \nabla\log p\|^2$ equals a tractable
  objective involving $\mathrm{tr}(\nabla s_\theta)$ plus a constant, by
  integration by parts. The derivation is two lines and is the crux of the whole
  field.
- **Denoising score matching.** Vincent, *A connection between score matching and
  denoising autoencoders*, Neural Comput. 23, 1661 (2011) — estimating the score
  of the noised distribution is *the same thing* as optimal denoising. Ties the
  climax of the book back to Chapter 7.
- **ELBO ≡ the diffusion loss.** Ho et al. (2020) show the variational bound
  reduces, after reweighting, to a simple $\|\epsilon - \epsilon_\theta\|^2$
  regression. Connect to Chapter 13: a diffusion model is a deep hierarchical
  VAE with a fixed encoder.
- **Jarzynski equality / annealed importance sampling** — the framing of
  Sohl-Dickstein et al. (2015), which is where the whole idea came from.

### Papers
- Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli, *Deep unsupervised learning
  using nonequilibrium thermodynamics* (ICML 2015) — **the origin, and written
  by physicists in physics language.** Teach from this one.
- Song & Ermon, *Generative modeling by estimating gradients of the data
  distribution* (NeurIPS 2019) — noise-conditional score networks and annealed
  Langevin sampling.
- Ho, Jain & Abbeel, *Denoising diffusion probabilistic models* (NeurIPS 2020) —
  the practical formulation.
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole, *Score-based generative
  modeling through stochastic differential equations* (ICLR 2021) — **the
  unifying continuous-time paper**; SDE/ODE duality, exact likelihoods.
- Karras, Aittala, Aila & Laine, *Elucidating the design space of diffusion-based
  generative models* (NeurIPS 2022) — disentangles the many conventions into a
  clean parametrisation; the paper to read before implementing.
- Lipman et al., *Flow matching for generative modeling* (ICLR 2023) and
  Albergo & Vanden-Eijnden, *Stochastic interpolants* (ICLR 2023) — the modern
  synthesis of flows and diffusions, the latter by physicists/applied
  mathematicians.

### Physics literature
- Wang, Aarts & Zhou, *Diffusion models as stochastic quantization in lattice
  field theory*, JHEP 05 (2024) 060 — the diffusion process *is* stochastic
  quantisation; the loop closes completely.
- Watson et al., *De novo design of protein structure and function with
  RFdiffusion*, Nature 620, 1089 (2023) — for scope.

---

# TIER 5 — Physics-specific payoff chapters (choose two or three; do not cover all)

These are the chapters where the reader stops learning ML and starts doing
physics with it. Selection should follow your own research taste — a "nutshell"
book is better served by two deep chapters than five shallow ones.

---

## 17a — Neural ODEs and Continuous-Depth Models
- **Idea:** take $\Delta\ell\to 0$ in Chapter 6. Depth becomes continuous time;
  the network parametrises a vector field; training is an optimal-control
  problem.
- **Theorems:** the **adjoint method** for gradients — which is
  **Pontryagin's maximum principle**, and identical to the adjoint-state method
  used in geophysical and variational data assimilation; memory cost becomes
  $O(1)$ in depth. **Picard–Lindelöf** for existence/uniqueness of the flow.
  **Dupont, Doucet & Teh, *Augmented Neural ODEs* (NeurIPS 2019):** the flow map
  is a homeomorphism, so trajectories cannot cross — hence there exist simple
  functions (e.g. a 1D reflection) a neural ODE *cannot* represent. A sharp and
  instructive negative result about a topological constraint.
- **Papers:** Chen, Rubanova, Bettencourt & Duvenaud, *Neural ordinary
  differential equations* (NeurIPS 2018); Massaroli et al., *Dissecting neural
  ODEs* (NeurIPS 2020); Kidger, *On neural differential equations* (DPhil
  thesis, Oxford, 2022 — the best single reference).

## 17b — Equivariant and Geometric Networks *(strongest candidate: this is the punchline of §0)*
- **Idea:** demand $f(\rho_{\rm in}(g)x) = \rho_{\rm out}(g) f(x)$ for a group $G$
  — E(3), SO(3), the Lorentz group, a gauge group, a permutation group. The
  symmetry is then exact by construction, not learned from data, and sample
  efficiency improves dramatically.
- **Theorems:** Kondor & Trivedi (ICML 2018) — equivariant linear maps are group
  convolutions; **Peter–Weyl** and **Schur's lemma** as the harmonic-analysis
  backbone of steerable filters; **Wigner–Eckart theorem for equivariant
  kernels** (Lang & Weiler, ICLR 2021) — the kernel basis is fixed by
  Clebsch–Gordan coefficients. Your reader learned Wigner–Eckart in quantum
  mechanics; discovering that it *constructs a neural network layer* is the
  single most delightful moment available in this entire book.
- **Papers:** Cohen & Welling, *Group equivariant convolutional networks*
  (ICML 2016); Cohen & Welling, *Steerable CNNs* (ICLR 2017); Weiler & Cesa,
  *General E(2)-equivariant steerable CNNs* (NeurIPS 2019); Thomas et al.,
  *Tensor field networks* (2018, arXiv:1802.08219); Fuchs et al., *SE(3)-
  Transformers* (NeurIPS 2020); Satorras, Hoogeboom & Welling, *E(n)-equivariant
  graph neural networks* (ICML 2021); Bronstein, Bruna, Cohen & Veličković,
  *Geometric deep learning: grids, groups, graphs, geodesics and gauges*
  (2021, arXiv:2104.13478) — the manifesto, and the natural companion volume to
  your own book.
- **Physics:** Batzner et al., *NequIP*, Nature Comm. 13, 2453 (2022) and
  Batatia et al., *MACE* (NeurIPS 2022) — E(3)-equivariant interatomic
  potentials, now standard in computational chemistry; Bogatskiy et al.,
  *Lorentz group equivariant neural network for particle physics* (ICML 2020);
  Favoni, Ipp, Müller & Schuh, *Lattice gauge equivariant convolutional neural
  networks*, Phys. Rev. Lett. 128, 032003 (2022) — **gauge equivariance as an
  architecture**; Cohen, Weiler, Kicanaoglu & Welling, *Gauge equivariant
  convolutional networks and the icosahedral CNN* (ICML 2019).

## 17c — Physics-Informed Neural Networks (PINNs)
- **Idea:** put the PDE in the loss. Represent the solution as $u_\theta(x,t)$ and
  penalise the residual $\mathcal{R}[u_\theta]$ at collocation points, using
  automatic differentiation for the derivatives. No mesh, no training data
  beyond boundary/initial conditions. Also the honest inverse-problem tool:
  fit unknown coefficients jointly with the field.
- **Theorems / caveats:** Shin, Darbon & Karniadakis, *On the convergence of
  physics-informed neural networks for linear second-order elliptic and
  parabolic type PDEs*, Commun. Comput. Phys. 28, 2042 (2020) — consistency
  results; Mishra & Molinaro for generalisation error bounds. **Be honest about
  the failure modes:** Krishnapriyan et al., *Characterizing possible failure
  modes in physics-informed neural networks* (NeurIPS 2021) — PINNs fail on
  convection-dominated and multiscale problems; Wang, Teng & Perdikaris,
  *Understanding and mitigating gradient pathologies in PINNs*, SIAM J. Sci.
  Comput. 43, A3055 (2021) — an NTK analysis of the loss-term imbalance. A
  chapter that presents PINNs uncritically would misinform a reader who might
  actually use them.
- **Papers:** Dissanayake & Phan-Thien (1994) and Lagaris, Likas & Fotiadis,
  IEEE TNN 9, 987 (1998) — the pre-history, worth citing for honesty;
  Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*,
  J. Comput. Phys. 378, 686 (2019) — the paper that launched the field;
  Karniadakis et al., *Physics-informed machine learning*, Nature Rev. Phys. 3,
  422 (2021) — the review.

## 17d — Operator Learning
- **Idea:** learn a map between *function spaces*, $\mathcal{G}: a \mapsto u$
  (e.g. coefficient field → PDE solution), so that one trained model solves a
  whole family of PDEs and is resolution-independent.
- **Theorems:** **Chen & Chen**, *Universal approximation to nonlinear operators
  by neural networks with arbitrary activation functions*, IEEE TNN 6, 911
  (1995) — the 1995 theorem DeepONet is built on, and a nice demonstration that
  the theory long preceded the practice; Kovachki, Lanthaler & Mishra, *On
  universal approximation and error bounds for Fourier neural operators*, JMLR
  22, 290 (2021).
- **Papers:** Lu, Jin, Pang, Zhang & Karniadakis, *DeepONet*, Nature Mach.
  Intell. 3, 218 (2021); Li et al., *Fourier neural operator for parametric
  partial differential equations* (ICLR 2021) — **convolution as multiplication
  in Fourier space, i.e. a learned spectral method**, which closes the loop with
  the convolution theorem in Chapter 8; Kovachki et al., *Neural operator:
  learning maps between function spaces*, JMLR 24, 89 (2023).

## 17e — Hamiltonian, Lagrangian and Symplectic Networks
- **Idea:** don't learn the dynamics — learn the *generator*. Parametrise
  $H_\theta(q,p)$ and integrate $\dot q = \partial_p H$, $\dot p = -\partial_q H$.
  Energy conservation and symplectic structure hold by construction rather than
  approximately.
- **Theorems:** **Liouville's theorem** and symplecticity (preserved exactly by
  the architecture + a symplectic integrator); **Noether's theorem** — imposed
  symmetries of $H_\theta$ give exactly conserved quantities; universality of
  **SympNets** for symplectic maps (Jin, Zhang, Zhu & Karniadakis, Neural
  Networks 132, 166 (2020)).
- **Papers:** Greydanus, Dzamba & Yosinski, *Hamiltonian neural networks*
  (NeurIPS 2019); Cranmer et al., *Lagrangian neural networks* (ICLR 2020
  workshop) — works in arbitrary coordinates, no canonical momenta required;
  Chen, Zhang, Arjovsky & Bottou, *Symplectic recurrent neural networks*
  (ICLR 2020); Toth et al., *Hamiltonian generative networks* (ICLR 2020).

---

# Cross-cutting material you will need somewhere

Best placed as short appendices or a single "practicalities" chapter, so the
architecture chapters stay clean:

- **Optimisers.** SGD → momentum (a damped particle in a potential — literally,
  Polyak 1964) → Nesterov → RMSProp/Adagrad → Adam (Kingma & Ba, ICLR 2015) →
  AdamW (Loshchilov & Hutter, ICLR 2019). Your commented-out "Review of
  concepts" section already sketches this; the physics framing (damping,
  effective mass, preconditioning, temperature-like noise from mini-batching) is
  worth a page. See also Mandt, Hoffman & Blei on SGD as approximate Bayesian
  inference / an SDE with a temperature set by learning rate over batch size.
- **Generalization, and why the classical picture fails.** Zhang et al.,
  *Understanding deep learning requires rethinking generalization* (ICLR 2017) —
  networks fit random labels perfectly, so capacity-based bounds cannot explain
  anything; Belkin, Hsu, Ma & Mandal, *Reconciling modern machine learning
  practice and the bias–variance trade-off*, PNAS 116, 15849 (2019) — **double
  descent**, which is a phase transition and will fascinate your reader;
  Bahri et al., *Statistical mechanics of deep learning*, Annu. Rev. Condens.
  Matter Phys. 11, 501 (2020).
- **The infinite-width limit.** Neal (1996) for the shallow case; Lee et al.,
  *Deep neural networks as Gaussian processes* (ICLR 2018); Jacot, Gabriel &
  Hongler, *Neural tangent kernel* (NeurIPS 2018); Roberts, Yaida & Hanin,
  *The Principles of Deep Learning Theory* (Cambridge, 2022) — an effective
  field theory / $1/n$ expansion treatment of deep networks, written by
  physicists for physicists. **If your book has an intellectual sibling, it is
  this one** — worth an explicit "where to go next" pointer.
- **Scaling laws.** Kaplan et al. (2020); Hoffmann et al. (*Chinchilla*, 2022).
  Power laws with clean exponents; irresistible to a physicist and genuinely
  useful for planning experiments.

---

# General references to recommend to the reader

- **Mehta, Bukov, Wang, Day, Richardson, Fisher & Schwab, *A high-bias,
  low-variance introduction to machine learning for physicists*, Physics Reports
  810, 1 (2019).** The closest existing work to your book, with notebooks. Cite
  it in the preface, and position your book as the complement: they survey
  broadly from a statistical-mechanics standpoint; you build architectures from
  one neuron upward in index notation.
- Carleo, Cirac, Cranmer, Daudet, Schuld, Tishby, Vogt-Maranto & Zdeborová,
  *Machine learning and the physical sciences*, Rev. Mod. Phys. 91, 045002
  (2019). The map of applications.
- Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press, 2016) — the
  standard reference, now dated on architectures but still excellent on
  fundamentals.
- Prince, *Understanding Deep Learning* (MIT Press, 2023) — best modern
  figures, freely available.
- Bishop & Bishop, *Deep Learning: Foundations and Concepts* (Springer, 2024).
- Murphy, *Probabilistic Machine Learning: Advanced Topics* (MIT Press, 2023) —
  the reference for Tier 4.
- Dawid et al., *Modern applications of machine learning in quantum sciences*
  (2022, arXiv:2204.04198) — long lecture notes, quantum-focused.

---

# Sequencing summary

```
Chap 1–4  (written)   1 neuron → 2 neurons → dense MLP → concrete examples
   │
   ├── TIER 1  5. Depth & why it hurts        (bridge: Jacobian products, init, norm)
   │           6. Residual connections        (identity in the Jacobian; depth = time)
   │           7. Autoencoders                (bottleneck; unsupervised; PCA theorem)
   │
   ├── TIER 2  8. CNNs                        (Toeplitz + sharing; stencils; equivariance)
   │           9. RNNs                        (sharing in time; spectral radius; Lyapunov)
   │          10. Hopfield / Boltzmann        (optional aside: symmetric w ⇒ energy; Ising)
   │
   ├── TIER 3 11. Graph neural networks       (arbitrary adjacency; permutation equivariance)
   │          12. Attention / transformers    (input-dependent w; complete graph)
   │
   ├── TIER 4 13. VAE                         (ELBO = variational free energy)
   │          14. GANs                        (a game, not a potential)
   │          15. Normalizing flows           (change of variables; Boltzmann generators)
   │          16. Diffusion models            (Langevin, Fokker–Planck, score matching)
   │
   └── TIER 5 pick 2–3: neural ODEs · equivariant nets · PINNs · operator learning ·
                        Hamiltonian nets
```

**Hard dependencies:** 5 → 6 → (9 gating, 16 U-Net, 17a). 8 → 11 → 12.
7 → 13 → 16. 15 → 16. 10 → 12 (the Hopfield/attention equivalence).
8 → 17b (equivariance generalises translation) and 8 → 17d (convolution theorem
→ spectral methods).

**Two defensible reorderings.** (i) Swap 7 and 8 if you would rather hit
convolutions while the dense weight matrix is still fresh, and use autoencoders
later as the on-ramp to Tier 4. (ii) Attention before graphs — transformers are
attention on the complete graph, so either order is coherent, but 11 → 12 makes
attention feel *derived* rather than magical, which suits this book's method.

**Minimal path**, if the "nutshell" promise starts to strain: 5, 6, 8, 9, 12,
15/16, plus one Tier-5 chapter chosen for your own research area. That is a
complete and honest modern education in architectures, and it preserves the
book's central virtue — that every chapter is the smallest possible step from
the previous one.
