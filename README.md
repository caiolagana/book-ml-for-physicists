# Machine Learning for Physicists

*by Caio Laganá*

A book for physicists who already have the mathematical intuition for machine
learning and just need someone to connect it to the vocabulary and the code.

- **Sources:** [book.tex](book.tex)
- **Rendered PDF:** [book.pdf](book.pdf)

Build it locally with:

```bash
pdflatex book.tex && pdflatex book.tex   # twice, to resolve the ToC and refs
```

---

## Preface

So, you're a physicist looking to dive into the world of machine learning. You
have a deep understanding of calculus, are familiar with experimental
data-fitting methods, and have mastered concepts like entropy and
non-linearity — yet you have only a vague idea of how a machine learning
algorithm is actually written and how it works? Then this book is for you. And
the good news is: your learning curve will grow fast. I've been in your shoes.
I used to view machine learning as a distant horizon, but I soon realized that
the knowledge I'd gained in physics made it incredibly easy to quickly master
the concepts and engineering behind the field. So, come along and be amazed.

---

## Chapter 1 — Single-neuron ML

You have been using a single-neuron network since the first time you fitted
experimental data, but perhaps no one ever told you that. Let's recall the
basics. You have a set of experimental data points

$$(x_i, y_i)$$

through which to fit a linear function

$$f(x) = ax + b.$$

The Mean Squared Error, defined as

$$
\begin{aligned}
\mathrm{MSE}(a,b) &= \frac{1}{N}\sum_{i=1}^{N}\left[y_i - f(x_i)\right]^2 \\
                  &= \frac{1}{N}\sum_{i=1}^{N}\left[y_i - (ax_i + b)\right]^2
\end{aligned}
$$

tells us how distant the curve $f(x)$ is from the datapoints. Fitting $f(x)$ to
the datapoints $(x_i, y_i)$ means minimizing $\mathrm{MSE}(a,b)$, that is,
finding $a$ and $b$ such that

$$\frac{\partial}{\partial(a,b)}\mathrm{MSE}(a,b) = 0.$$

No surprises so far, right? Well, what if I tell you the above is the simplest
possible neural network? Indeed, it is a single-neuron network, where the linear
function $f(x) = ax + b$ is the **activation function**, the mean squared error
$\mathrm{MSE}(a,b)$ is the **loss function**, and setting its derivative to zero
is a very rudimentary **backpropagation** method.

Solving the zero-derivative condition for $a$ and $b$, which in this simple case
is possible analytically, fixes the parameters to $a_0$ and $b_0$

$$
\begin{aligned}
a_0 &= \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2} \\
b_0 &= \bar{y} - a_0\bar{x}
\end{aligned}
$$

and we end up with a model $f_0(x) = a_0 x + b_0$ (here $\bar{x}$ and $\bar{y}$
are the mean values of the $x_i$ and $y_i$ datapoints). This model is a single
neuron, no hidden layers: it receives one input parameter and outputs a number.
$f_0$ has learned from the data $(x_i, y_i)$ through the zero-derivative
condition and can now predict on any given $x$. Pretty cool, huh? The concepts
were sitting there all the time. Take a moment to re-read the definitions and
let's jump to the second simplest model: a two-neuron network.

---

## Chapter 2 — Two-neuron ML

Let us now concatenate two neurons.

*(in progress)*
