# ZetaOptimizer: A Riemann Zeta Function‑Inspired Optimizer for Deep Learning

## 🧑‍💻 Author

**Samiksha BC**
Indiana University


**ZetaOptimizer** is a novel stochastic‑gradient optimizer that integrates the Riemann zeta function $\zeta(s)$ into the update rule.  It blends Adam‑style momentum with dynamically scheduled zeta scaling, adaptive damping, and cosine‑similarity‑based boost.

> **Status**: Research prototype – early empirical results on **MNIST**,  and **CIFAR‑100** show consistent gains over Adam and momentum‑SGD.

---

## 🔑 Key Features

* **Zeta‑scaled learning rate** – update magnitude modulated by $\zeta(s)$ with a time‑varying exponent `s`.
* **Hybrid Adam × Zeta step** – weighted mix (`adam_mix`) of Adam update and zeta‑scaled gradient.
* **Adaptive damping** – zeta damp factor driven by gradient norm and loss EMA.
* **Cosine‑similarity momentum boost** – encourages consistent gradient directions.
* **Gradient clipping & learning‑rate cosine decay** for extra stability.

---

## 📦 Installation

```bash
pip install torch torchvision scipy
```

Python ≥ 3.9 recommended.  A CUDA‑capable GPU is strongly encouraged; CPU also works but will be slow for CIFAR‑scale experiments.

---

## 📁 Datasets & Notebooks

| Notebook                   | Dataset(s)      | Purpose                                        |
| -------------------------- | --------------- | ---------------------------------------------- |
| `Mnist_and_CIFAR‑10.ipynb` | MNIST, CIFAR‑10 | Quick reproduction of basic results            |
| `CIFAR‑100.ipynb`          | CIFAR‑100       | Larger‑scale test with 100‑class image dataset |

Datasets are downloaded automatically via **torchvision**.

---

## 📊 Result Snapshot (5–20 Epoch Runs)

| Optimizer         | MNIST Acc.  | CIFAR‑10 Acc. | CIFAR‑100 Acc. |
| ----------------- | ----------- | ------------- | -------------- |
| Adam              | 97.53 %     | 50.72 %       | 21.70 %        |
| SGD (+ momentum)  | 97.69 %     | 45.32 %       | 22.51 %        |
| **ZetaOptimizer** | **97.62 %** | **52.20 %**   | **25.71 %**    |

These early runs use simple fully‑connected networks; deeper CNNs are expected to widen the performance gap.

---

## 🚀 Quick Start

```bash
python main.py  # trains on MNIST, CIFAR‑10, CIFAR‑100 sequentially and prints results
```

Tune hyper‑parameters such as `lr`, `adam_mix`, `s_min/s_max`, or network architecture to explore further.

---


