# Evaluating Explainable AI. A Comparative Study of SENN, IG, and LIME
By Alessandra Gandini and Gaudenzia Genoni  
University of Trento
***

The study compares the intelligibility and faithfulness of explanations from a self-explainable neural
network (SENN) and two post-hoc methods—Integrated Gradients (IG) and LIME—on MNIST and
Confounded MNIST, a dataset designed to evaluate model reliance on spurious features. Through
a primarily qualitative analysis, supported by quantitative measures, we show that SENN fails to
provide meaningful explanations, while IG and LIME offer more faithful and interpretable attribu-
tions. Confounded MNIST reveals the Clever Hans effect, underscoring the need for robust evaluation
methods in Explainable AI.

==> To reproduce the experiments, run Notebook_1_MNIST.ipynb and Notebook_2_CONFOUNDED.ipynb (using Google Colab is suggested).
