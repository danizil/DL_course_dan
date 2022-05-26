r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. **True**: we cannot generalize the out of sample error from the test set, we need a validation set to do that.
2. **False**: There are preferable splits of the data for better out of sample estimate. 
    The training set should be bigger than the validation e.g. train set is 4 times bigger than the validation
3.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

This is because we can rewrite the expression as:  
$$
L_{i}(\mat{W}) =  \Delta\sum_{j \neq y_i} \max\left(0, 1 + \frac{\vectr{w_j}}{\Delta} \vec{x_i} - \frac{\vectr{w_{y_i}}}{\Delta} \vec{x_i}\right) = 
\Delta L_{i}^*(\mat {\frac{W}{\Delta}})
$$  
where  
$$L_{i}^* = \sum_{j \neq y_i} \max\left(0, 1 + \frac{\vectr{w_j}}{\Delta} \vec{x_i} - \frac{\vectr{w_{y_i}}}{\Delta} \vec{x_i}\right)$$  
 
So the entire loss becomes:  

$$
L(\mat{\frac{W}{\Delta}}) =
\frac{\Delta}{N} \sum_{i=1}^{N} L_{i}^*(\mat{\frac{W}{\Delta}})
+
\frac{\lambda}{2} \norm{\mat{\frac{W}{\Delta}}}^2
$$  
Having ${\Delta}$ before the expression has no effect on the loss's relative minimization, 
and having it in the denominator of the weights is a simple scaling that will depend only on
the regularization. 


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
