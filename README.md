# MAA106 - Introduction to Numerical Analysis 数值分析导入

## Professors
- [Maxime Breden](https://sites.google.com/site/maximebreden/)

## Course Organization
- 7 lectures (2h) and 7 Tutorials (2h) 28 hours in total with 3 ECTS

## 📚 Objective: Introduction to computational mathematics
* Practical knowledge of basic (but fundamental) mathematical algorithms
* Theoretical study: introduction to the notions of error / convergence / speed of convergence
* Practical implementation: lab sessions using Python and Jupyter notebooks

## 📚 Arrangement of this course

* [x] Chapter 0: Introduction to numerical analysis and introduction to **Numpy** and **matplotlib** (1 lec, 1 TD)
* [x] Chapter 1: solving equations of one variable (2 lec, 2 TD)
* [ ] Chapter 2: polynomial approximation (2 lec, 2 TD)
* [ ] Chapter 3: numerical integration (2 lec, 2 TD)

### ✏️  Chapter 0: Introduction to numerical analysis and introduction to **Numpy** and **matplotlib**
- Machine Number
- Introduction to numerical analysis
- Introduction to Numpy and Matplotlib

```python
t = np.linspace(0, 12 * pi, 1000)
x = np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) + np.sin(t / 12))
y = np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) + np.sin(t / 12))

plt.figure()
plt.plot(x, y)  # 绘制图像
plt.axis('equal')  # 使x轴和y轴的单位长度相等
plt.title("The butterfly curve!")
plt.show()
```


### ✏️  Chapter 1: solving equations of one variable
- Convergence / order of convergence
- Error estimator
- Stopping criterion
- the bisection method 二分法
- Fixed point iterations 定点迭代法
- The Newton-Raphson method 牛顿二分法

### ✏️  Chapter 2: polynomial approximation
### ✏️  Chapter 3: numerical integration

## Tools of this course
JupyterLab, Python, Numpy, matplotlib, mathematical analysis

