import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Derivative → Original Function Builder", layout="wide")
st.title("�� Derivative → Original Function Constructor")
st.markdown(
"""
This app reconstructs an **original function** from a given
- first derivative **f'(x)** or
- second derivative **f''(x)**
You may also supply **initial conditions** and **zeros of the derivative**
to fully determine the function.
Graphs are vertically aligned for easy comparison.
"""
)
# -----------------------------
# Symbol setup
# -----------------------------
x = sp.symbols('x')
C1, C2 = sp.symbols('C1 C2')
# -----------------------------
# Inputs
# -----------------------------
col1, col2 = st.columns(2)
with col1:
derivative_type = st.radio(
"What are you providing?",
["First Derivative f'(x)", "Second Derivative f''(x)"]
)
derivative_input = st.text_input(
"Enter the derivative expression (use x):",
value="2*x"
)
with col2:
x_min, x_max = st.slider(
"x-range for graphing",
-10.0, 10.0, (-5.0, 5.0)
)
# Initial conditions
st.subheader("Initial Conditions (optional)")
ic_text = st.text_area(
"Enter conditions (one per line, examples: f(0)=1, f'(2)=0):",
""
)
# Zeros of derivative
zeros_text = st.text_input(
"Zeros of f'(x) (comma-separated, optional):",
""
)
# -----------------------------
# Helper functions
# -----------------------------
def parse_conditions(text, f_expr, fp_expr):
equations = []
for line in text.splitlines():
if "=" not in line:
continue
left, right = line.split("=")
right = float(right.strip())
if "f'(" in left:
val = float(left[left.find("(")+1:left.find(")")])
equations.append(fp_expr.subs(x, val) - right)
elif "f(" in left:
val = float(left[left.find("(")+1:left.find(")")])
equations.append(f_expr.subs(x, val) - right)
return equations
# -----------------------------
# Main computation
# -----------------------------
try:
derivative_expr = sp.sympify(derivative_input)
if derivative_type == "First Derivative f'(x)":
f_prime = derivative_expr
f = sp.integrate(f_prime, x) + C1
f_double_prime = sp.diff(f_prime, x)
constants = [C1]
else:
f_double_prime = derivative_expr
f_prime = sp.integrate(f_double_prime, x) + C1
f = sp.integrate(f_prime, x) + C2
constants = [C1, C2]
# Build equations
equations = []
# Initial conditions
equations += parse_conditions(ic_text, f, f_prime)
# Zeros of derivative
if zeros_text.strip():
for z in zeros_text.split(","):
z = float(z.strip())
equations.append(f_prime.subs(x, z))
# Solve constants
if equations and constants:
solution = sp.solve(equations, constants, dict=True)
if solution:
sol = solution[0]
f = f.subs(sol)
f_prime = f_prime.subs(sol)
f_double_prime = f_double_prime.subs(sol)
# Lambdify
f_func = sp.lambdify(x, f, "numpy")
fp_func = sp.lambdify(x, f_prime, "numpy")
fpp_func = sp.lambdify(x, f_double_prime, "numpy")
xs = np.linspace(x_min, x_max, 600)
ys = f_func(xs)
yps = fp_func(xs)
ypps = fpp_func(xs)
# -----------------------------
# Critical & inflection points
# -----------------------------
critical_points = sp.solve(f_prime, x)
inflection_points = sp.solve(f_double_prime, x)
critical_points = [float(c) for c in critical_points if c.is_real]
inflection_points = [float(i) for i in inflection_points if i.is_real]
# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
axes[0].plot(xs, ys)
axes[0].set_ylabel("f(x)")
axes[0].grid(True)
axes[1].plot(xs, yps)
axes[1].set_ylabel("f'(x)")
axes[1].grid(True)
axes[2].plot(xs, ypps)
axes[2].set_ylabel("f''(x)")
axes[2].set_xlabel("x")
axes[2].grid(True)
# Connector lines
for c in critical_points:
if x_min <= c <= x_max:
axes[0].plot([c, c], [f_func(c), 0], linestyle="--")
axes[1].plot([c, c], [0, 0], linestyle="--")
for i in inflection_points:
if x_min <= i <= x_max:
axes[0].plot([i, i], [f_func(i), 0], linestyle="--")
axes[2].plot([i, i], [0, 0], linestyle="--")
st.pyplot(fig)
# -----------------------------
# Display formulas
# -----------------------------
st.subheader("Reconstructed Functions")
st.latex(f"f(x) = {sp.latex(f)}")
st.latex(f"f'(x) = {sp.latex(f_prime)}")
st.latex(f"f''(x) = {sp.latex(f_double_prime)}")
except Exception as e:
st.error(f"Error: {e}")
