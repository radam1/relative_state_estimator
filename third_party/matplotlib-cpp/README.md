# matplotlib-cpp (vendored)

Header-only C++ wrapper around Python's matplotlib, from
https://github.com/lava/matplotlib-cpp (master, MIT licence -- see LICENSE).

Local patch:
- `subplot()` passed nrows / ncols / plot_number as Python floats, which
  matplotlib >= 3 rejects ("Call to subplot() failed").  They are now passed
  with `PyLong_FromLong`.

`src/plotter.cpp` includes it with `WITHOUT_NUMPY`, so only the Python
headers and a Python with matplotlib are needed (no numpy C API).
