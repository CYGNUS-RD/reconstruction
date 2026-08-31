echo "cythonizing the noise part"
cython cython_cygno.pyx
python setup.py build_ext --inplace

