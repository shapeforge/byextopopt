ByExTopopt
=========

Code for our paper:

> [Structure and Appearance Optimization for Controllable Shape Design][1]<br>
> Jonàs Martínez, Jérémie Dumas, Sylvain Lefebvre, Li-Yi Wei<br>
> ACM Transactions on Graphics (TOG) — Proceedings of ACM SIGGRAPH Asia 2015<br>
> Volume 34 Issue 6, December 2015

[1]: https://sites.google.com/site/jonasmartinezbayona/structure_appearance

Dependencies
------------

- Python 2 or 3
- [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt)
- Numpy
- Matplotlib
- [CVXOPT](http://cvxopt.org/)
- [PyOpenCL](https://mathema.tician.de/software/pyopencl/)

You need to compile and install NLopt with its Python bindings. Moreover, you will also need PyOpenCL to run the OpenCL PatchMatch code used by the appearance function. Cholmod (through CXVOPT) is used for the solving the linear system in the FEM elasticity problem. MMA (through NLopt) is used to solve the constrained non-linear minimization problem.


Usage
-----

```
python main.py input/l_beam.json output.png
```


License
-------

The code in this project is distributed under MIT license.
The topology optimization code used in this project is based on the Python implementation of the SIMP method available on the [DTU TopOpt](http://www.topopt.dtu.dk/?q=node/881) research group.
We also use a modified version of [OpenCL implementation](https://github.com/abiusx/CLPatchMatch) of PatchMatch by @abiusx.


Acknowledgments
---------------

This work was supported by ERC grant ShapeForge (StG-2012-307877) and general research fund Dynamic Element Textures (HKU 717112E).
