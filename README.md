# InverseDesign

A collection of part of the data analysis used for my project in inverse design. The code specifically is used to investigate the relationship between 2D and 3D solution spaces at local minima.

The backend cannot be made publicly available.

The projects considers light going through a optic chip. To investigate how the light passes through, we consider a linear equation of the form Ae = b for some matrix A and vectors e,b. e represents the electric field. A depends on how the cip is configured by eletric permitivities in the chip. Let p be the vector of around 10,000 permittivity configurations in the chip. The goal is to optimize some function f that depends on the electric field e. As e depends on the permittivity p, then f = f(p) is a function of p.

Due to the size of p, the solution space is enormous and is very difficult to quantify. We want to investigate how narrow the local minima are when doing a 2D vs a 3D simulation. This is done by measuring the size of the second derivative matrix d²/dp_kdp_l f(p). To compute this second derivative, we compute de/dp_k for all indeces p_k in p. This requres solving the equation A de/dp_k = dA/dp_k e for all indeces p_k in p. This is done in simulate_de_dpk.py

Next, the second derivative itself needs to be computed. This requires loading about 150GB onto the RAM from the previous process. Due to the total amount of computations necessary, this is done on the GPU. The data is loaded in blocks using multiprocessing, copied on the GPU, computed, and then the memory is released.
