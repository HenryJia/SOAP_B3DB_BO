# SOAP_BO
A Bayesian Optimisation Approach to optimise the SOAP descriptor. 

## Motivation
The Smooth Overlap of Atomic Positions (SOAP) descriptor [[1]](#1) is a set of mathematical objects that can be used to
represents and/or extract information from molecular structures. The SOAP descriptor has been used to build machine
learning-based interatomic potentials for a number of materials. More relevant to the `SOAP_BO` code, however, is the
usage of the SOAP descriptor to predict the functional properties of molecular structures by means of machine learning
algorithms. 
  
  At its core, the SOAP descriptor can be thought as a representation of the local atomic environments within a certain
molecular structure. Said representation is obtained by using a local expansion of a Gaussian smeared atomic density
with orthonormal functions based on spherical harmonics and radial basis functions. 
  
  In order to obtain a SOAP descriptor, one has to pick one or more atomic species as centre(s) of the local atomic
environments and one or more species as the neighbors of the central atom that define said environment. It is not
uncommon to use multiple SOAP descriptors, characterised by different choices of centres and neighbors, as they do
contain information that might be hidden when simply choosing every atomic species in the molecule as both centre and
neighbour. 
  
  A number of parameters are needed to define a SOAP descriptor, most prominently:
* **n_max**: The number of radial basis functions
* **l_max**: The maximum degree of the spherical harmonics
* **cutoff**: The spatial extent (in Å) of the local atomic environment
* **atom_sigma**: The standard deviation (in Å) of the Gaussian functions
  
  The choice of these parameters is not straightforward, and it is key to the accuracy and the predictive power of the
SOAP descriptor. Physical intuition can provide a starting point, particularly in terms of the choice of `cutoff` (based
on e.g. the extent of the moelcular structures in question), but finding (one of the) best combination(s) of these
parameters often requires additional effort. A simple grid search in this 4-parameter space is a possibility,
particularly if conducted in a randomised fashion, as the typical search space is simply too vast to be systematically
explored. A typical search space would be defined by:
* 2 < **n_max** 10
* 2 < **l_max** 10
* 5 **cutoff** 20 [Å]
* 0.1 < **atom_sigma** < 1.5 [Å] The size of the grid obviously depends on the granularity of the variation with respect
  to each parameter, but typical sizes would ential be around 15,000 combinations.
  
  The computational effort required to assess the performance of a given SOAP descriptor varies according to the size of
the dataset, the extent of the moelcular structures in question, and the number of centers and neighbors. In addition,
different choices of the above mentioned four parameters will massively impact the resulting dimensionality of the SOAP
vector, and thus the computational effort. Even when dealing with very small datasets (e.g. 100 molecules), a randomised
grid search is needed to try and identify a sufficiently accurate combination of these SOAP parameters.
  
  In addition, optimising the SOAP parameters for different SOAPS via a randomised grid search *at the same time* would
involve an intractacble number of potential combinations.


  ## References
  <a id="1">[1]</a> 
De, S., Bartók, A. P., Csányi, G. & Ceriotti, M. Comparing molecules and solids across structural and alchemical space. Phys. Chem. Chem. Phys. 18, 13754–13769 (2016).

  <a id="2">[2]</a>
Darby, J. P., Kermode, J. R. & Csányi, G. Compressing local atomic neighbourhood descriptors. (2021) doi:10.48550/arXiv.2112.13055.

