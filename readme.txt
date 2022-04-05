This README file was generated on 2022-04-05 by Knut Ola Dølven.

-------------------
GENERAL INFORMATION
-------------------
// Repository for code and data accompanying Dølven et al., (2021) (see bottom for publication info)

// Contact Information
     // Name: Knut Ola Dølven	
     // Institution: UiT, The Arctic University of Tromsø	
     // Email: knut.o.dolven@uit.no
     // ORCID: 0000-0002-5315-4834

// Contributors (code): Knut Ola Dølven, Juha Vierinen (https://github.com/jvierine)
// Contributors (data): Roberto Grilli, Jack Triest
// Controbutors (method developement): Knut Ola Dølven, Juha Vierinen, Roberto Grilli, Jack Tries, Bénédicte Férre

// For date of data collection, geographic location, funding sources, and description of data: See Dølven et al. (2021) 

--------------------------
METHODOLOGICAL INFORMATION
--------------------------

Reconstruction of a fast response signal from slow response sensor data is achieved using statistical inverse theory.
We apply a weighted linear least squares estimator and the growth-law as measurement model. Regularization of the solution is 
done using model sparsity, assuming changes occurs with a particular time-step, or tikhonov regularization (optional in code).
The amount of regularization is optimized using L-curve analysis, but can also be selected manually based on domain-specific 
knowledge. See commentary in deconv.py for specifics and DØlven et al. (2021) for a full detailed description.

--------------------
DATA & FILE OVERVIEW
--------------------

field_data.txt - data used in the field experiment in Dølven et al. (2022)

Lab_experiment_data - data used in the laboratory experiment in Dølven et al. (2022)

RTdetermination.csv - data used to determine the response time of the EB sensor
for the field experiment in Dølven et al. (2022)

deconv.py - Pytyon file containing all functions used in the deconv_master function
which does deconvolution as presented in the manuscript Dølven et al., 2021. Also include 
functions used to produce the examples in the manuscript. See file for description of 
content. 


---------------------------
PUBLICATION AND HOW TO CITE
---------------------------

Dølven, K. O., Vierinen, J., Grilli, R., Triest, J., and Ferré, B.: Response time correction of slow 
response sensor data by deconvolution of the growth-law equation, Geosci. Instrum. Method. Data Syst. 
Discuss. [preprint], https://doi.org/10.5194/gi-2021-28, in review, 2021.
