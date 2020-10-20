# Online Determinant Maximization

The repository contains the code to accompany the paper "Online MAP Inference of Determinantal Point Processes",  by Aditya Bhaskara, Amin Karbasi, Silvio Lattanzi and Morteza Zadimoghaddam. The paper will appear in NeurIPS 2020 and the link to the final PDF will be added here when available.

The code implements an online variant of the swap algorithm (each time a column arrives, see if it can replace one of the chosen columns). A variant of it which includes a "stash" is also implemented. Here, the "replaced" columns are placed into a stash and are used as candidates for swaps in the future. Please see the paper for details.

----
