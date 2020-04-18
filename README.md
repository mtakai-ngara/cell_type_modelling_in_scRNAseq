Simulating different parameters for cell type discovery in scRNAseq
===================

Project overview 
-----------------------------------------
Single-cell approaches are currently being broadly used to enumerate and characterize cell types and transcriptional states in man and model organisms. In particular single-cell RNA-sequencing has reached sufficient accuracy and cellular throughput to enable the generation of systematic atlases of cell types and states present across tissues, organs and whole organisms. Captured and sequenced cells are being assigned to discrete or hierarchical clusters and later compared to known cell types and cellular phenotypes. 

Here, we used a simulation strategy to define the extent of biological differences in cellular transcriptomes required for this paradigm of unbiased exploration of cells to correctly group cells, which shed light on biological differences that currently can’t be resolved using single-cell RNA-sequencing. We demonstrate that methods with highest sensitivity had the greatest ability to identify subtypes for genes expressed at lower or intermediate levels, whereas all methods could identify subtypes when perturbing the most highly expressed genes. Additionally, we explored how normalization methods affected sub-type discovery. Surprisingly, rather large biological fold-change differences among large numbers of genes are often unresolved, pointing towards large numbers of false negative subtypes in current methodology.

Repository contents
----
src : All source code

doc : All documentations

lib : All dependencies and libraries

tools : Common utility codes

build : Different releases and test versions

notebook : Notebook demos for various analysis
