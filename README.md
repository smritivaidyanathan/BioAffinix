# BioAffinix
A "bio-cheminformatics" project in progress to predict small molecule-protein interaction strength through a neural net trained on existing interactions from ChEMBL, amino acid sequences, and SMILES strings for a small set of proteins.

**11β-Hydroxysteroid Dehydrogenase Type 1 (11β-HSD1)** is an enzyme crucial for regulating cortisol, a hormone involved in metabolism, immune response, and stress regulation. Overactivity of 11β-HSD1 is linked to metabolic syndrome, obesity, type 2 diabetes, and inflammation, making it a key target for therapeutic interventions.

This personal project aims to predict the inhibitory potential of small molecules against 11β-HSD1, which could be valuable for drug development. The model will take the molecular structure of small molecules as input and output a measure of their inhibitory power against 11β-HSD1.

**Dataset**: 4,500+ embeddings generated from SMILES strings and corresponding IC50 values.

**Current Approach**: Utilizing embeddings generated from the MolecularTransformerEmbeddings pre-trained transformer (https://github.com/mpcrlab/MolecularTransformerEmbeddings.git) and training a CNN with Adam Optimizer to predict IC50 values. (Reference: "Predicting Binding from Screening Assays with Transformer Network Embeddings")

**References**
Morris P, St Clair R, Barenholtz E, Hahn WE. Predicting Binding from Screening Assays with Transformer Network Embeddings. ChemRxiv. 2020; doi:10.26434/chemrxiv.11625885.v1 This content is a preprint and has not been peer-reviewed.


Chapman, K., Holmes, M., & Seckl, J. (2013). 11β-Hydroxysteroid dehydrogenases: intracellular gate-keepers of tissue glucocorticoid action. Physiological Reviews, 93(3), 1139-1206. DOI: 10.1152/physrev.00020.2012


Hardy, R., & Cooper, M. S. (2009). Glucocorticoid-induced osteoporosis: molecular mechanisms and therapeutic strategies. Advances in Experimental Medicine and Biology, 872, 45-58. DOI: 10.1007/978-1-4614-0432-6_3


Morgan, S. A., McCabe, E. L., Gathercole, L. L., Hassan-Smith, Z. K., Larner, D. P., Bujalska, I. J., ... & Lavery, G. G. (2014). 11β-HSD1 is the major regulator of the tissue-specific effects of circulating glucocorticoid excess. Proceedings of the National Academy of Sciences, 111(24), E2482-E2491. DOI: 10.1073/pnas.1323681111




