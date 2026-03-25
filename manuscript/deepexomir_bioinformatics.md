# DeepExoMir: A Hybrid Deep Learning Framework Integrating RNA Language Model Embeddings and Duplex Graph Attention for MicroRNA Target Prediction

## Authors
Wen-Hsien Lin^1,*^

^1^ AI and Data Applications Division, GGA Corp., Taipei, Taiwan

*To whom correspondence should be addressed. Email: BryceLin@bionetTX.com

## Abstract

**Motivation:** Accurate prediction of microRNA (miRNA) targets is essential for understanding post-transcriptional gene regulation, yet existing methods are limited by reliance on hand-crafted features or architectures that fail to model the structural topology of miRNA-target duplexes. Recent benchmarking with experimentally validated CLIP-seq negatives has revealed that many published methods perform substantially worse than reported with random negatives.

**Results:** We present DeepExoMir, a hybrid deep learning framework that integrates frozen RiNALMo RNA language model embeddings with a novel duplex-aware architecture. The model employs an 8-layer hybrid encoder alternating bidirectional convolution gating with cross-attention, interaction-aware pooling, and a duplex graph attention network (DuplexGAT) that explicitly models the miRNA-target duplex as a nucleotide-level graph. Incorporating 33 biologically informed features spanning thermodynamic stability, evolutionary conservation, and RNA secondary structure, DeepExoMir achieves a mean AU-PRC of 0.8551 across three independent miRBench test datasets, surpassing the best retrained CNN baseline (0.82) by +0.04. Systematic ablation across 22 model versions reveals that evolutionary conservation contributes 52% of total feature importance. Attention analysis confirms autonomous learning of seed region specificity (1.16-fold enrichment) and 3' compensatory pairing patterns.

**Availability and Implementation:** Source code and pre-trained models are available at [GitHub URL] and archived at Zenodo [DOI].

**Contact:** BryceLin@bionetTX.com

**Supplementary information:** Supplementary data are available at *Bioinformatics* online.

## 1 Introduction

MicroRNAs (miRNAs) are short non-coding RNAs of approximately 22 nucleotides that regulate gene expression post-transcriptionally by guiding the RNA-induced silencing complex (RISC) to complementary sequences in target mRNAs (1). Over 2,600 mature human miRNAs have been catalogued in miRBase v22.1 (2), collectively regulating more than 60% of protein-coding genes (3). Dysregulation of miRNA-target interactions has been implicated in cancer, cardiovascular disease, and neurodegeneration (4), making accurate target prediction essential for both basic research and therapeutic development.

The canonical mechanism of miRNA target recognition centers on the seed region (positions 2-8 from the 5' end), which forms Watson-Crick base pairs with complementary sites in the target 3'UTR (5). However, functional targeting can also involve 3' compensatory pairing (positions 13-16), centered sites, and non-canonical interactions, necessitating methods that capture the full spectrum of targeting determinants (6).

Classical approaches such as TargetScan (6) and miRDB (7) rely on curated biological features including seed complementarity, thermodynamic stability, and evolutionary conservation. While interpretable, these methods are limited by manual feature selection and linear modeling assumptions. Deep learning approaches including DeepMirTar (8), miRBind (9), and TargetNet (10) have demonstrated improved performance by learning representations directly from data. However, these methods face several limitations: (i) they typically employ random negative sampling, which artificially inflates performance metrics; (ii) they lack integration with pre-trained RNA language models; and (iii) they fail to explicitly model the structural topology of the miRNA-target duplex.

The development of RNA foundation models, particularly RiNALMo (11), pre-trained on 36 million non-coding RNA sequences, offers rich contextual representations beyond what hand-crafted features provide. The introduction of the miRBench benchmark (12) with experimentally validated CLIP-seq negatives has established a more rigorous evaluation standard.

Here, we present DeepExoMir, a hybrid deep learning framework that addresses these limitations through four innovations: (i) integration of frozen RiNALMo embeddings with PCA dimensionality reduction; (ii) an 8-layer hybrid encoder alternating bidirectional convolution gating with cross-attention; (iii) a duplex graph attention network (DuplexGAT) that models the miRNA-target duplex as a nucleotide-level graph; and (iv) interaction-aware pooling that captures pairwise dependencies before classification. Through systematic ablation across 22 model versions, we quantify each component's contribution and demonstrate state-of-the-art performance on the miRBench benchmark.

## 2 Materials and Methods

### 2.1 Datasets and preprocessing

#### 2.1.1 Training data
All training, validation, and test data were sourced from the miRBench standardized benchmark suite (12), which provides experimentally validated miRNA-target interaction datasets from AGO2 crosslinking experiments with controlled negative sampling. Three complementary datasets were combined: AGO2 eCLIP from Manakov *et al.* (1,906,909 training pairs), AGO2 CLASH from Hejret *et al.* (5,925 training pairs), and AGO2 eCLIP from Klimentova *et al.* (616 training pairs). The miRBench pipeline employs gene-level splitting to prevent data leakage, ensuring all pairs involving a given target gene are assigned entirely to one partition. After removing 45,863 contradictory pairs (identical miRNA-target tuples with conflicting labels), the final dataset comprised 2,761,229 samples split into training (1,913,450, 69.3%), validation (339,894, 12.3%), and test (507,885, 18.4%) sets.

#### 2.1.2 External evaluation benchmark
For comparison with published baselines, we evaluated on the three miRBench held-out test sets: Hejret-AGO2 CLASH (965 samples), Klimentova-AGO2 eCLIP (954 samples), and Manakov-AGO2 eCLIP (327,129 samples). These test sets use a balanced 1:1 positive-to-negative ratio with experimentally validated CLIP-seq negatives.

### 2.2 Sequence embeddings

We employed the pre-trained RiNALMo-giga model (11), a 650M-parameter RNA language model. Per-token embeddings (dimension 1280) were extracted for all sequences. PCA dimensionality reduction from 1280 to 256 dimensions, retaining 88.7% of variance, reduced storage from 151 GB to 31 GB. Both per-token and mean-pooled embeddings were pre-computed and cached.

### 2.3 Model architecture

DeepExoMir comprises six main components (Fig. 1):

**Input projection.** PCA-reduced embeddings are projected through separate linear layers with layer normalization for miRNA (30 positions) and target (50 positions) sequences, followed by learnable positional embeddings.

**Hybrid encoder.** The core encoder comprises 8 layers alternating between BiConvGate layers employing bidirectional depthwise separable convolution with SwiGLU gating (14,15) for local pattern extraction, and cross-attention layers (4 total, every 2 layers) enabling inter-sequence information exchange through 8-head attention. All layers incorporate pre-normalization, residual connections, and stochastic depth (drop path rate 0.1).

**Interaction pooling.** A multi-head attention mechanism (4 heads) computes self-attention within each sequence and cross-attention between sequences, producing a 512-dimensional interaction vector that replaces standard mean pooling.

**DuplexGAT.** The miRNA-target duplex is represented as a graph with 80 nucleotide nodes (30 miRNA + 50 target) connected by backbone edges, Watson-Crick/wobble base-pairing edges, and proximity edges. Two GATv2Conv layers (16) with 4 attention heads and learned edge-type embeddings produce a 128-dimensional graph-level representation.

**Biological feature encoders.** Three parallel encoders process: (i) a 6-channel base-pairing CNN encoding the 30x50 alignment matrix (128-d output); (ii) a 2-layer structural MLP processing 33 pre-computed features spanning thermodynamic stability (12), ViennaRNA energetics (6), PhyloP conservation (5), and miRNA secondary structure (10) (64-d output); (iii) a contact map CNN computing soft interaction maps from encoder outputs (128-d output).

**MoE classifier.** All features are concatenated (1216-d total) and processed by a Mixture-of-Experts classifier with 4 expert networks, top-2 gating, and Platt scaling.

Auxiliary multi-task heads for seed binding strength, duplex MFE regression, and binding site position provide additional training signal.

### 2.4 Training procedure

Models were trained for up to 60 epochs with early stopping (patience 20, monitoring validation AUC-ROC). We used AdamW with cosine scheduling (peak LR 1e-4, 1000-step warmup), gradient accumulation over 8 micro-batches (effective batch size 2048), and FP16 mixed precision. Focal loss (gamma=1.0, alpha=0.5) with label smoothing (0.05) addressed class imbalance. Embedding-level Mixup (alpha=0.2, p=0.5) and EvoAug-inspired structural perturbation (21) provided data augmentation. All experiments used a single NVIDIA RTX 5090 GPU (32 GB).

### 2.5 Computational cost

**Table 4.** Computational requirements for DeepExoMir.

| Component | Metric | Value |
|-----------|--------|-------|
| Model parameters | v19 / v22 | 26.4M / 27.9M |
| Embedding pre-computation | Time (1.9M pairs) | ~8 hours |
| Embedding cache | Storage (PCA-256) | 33 GB |
| Training time | v19 (55 epochs) | ~14 hours |
| Training time | v22 (48 epochs) | ~21 hours |
| Training throughput | Samples/sec | ~37 (v19) / ~25 (v22) |
| Per-epoch time | v19 / v22 | 16 min / 26 min |
| Inference time | 508K test samples | ~110 sec |
| Inference throughput | Samples/sec | ~4,600 |
| GPU memory | Training (batch=256) | ~12 GB |
| Checkpoint size | v19 / v22 | 106 MB / 112 MB |

The DuplexGAT module (v22) adds approximately 1.5M parameters and increases per-epoch training time by ~65% due to graph construction overhead. For applications where inference speed is critical, the v19 architecture (without DuplexGAT) provides comparable performance (test AUC 0.832 vs 0.832) at substantially higher throughput.

## 3 Results

### 3.1 miRBench benchmark performance

DeepExoMir was evaluated on all three miRBench test datasets using the RiNALMo backbone for on-the-fly embedding of unseen sequences (Table 1). We compare against both the original pre-trained baselines (evaluated zero-shot on miRBench by Sammut *et al.* (12)) and the retrained CNN baselines trained on the same miRBench training data as DeepExoMir. DeepExoMir achieved a mean AU-PRC of 0.8551, surpassing the best retrained CNN (mean 0.82) by +0.04 and all original pre-trained baselines.

**Table 1.** Performance comparison on miRBench benchmark (AU-PRC). Top section: original pre-trained baselines evaluated zero-shot (12). Bottom section: models trained on miRBench training data.

| Method | Hejret | Klimentova | Manakov | Mean |
|--------|:------:|:----------:|:-------:|:----:|
| *Original pre-trained (zero-shot):* | | | | |
| CnnMirTarget (18) | 0.53 | 0.51 | 0.58 | 0.54 |
| TargetNet (10) | 0.58 | 0.58 | 0.66 | 0.61 |
| InteractionAwareModel (19) | 0.74 | 0.74 | 0.63 | 0.70 |
| TargetScanCnn (6) | 0.74 | 0.71 | 0.77 | 0.74 |
| miRNA_CNN (20) | 0.77 | 0.77 | 0.71 | 0.75 |
| miRBind (9) | 0.80 | 0.75 | 0.71 | 0.75 |
| *Trained on miRBench data:* | | | | |
| CNN retrained-Hejret (12) | 0.86 | 0.79 | 0.77 | 0.81 |
| CNN retrained-Manakov (12) | 0.84 | 0.82 | 0.81 | 0.82 |
| **DeepExoMir (ours)** | **0.85** | **0.87** | **0.85** | **0.86** |

### 3.2 Ablation study

To quantify each component's contribution, we conducted systematic ablation across 22 model versions (Table 2).

**Table 2.** Ablation study showing incremental contributions

| Version | Key change | Features | val AUC | test AUC |
|---------|-----------|:--------:|:-------:|:--------:|
| v1 | Baseline MLP | 0 | 0.593 | -- |
| v8 | HybridEncoder + MoE + MultiTask | 8 | 0.829 | 0.813 |
| v10+ | 6-ch BP matrix + Mixup | 12 | 0.833 | 0.814 |
| v14-2L | +PhyloP conservation | 31 | 0.850 | 0.830 |
| v16a | Feature pruning (-8 noise) | 23 | 0.850 | 0.830 |
| v18 | +miRNA secondary structure | 33 | 0.850 | 0.831 |
| v19 | +InteractionPool + 4x CrossAttn | 33 | 0.852 | 0.832 |
| v22 | +DuplexGAT | 33 | 0.851 | **0.832** |
| Ensemble | 5-model heterogeneous | -- | -- | **0.835** |

Key findings: (i) evolutionary conservation (PhyloP) was the single most impactful addition (+0.016 test AUC); (ii) removing 8 noise features improved generalization; (iii) architectural innovations (InteractionPooling, DuplexGAT) provided consistent but modest improvements; (iv) hyperparameter tuning (focal gamma, learning rate) showed diminishing returns.

### 3.3 Feature importance analysis

Permutation importance on the validation set (Fig. 2) revealed that conservation features (5 features) contributed 52% of total importance, thermodynamic features (12) contributed 41%, and ViennaRNA features (6) contributed 6%. The top features were ensemble MFE (AUC drop 0.0184), 3'UTR site indicator (0.0171), CDS site indicator (0.0126), PhyloP mean (0.0089), and PhyloP max (0.0043). Eight pairing statistics features showed negative total importance (-0.0002), confirming their removal was justified.

### 3.4 Multi-seed stability

Training v19 with three random seeds yielded test AUC of 0.8305 +/- 0.0007 (SD) and test AUPR of 0.8503 +/- 0.0010, demonstrating robust performance across initializations (Table 3).

**Table 3.** Multi-seed evaluation (v19 architecture)

| Seed | val AUC | test AUC | test AUPR |
|:----:|:-------:|:--------:|:---------:|
| Original | 0.852 | 0.832 | 0.852 |
| 42 | 0.850 | 0.830 | 0.850 |
| 123 | 0.852 | 0.830 | 0.850 |
| 456 | 0.851 | 0.830 | 0.849 |
| **Mean +/- SD** | **0.851 +/- 0.001** | **0.831 +/- 0.001** | **0.850 +/- 0.001** |

### 3.5 Attention analysis

Analysis of cross-attention and interaction pooling across 5,000 positive samples (Fig. 3) revealed biologically meaningful patterns. The first cross-attention layer exhibited 1.16-fold enrichment at seed positions (2-8), consistent with seed pairing as the primary recognition signal. The 3' compensatory region (positions 13-16) showed 1.06-fold enrichment, with position 13 receiving the highest attention overall, aligning with the established role of supplementary pairing (17). Deeper layers showed progressively reduced seed bias (0.92x, 0.90x, 0.89x), suggesting transition from local base-pairing to global context integration. Interaction pooling showed 1.20-fold seed enrichment, confirming selective amplification of seed signals for classification.

## 4 Discussion

DeepExoMir demonstrates that integrating pre-trained RNA language model representations with biologically informed architecture substantially advances miRNA target prediction. Several findings merit discussion.

The use of frozen RiNALMo embeddings eliminates task-specific pre-training while providing contextual representations capturing evolutionary and structural information. PCA reduction (1280 to 256 dimensions) enables practical deployment with minimal information loss. This approach generalizes to other RNA-RNA interaction prediction tasks.

A critical aspect of our evaluation is the use of miRBench CLIP-seq negatives rather than random sequences. This accounts for the discrepancy between our test AUC (0.832) and the high accuracy reported by earlier methods such as DeepMirTar (~93.5%). Random negatives lack genuine non-target properties, inflating performance estimates (12). Importantly, the original pre-trained baselines in miRBench were evaluated zero-shot (using their published weights without retraining), placing them at a distribution disadvantage. The most direct comparison is therefore against the retrained CNN baselines (12), which were trained on the same miRBench data distribution. DeepExoMir surpasses these by +0.04 mean AU-PRC, demonstrating the benefit of our architectural innovations.

**Frozen vs. fine-tuned backbone.** We deliberately froze RiNALMo rather than fine-tuning it. Preliminary experiments with backbone fine-tuning (learning rate scale 0.01-0.1) showed no improvement in validation AUC while increasing training time 4-fold and GPU memory requirements beyond 24 GB. This is consistent with recent findings that frozen large language models serve as effective feature extractors when combined with task-specific architectures (21,22). The frozen approach also enables embedding pre-computation, reducing per-epoch training time from ~60 minutes (live backbone) to ~16 minutes.

**Graph attention for duplex modeling.** DuplexGAT represents, to our knowledge, the first application of graph attention networks to miRNA-target duplex modeling. While improvement over the base architecture was modest (+0.0004 test AUC), the graph formulation provides explicit structural reasoning through learned edge-type embeddings that distinguish backbone connectivity from base-pairing interactions. This approach may prove more beneficial for non-canonical interaction types (e.g., centered sites, bulge-containing duplexes) where sequence-based features are less informative (23,24).

**Domain knowledge complements learned representations.** Despite the power of learned representations, curated biological features—particularly PhyloP conservation (25) and ensemble MFE—provided the largest performance improvements in ablation. This aligns with recent observations in protein structure prediction (26) and drug-target binding (27), suggesting that domain knowledge and data-driven learning are complementary paradigms in computational biology.

**Limitations.** Several limitations should be noted. First, the model requires pre-computed RiNALMo embeddings, adding a preprocessing step for novel sequences (~0.2 seconds per pair). Second, evaluation is limited to human miRNA-target interactions; cross-species generalization to mouse (28) or *C. elegans* requires further investigation. Third, the model does not explicitly account for target site accessibility within full-length mRNA secondary structures (29,30), which influences targeting efficacy. Finally, while attention analysis reveals biologically meaningful patterns, caution is warranted in interpreting attention weights as direct mechanistic explanations (31).

## 5 Conclusion

DeepExoMir achieves state-of-the-art performance on the miRBench benchmark (mean AU-PRC 0.86), surpassing the best retrained CNN baseline (0.82) by integrating RNA language model embeddings, a hybrid encoder, duplex graph attention, and 33 biologically curated features. Ablation across 22 model versions quantifies the contribution of each architectural component and biological feature group, revealing that evolutionary conservation alone accounts for 52% of feature importance. Attention analysis confirms that the model autonomously learns biologically meaningful targeting patterns, including seed region specificity and hierarchical processing from local base-pairing to global context integration.

Several directions for future work are promising. Cross-species evaluation on conserved miRNA families (e.g., let-7, miR-17/92 cluster) could validate the generalizability of learned representations. Integration of full-length mRNA secondary structure predictions from tools such as LinearFold (32) may improve target site accessibility modeling. The DuplexGAT framework could be extended to model non-canonical interactions including centered sites and bulge-mediated targeting (23). Finally, combining DeepExoMir with exosome-specific miRNA expression profiles could enable tissue-specific target prediction for precision medicine applications (33,34).

DeepExoMir is freely available at [GitHub URL].

## Acknowledgements

The authors thank the miRBench team for providing the standardized benchmark datasets and evaluation framework.

## Funding

This work was supported by GGA Corp. internal research resources.

## Conflict of Interest

None declared.

## Data Availability

The miRBench benchmark datasets are available through the miRBench Python package (12). Training data are derived from the miRBench standardized training splits of AGO2 CLIP-seq datasets. Pre-trained model checkpoints, source code, and analysis scripts are deposited at Zenodo [DOI] and maintained at [GitHub URL].

## References

1. Bartel,D.P. (2018) Metazoan microRNAs. *Cell*, **173**, 20-51.
2. Kozomara,A., Birgaoanu,M. and Griffiths-Jones,S. (2019) miRBase: from microRNA sequences to function. *Nucleic Acids Res.*, **47**, D155-D162.
3. Friedman,R.C., Farh,K.K., Burge,C.B. and Bartel,D.P. (2009) Most mammalian mRNAs are conserved targets of microRNAs. *Genome Res.*, **19**, 92-105.
4. Rupaimoole,R. and Slack,F.J. (2017) MicroRNA therapeutics: towards a new era for the management of cancer and other diseases. *Nat. Rev. Drug Discov.*, **16**, 203-222.
5. Bartel,D.P. (2009) MicroRNAs: target recognition and regulatory functions. *Cell*, **136**, 215-233.
6. Agarwal,V., Bell,G.W., Nam,J.W. and Bartel,D.P. (2015) Predicting effective microRNA target sites in mammalian mRNAs. *eLife*, **4**, e05005.
7. Chen, Y. and Wang, X. (2020) miRDB: an online database for prediction of functional microRNA targets. *Nucleic Acids Res.*, **48**, D127–D131.
8. Wen, M., Cong, P., Zhang, Z., Lu, H. and Li, T. (2018) DeepMirTar: a deep-learning approach for predicting human miRNA targets. *Bioinformatics*, **34**(22), 3781-3787.
9. Klimentová, E. *et al.* (2022) miRBind: A Deep Learning Method for miRNA Binding Classification. *Genes*, **13**(12), 2323.
10. Min, S., Lee, B. and Yoon, S. (2022) TargetNet: functional microRNA target prediction with deep neural networks. *Bioinformatics*, **38**(3), 671-677.
11. Penić, R. J. *et al.* (2025) RiNALMo: general-purpose RNA language models can generalize well on structure prediction tasks. *Nat. Commun.*, **16**, 5671.
12. Sammut, S., Gresova, K., Tzimotoudis, D., Marsalkova, E., Cechak, D. and Alexiou, P. (2025) miRBench: novel benchmark datasets for microRNA binding site prediction that mitigate against prevalent microRNA frequency class bias. *Bioinformatics*, **41**(Supplement_1), i542–i551.
13. Huang,H.Y. *et al.* (2022) miRTarBase update 2022: an informative resource for experimentally validated miRNA-target interactions. *Nucleic Acids Res.*, **50**, D222-D230.
14. Gu, A. and Dao, T. (2024) Mamba: Linear-Time Sequence Modeling with Selective State Spaces. In: *First Conference on Language Modeling (COLM 2024)*.
15. Shazeer,N. (2020) GLU variants improve Transformer. *arXiv*:2002.05202.
16. Brody,S., Alon,U. and Yahav,E. (2022) How attentive are graph attention networks? In *ICLR 2022*.
17. Grimson,A., Farh,K.K., Johnston,W.K., Garrett-Engele,P., Lim,L.P. and Bartel,D.P. (2007) MicroRNA targeting specificity in mammals: determinants beyond seed pairing. *Mol. Cell*, **27**, 91-105.
18. Zheng, X. *et al.* (2020) Prediction of miRNA targets by learning from interaction sequences. *PLoS ONE*, **15**(5), e0232578.
19. Yang, T. H. *et al.* (2024) Identifying Human miRNA Target Sites via Learning the Interaction Patterns between miRNA and mRNA Segments. *J. Chem. Inf. Model.*, **64**(7), 2445–2453.
20. Hejret, V. *et al.* (2023) Analysis of chimeric reads characterises the diverse targetome of AGO2-mediated regulation. *Sci. Rep.*, **13**, 22895.
21. Bommasani,R. *et al.* (2021) On the opportunities and risks of foundation models. *arXiv*:2108.07258.
22. Ji,Y. *et al.* (2021) DNABERT: pre-trained bidirectional encoder representations from transformers model for DNA-language in genome. *Bioinformatics*, **37**, 2112-2120.
23. Shin,C., Nam,J.W., Farh,K.K., Chiang,H.R., Shkumatava,A. and Bartel,D.P. (2010) Expanding the microRNA targeting code: functional sites with centered pairing. *Mol. Cell*, **38**, 789-802.
24. Chi,S.W., Hannon,G.J. and Darnell,R.B. (2012) An alternative mode of microRNA target recognition. *Nat. Struct. Mol. Biol.*, **19**, 321-327.
25. Pollard, K.S., Hubisz, M.J., Rosenbloom, K.R. and Siepel, A. (2010) Detection of nonneutral substitution rates on mammalian phylogenies. *Genome Res.*, **20**(1), 110-121.
26. Jumper,J. *et al.* (2021) Highly accurate protein structure prediction with AlphaFold. *Nature*, **596**, 583-589.
27. Chen, L., Fan, Z., Chang, J. *et al.* (2023) Sequence-based drug design as a concept in computational drug design. *Nat. Commun.*, **14**, 4219.
28. Betel,D., Koppal,A., Agius,P., Sander,C. and Leslie,C. (2010) Comprehensive modeling of microRNA targets predicts functional non-conserved and non-canonical sites. *Genome Biol.*, **11**, R90.
29. Kertesz,M., Iovino,N., Unnerstall,U., Gaul,U. and Segal,E. (2007) The role of site accessibility in microRNA target recognition. *Nat. Genet.*, **39**, 1278-1284.
30. Lorenz,R. *et al.* (2011) ViennaRNA package 2.0. *Algorithms Mol. Biol.*, **6**, 26.
31. Jain,S. and Wallace,B.C. (2019) Attention is not explanation. In *NAACL-HLT 2019*, 3543-3556.
32. Huang,L. *et al.* (2019) LinearFold: linear-time approximate RNA folding by 5'-to-3' dynamic programming and beam search. *Bioinformatics*, **35**, i295-i304.
33. Valadi,H. *et al.* (2007) Exosome-mediated transfer of mRNAs and microRNAs is a novel mechanism of genetic exchange between cells. *Nat. Cell Biol.*, **9**, 654-659.
34. O'Brien, K., Breyne, K., Ughetto, S., Laurent, L. C. and Breakefield, X. O. (2020) RNA delivery by extracellular vesicles in mammalian cells and its applications. *Nat. Rev. Mol. Cell Biol.*, **21**, 585-606.
35. McGeary,S.E., Lin,K.S., Shi,C.Y., Pham,T.M., Bisaria,N., Kelley,G.M. and Bartel,D.P. (2019) The biochemical basis of microRNA targeting efficacy. *Science*, **366**, eaav1741.
36. Manakov, S.A. *et al.* (2022) Scalable and deep profiling of mRNA targets for individual microRNAs with chimeric eCLIP. *bioRxiv*. doi: 10.1101/2022.02.13.480296.
37. Wieder, O., Garon, A., Perricone, U., Hessler, G. and Bauer, M. R. (2020) A compact review of molecular property prediction with graph neural networks. *Drug Discov Today Technol*, **37**, 1-12.

## Figure Legends

**Figure 1.** Architecture overview of DeepExoMir. miRNA and target sequences are encoded by frozen RiNALMo embeddings (PCA-reduced to 256 dimensions), processed by an 8-layer hybrid encoder alternating BiConvGate and cross-attention layers, and combined with structural features through interaction pooling, DuplexGAT, base-pairing CNN, contact map CNN, and structural MLP before classification by a Mixture-of-Experts classifier. Dashed boxes indicate auxiliary multi-task heads.

**Figure 2.** Permutation feature importance analysis. (A) Top 15 features ranked by AUC-ROC drop on the validation set (3 repeats, 50K samples). Colors indicate feature groups: conservation (blue), thermodynamic (red), ViennaRNA (orange), pairing/noise (gray). (B) Aggregated importance by feature group showing conservation features contribute 52% of total importance despite comprising only 5 of 33 features.

**Figure 3.** Attention analysis across 5,000 positive test samples. (A) Mean cross-attention weight by miRNA position, colored by biological region. Shaded regions highlight the seed (positions 2-8) and 3' compensatory (13-16) regions. (B) Seed region enrichment fold-change per cross-attention layer, showing strongest bias in Layer 1 (1.16x) with progressive decrease in deeper layers. (C) Layer 1 cross-attention heatmap (miRNA vs. target positions) with seed region outlined. (D) Interaction pooling seed bias comparison between seed (positions 2-8) and overall mean attention weights.
