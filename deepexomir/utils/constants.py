"""Domain-specific constants for DeepExoMir.

Aesthetic medicine pathway mappings connecting miRNA targets to
XunLian Group's exosome product applications.
"""

# RNA nucleotide mappings
RNA_BASES = ["A", "U", "G", "C"]
DNA_TO_RNA = str.maketrans("Tt", "Uu")
COMPLEMENT_RNA = str.maketrans("AUGCaugc", "UACGuacg")

# Seed region positions (1-indexed as in biology, 0-indexed in code)
SEED_START = 1  # position 2 in biology = index 1
SEED_END = 7    # position 8 in biology = index 7

# Base-pairing scores for the base-pairing matrix
BP_SCORES = {
    ("A", "U"): 1.0, ("U", "A"): 1.0,
    ("G", "C"): 1.0, ("C", "G"): 1.0,
    ("G", "U"): 0.5, ("U", "G"): 0.5,  # wobble pair
}
BP_MISMATCH = -1.0
BP_GAP = -0.5

# Seed match types
SEED_TYPES = ["8mer", "7mer-m8", "7mer-A1", "6mer", "6mer-GU", "non-canonical", "none"]

# ============================================================================
# Aesthetic Medicine Pathway Mappings
# ============================================================================

AESTHETIC_PATHWAYS = {
    "whitening": {
        "display_name": "Skin Whitening / Melanogenesis Inhibition",
        "display_name_zh": "美白 / 黑色素生成抑制",
        "kegg_pathways": [
            "hsa04916",  # Melanogenesis
        ],
        "key_genes": [
            "MITF", "TYR", "TYRP1", "DCT", "MC1R", "CREB1",
            "KIT", "KITLG", "POMC", "SLC45A2", "OCA2",
        ],
        "key_mirnas": [
            "hsa-miR-330-5p", "hsa-miR-181a-5p", "hsa-miR-137",
            "hsa-miR-141-3p", "hsa-miR-200a-3p", "hsa-miR-145-5p",
            "hsa-miR-125b-5p", "hsa-miR-340-5p", "hsa-miR-25-3p",
            "hsa-miR-148a-3p", "hsa-miR-155-5p", "hsa-miR-182-5p",
            "hsa-miR-218-5p", "hsa-miR-508-3p", "hsa-miR-211-5p",
        ],
        "mechanism": "Downregulation of melanogenesis via MITF/TYR axis",
    },
    "scar_removal": {
        "display_name": "Scar Removal / Anti-Fibrosis",
        "display_name_zh": "淡疤 / 抗纖維化",
        "kegg_pathways": [
            "hsa04350",  # TGF-beta signaling pathway
            "hsa04310",  # Wnt signaling pathway
        ],
        "key_genes": [
            "TGFB1", "TGFB2", "TGFBR1", "TGFBR2",
            "SMAD2", "SMAD3", "SMAD7",
            "COL1A1", "COL1A2", "COL3A1",
            "ACTA2", "FN1", "CTGF", "SERPINH1",
        ],
        "key_mirnas": [
            "hsa-miR-29a-3p", "hsa-miR-29b-3p", "hsa-miR-29c-3p",
            "hsa-miR-21-5p", "hsa-miR-192-5p", "hsa-miR-200b-3p",
            "hsa-miR-149-5p", "hsa-miR-92a-3p",
        ],
        "mechanism": "Inhibition of TGF-beta/Smad signaling and collagen overproduction",
    },
    "hair_restoration": {
        "display_name": "Gray Hair Reversal / Hair Restoration",
        "display_name_zh": "白髮逆轉 / 毛髮再生",
        "kegg_pathways": [
            "hsa04310",  # Wnt signaling pathway
            "hsa04151",  # PI3K-Akt signaling pathway
            "hsa04060",  # Cytokine-cytokine receptor interaction
        ],
        "key_genes": [
            "CTNNB1", "WNT3A", "WNT10B", "LEF1", "DKK1",
            "BMP4", "SHH", "VEGFA", "FGF7", "FGF10",
            "NOTCH1", "JAG1",
        ],
        "key_mirnas": [
            "hsa-miR-205-5p", "hsa-miR-218-5p", "hsa-miR-140-5p",
            "hsa-miR-31-5p", "hsa-miR-214-5p", "hsa-miR-24-3p",
        ],
        "mechanism": "Activation of Wnt/beta-catenin pathway for follicle regeneration",
    },
    "anti_aging": {
        "display_name": "Cell Rejuvenation / Anti-Aging",
        "display_name_zh": "細胞回春 / 抗衰老",
        "kegg_pathways": [
            "hsa04115",  # p53 signaling pathway
            "hsa04210",  # Apoptosis
            "hsa04211",  # Longevity regulating pathway
            "hsa04150",  # mTOR signaling pathway
        ],
        "key_genes": [
            "TP53", "SIRT1", "SIRT6", "CDKN1A", "CDKN2A",
            "MDM2", "TERT", "FOXO3", "MTOR", "IGF1R",
            "RB1", "E2F1",
        ],
        "key_mirnas": [
            "hsa-miR-146a-5p", "hsa-miR-34a-5p", "hsa-miR-145-5p",
            "hsa-miR-21-5p", "hsa-miR-155-5p", "hsa-miR-16-5p",
            "hsa-miR-22-5p", "hsa-miR-199a-5p",
        ],
        "mechanism": "Modulation of p53/SIRT1 senescence pathways and telomere maintenance",
    },
    "skin_refinement": {
        "display_name": "Skin Refinement / Collagen Promotion",
        "display_name_zh": "肌膚細緻 / 膠原蛋白促進",
        "kegg_pathways": [
            "hsa04510",  # Focal adhesion
            "hsa04512",  # ECM-receptor interaction
        ],
        "key_genes": [
            "COL1A1", "COL1A2", "COL3A1", "COL4A1",
            "ELN", "FBN1",
            "MMP1", "MMP2", "MMP3", "MMP9",
            "TIMP1", "TIMP2",
        ],
        "key_mirnas": [
            "hsa-miR-29a-3p", "hsa-miR-29b-3p", "hsa-miR-29c-3p",
            "hsa-miR-133a-3p", "hsa-miR-152-3p", "hsa-miR-196a-5p",
        ],
        "mechanism": "Balance of collagen synthesis and MMP-mediated degradation",
    },
}

# Aggregate all aesthetic key genes and miRNAs for quick lookup
ALL_AESTHETIC_GENES = set()
ALL_AESTHETIC_MIRNAS = set()
GENE_TO_CATEGORIES = {}
MIRNA_TO_CATEGORIES = {}

for category, info in AESTHETIC_PATHWAYS.items():
    for gene in info["key_genes"]:
        ALL_AESTHETIC_GENES.add(gene)
        GENE_TO_CATEGORIES.setdefault(gene, []).append(category)
    for mirna in info["key_mirnas"]:
        ALL_AESTHETIC_MIRNAS.add(mirna)
        MIRNA_TO_CATEGORIES.setdefault(mirna, []).append(category)

ALL_KEGG_PATHWAYS = set()
for info in AESTHETIC_PATHWAYS.values():
    ALL_KEGG_PATHWAYS.update(info["kegg_pathways"])
