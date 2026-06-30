# SNP Barcode Ablation Study

A research pipeline for converting SNP variation into grayscale barcode images
and evaluating whether barcode design components improve genomic
classification.

The project is designed around a controlled ablation study. Instead of only
showing that SNPs can be converted into images, the experiments test which
parts of the image representation contribute to classification performance.

## Project Goal

The main goal is to develop and evaluate an explainable SNP-barcode imaging
representation for tasks such as:

- antimicrobial resistance prediction
- genomic strain classification
- lineage, clade, or sequence-type classification
- pathogenic versus non-pathogenic classification
- outbreak versus non-outbreak classification

The proposed barcode preserves three information sources:

1. Local nucleotide context around each SNP.
2. Visual emphasis of the SNP center position.
3. Global genomic spacing between consecutive SNPs.

The central research claim should be tested as:

```text
The complete SNP barcode improves classification because it combines local SNP
context, SNP-position emphasis, and genomic distance information.
```

## Current Project Status

Implemented:

- E1, E2, and E3 barcode generation.
- FASTA and VCF parsing with a small-file backend.
- Optional `pysam` backend for larger indexed genome files.
- Genotype-aware ALT allele selection from VCF samples.
- Example reference FASTA and SNP VCF.
- Example generated barcode PNGs and JSON metadata.
- Experiment design matrix.

Not implemented yet:

- CNN training scripts.
- Traditional ML baseline scripts.
- Explainability scripts such as Grad-CAM or occlusion sensitivity.
- Real biological dataset and labels.
- Final result tables for a paper or thesis.

This means the repository is ready for dataset preparation and experimental
modeling, but it should not yet be presented as a completed classification
study.

## Repository Layout

```text
snp_barcode_ablation_study/
|-- README.md
|-- requirements.txt
|-- experiment_matrix.csv
|-- examples/
|   |-- reference.fasta
|   `-- strain_1.snps.vcf
|-- scripts/
|   `-- generate_ablation_barcodes.py
`-- barcodes/
    |-- E1_context_only/
    |-- E2_context_scaled/
    `-- E3_full_barcode/
```

Large real datasets should not be committed directly unless the repository is
configured for Git LFS. Keep large FASTA, VCF, BAM, and image datasets outside
the repo and reference them through a manifest file.

## Installation

Create an environment and install the required Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For real genome-scale files, also install `pysam`:

```bash
python -m pip install pysam
```

The included smoke example can run without `pysam` by using the simple reader
backend.

## Input Dataset

For a real experiment, prepare one dataset manifest with one row per strain or
sample:

| sample_id | strain_name | reference | vcf_path | label |
| --- | --- | --- | --- | --- |
| S1 | strain_1 | reference.fasta | strain_1.snps.vcf.gz | Resistant |
| S2 | strain_2 | reference.fasta | strain_2.snps.vcf.gz | Susceptible |
| S3 | strain_3 | reference.fasta | strain_3.snps.vcf.gz | Resistant |

The `label` column depends on the biological task:

| Task | Label examples |
| --- | --- |
| AMR prediction | Resistant, Susceptible |
| Strain typing | lineage, clade, ST type |
| Pathogenicity prediction | pathogenic, non-pathogenic |
| Outbreak tracking | outbreak, non-outbreak |
| Species or subspecies classification | species or class name |

For a PhD or paper, AMR prediction or strain classification is usually stronger
than barcode generation alone because it connects the representation to a
biological prediction problem.

## Barcode Variants

The repository generates three barcode datasets from the same SNP records.
Only the barcode representation changes across experiments.

| Experiment | Input representation | SNP scaling | Gap encoding | Boundary markers | Role |
| --- | --- | ---: | ---: | ---: | --- |
| E1 | SNP-context grayscale barcode | No | No | No | Baseline barcode |
| E2 | SNP-context grayscale barcode with center scaling | Yes | No | No | Ablation model |
| E3 | Full genomic barcode | Yes | Yes | Yes | Proposed model |

E3 is the final proposed barcode model. E1 and E2 are ablation experiments used
to quantify the contribution of local SNP context and SNP-center emphasis.

## Quick Start With Example Data

Run the included smoke example:

```bash
python scripts/generate_ablation_barcodes.py \
  --reference examples/reference.fasta \
  --vcf examples/strain_1.snps.vcf \
  --sample strain_1 \
  --output-root barcodes \
  --flank 5 \
  --scaled-snp-scale 3 \
  --backend simple
```

Expected outputs:

```text
barcodes/E1_context_only/strain_1_E1_context_only.png
barcodes/E1_context_only/strain_1_E1_context_only.json
barcodes/E2_context_scaled/strain_1_E2_context_scaled.png
barcodes/E2_context_scaled/strain_1_E2_context_scaled.json
barcodes/E3_full_barcode/strain_1_E3_full_barcode.png
barcodes/E3_full_barcode/strain_1_E3_full_barcode.json
```

Expected behavior:

```text
E1 encodes the SNP-context columns only.
E2 encodes the same SNPs with center-row scaling.
E3 encodes the same SNPs with center-row scaling, gap encoding, and markers.
```

The example VCF includes one indel and one reference-genotype record so the
filtering logic can be checked.

## Generate Barcodes For A Real Strain

Run this command for each strain-level VCF:

```bash
python scripts/generate_ablation_barcodes.py \
  --reference path/to/reference.fasta \
  --vcf path/to/strain_1.snps.vcf.gz \
  --sample strain_1 \
  --output-root barcodes \
  --flank 50 \
  --scaled-snp-scale 10
```

If the VCF contains exactly one sample, the script can infer it automatically.
If the VCF contains multiple samples, pass `--sample`.

To ignore genotypes and always use the first ALT allele:

```bash
python scripts/generate_ablation_barcodes.py \
  --reference path/to/reference.fasta \
  --vcf path/to/strain_1.snps.vcf.gz \
  --ignore-genotypes \
  --output-root barcodes
```

Each image has a matching JSON file that records:

- encoded SNP count
- image height and width
- skipped VCF records
- selected sample
- barcode components used
- grayscale mapping

## Encoding Rules

Base-to-grayscale mapping:

```text
A = 255
C = 85
G = 170
T = 0
N = 128
```

Reserved grayscale values:

```text
gap columns      = 32
boundary markers = 220
```

Inter-SNP genomic gaps follow the non-overlapping bins used in the Objective 1
presentation:

```text
0 px = 0-10 bp
1 px = 11-100 bp
2 px = 101-1,000 bp
3 px = 1,001-10,000 bp
4 px = 10,001-100,000 bp
```

For E1, image height is:

```text
2 * flank + 1
```

For E2 and E3, image height is:

```text
flank + scaled_snp_scale + flank
```

With `flank = 50` and `scaled_snp_scale = 10`, E1 height is 101 pixels and
E2/E3 height is 110 pixels.

## Verification

Run the included boundary test for gap encoding:

```bash
python -m unittest discover -s tests
```

## Experimental Design

Train the same CNN architecture separately on E1, E2, and E3:

```text
CNN on E1_context_only
CNN on E2_context_scaled
CNN on E3_full_barcode
```

Keep the following fixed across all three experiments:

- sample set
- labels
- train/validation/test split
- CNN architecture
- optimizer
- learning rate
- batch size
- number of epochs
- augmentation rules
- evaluation metrics
- random seed

Only the input barcode type should change. This makes the comparison a true
ablation study.

Because image sizes can differ between variants, choose one consistent input
handling strategy:

```text
Option A: resize every barcode to the same CNN input size, such as 224 x 224.
Option B: use adaptive pooling so the CNN can accept variable image dimensions.
```

Use the same strategy for E1, E2, and E3.

## Recommended Baselines

To make the study stronger, compare barcode CNNs with traditional SNP-vector
machine learning baselines:

| Model | Input |
| --- | --- |
| Random Forest | SNP presence/absence vector |
| XGBoost | SNP feature vector |
| SVM | SNP feature vector |
| CNN-E1 | Context-only barcode |
| CNN-E2 | Context plus SNP scaling barcode |
| CNN-E3 | Full barcode with gap encoding |

This comparison tests whether the image representation adds value beyond
standard tabular SNP features.

## Explainability Plan

Because the barcode is image-based, the CNN can be interpreted with visual
explainability methods:

| Method | Purpose |
| --- | --- |
| Grad-CAM | Identifies barcode regions that influenced CNN predictions |
| Integrated Gradients | Scores important barcode pixels |
| Occlusion sensitivity | Tests performance changes after masking barcode regions |
| SHAP | Explains traditional SNP-vector baseline models |

Recommended thesis wording:

```text
Explainability analysis was performed to identify whether the CNN focused on
SNP-context regions, scaled SNP rows, or inter-SNP gap patterns during
classification.
```

## Results To Report

At minimum, report:

```text
experiment_id
input_representation
model
accuracy
precision
recall
f1_score
auc
train_split
validation_split
test_split
random_seed
notes
```

Use `experiment_matrix.csv` as the design table. The key result is whether E3
improves over E1 and E2 under the same training and evaluation setup.

## Suggested Research Objectives

| Objective | Work |
| --- | --- |
| Objective 1 | SNP-barcode design and encoding logic |
| Objective 2 | Ablation study of barcode components E1, E2, and E3 |
| Objective 3 | CNN-based classification using barcode images |
| Objective 4 | Comparison with traditional SNP-vector ML models |
| Objective 5 | Explainability of barcode-based genomic decisions |
| Objective 6 | Robustness testing across strains, references, and SNP density |

Possible title:

```text
Explainable SNP-Barcode Imaging for Genomic Strain Classification and
Antimicrobial Resistance Prediction Using Deep Learning
```

## Immediate Next Steps

1. Finalize the biological prediction task.
2. Prepare `labels.csv` or a dataset manifest with reference, VCF, and label
   columns.
3. Generate E1, E2, and E3 barcode images for every strain.
4. Train the same CNN separately on E1, E2, and E3.
5. Train traditional ML baselines using SNP vectors.
6. Compare Accuracy, Precision, Recall, F1, and AUC.
7. Add Grad-CAM or occlusion sensitivity.
8. Write the ablation-result section based on whether E3 improves over E1 and
   E2.

## Thesis-Ready Methodology Text

```text
To evaluate the contribution of each barcode design component, three
independent barcode variants were generated and evaluated separately. The first
variant encoded only the SNP-centred nucleotide context using grayscale
intensity mapping. The second variant incorporated SNP-centre scaling, where
the central variant nucleotide was vertically enlarged to increase feature
prominence during model learning. The third variant represented the complete
proposed barcode design, integrating SNP-context encoding, SNP-centre scaling,
logarithmic inter-SNP distance gaps, and barcode boundary markers. Each barcode
variant was used as an independent input to the same deep learning
architecture, allowing direct comparison of the discriminative contribution of
local sequence context, SNP emphasis, and spatial SNP distribution.
```

## Research Novelty

The novelty should not be stated only as:

```text
We converted SNPs into barcode images.
```

The stronger claim is:

```text
We designed a multi-component SNP barcode representation that preserves local
nucleotide context, variant-position emphasis, and genomic distance
distribution, and validated each component through a controlled ablation study.
```
