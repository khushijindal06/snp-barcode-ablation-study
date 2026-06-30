# SNP Barcode Ablation Study

This folder turns the proposed idea into a controlled ablation-study package.
The framing is research-appropriate: instead of saying that three models are
run separately for convenience, the experiments quantify the contribution of
each barcode design component.

## Checked Research Framing

Use this wording in the thesis or paper:

```text
An ablation study was conducted to quantify the individual and combined
contribution of each barcode design component.
```

Do not frame it as:

```text
Three separate models were trained because the barcode has three parts.
```

The ablation framing is stronger because E1, E2, and E3 isolate specific
information sources while keeping the model architecture and training protocol
the same.

## Experiments

| Experiment | Input representation | SNP scaling | Gap encoding | Boundary markers | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| E1 | SNP-context grayscale barcode | No | No | No | Tests local nucleotide context |
| E2 | SNP-context grayscale barcode with center scaling | Yes | No | No | Tests SNP-position emphasis |
| E3 | Full genomic barcode | Yes | Yes | Yes | Tests the complete proposed method |

E3 should be presented as the final proposed barcode model. E1 and E2 are
ablation or independent-validation experiments.

## Folder Layout

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

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install pysam  # recommended for large/indexed genome files
```

The script can run without `pysam` on small FASTA/VCF files by using its simple
reader backend. For real genome-scale work, install `pysam`.

## Generate The Three Barcode Variants

Run this from inside `snp_barcode_ablation_study/`:

```bash
python scripts/generate_ablation_barcodes.py \
  --reference path/to/reference.fasta \
  --vcf path/to/strain_1.snps.vcf.gz \
  --sample strain_1 \
  --output-root barcodes \
  --flank 50 \
  --scaled-snp-scale 10
```

The output will be:

```text
barcodes/E1_context_only/strain_1_E1_context_only.png
barcodes/E2_context_scaled/strain_1_E2_context_scaled.png
barcodes/E3_full_barcode/strain_1_E3_full_barcode.png
```

Each PNG has a matching JSON metadata file that records encoded SNP count,
image size, skipped records, component switches, and grayscale mapping.

## Run The Included Smoke Example

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

Expected behavior:

```text
E1 encodes 3 SNPs without scaling, gaps, or markers.
E2 encodes the same 3 SNPs with center-row scaling.
E3 encodes the same 3 SNPs with center-row scaling, gap encoding, and markers.
```

The example VCF also contains one indel and one reference genotype so the
filtering logic can be checked.

## Encoding Rules

Base-to-grayscale mapping:

```text
A = 255
C = 85
G = 170
T = 0
N = 128
```

Reserved values:

```text
gap columns      = 32
boundary markers = 220
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

## Training Protocol

Use the same samples, labels, train/validation/test split, CNN architecture,
optimizer, learning rate, epochs, batch size, augmentation rules, and evaluation
metrics for E1, E2, and E3.

Because the raw barcode sizes can differ between variants, handle input size in
one controlled way:

```text
Option A: resize every barcode to the same CNN input size, such as 224 x 224.
Option B: use a CNN with adaptive pooling so it can accept variable dimensions.
```

Use the same option for all three experiments.

Recommended comparison:

```text
Baseline 1: Traditional ML on SNP feature vector
Baseline 2: CNN on E1 SNP-context barcode only
Ablation 1: CNN on E2 SNP-context + SNP scaling
Proposed:   CNN on E3 full barcode with gap encoding
```

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

## Reporting Template

Use the included `experiment_matrix.csv` for the design table. For the results,
report at least:

```text
experiment_id
input_representation
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

The central claim should be based on whether E3 improves over E1 and E2 under
the same evaluation setup.
