# TC-MGC: Text-Conditioned Multi-Grained Contrastive Learning for Text-Video Retrieval

[Xiaolun Jing](https://scholar.google.com/citations?hl=zh-CN&user=LsozN5kAAAAJ), Genke Yang, Jian Chu

This is the official code implementation of the paper "TC-MGC: Text-Conditioned Multi-Grained Contrastive Learning for Text-Video Retrieval", the checkpoint will be released soon.

We are continuously refactoring our code, be patient and wait for the latest updates!

## :star: Overview
Motivated by the success of coarse-grained or fine-grained contrast in text-video retrieval, there emerge multi-grained contrastive learning methods which focus on the integration of contrasts with different granularity. However, due to the wider semantic range of videos, the text-agnostic video representations might encode misleading information not described in texts, thus impeding the model from capturing precise cross-modal semantic correspondence. To this end, we propose a TextConditioned Multi-Grained Contrast framework, dubbed TC-MGC. Specifically, our model employs a language-video attention block to generate aggregated frame and video representations conditioned on the word’s and text’s attention weights over frames. To filter unnecessary similarity interactions and decrease trainable parameters in the Interactive Similarity Aggregation (ISA) module, we design a Similarity Reorganization (SR) module to identify attentive similarities and reorganize cross-modal similarity vectors and matrices. Next, we argue that the imbalance problem among multi-grained similarities may result in over- and under-representation issues. We thereby introduce an auxiliary Similarity Decorrelation Regularization (SDR) loss to facilitate cooperative relationship utilization by similarity variance minimization on matching text-video pairs. Finally, we present a Linear Softmax Aggregation (LSA) module to explicitly encourage the interactions between multiple similarities and promote the usage of multi-grained information. Empirically, TC-MGC achieves competitive results on multiple text-video retrieval benchmarks, outperforming X-CLIP model by +2.8% (+1.3%), +2.2% (+1.0%), +1.5% (+0.9%) relative (absolute) improvements in text-to-video retrieval R@1 on MSRVTT, DiDeMo and VATEX, respectively.

![image](https://github.com/JingXiaolun/TC-MGC/blob/master/image/sen_frame_and_word_frame_contrast.jpg?raw=true)
Fig. 1. Illustration of the multi-grained contrasts between frame and sentence (word) representations, including sentence-frame (crossgrained) and frame-word (fine-grained) contrasts. The arrows indicate that the texts are semantic-relevant to sub-regions of videos.

## :herb: Method
![image](https://github.com/JingXiaolun/TC-MGC/blob/master/image/main_structure.jpg?raw=true)
Fig. 2. The pipeline of TC-MGC. Given pair-wise text-video data, CLIP encoders simultaneously extract textual and visual representations, of which the extracted frame features are fed into the temporal encoder block for sequential modeling. Through language-video attention block, video representations with different granularity are regenerated in a text-guided manner. Finally, multi-grained interaction is implemented on the textual representations and text-conditioned visual representations to obtain the similarity score.

### Language-Video Attention (LVA)
![image](https://github.com/JingXiaolun/TC-MGC/blob/master/image/language_video_attention.jpg?raw=true)
Fig. 3. The Diagram of language-video attention block, which aims to generate coarse-grained (video) and fine-grained (frame) representations in a text-guided manner.

### Multi-Grained Interaction
![image](https://github.com/JingXiaolun/TC-MGC/blob/master/image/multi_grained_interaction.jpg?raw=true)
Fig. 4. The illustration of multi-grained interaction mechanism. We first use matrix multiplication to obtain video-sentence similarity score, video-word and sentence-frame similarity vectors, frame-word similarity matrices respectively, followed by SR and Bi-SR modules to achieve similarity vectors and matrices reorganization. Next, we perform ISA and Bi-ISA modules on the reorganized similarity vectors and matrices to generate instance-level scores. Finally, we employ LSA module to achieve multi-grained scores aggregation.

### Similarity Reorganization (SR)
![image](https://github.com/JingXiaolun/TC-MGC/blob/master/image/similarity_reorganization.jpg?raw=true)
Fig. 5. Similarity Reorganization modules (SR). (a) We identify and rearrange the attentive similarities as the reorganized video-word vector. (b) We preserve the attentive similarities and fuse the inattentive similarities into one similarity, which are concatenated to generate the reorganized sentence-frame vector. (c) We apply the Bi-SR module on the word and frame directions respectively to obtain the reorganized
frame-word matrix.

### Interactive Similarity Aggregation (ISA)
![image](https://github.com/JingXiaolun/TC-MGC/blob/master/image/ISA.jpg?raw=true)
Fig. 6. Interactive Similarity Aggregation module (ISA). (a) We employ the ISA module to aggregate the reorganized video-word vector into the video-word Score. (b) We adopt the ISA module
to obtain the sentence-frame Score from the reorganized sentenceframe vector. (c) We leverage the Bi-ISA module to aggregate the reorganized frame-word matrix into the frame-word score.

### Linear Softmax Aggregation (LSA)
![image](https://github.com/JingXiaolun/TC-MGC/blob/master/image/LSA.jpg?raw=true)
Fig. 7. The overview of LSA, which leverages the cascade of linear and softmax layers to calculate the weights of different instance-level scores.

## :mag: Usage
### Requirement
```bash
pip install -r requirements.txt
```
### Datasets
We train our model on MSR-VTT, DiDeMo and VATEX datasets respectively. Please refer to this [repo](https://github.com/ArrowLuo/CLIP4Clip) for data preparation.

## How to Run
Download CLIP (ViT-B/32) weight,
```bash
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```
or, download CLIP (ViT-B/16) weight,
```bash 
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```
Then, run

**MSR-VTT**

```bash
# ViT-B/32
sh scripts/run_xclip_msrvtt_vit32.sh

# ViT-B/16
sh scripts/run_xclip_msrvtt_vit16.sh
```

**MSVD**

```bash
# ViT-B/32
sh scripts/run_xclip_msvd_vit32.sh

# ViT-B/16
sh scripts/run_xclip_msvd_vit16.sh
```

**LSMDC**

```bash
# ViT-B/32
sh scripts/run_xclip_lsmdc_vit32.sh

# ViT-B/16
sh scripts/run_xclip_lsmdc_vit16.sh
```

**DiDeMo**

```bash
# ViT-B/32
sh scripts/run_xclip_didemo_vit32.sh

# ViT-B/16
sh scripts/run_xclip_didemo_vit16.sh
```

**ActivityNet**

```bash
# ViT-B/32
sh scripts/run_xclip_actnet_vit32.sh

# ViT-B/16
sh scripts/run_xclip_actnet_vit16.sh
```

## Acknowledgments

The implementation of TC-MGC relies on resources from [X-CLIP](https://github.com/xuguohai/X-CLIP "X-CLIP") and [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip "CLIP4Clip"). We thank the original authors for their open-sourcing.
