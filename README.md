# 【INFFUS'2025 :fire:】TC-MGC: Text-Conditioned Multi-Grained Contrastive Learning for Text-Video Retrieval

<p align="center">
    <a href="https://doi.org/10.1016/j.inffus.2025.103151">
    <img src="https://img.shields.io/badge/INFFUS-2025-yellow.svg" alt="Build Status">
    
    </a>
    <a href="https://arxiv.org/abs/2504.04707">
    <img src="https://img.shields.io/badge/Paper-arxiv.2504.04707-b31b1b.svg" alt="Build Status">
    </a>
</p>

The implementation of INFFUS 2025 paper [TC-MGC: Text-Conditioned Multi-Grained Contrastive Learning for Text-Video Retrieval](https://arxiv.org/abs/2504.04707). 

## :pushpin: Citation
If you find our method useful in your work, please cite:
```bibtex
@article{jing2025tc,
  title={TC-MGC: Text-conditioned multi-grained contrastive learning for text-video retrieval},
  author={Jing, Xiaolun and Yang, Genke and Chu, Jian},
  journal={Information Fusion},
  pages={103151},
  year={2025},
  publisher={Elsevier}
}
```

## :star: Overview
Motivated by the success of coarse-grained or fine-grained contrast in text-video retrieval, there emerge multi-grained contrastive learning methods which focus on the integration of contrasts with different granularity. However, due to the wider semantic range of videos, the text-agnostic video representations might encode misleading information not described in texts, thus impeding the model from capturing precise cross-modal semantic correspondence. To this end, we propose a Text-Conditioned Multi-Grained Contrast framework, dubbed TC-MGC. Specifically, our model employs a language-video attention block to generate aggregated frame and video representations conditioned on the word's and text's attention weights over frames. To filter unnecessary similarity interactions and decrease trainable parameters in the Interactive Similarity Aggregation (ISA) module, we design a Similarity Reorganization (SR) module to identify attentive similarities and reorganize cross-modal similarity vectors and matrices. Next, we argue that the imbalance problem among multi-grained similarities may result in over- and under-representation issues. We thereby introduce an auxiliary Similarity Decorrelation Regularization (SDR) loss to facilitate cooperative relationship utilization by similarity variance minimization on matching text-video pairs. Finally, we present a Linear Softmax Aggregation (LSA) module to explicitly encourage the interactions between multiple similarities and promote the usage of multi-grained information. Empirically, TC-MGC achieves competitive results on multiple text-video retrieval benchmarks, outperforming X-CLIP model by +2.8% (+1.3%), +2.2% (+1.0%), +1.5% (+0.9%) relative (absolute) improvements in text-to-video retrieval R@1 on MSR-VTT, DiDeMo and VATEX, respectively. 

![image](https://github.com/JingXiaolun/TC-MGC/blob/master/image/sen_frame_and_word_frame_contrast.jpg?raw=true)
Fig. 1. Illustration of the multi-grained contrasts between frame and sentence (word) representations, including sentence-frame (cross-grained) and frame-word (fine-grained) contrasts. The connections indicate that the texts are semantic-relevant to sub-regions of videos.

## :herb: Method

### Framework
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

## :rocket: Quick Start
### Setup code environment
```bash
pip install -r requirements.txt
```
### Datasets
We train our model on MSR-VTT, DiDeMo and VATEX datasets respectively. Please refer to this [repo](https://github.com/ArrowLuo/CLIP4Clip) for data preparation.

| Datasets  | Google Cloud    | Baidu Yun | Peking University Yun|
|:------:|:------:|:------:|:------:|
| MSR-VTT  | [Download](https://drive.google.com/drive/folders/1LYVUCPRxpKMRjCSfB_Gz-ugQa88FqDu_)  | [Download](https://pan.baidu.com/share/init?surl=Gdf6ivybZkpua5z1HsCWRA&pwd=enav) | [Download](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA6A028EE7EF5C48A788118B82D6ABE0C5?_tb=none&expires_at=1970-01-01T08%3A00%3A00%2B08%3A00&item_type=folder&password_required=false&title=MSRVTT&type=anonymous) |
| DiDeMo  | TODO  | [Download](https://pan.baidu.com/share/init?surl=Tsy9nb1hWzeXaZ4xr7qoTg&pwd=c842) | [Download](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA14E48D1333114022B736291D60350FA5?_tb=none&expires_at=1970-01-01T08%3A00%3A00%2B08%3A00&item_type=folder&password_required=false&title=didemo&type=anonymous) |
| VATEX  | TODO  | TODO | TODO |

### How to Run
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
python -m torch.distributed.launch --nproc_per_node=4 --master_port='30500' \
main_tcmgc.py --do_train --num_thread_reader=8 \
--epochs=5 --batch_size=128 --batch_size_val 64 --n_display=50 \
--train_csv ${FILE_DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ../DataSet/MSRVTT/data/file/MSRVTT_JSFUSION_test.csv \
--data_path ../DataSet/MSRVTT/data/file/MSRVTT_data.json \
--features_path ../DataSet/MSRVTT/data/file/clip4clip_video_frame_input \
--output_dir ../Model/tcmgc_msrvtt_vit32 \
--log_dir ../Log/tcmgc_msrvtt_vit32 \
--visualize_dir ../Visualize/tcmgc_msrvtt_vit32 \
--lr 1e-4 --max_words 32 --max_frames 12 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--conditional_flag --sdr_loss_flag \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32
```

**DiDeMo**

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port='30501' \
main_glccl.py --do_train --num_thread_reader=8 \
--epochs=20 --batch_size=64 --batch_size_val 32 --n_display=50 \
--data_path ../DataSet/DiDeMo/data/compressed/split_file \
--features_path ../DataSet/DiDeMo/data/compressed/split_video \
--output_dir ../Model/tcmgc_didemo_vit32 \
--log_dir ../Log/tcmgc_didemo_vit32 \
--visualize_dir ../Visualize/tcmgc_didemo_vit32 \
--lr 1e-4 --max_words 64 --max_frames 64 \
--datatype didemo \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--conditional_flag --sdr_loss_flag \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32
```

**VATEX**

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port='30502' \
main_glccl.py --do_train --num_thread_reader=8 \
--epochs=5 --batch_size=128 --batch_size_val 128 --n_display=50 \
--data_path ../DataSet/VATEX/data/compressed/split_file \
--features_path ../DataSet/VATEX/data/compressed/clip4clip_video_frame_input \
--output_dir ../Model/tcmgc_vatex_vit32 \
--log_dir ../Log/tcmgc_vatex_vit32 \
--visualize_dir ../Visualize/tcmgc_vatex_vit32 \
--lr 1e-4 --max_words 32 --max_frames 12 \
--datatype vatex \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--conditional_flag --sdr_loss_flag \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32
```

## :telescope: Experiments

### MSR-VTT
![image](https://github.com/JingXiaolun/TC-MGC/blob/master/image/MSRVTT.png?raw=true)

### DiDeMo
![image](https://github.com/JingXiaolun/TC-MGC/blob/master/image/DiDeMo.png?raw=true)

### VATEX
![image](https://github.com/JingXiaolun/TC-MGC/blob/master/image/VATEX.png?raw=true)

## :heart_eyes: Visualization
![image](https://github.com/JingXiaolun/TC-MGC/blob/master/image/Visualization.jpg?raw=true)

## :reminder_ribbon: Acknowledgments

Our code is based on  [X-CLIP](https://github.com/xuguohai/X-CLIP "X-CLIP") and [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip "CLIP4Clip"). We thank the original authors for their open-sourcing.
