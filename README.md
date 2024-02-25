# Road Network and Lnae Topology Extraction

> **ICCV 2023 (Oral)** [**Translating Images to Road Network: A Non-Autoregressive Sequence-to-Sequence Approach**](https://arxiv.org/abs/2402.08207),            
> Jiachen Lu, Renyuan Peng, Xinyue Cai, Hang Xu, Hongyang Li, Feng Wen, Wei Zhang, [Li Zhang](https://lzrobots.github.io)  
> **Fudan University, Huawei Noah’s Ark Lab, Shanghai AI Lab**

> **AAAI 2024** [**LaneGraph2Seq: Lane Topology Extraction with Language Model via Vertex-Edge Encoding and Connectivity Enhancement**](https://arxiv.org/abs/2401.17609),            
> Renyuan Peng, Xinyue Cai, Hang Xu, Jiachen Lu, Feng Wen, Wei Zhang, [Li Zhang](https://lzrobots.github.io)  
> **Fudan University, Huawei Noah’s Ark Lab**

## Get Started
Please checkout for [get_started.md](get_started.md)
## Training
For ICCV: Translating Images to Road Network: A Non-Autoregressive Sequence-to-Sequence Approach
```
./tools/dist_train.sh projects/RoadNetwork/configs/rntr_ar_roadseg/lss_ar_rntr_changeloss_test_fp16_torch2.py 8
```
For AAAI: LaneGraph2Seq: Lane Topology Extraction with Language Model via Vertex-Edge Encoding and Connectivity Enhancement
```
./tools/dist_train.sh projects/RoadNetwork/configs/lanegraph2seq/langraph2seq_fp16_torch2.py 8
```

## Acknowledgements
We thank numerous excellent works and open-source codebases:
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [PETR](https://github.com/megvii-research/PETR)
- [DETR3D](https://github.com/WangYueFt/detr3d)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)

## 📜 BibTex

```bibtex
@inproceedings{lu2023translating,
  title={Translating Images to Road Network: A Non-Autoregressive Sequence-to-Sequence Approach},
  author={Lu, Jiachen and Peng, Renyuan and Cai, Xinyue and Xu, Hang and Li, Hongyang and Wen, Feng and Zhang, Wei and Zhang, Li},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={23--33},
  year={2023}
}
}
```


```bibtex
@inproceedings{peng2024lanegraph2seq,
  title={LaneGraph2Seq: Lane Topology Extraction with Language Model via Vertex-Edge Encoding and Connectivity Enhancement},
  author={Peng, Renyuan and Cai, Xinyue and Xu, Hang and Lu, Jiachen and Wen, Feng and Zhang, Wei and Zhang, Li},
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
  year={2024}
}
```
