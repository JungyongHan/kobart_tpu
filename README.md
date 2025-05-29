# KoBART-summarization for TPU

This is a TPU-compatible version of the KoBART summarization model, optimized for TPUv4-32 (4 VMs, 16 cores).

## Environment Setup

### TPU Configuration
- TPUv4-32 (4 VMs, 16 cores)
- PyTorch/XLA version 2.7.0

## Installation

```bash
# Install KoBART and TPU dependencies
bash install_kobart.sh

# Install other requirements
pip install -r requirements.txt
```

## Load KoBART
- Using huggingface.co binary
  - https://huggingface.co/gogamza/kobart-base-v1

## Data
- [Dacon 한국어 문서 생성요약 AI 경진대회](https://dacon.io/competitions/official/235673/overview/) 의 학습 데이터를 활용함
- 학습 데이터에서 임의로 Train / Test 데이터를 생성함
- 데이터 탐색에 용이하게 tsv 형태로 데이터를 변환함
- Data 구조
    - Train Data : 34,242
    - Test Data : 8,501
- default로 data/train.tsv, data/test.tsv 형태로 저장함
  
| news  | summary |
|-------|--------:|
| 뉴스원문| 요약문 |  

- 참조 데이터
  - AIHUB 문서 요약 데이터 (https://aihub.or.kr/aidata/8054)

## How to Train
- KoBART summarization fine-tuning on TPU

```bash
# Run training script
bash run_train.sh
```

The training script is configured for TPUv4-32 with the following parameters:
- 32 TPU cores
- Batch size of 32 per core
- BFloat16 mixed precision
- Gradient clipping at 1.0
- 100 max epochs

## TPU-Specific Optimizations

1. **XLA Compilation**: The model uses XLA (Accelerated Linear Algebra) for optimized compilation on TPUs.

2. **Distributed Training**: Implemented using `torch_xla.distributed.xla_multiprocessing` to efficiently utilize all TPU cores.

3. **BFloat16 Precision**: Uses BFloat16 mixed precision for faster training while maintaining numerical stability.

4. **Optimized Data Loading**: Uses `torch_xla.distributed.parallel_loader.MpDeviceLoader` for efficient data loading on TPUs.

5. **Synchronized Batch Normalization**: Implements synchronized batch normalization across TPU cores.

6. **XLA Mark Step**: Strategic placement of `xm.mark_step()` calls to optimize TPU execution.

## Demo
- 학습한 model binary 추출 작업이 필요함
   - pytorch-lightning binary --> huggingface binary로 추출 작업 필요
   - hparams의 경우에는 <b>./logs/tb_logs/default/version_0/hparams.yaml</b> 파일을 활용
   - model_binary 의 경우에는 <b>./checkpoint/model_chp</b> 안에 있는 .ckpt 파일을 활용
   - 변환 코드를 실행하면 <b>./kobart_summary</b> 에 model binary 가 추출 됨
  
```
python get_model_binary.py --model_binary model_binary_path
```

- streamlit을 활용하여 Demo 실행 (TPU 지원)
    - 실행 시 <b>http://localhost:8501/</b> 로 Demo page가 실행됨
```
streamlit run infer.py
```