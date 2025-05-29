import torch
import streamlit as st
from transformers.models.bart import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast

# Import TPU-specific modules
import torch_xla.core.xla_model as xm

@st.cache
def load_model():
    # Get the TPU device
    device = xm.xla_device()
    
    # Load model and move to TPU
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    model = model.to(device)
    return model

model = load_model()
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
special_tokens_dict = {'additional_special_tokens': ['<newline>']}
tokenizer.add_special_tokens(special_tokens_dict)
st.title("KoBART 요약 Test (TPU Version)")
text = st.text_area("뉴스 입력:")

st.markdown("## 뉴스 원문")
st.write(text)

if text:
    text = text.replace('\n', '')
    st.markdown("## KoBART 요약 결과")
    with st.spinner('processing..'):
        # Get the TPU device
        device = xm.xla_device()
        
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        
        # Move input to TPU
        input_ids = input_ids.to(device)
        
        # Generate output and mark step for TPU
        output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
        xm.mark_step()
        
        # Move output back to CPU for decoding
        output = output.cpu()
        
        output = tokenizer.decode(output[0], skip_special_tokens=False)
        # ouput에서 다른 특수토큰들 제거
        output = output.replace('<s>', '')
        output = output.replace('</s>', '')
        output = output.replace('<pad>', '')
        output = output.replace('<unk>', '')
        output = output.replace('<newline>', '\n')
    st.write(output)