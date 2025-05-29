import argparse
from model import KoBARTConditionalGeneration
from transformers.models.bart import BartForConditionalGeneration
import yaml
import torch_xla.core.xla_model as xm

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, type=str)
parser.add_argument("--model_binary", default=None, type=str)
parser.add_argument("--output_dir", default='kobart_summary', type=str)
args = parser.parse_args()

# Load the checkpoint
inf = KoBARTConditionalGeneration.load_from_checkpoint(args.model_binary)

# Save the model for HuggingFace compatibility
inf.model.save_pretrained(args.output_dir)

print(f"Model binary successfully extracted to {args.output_dir}")