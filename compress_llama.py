from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from llama_comp.compression_utils import CompressedLlamaForCausalLM, CompressedLlamaModel

#pretrained = "meta-llama/Llama-2-7b-hf"
pretrained = "meta-llama/Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)

#rank=2750, 0.949,
#rank=2480+40, 0.899
#rank=2250+60, 0.85
#rank=2050, 0.80
#rank=1800, 0.75
#rank=1600, 0.70
trunc_raw = {i:2750 for i in fisher.keys()}

model_ = CompressedLlamaForCausalLM.from_pretrained(
                    pretrained,
                    trust_remote_code=True,
                    rank=trunc_raw,
                    layer_mask = r'.*/(up_proj|down_proj|gate_proj)',
                    #layer_mask = r'.*/(up_proj|down_proj|gate_proj|q_proj|k_proj|o_proj|v_proj)',
                    compression_type='svd',
                )

model_.to_compression(compress=True, 
                    weight=fisher if fisher else None,
                    )

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
pytorch_total_params

print('Model after compression',sum(p.numel() for p in model_.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters() if p.requires_grad))

model_.save_pretrained(r"./compressed_llamas/llama-Llama-3-8b-hf-09-svd")
tokenizer.save_pretrained(r"./compressed_llamas/llama-Llama-3-8b-hf-09-svd")
