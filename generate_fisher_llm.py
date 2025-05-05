import os
from collections import defaultdict
from functools import reduce
from trl import SFTConfig, SFTTrainer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def get_module_by_name(module, access_string):
     names = access_string.split(sep='.')
     return reduce(getattr, names, module)

class CustomTrainer(SFTTrainer):
    def make_grad_bank(self):
        self.mass = dict() #defaultdict(torch.tensor)
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if get_module_by_name(self.model, name).weight.requires_grad:
                    self.mass[name] = torch.zeros_like(get_module_by_name(self.model, name).weight, 
                                                       dtype=torch.float).cpu()
        self.avg_counter = 0
        
    def training_step(
        self, model, inputs, num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        kwargs = {}

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if get_module_by_name(model, name).weight.requires_grad:
                        #self.mass[name] += get_module_by_name(model, name).weight.grad.detach().cpu().double()**2
                        new_var = get_module_by_name(model, name).weight.grad.detach().cpu()**2
                        self.mass[name] += new_var
            self.avg_counter += 1


            # Finally we need to normalize the loss for reporting
            if num_items_in_batch is None:
                return loss.detach() / self.args.gradient_accumulation_steps
            return loss.detach()


        return loss.detach() / self.args.gradient_accumulation_steps

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

#dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
#dataset['train'] = dataset['train'].select(range(4000*2))
#dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

#dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
#dataset = dataset.select(range(4000*4))

ds = load_dataset("robbiegwaldd/dclm-micro")
dataset = ds['train'].select(range(150000))

peft_config = LoraConfig(
    r=64, 
    lora_alpha=64,
    use_rslora=True,
    #target_modules = ['down_proj', 'gate_proj', 'up_proj'],
    target_modules="all-linear",
    lora_dropout=0.0, 
    bias="none", 
    #modules_to_save = ["lm_head", "embed_tokens"],        # needed because we added new tokens to tokenizer/model
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
    max_seq_length=512,
    output_dir="./_tmp6_llama2_7b_all-linear",
    report_to='none',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    eval_strategy="no",
    save_strategy="no",
    # remove_unused_columns=False,
    # load_best_model_at_end = True,
    dataset_text_field="text",
    learning_rate=5e-5,
    warmup_steps=0,
    seed=0,
    lr_scheduler_type="constant",
    bf16=True,
    )

trainer = CustomTrainer(
    "meta-llama/Llama-2-7b-hf",
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
)
trainer.make_grad_bank()

trainer.train()
#"meta-llama/Llama-3.1-8B",
#"meta-llama/Llama-2-7b-hf",

import pickle
with open(trainer.args.output_dir+f'/fisher_{trainer.avg_counter}.pkl', 'wb') as fp:
    pickle.dump(trainer.mass, fp)
