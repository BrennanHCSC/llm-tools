import torch
import gc
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import logging
import os
import huggingface_hub

from prompter import Prompter

class Model:
    def __init__(self, 
                 base_model = 'decapoda-research/llama-7b-hf', 
                 lora_weights = 'tloen/alpaca-lora-7b', 
                 promt_template = ""):
        self.load_model(base_model, lora_weights)
        self.promt_template = promt_template
        self.prompter = Prompter(promt_template)

    '''
    Load the model based on https://github.com/tloen/alpaca-lora/blob/main/generate.py#L40
    '''
    def load_model(self, base_model, lora_weights):
        self.base_model = base_model
        self.lora_weights = lora_weights
        logging.info('Base model name: ' + str(self.base_model))
        logging.info('Lora weights name: ' + str(self.lora_weights))
        logging.info('Loading tokenizer')
        self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model)
        logging.info('Loading tokenizer -- Complete')
        logging.info('Loading model')
        
        try:
            self.loadModelForDevice("auto")
        except Exception as e:
            logging.warn('Loading model, device = ' + str(self.device) + '-- Failed')
            logging.warn(str(e))
            try:
                self.loadModelForDevice("cuda")
            except Exception as e:
                logging.warn('Loading model, device = ' + str(self.device) + '-- Failed')
                logging.warn(str(e))
                self.loadModelForDevice({'': "cpu"})
        
        self.logMemoryUse()
        logging.info('Loading -- Complete')


    def evaluate(
        self,
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        batch_size = 1,
        **kwargs,
    ):
        prompt = self.prompter.generate_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        self.model.to('cuda')
        input_ids = inputs["input_ids"].to(self.device)
        gc.collect()
        torch.cuda.empty_cache()

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids.to('cuda'),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        yield self.prompter.get_response(output)
       

    def getPromptReply(self, prompt):
        reply = self.evaluate(prompt)
        out = ""
        for i in reply:
            out += i
        return out


    def logMemoryUse(self):
        (free_gpu_memory, total_gpu_memory) = torch.cuda.mem_get_info()
        logging.info('Available CUDA memory: ' + str(free_gpu_memory/ 1024**2) + " MB")
        logging.info('Total CUDA memory: ' + str(total_gpu_memory/ 1024**2) + " MB")
        logging.info('total CUDA memory allocated is: {:6.0f}'.format(torch.cuda.memory_allocated() / 1024**2), 'MB') # Returns the current GPU memory occupied by tensors in bytes for a given device.
        logging.info('maximum CUDA memory allocated is: {:6.0f}'.format(torch.cuda.max_memory_allocated() / 1024**2), 'MB') # Returns the maximum GPU memory occupied by tensors in bytes for a given device.
        logging.info('total CUDA memory reserved is: {:6.0f}' .format(torch.cuda.memory_reserved()  / 1024**2), 'MB')  # Returns the current GPU memory managed by the caching allocator in bytes for a given device.
        logging.info('maximum CUDA memory reserved is: {:6.0f}' .format(torch.cuda.max_memory_reserved()  / 1024**2), 'MB') # Returns the maximum GPU memory managed by the caching allocator in bytes for a given device.


    def loadModelForDevice(self, device):
        self.device = device
        logging.info('Loading base model, device = ' + str(self.device))
        self.model = LlamaForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit = False,
            llm_int8_enable_fp32_cpu_offload = True,
            torch_dtype = torch.float16,
            device_map = "auto" if(self.device=="cuda") else self.device
        )
        logging.info('Loading base model, device = ' + str(self.device) + " -- Complete")
        logging.info('Loading lora weights, device = ' + str(self.device))
        self.model = PeftModel.from_pretrained(
            self.model,
            self.lora_weights,
            torch_dtype = torch.float16,
            #device_map = self.device
        )
        logging.info('Loading lora weights, device = ' + str(self.device) + " -- Complete")
