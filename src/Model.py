import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import logging

from prompter import Prompter

import huggingface_hub
import os

class Model:

    def __init__(self, forceCPU = True, promptTemplate = ""):
        if forceCPU:
            self.device = "cpu"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        try:
            if torch.backends.mps.is_available():
                self.device = "mps"
        except:  # noqa: E722
            pass

        self.loadModel()
        self.promptTemplate = promptTemplate
        self.prompter = Prompter(self.promptTemplate)

    '''
    Load the model based on https://github.com/tloen/alpaca-lora/blob/main/generate.py#L40
    '''
    def loadModel(self, 
                  baseModel = 'decapoda-research/llama-7b-hf', 
                  loraWeights = 'tloen/alpaca-lora-7b'):
        self.baseModel = baseModel
        self.loraWeights = loraWeights
        logging.info('Base model name: ' + str(self.baseModel))
        logging.info('Lora weights name: ' + str(self.loraWeights))
        logging.info('Loading tokenizer')
        self.tokenizer = LlamaTokenizer.from_pretrained(self.baseModel)
        logging.info('Loading tokenizer -- Complete')
        logging.info('Loading model')
        self.loadModelForDevice()
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
        **kwargs,
    ):
        prompt = self.prompter.generate_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
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
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        yield self.prompter.get_response(output)
       
    def loadModelForDevice(self, load_8bit = False):
        logging.info('Loading ' + str(self.device) + ' model')
        if self.device == "cuda":
            logging.info('Loading base model')
            self.model = LlamaForCausalLM.from_pretrained(
                self.baseModel,
                load_in_8bit = load_8bit,
                torch_dtype = torch.float16,
                device_map="auto",
            )
            logging.info('Loading base model -- Complete')
            logging.info('Loading lora weights')
            self.model = PeftModel.from_pretrained(
                self.model,
                self.loraWeights,
                torch_dtype = torch.float16,
            )
            logging.info('Loading lora weights -- Complete')
        elif self.device == "mps":
            logging.info('Loading base model')
            self.model = LlamaForCausalLM.from_pretrained(
                self.baseModel,
                device_map = {"": self.device},
                torch_dtype = torch.float16,
            )
            logging.info('Loading base model -- Complete')
            logging.info('Loading lora weights')
            self.model = PeftModel.from_pretrained(
                self.model,
                self.loraWeights,
                device_map = {"": self.device},
                torch_dtype = torch.float16,
            )
            logging.info('Loading lora weights -- Complete')
        else:
            logging.info('Loading base model')
            self.model = LlamaForCausalLM.from_pretrained(
                self.baseModel, device_map = {"": self.device}, low_cpu_mem_usage = True
            )
            logging.info('Loading base model -- Complete')
            logging.info('Loading lora weights')
            self.model = PeftModel.from_pretrained(
                self.model,
                self.loraWeights,
                device_map = {"": self.device},
            )
            logging.info('Loading lora weights -- Complete')

    def getPromptReply(self, prompt):
        reply = self.evaluate(prompt)
        out = ""
        for i in reply:
            out += i
        return out
