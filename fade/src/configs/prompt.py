from typing import Literal, Optional, Union

import yaml
from pathlib import Path
import pandas as pd
import random

from pydantic import BaseModel, root_validator
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import torch.nn.functional as F 
from src.misc.clip_templates import imagenet_templates
from src.engine.train_util import encode_prompts

ACTION_TYPES = Literal[
    "erase",
    "erase_with_retain",
]

class PromptEmbedsXL:
    text_embeds: torch.FloatTensor
    pooled_embeds: torch.FloatTensor

    def __init__(self, embeds) -> None:
        self.text_embeds, self.pooled_embeds = embeds

PROMPT_EMBEDDING = Union[torch.FloatTensor, PromptEmbedsXL]


class PromptEmbedsCache:
    prompts: dict[str, PROMPT_EMBEDDING] = {}

    def __setitem__(self, __name: str, __value: PROMPT_EMBEDDING) -> None:
        self.prompts[__name] = __value

    def __getitem__(self, __name: str) -> Optional[PROMPT_EMBEDDING]:
        if __name in self.prompts:
            return self.prompts[__name]
        else:
            return None


class PromptSettings(BaseModel):  # yaml
    target: str
    positive: str = None  # if None, target will be used
    unconditional: str = ""  # default is ""
    neutral: str = None  # if None, unconditional will be used
    retain_one: str
    retain_two: str
    retain_three: str 
    retain_four: str 
    retain_five: str
    action: ACTION_TYPES = "erase"  # default is "erase"
    guidance_scale: float = 1.0  # default is 1.0
    resolution: int = 512  # default is 512
    dynamic_resolution: bool = False  # default is False
    batch_size: int = 1  # default is 1
    dynamic_crops: bool = False  # default is False. only used when model is XL
    use_template: bool = False  # default is False
    
    retain_strength: float = 1000.0
    guide_strength: float = 100.0
    exp_strength: float = 2.0
    sampling_batch_size: int = 4

    seed: int = None
    case_number: int = 0

    @root_validator(pre=True)
    def fill_prompts(cls, values):
        keys = values.keys()
        if "target" not in keys:
            raise ValueError("target must be specified")
        if "positive" not in keys:
            values["positive"] = values["target"]
        if "unconditional" not in keys:
            values["unconditional"] = ""
        if "neutral" not in keys:
            values["neutral"] = values["unconditional"]

        return values


class PromptEmbedsPair:
    target: PROMPT_EMBEDDING  # the concept that do not want to generate 
    positive: PROMPT_EMBEDDING  # generate the concept
    unconditional: PROMPT_EMBEDDING  # uncondition (default should be empty)
    neutral: PROMPT_EMBEDDING  # base condition (default should be empty)

    retain_one: PROMPT_EMBEDDING
    retain_two: PROMPT_EMBEDDING
    retain_three: PROMPT_EMBEDDING
    retain_four: PROMPT_EMBEDDING
    retain_five: PROMPT_EMBEDDING
    use_template: bool = False  # use clip template or not


    guidance_scale: float
    resolution: int
    dynamic_resolution: bool
    batch_size: int
    dynamic_crops: bool

    loss_fn: torch.nn.Module
    action: ACTION_TYPES

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        target: PROMPT_EMBEDDING,
        positive: PROMPT_EMBEDDING,
        unconditional: PROMPT_EMBEDDING,
        neutral: PROMPT_EMBEDDING,
        retain_one: PROMPT_EMBEDDING,
    retain_two: PROMPT_EMBEDDING,
    retain_three: PROMPT_EMBEDDING,
    retain_four: PROMPT_EMBEDDING,
    retain_five: PROMPT_EMBEDDING,
        settings: PromptSettings,
    ) -> None:
        self.loss_fn = loss_fn
        self.target = target
        self.positive = positive
        self.unconditional = unconditional
        self.neutral = neutral
        self.retain_one = retain_one
        self.retain_two = retain_two 
        self.retain_three = retain_three 
        self.retain_four = retain_four 
        self.retain_five = retain_five
        
        self.settings = settings

        self.use_template = settings.use_template
        self.guidance_scale = settings.guidance_scale
        self.resolution = settings.resolution
        self.dynamic_resolution = settings.dynamic_resolution
        self.batch_size = settings.batch_size
        self.dynamic_crops = settings.dynamic_crops
        self.action = settings.action
        
        self.retain_strength = settings.retain_strength
        self.guide_strength = settings.guide_strength 
        self.exp_strength = settings.exp_strength
        self.sampling_batch_size = settings.sampling_batch_size
        
        
    def _prepare_embeddings(
        self, 
        cache: PromptEmbedsCache,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
    ):
        """
        Prepare embeddings for training. When use_template is True, the embeddings will be
        format using a template, and then be processed by the model.
        """
        if not self.use_template:
            return
        template = random.choice(imagenet_templates)
        target_prompt = template.format(self.settings.target)
        if cache[target_prompt]:
            self.target = cache[target_prompt]
        else:
            self.target = encode_prompts(tokenizer, text_encoder, [target_prompt])
        
    
        
    def _erase_with_retain(
        self,
        target_latents: torch.FloatTensor,  
        positive_latents: torch.FloatTensor,  
        neutral_latents: torch.FloatTensor,  
        retain_latents: list,
        retain_latents_ori: list,
        anchor_latents: torch.FloatTensor, 
        anchor_latents_ori: torch.FloatTensor, 
        negative_latents: list,
        **kwargs,
    ):
        # anchoring_loss = self.loss_fn(anchor_latents, anchor_latents_ori)
       #print(retain_latents[0].shape,retain_latents_ori[1].shape)
        retain_loss = None 
        for tgt,ori in zip(retain_latents,retain_latents_ori):
            if retain_loss is None:
                retain_loss = self.loss_fn(tgt,ori)
            else:
                retain_loss += self.loss_fn(tgt,ori)
        
        pos_distance = None 
        for pl in retain_latents_ori:
            if pos_distance is None :
                pos_distance = F.pairwise_distance(target_latents,pl).mean()
                # print("pd :: ",pos_distance.shape)
            else:
                pos_distance += F.pairwise_distance(target_latents,pl).mean()
        
        neg_distance = None 
        for nl in negative_latents:
            if neg_distance is None :
                neg_distance = F.pairwise_distance(target_latents,nl).mean()
            else:
                neg_distance += F.pairwise_distance(target_latents,nl).mean()

        margin = 1.0
        exp_loss = F.relu(pos_distance-neg_distance+margin)
       
        guidance_loss = self.loss_fn(
            target_latents,
            neutral_latents
        )
        
        
        losses = {
            "loss": self.guide_strength*guidance_loss + self.retain_strength * retain_loss + self.exp_strength*exp_loss,
            "loss/guidance": guidance_loss,
            "loss/retain": retain_loss,
            "loss/exp_loss":exp_loss
        }
        return losses

    def loss(
        self,
        **kwargs,
    ):
        if self.action == "erase":
            return self._erase(**kwargs)
        elif self.action == "erase_with_retain":
            return self._erase_with_retain(**kwargs)
        else:
            raise ValueError("action must be erase or erase_with_retain")


def load_prompts_from_yaml(path: str | Path) -> list[PromptSettings]:
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)

    if len(prompts) == 0:
        raise ValueError("prompts file is empty")

    prompt_settings = [PromptSettings(**prompt) for prompt in prompts]

    return prompt_settings

def load_prompts_from_table(path: str | Path) -> list[PromptSettings]:
    # check if the file ends with .csv
    if not path.endswith(".csv"):
        raise ValueError("prompts file must be a csv file")
    df = pd.read_csv(path)
    prompt_settings = []
    for _, row in df.iterrows():
        prompt_settings.append(PromptSettings(**dict(
            target=str(row.prompt),
            seed=int(row.get('sd_seed', row.evaluation_seed)),
            case_number=int(row.get('case_number', -1)),
        )))
    return prompt_settings


