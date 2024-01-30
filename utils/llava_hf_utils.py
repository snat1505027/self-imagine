"""
Utilities related to the HF version of LLAVA.
Tested on transformers 4.36.2

!pip install transformers --upgrade
"""

from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from typing import Dict


def get_model_and_processor(model_name: str = "llava-hf/llava-1.5-7b-hf"):
    """
    Returns the LLAVA model and processor.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = LlavaForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def get_image_features(model: LlavaForConditionalGeneration, inputs: dict) -> torch.Tensor:
    """
    Extracts image features using a specified vision model.

    This function processes input images through a vision model (like a tower of CLIP),
    extracts the hidden states from a specified layer, and then projects these features
    using the model's multi-modal projector.

    Args:
    inputs (dict): A dictionary containing input data for the model.
                   Expected to have a key "pixel_values" with tensor values representing the image.

    Returns:
    Tensor: The projected image features extracted from the specified layer of the model.
    """
    pixel_values = inputs["pixel_values"]
    image_outputs = model.vision_tower(pixel_values, output_hidden_states=True).hidden_states
    selected_image_features = image_outputs[model.config.vision_feature_layer][:, 1:]
    image_features = model.multi_modal_projector(selected_image_features)
    return image_features


def get_input_text_ids_locations(attentions: torch.Tensor,
                                     input_ids:  torch.Tensor,
                                     image_features:  torch.Tensor,
                                     model: LlavaForConditionalGeneration) -> tuple:

    """
    Computes the positions where text tokens should be in the merged image-text sequence, considering special image tokens
    and the number of image patches.

    Reference:
    https://github.com/huggingface/transformers/blob/7226f3d2b06316a1623b7b557cc32f360a854860/src/transformers/models/llava/modeling_llava.py#L279
    """

    # calculate where in the output can we expect the image tokens to be?
    num_images, num_image_patches, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(model.pad_token_id))
    # 1. Create a mask to know where special image tokens are
    special_image_token_mask = input_ids == model.config.image_token_index
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    # Compute the maximum embed dimension
    max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
    batch_indices, non_image_indices = torch.where(input_ids != model.config.image_token_index)

    # 2. Compute the positions where text should be written
    # Calculate new positions for text tokens in merged image-text sequence.
    # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
    # `torch.cumsum` computes how each image token shifts subsequent text token positions.
    # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions += nb_image_pad[:, None]  # offset for left padding
    text_ids = new_token_positions[batch_indices, non_image_indices]

    return batch_indices, text_ids

@torch.no_grad()
def get_attention_over_text_and_images(prompt: str,
        image: torch.Tensor,
        model: LlavaForConditionalGeneration,
        processor: AutoProcessor,
        max_length: int = 30,
        return_average: bool = False) -> Dict[str, torch.Tensor]:
  """Generates outputs for given text prompts and images, and calculates attention scores
    to text and images in the output.

    This function addresses the limitation of HuggingFace's default behavior which doesn't return
    the output attentions for generated tokens. It does so by:
    1. Generating output tokens (generated_ids).
    2. Performing a forward pass on the combined input_ids and generated_ids to obtain attention scores.


    Returns:
    Dict[str, Tensor]: A dictionary containing attention scores to image tokens ('attentions_to_image')
                        and text ('attentions_to_text')."""

  assert not isinstance(prompt, list), "This version only works for a batch size of 1"

  # Step 1: Generate output tokens (generated_ids)
  inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
  generate_ids = model.generate(**inputs, max_length=max_length)

  # NOTE: generate_ids already includes the input tokens

  # Step 2: Do a forward pass on input_ids + generated_ids, get the attention tokens.
  outs = model(input_ids=generate_ids,
               attention_mask=torch.ones_like(generate_ids),
                pixel_values=inputs['pixel_values'],
               return_dict=True,
               output_attentions=True)

  # Step 3: Extract the location of text_ids in the concatenated
  # representation (input_embeds + image_embeds) that would be have been to the language model

  image_features = get_image_features(model, inputs)

  batch_indices, text_ids = get_input_text_ids_locations(outs.attentions, generate_ids, image_features, model)



  # outs.logits.shape = (bsz, seq_len, vocab)
  all_possible_indices = torch.arange(outs.logits.shape[-2]).to(model.device)
  # image ids are all the locations that are not text_ids
  image_ids = all_possible_indices[~torch.isin(all_possible_indices, text_ids)]



  # split the input and output ids
  generated_text_ids = text_ids[inputs["input_ids"].shape[1]:]
  generated_batch_indices = batch_indices[inputs["input_ids"].shape[1]:]

  input_text_ids = text_ids[:inputs["input_ids"].shape[1]]
  input_batch_indices = batch_indices[:inputs["input_ids"].shape[1]]


  # Step 4: get attentions
  # print(len(outs.attentions)) # 32 layers
  # print(outs.attentions[-1].shape) # 32 heads for the 7b model

  last_layer_attn_avg_heads = outs.attentions[-1].mean(dim=1)


  # (num_generated_tokens, total_seq_len)
  attention_for_generated_tokens = last_layer_attn_avg_heads[generated_batch_indices, generated_text_ids]

  # (num_generated_tokens, 1), each tensor calculates the total attention that went to the input tokens
  attentions_to_text = attention_for_generated_tokens.gather(dim=1, index=input_text_ids.unsqueeze(0).expand(10, -1)).sum(-1)


  # (num_generated_tokens, 1), each tensor calculates the total attention that went to the image tokens
  attentions_to_image = attention_for_generated_tokens.gather(dim=1, index=image_ids.unsqueeze(0).expand(10, -1)).sum(-1)

  if return_average:
    attentions_to_text = attentions_to_text.mean()
    attentions_to_image = attentions_to_image.mean()

  return {
      "attentions_to_image": attentions_to_image,
      "attentions_to_text": attentions_to_text
  }



def test_get_attention_over_text_and_images():
    """Tests the function get_attention_over_text_and_images"""
    import requests
    from PIL import Image
    model, processor = get_model_and_processor()
    prompt = "<image>\nUSER: What's happening in the picture?\nASSISTANT:"
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    res = get_attention_over_text_and_images(prompt, image, processor=processor, model=model)
    """
    res will be:
    {'attentions_to_image': tensor([0.1066, 0.1398, 0.1511, 0.1730, 0.1498, 0.1213, 0.1306, 0.1292, 0.1368,
         0.1655], device='cuda:0'),
    'attentions_to_text': tensor([0.8139, 0.7184, 0.7009, 0.7216, 0.7201, 0.7093, 0.6814, 0.6757, 0.6996,
         0.6587], device='cuda:0')}

    Interpretation: for generating the first output token, ~81% (0.8139) of the attention went to the input text tokens,
    and ~10% (0.1066) of the attention went to the image tokens.
    """
    return res
