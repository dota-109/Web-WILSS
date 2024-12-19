import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import requests
import torch
import tqdm

def load_model():
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1
        # cache_dir="PATH/TO/CACHE/DIR"  # Defaults to ~/.cache
    )

    checkpoint_path = hf_hub_download("openflamingo/openFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    return model.cuda(), image_processor, tokenizer

def generate_caption(model, vision_x, lang_x, tokenizer):
    generated_text = model.generate(
        vision_x=vision_x.cuda(),
        lang_x=lang_x["input_ids"].cuda(),
        attention_mask=lang_x["attention_mask"].cuda(),
        max_new_tokens=40,
        num_beams=3
        # pad_token_id=50277
    )

    return tokenizer.decode(generated_text[0])[7:-14]

def predict_with_flamingo(query_image, model, image_processor, tokenizer):

    vision_x = image_processor(query_image).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # print(vision_x.shape)
    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    lang_x = tokenizer(
        ["<image>An image of"],
        return_tensors="pt",
    )

    return generate_caption(model, vision_x, lang_x, tokenizer)


def write_caption(image_path, file_name, ignore_len=False):
    if os.path.isfile(file_name):
        with open(file_name, 'r') as f:
            file_len = f.readlines()
    else:
        file = open(file_name,'w')
        file.close()
        file_len = []
    if not ignore_len:
        img_names = sorted(os.listdir(image_path))[len(file_len):]
    else:
        img_names = sorted(os.listdir(image_path))

    model,image_processor,tokenizer = load_model()

    for img_name in tqdm.tqdm(img_names):
        full_path = os.path.join(image_path, img_name)
        img = Image.open(full_path)
        result = predict_with_flamingo(img, model, image_processor, tokenizer)
        with open(file_name, 'a') as f:
            f.write(img_name)
            f.write(" ")
            f.write(result[:])
            f.write("\n")







if __name__ == "__main__":

    im_path = r"./data/pascal/JPEGImages"
    file_name = "./pascal_cap.txt"

    write_caption(im_path, file_name)