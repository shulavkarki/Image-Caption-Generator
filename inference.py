import pickle
import glob
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


import config
from models.model import CNNtoRNN

def caption_img(image, max_length=25):
    """
    
    :param image: the image you to caption
    :param max_length: The maximum length of the caption, defaults to 40 (optional)
    :return: A list of words
    """
    num_layers = 2
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    idx_str_map = {v:k for k, v in vocab.items()}
    model = CNNtoRNN(config.EMBED_SIZE, config.HIDDEN_SIZE, len(idx_str_map), num_layers)
    checkpoint = torch.load(config.MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    result_caption = []
    model.encoderCNN.eval()
    model.decoderRNN.eval()
    with torch.no_grad():
        x = model.encoderCNN(image).unsqueeze(0)
        states = (torch.zeros(num_layers, 1, model.decoderRNN.hidden_size),
                torch.zeros(num_layers, 1, model.decoderRNN.hidden_size))

        for _ in range(max_length):
            hiddens, states = model.decoderRNN.lstm(x, states)
            output = model.decoderRNN.linear(hiddens.squeeze(0))
            predicted = output.argmax(1)
            result_caption.append(predicted.item())
            x = model.decoderRNN.embedding(predicted).unsqueeze(0)

            if idx_str_map[predicted.item()] == "<end>":
                break

    return [idx_str_map[idx] for idx in result_caption]


def overlay_save_image(processed_caption, image, img_path):
    draw = ImageDraw.Draw(image)

    # Define the font and font size

    font = ImageFont.truetype("arial_narrow_7.ttf", 25)

    # Get the size of the caption text
    text_size = draw.textsize(processed_caption, font)

    # Define the position of the caption textyt
    x = image.width - text_size[0] - 10
    y = image.height - text_size[1] - 10

    # Draw a black background rectangle for the caption text
    draw.rectangle((x, y, image.width, image.height), fill=(0, 0, 0))

    # Draw the caption text
    draw.text((x, y), processed_caption, fill=(255, 255, 255), font=font)
    image.save(f"versions/v3/captioned_images/{img_path}")
    image.show()


if __name__ == "__main__":
    save_image = True
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    for img_path in glob.glob('inference_images/*'):
        image = Image.open(img_path)
        image_tensor = transform(image)
        caption = caption_img(image_tensor.unsqueeze(0))
        processed_caption = " ".join(caption[1:-1]).capitalize()
        if save_image:
            overlay_save_image(processed_caption, image, img_path.split('/')[1])
        print(processed_caption)