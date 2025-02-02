import torch

SRC_LANGUAGE = 'english'
TRG_LANGUAGE = 'japanese'

def get_torch_device():
    return torch.device('cpu')

def perform_sample_translation(model, input_text, text_transform, vocab_transform):
    device = get_torch_device()

    src_text = text_transform[SRC_LANGUAGE](input_text).to(device)
    trg_text = text_transform[TRG_LANGUAGE]("鼻").to(device)

    src_text = src_text.reshape(1, -1) 
    trg_text = trg_text.reshape(1, -1)

    model.eval()
    with torch.no_grad():
        output, _ = model(src_text, trg_text)

    output = output.squeeze(0)
    output = output[1:]
    output_max = output.argmax(1)
    mapping = vocab_transform[TRG_LANGUAGE].get_itos()
    return "".join([mapping[i] for i in output_max])