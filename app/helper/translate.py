import torch

SRC_LANGUAGE = 'english'
TRG_LANGUAGE = 'japanese'
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

# src and trg language text transforms to convert raw strings into tensors indices
def get_text_transform(token_transform, vocab_transform):
    text_transform = {}
    for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], vocab_transform[ln], tensor_transform)

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