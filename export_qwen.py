import torch
from transformers import AutoModelForCausalLM

MODEL_DIR = "./qwen2.5-0.5b"
ONNX_PATH = "qwen2.5-0.5b.onnx"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,
    local_files_only=True,
)

model.eval()
model.config.use_cache = False

print("Materializing weights...")
for _, param in model.named_parameters():
    param.data = param.data.contiguous()

dummy_input = torch.ones((1, 16), dtype=torch.long)

class ForwardWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
        )
        return outputs[0]  # logits

wrapped_model = ForwardWrapper(model)

print("Exporting to ONNX...")
torch.onnx.export(
    wrapped_model,
    args=(dummy_input,),
    f=ONNX_PATH,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {1: "seq_len"},
        "logits": {1: "seq_len"},
    },
    opset_version=17,
    do_constant_folding=False,
)

print("ONNX export complete:", ONNX_PATH)
