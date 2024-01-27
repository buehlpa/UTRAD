
import torch
import torch.nn.functional as F
from models import Create_nets
from datasets import Get_dataloader
from options import TrainOptions
from torchvision import models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = TrainOptions().parse()
torch.manual_seed(args.seed)

transformer = Create_nets(args).to(device)
checkpoint = torch.load('/home/bule/projects/UTRAD/Exp0-r18-cable/saved_models/checkpoint.pth')
transformer.load_state_dict(checkpoint['transformer'])
print(transformer)


backbone = models.resnet18(pretrained=True).to(device)
backbone.eval()
outputs = []
def hook(module, input, output):
    outputs.append(output)
backbone.layer1[-1].register_forward_hook(hook)
backbone.layer2[-1].register_forward_hook(hook)
backbone.layer3[-1].register_forward_hook(hook)

_, test_dataloader = Get_dataloader(args)

def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).to(device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z
    
transformer.eval()

for i,(name ,batch, ground_truth, gt) in enumerate(test_dataloader): 
    if i == 1:
        break                      
    with torch.no_grad():
        
        num = 4
        norm_range = [(0.9,1.3),(0.9,1.3),(0.9,1.3),(0.9,1.3),(1.1,1.5),]

        inputs = batch.to(device)
        ground_truth = ground_truth.to(device)
        
        outputs = []
        _ = backbone(inputs)                 ## with the hook funciton in the backbone the feature maps of the backbone are put in the list outputs
        outputs = embedding_concat(embedding_concat(outputs[0],outputs[1]),outputs[2])
        # recon, std = transformer(outputs)

        torch.onnx.export(transformer, outputs, "transformer_model.onnx", 
                        export_params=True,
                        input_names=['input'],
                        output_names=['output'],)