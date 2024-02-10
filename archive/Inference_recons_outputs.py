
import torch
import torch.nn.functional as F
from models import Create_nets
from datasets import Get_dataloader
from options import TrainOptions
from torchvision import models
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = TrainOptions().parse()
torch.manual_seed(args.seed)

## laod a saved model to inspect 

category= "cable"
SAVE_PATH='./inspects'

## Load Pretrained  Trafo model
transformer = Create_nets(args).to(device)
checkpoint = torch.load(f'./Exp0-r18-{category}/saved_models/checkpoint.pth')
transformer.load_state_dict(checkpoint['transformer'])
#print(transformer)

# Backbone hooks
backbone = models.resnet18(pretrained=True).to(device)
backbone.eval()
outputs = []
def hook(module, input, output):
    outputs.append(output)
backbone.layer1[-1].register_forward_hook(hook)
backbone.layer2[-1].register_forward_hook(hook)
backbone.layer3[-1].register_forward_hook(hook)

# train tes loader
train_dataloader, test_dataloader = Get_dataloader(args)

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
for i,(name ,batch) in enumerate(train_dataloader): # test (name ,batch, ground_truth, gt) , train (filename, batch)
    
    if i == 2:
        break  
    print(name)                    
    with torch.no_grad():
        inputs = batch.to(device)
        print(inputs.shape)
        print(inputs.type())
        outputs = []
        _ = backbone(inputs)     
        print(outputs[0].shape)
        print(outputs[1].shape)
        print(outputs[2].shape)
        ## with the hook funciton in the backbone the feature maps of the backbone are put in the list outputs
        outputs = embedding_concat(embedding_concat(outputs[0],outputs[1]),outputs[2])
        recon, std = transformer(outputs)
        
        # torch.save(outputs, os.path.join(SAVE_PATH,f'outputs_train_{category}.pth'))
        # torch.save(recon,   os.path.join(SAVE_PATH,f'recons_train_{category}.pth'))
        
        
for i,(name ,batch, ground_truth, gt) in enumerate(test_dataloader): # test (name ,batch, ground_truth, gt) , train (filename, batch)
    if i == 1:
        break  
    print(name)
                    
    with torch.no_grad():

        inputs = batch.to(device)
        outputs = []
        _ = backbone(inputs)     
        print(outputs[0].shape)
        print(outputs[1].shape)
        print(outputs[2].shape)
        ## with the hook funciton in the backbone the feature maps of the backbone are put in the list outputs
        outputs = embedding_concat(embedding_concat(outputs[0],outputs[1]),outputs[2])
        recon, std = transformer(outputs)
        # torch.save(outputs, os.path.join(SAVE_PATH,f'outputs_test_{category}.pth'))
        # torch.save(recon,   os.path.join(SAVE_PATH,f'recons_test_{category}.pth'))
        
        #### uncommetn to  save onnx model
        # torch.onnx.export(transformer, outputs, "transformer_model.onnx", 
        #                 export_params=True,
        #                 input_names=['input'],
        #                 output_names=['output'],)