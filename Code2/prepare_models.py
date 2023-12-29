import os
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torch.nn import functional as F


def prep_models(context_model='resnet18', body_model='resnet18', model_dir='C:/Users/Daisy/Python_code/Emotions/pythonProject2/Data/models'):
    # create the network architecture
    model_context = models.__dict__[context_model](num_classes=365)
    checkpoint = torch.load(os.path.join(model_dir,'%s_places365.pth.tar' % context_model), map_location=lambda storage,
                                                           loc: storage)  # model trained in GPU could be deployed in CPU machine like this!
    if context_model == 'densenet161':
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        state_dict = {str.replace(k, 'norm.', 'norm'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'conv.', 'conv'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'normweight', 'norm.weight'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'normrunning', 'norm.running'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'normbias', 'norm.bias'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'convweight', 'conv.weight'): v for k, v in state_dict.items()}
    else:
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
            'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
    model_context.load_state_dict(state_dict)
    model_context.eval()
    model_context.cpu()
    torch.save(model_context, os.path.join(model_dir, 'context_model' + '.pth'))

    print('completed preparing context model')

    model_body = models.__dict__[body_model](pretrained=True)
    model_body.cpu()
    torch.save(model_body, os.path.join(model_dir, 'body_model' + '.pth'))

    print('completed preparing body model')
    return model_context, model_body


if __name__ == '__main__':
    prep_models(model_dir='C:/Users/Daisy/Python_code/Emotions/pythonProject2/Data/models')