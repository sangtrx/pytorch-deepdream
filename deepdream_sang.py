"""
    This file contains the implementation of the DeepDream algorithm.

    If you have problems understanding any parts of the code,
    go ahead and experiment with functions in the playground.py file.
"""

import os
import argparse
import shutil
import time


import numpy as np
import torch
import cv2 as cv


import utils.utils as utils
from utils.constants import *
import utils.video_utils as video_utils
from collections import namedtuple

# loss.backward(layer) <- original implementation did it like this it's equivalent to MSE(reduction='sum')/2
def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration):
    # Step 0: Feed forward pass
    out = model(input_tensor)

    # Step 1: Grab activations/feature maps of interest
    activations = [out[layer_id_to_use] for layer_id_to_use in layer_ids_to_use]

    # Step 2: Calculate loss over activations
    losses = []
    for layer_activation in activations:
        # Use torch.norm(torch.flatten(layer_activation), p) with p=2 for L2 loss and p=1 for L1 loss.
        # But I'll use the MSE as it works really good, I didn't notice any serious change when going to L1/L2.
        # using torch.zeros_like as if we wanted to make activations as small as possible but we'll do gradient ascent
        # and that will cause it to actually amplify whatever the network "sees" thus yielding the famous DeepDream look
        loss_component = torch.nn.MSELoss(reduction='mean')(layer_activation, torch.zeros_like(layer_activation))
        losses.append(loss_component)

    loss = torch.mean(torch.stack(losses))
    loss.backward()

    # Step 3: Process image gradients (smoothing + normalization)
    grad = input_tensor.grad.data

    # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
    # sigma is calculated using an arbitrary heuristic feel free to experiment
    sigma = ((iteration + 1) / config['num_gradient_ascent_iterations']) * 2.0 + config['smoothing_coefficient']
    smooth_grad = utils.CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well

    # Normalize the gradients (make them have mean = 0 and std = 1)
    # I didn't notice any big difference normalizing the mean as well - feel free to experiment
    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    smooth_grad = smooth_grad - g_mean
    smooth_grad = smooth_grad / g_std

    # Step 4: Update image using the calculated gradients (gradient ascent step)
    input_tensor.data += config['lr'] * smooth_grad

    # Step 5: Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
    input_tensor.grad.data.zero_()
    input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)


def deep_dream_static_image(config, img):
    model = utils.fetch_and_prepare_model(config['model_name'], config['pretrained_weights'], DEVICE)
    try:
        layer_ids_to_use = [model.layer_names.index(layer_name) for layer_name in config['layers_to_use']]
    except Exception as e:  # making sure you set the correct layer name for this specific model
        print(f'Invalid layer names {[layer_name for layer_name in config["layers_to_use"]]}.')
        print(f'Available layers for model {config["model_name"]} are {model.layer_names}.')
        return

    if img is None:  # load either the provided image or start from a pure noise image
        img_path = utils.parse_input_file(config['input'])
        # load a numpy, [0, 1] range, channel-last, RGB image
        img = utils.load_image(img_path, target_shape=config['img_width'])
        if config['use_noise']:
            shape = img.shape
            img = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32)

    img = utils.pre_process_numpy_img(img)
    base_shape = img.shape[:-1]  # save initial height and width

    # Note: simply rescaling the whole result (and not only details, see original implementation) gave me better results
    # Going from smaller to bigger resolution (from pyramid top to bottom)
    for pyramid_level in range(config['pyramid_size']):
        new_shape = utils.get_new_shape(config, base_shape, pyramid_level)
        img = cv.resize(img, (new_shape[1], new_shape[0]))
        input_tensor = utils.pytorch_input_adapter(img, DEVICE)

        for iteration in range(config['num_gradient_ascent_iterations']):
            h_shift, w_shift = np.random.randint(-config['spatial_shift_size'], config['spatial_shift_size'] + 1, 2)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift)

            gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration)

            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift, should_undo=True)

        img = utils.pytorch_output_adapter(input_tensor)

    return utils.post_process_numpy_img(img)


def deep_dream_video_ouroboros(config):
    """
    Feeds the output dreamed image back to the input and repeat

    Name etymology for nerds: https://en.wikipedia.org/wiki/Ouroboros

    """
    ts = time.time()
    assert any([config['input_name'].lower().endswith(img_ext) for img_ext in SUPPORTED_IMAGE_FORMATS]), \
        f'Expected an image, but got {config["input_name"]}. Supported image formats {SUPPORTED_IMAGE_FORMATS}.'

    utils.print_ouroboros_video_header(config)  # print some ouroboros-related metadata to the console

    img_path = utils.parse_input_file(config['input'])
    # load numpy, [0, 1] range, channel-last, RGB image
    # use_noise and consequently None value, will cause it to initialize the frame with uniform, [0, 1] range, noise
    frame = None if config['use_noise'] else utils.load_image(img_path, target_shape=config['img_width'])

    for frame_id in range(config['ouroboros_length']):
        print(f'Ouroboros iteration {frame_id+1}.')
        # Step 1: apply DeepDream and feed the last iteration's output to the input
        frame = deep_dream_static_image(config, frame)
        dump_path = utils.save_and_maybe_display_image(config, frame, name_modifier=frame_id)
        print(f'Saved ouroboros frame to: {os.path.relpath(dump_path)}\n')

        # Step 2: transform frame e.g. central zoom, spiral, etc.
        # Note: this part makes amplifies the psychodelic-like appearance
        frame = utils.transform_frame(config, frame)

    video_utils.create_video_from_intermediate_results(config)
    print(f'time elapsed = {time.time()-ts} seconds.')

class ResNet50(torch.nn.Module):

    def __init__(self, pretrained_weights, requires_grad=False, show_progress=False):
        super().__init__()
        if pretrained_weights == SupportedPretrainedWeights.IMAGENET.name:
            resnet50 = models.resnet50(pretrained=True, progress=show_progress).eval()
            
        elif pretrained_weights == SupportedPretrainedWeights.PLACES_365.name:
            resnet50 = models.resnet50(pretrained=False, progress=show_progress).eval()

            binary_name = 'resnet50_places365.pth.tar'
            resnet50_places365_binary_path = os.path.join(BINARIES_PATH, binary_name)

            if os.path.exists(resnet50_places365_binary_path):
                state_dict = torch.load(resnet50_places365_binary_path)['state_dict']
            else:
                binary_url = r'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
                print(f'Downloading {binary_name} from {binary_url} it may take some time.')
                download_url_to_file(binary_url, resnet50_places365_binary_path)
                print('Done downloading.')
                state_dict = torch.load(resnet50_places365_binary_path)['state_dict']

            new_state_dict = {}  # modify key names and make it compatible with current PyTorch model naming scheme
            for old_key in state_dict.keys():
                new_key = old_key[7:]
                new_state_dict[new_key] = state_dict[old_key]

            resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 365)
            resnet50.load_state_dict(new_state_dict, strict=True)
        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} model.')

        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        # 3
        self.layer10 = resnet50.layer1[0]
        self.layer11 = resnet50.layer1[1]
        self.layer12 = resnet50.layer1[2]

        # 4
        self.layer20 = resnet50.layer2[0]
        self.layer21 = resnet50.layer2[1]
        self.layer22 = resnet50.layer2[2]
        self.layer23 = resnet50.layer2[3]

        # 6
        self.layer30 = resnet50.layer3[0]
        self.layer31 = resnet50.layer3[1]
        self.layer32 = resnet50.layer3[2]
        self.layer33 = resnet50.layer3[3]
        self.layer34 = resnet50.layer3[4]
        self.layer35 = resnet50.layer3[5]

        # 3
        self.layer40 = resnet50.layer4[0]
        self.layer41 = resnet50.layer4[1]
        # self.layer42 = resnet50.layer4[2]

        # Go even deeper into ResNet's BottleNeck module for layer 42
        self.layer42_conv1 = resnet50.layer4[2].conv1
        self.layer42_bn1 = resnet50.layer4[2].bn1
        self.layer42_conv2 = resnet50.layer4[2].conv2
        self.layer42_bn2 = resnet50.layer4[2].bn2
        self.layer42_conv3 = resnet50.layer4[2].conv3
        self.layer42_bn3 = resnet50.layer4[2].bn3
        self.layer42_relu = resnet50.layer4[2].relu

        # Set these to False so that PyTorch won't be including them in its autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # Feel free to experiment with different layers
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer10(x)
        layer10 = x
        x = self.layer11(x)
        layer11 = x
        x = self.layer12(x)
        layer12 = x
        x = self.layer20(x)
        layer20 = x
        x = self.layer21(x)
        layer21 = x
        x = self.layer22(x)
        layer22 = x
        x = self.layer23(x)
        layer23 = x
        x = self.layer30(x)
        layer30 = x
        x = self.layer31(x)
        layer31 = x
        x = self.layer32(x)
        layer32 = x
        x = self.layer33(x)
        layer33 = x
        x = self.layer34(x)
        layer34 = x
        x = self.layer35(x)
        layer35 = x
        x = self.layer40(x)
        layer40 = x
        x = self.layer41(x)
        layer41 = x

        layer42_identity = layer41
        x = self.layer42_conv1(x)
        layer420 = x
        x = self.layer42_bn1(x)
        layer421 = x
        x = self.layer42_relu(x)
        layer422 = x
        x = self.layer42_conv2(x)
        layer423 = x
        x = self.layer42_bn2(x)
        layer424 = x
        x = self.layer42_relu(x)
        layer425 = x
        x = self.layer42_conv3(x)
        layer426 = x
        x = self.layer42_bn3(x)
        layer427 = x
        x += layer42_identity
        layer428 = x
        x = self.relu(x)
        layer429 = x

        # Feel free to experiment with different layers, layer35 is my favourite
        net_outputs = namedtuple("ResNet50Outputs", self.layer_names)
        # You can see the potential ambiguity arising here if we later want to reconstruct images purely from the filename
        out = net_outputs(layer10, layer23, layer34, layer40)
        return out

def deep_dream_video(config):
    video_path = utils.parse_input_file(config['input'])
    tmp_input_dir = os.path.join(OUT_VIDEOS_PATH, 'tmp_input')
    tmp_output_dir = os.path.join(OUT_VIDEOS_PATH, 'tmp_out')
    config['dump_dir'] = tmp_output_dir
    os.makedirs(tmp_input_dir, exist_ok=True)
    os.makedirs(tmp_output_dir, exist_ok=True)

    metadata = video_utils.extract_frames(video_path, tmp_input_dir)
    config['fps'] = metadata['fps']
    utils.print_deep_dream_video_header(config)

    last_img = None


    config['img_width'] = 960
    for frame_id, frame_name in enumerate(sorted(os.listdir(tmp_input_dir))):


        # Step 1: load the video frame
        print(f'Processing frame {frame_id}')
        frame_path = os.path.join(tmp_input_dir, frame_name)
        frame = utils.load_image(frame_path, target_shape=config['img_width'])

        # Step 2: potentially blend it with the last frame
        if config['blend'] is not None and last_img is not None:
            # blend: 1.0 - use the current frame, 0.0 - use the last frame, everything in between will blend the two
            frame = utils.linear_blend(last_img, frame, config['blend'])

        if frame_id < 35:
            last_img = frame
            dreamed_frame = frame
            dump_path = utils.save_and_maybe_display_image(config, dreamed_frame, name_modifier=frame_id)
            print(f'Saved DeepDream frame to: {os.path.relpath(dump_path)}\n')
            continue

        # Step 3: Send the blended frame to some good old DeepDreaming
        if frame_id in range(35,44):
            factor = 0
        elif frame_id in range(44,55):
            factor = 1
        elif frame_id in range(55,65):
            factor = 2
        elif frame_id in range(65,75):
            factor = 3
        elif frame_id in range(75,85):
            factor = 4
        else:
            factor = 5

        config['model_name'] = SupportedModels.RESNET50.name
        config['pretrained_weights'] = SupportedPretrainedWeights.PLACES_365.name
        config['layers_to_use'] = ['layer3']  # layer34 was used
        config['pyramid_size'] = 1 + factor #1-4
        config['pyramid_ratio'] = 1.5 + 0.1*factor#1.8
        config['num_gradient_ascent_iterations'] = 7 + factor#10
        config['lr'] = 0.09
        config['spatial_shift_size'] = 30 + factor*2#40

        dreamed_frame = deep_dream_static_image(config, frame)

        # Step 4: save the frame and keep the reference
        last_img = dreamed_frame
        dump_path = utils.save_and_maybe_display_image(config, dreamed_frame, name_modifier=frame_id)
        print(f'Saved DeepDream frame to: {os.path.relpath(dump_path)}\n')

    video_utils.create_video_from_intermediate_results(config)

    shutil.rmtree(tmp_input_dir)  # remove tmp files
    print(f'Deleted tmp frame dump directory {tmp_input_dir}.')


if __name__ == "__main__":

    # Only a small subset is exposed by design to avoid cluttering
    parser = argparse.ArgumentParser()

    # Common params
    parser.add_argument("--input", type=str, help="Input IMAGE or VIDEO name that will be used for dreaming", default='figures.jpg')
    parser.add_argument("--img_width", type=int, help="Resize input image to this width", default=600)
    parser.add_argument("--layers_to_use", type=str, nargs='+', help="Layer whose activations we should maximize while dreaming", default=['relu3_3'])
    parser.add_argument("--model_name", choices=[m.name for m in SupportedModels],
                        help="Neural network (model) to use for dreaming", default=SupportedModels.VGG16_EXPERIMENTAL.name)
    parser.add_argument("--pretrained_weights", choices=[pw.name for pw in SupportedPretrainedWeights],
                        help="Pretrained weights to use for the above model", default=SupportedPretrainedWeights.IMAGENET.name)

    # Main params for experimentation (especially pyramid_size and pyramid_ratio)
    parser.add_argument("--pyramid_size", type=int, help="Number of images in an image pyramid", default=4)
    parser.add_argument("--pyramid_ratio", type=float, help="Ratio of image sizes in the pyramid", default=1.8)
    parser.add_argument("--num_gradient_ascent_iterations", type=int, help="Number of gradient ascent iterations", default=10)
    parser.add_argument("--lr", type=float, help="Learning rate i.e. step size in gradient ascent", default=0.09)

    # deep_dream_video_ouroboros specific arguments (ignore for other 2 functions)
    parser.add_argument("--create_ouroboros", action='store_true', help="Create Ouroboros video (default False)")
    parser.add_argument("--ouroboros_length", type=int, help="Number of video frames in ouroboros video", default=30)
    parser.add_argument("--fps", type=int, help="Number of frames per second", default=30)
    parser.add_argument("--frame_transform", choices=[t.name for t in TRANSFORMS],
                        help="Transform used to transform the output frame and feed it back to the network input",
                        default=TRANSFORMS.ZOOM_ROTATE.name)

    # deep_dream_video specific arguments (ignore for other 2 functions)
    parser.add_argument("--blend", type=float, help="Blend coefficient for video creation", default=0.85)

    # You usually won't need to change these as often
    parser.add_argument("--should_display", action='store_true', help="Display intermediate dreaming results (default False)")
    parser.add_argument("--spatial_shift_size", type=int, help='Number of pixels to randomly shift image before grad ascent', default=32)
    parser.add_argument("--smoothing_coefficient", type=float, help='Directly controls standard deviation for gradient smoothing', default=0.5)
    parser.add_argument("--use_noise", action='store_true', help="Use noise as a starting point instead of input image (default False)")
    args = parser.parse_args()

    # Wrapping configuration into a dictionary
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config['dump_dir'] = OUT_VIDEOS_PATH if config['create_ouroboros'] else OUT_IMAGES_PATH
    config['dump_dir'] = os.path.join(config['dump_dir'], f'{config["model_name"]}_{config["pretrained_weights"]}')
    config['input_name'] = os.path.basename(config['input'])

    # Create Ouroboros video (feeding neural network's output to it's input)
    if config['create_ouroboros']:
        deep_dream_video_ouroboros(config)

    # Create a blended DeepDream video
    elif any([config['input_name'].lower().endswith(video_ext) for video_ext in SUPPORTED_VIDEO_FORMATS]):  # only support mp4 atm
        deep_dream_video(config)

    else:  # Create a static DeepDream image
        print('Dreaming started!')
        img = deep_dream_static_image(config, img=None)  # img=None -> will be loaded inside of deep_dream_static_image
        dump_path = utils.save_and_maybe_display_image(config, img)
        print(f'Saved DeepDream static image to: {os.path.relpath(dump_path)}\n')

