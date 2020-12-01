import torch.backends.cudnn as cudnn
from sklearn.metrics.ranking import roc_auc_score
import os
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision

DISEASES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

NUM_CLASSES = 14
TRANS_RESIZE = 256
TRANS_CROP = 224

class DenseNet121(nn.Module):
    def __init__(self, num_classes, trained):
        super(DenseNet121, self).__init__()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=trained)
        num_fc_kernels = self.dense_net_121.classifier.in_features
        self.dense_net_121.classifier = nn.Sequential(nn.Linear(num_fc_kernels, num_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.dense_net_121(x)
        return x


class CheXNet:
    def __init__(self, mode='test', checkpoint=None):
        if checkpoint is None:
            raise ValueError(1)

        self.nn_architecture = 'DENSE-NET-121'
        self.num_classes = NUM_CLASSES
        self.trans_resize = TRANS_RESIZE
        self.trans_crop = TRANS_CROP
        self.pre_checkpoint = checkpoint
        if mode == 'test':
            network_model = DenseNet121(self.num_classes, True).cpu()
            cudnn.benchmark = True

            network_model = torch.nn.DataParallel(network_model).cpu()
            model_checkpoint = torch.load(self.pre_checkpoint, map_location='cpu')
            network_model.load_state_dict(model_checkpoint['state_dict'])
            self.loaded_model = network_model
            self.loaded_model.eval()

    def compute_auroc(self, ground_truth, prediction):
        out_auroc = []
        np_ground_truth = ground_truth.cpu().numpy()
        np_prediction = prediction.cpu().numpy()
        for i in range(self.num_classes):
            # calculate the roc_auc_score of each class
            try:
                out_auroc.append(roc_auc_score(np_ground_truth[:, i], np_prediction[:, i]))
            except ValueError:
                out_auroc.append(np.NAN)
        return out_auroc

    def get_transform_sequence(self):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transform_list = list()
        transform_list.append(transforms.Resize(self.trans_resize))
        transform_list.append(transforms.TenCrop(self.trans_crop))
        transform_list.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop)
                                                                           for crop in crops])))
        transform_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transform_sequence = transforms.Compose(transform_list)
        return transform_sequence

    def test_one(self, file_names):
        transform_seq = self.get_transform_sequence()
        output_ground_truth = torch.FloatTensor().cpu()
        output_prediction = torch.FloatTensor().cpu()
        data_loader = get_img_file_dataloader(file_names, transform_seq)
        # data_loader = get_image_dataloader(batch_size=SAMPLE_DICT['Batch Size'], shuffle=False,
        #                                    num_workers=1, transform_seq=transform_seq)

        with torch.autograd.no_grad():
            for batch_index, (image, label) in enumerate(data_loader):
                label = label.cpu()
                output_ground_truth = torch.cat((output_ground_truth, label), 0)
                batch_size, n_crops, num_channels, height, width = image.size()
                var_image = torch.autograd.Variable(image.view(-1, num_channels, height, width).cpu())
                out = self.loaded_model(var_image)
                out_mean = out.view(batch_size, n_crops, -1).mean(1)
                output_prediction = torch.cat((output_prediction, out_mean.data), 0)

        return output_prediction
        print("output_prediction", output_prediction)
        print('')



class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file, transform=None):
        self.list_image_paths = []
        self.list_image_labels = []
        self.transform = transform
        with open(path_to_dataset_file, "r") as file_descriptor:
            lines = file_descriptor.readlines()
            for line in lines:
                line_items = line.split()
                image_path = os.path.join(path_to_img_dir, line_items[0])
                image_label = line_items[1:]
                image_label = [int(i) for i in image_label]
                self.list_image_paths.append(image_path)
                self.list_image_labels.append(image_label)

    def __getitem__(self, index):

        image_path = self.list_image_paths[index]
        image_data = Image.open(image_path).convert('RGB')
        image_label = torch.FloatTensor(self.list_image_labels[index])

        if self.transform:
            image_data = self.transform(image_data)

        return image_data, image_label

    def __len__(self):
        return len(self.list_image_paths)


def get_img_file_dataloader(file_names, transform_seq):
    import tempfile
    f_name = tempfile.mktemp()
    f_data = "\n".join(f + " 0" * 14 for f in file_names)
    f = open(f_name, "w")
    f.write(f_data)
    f.close()
    dataset = DatasetGenerator(path_to_img_dir=".",
                     path_to_dataset_file=f_name, transform=transform_seq)
    import os
    os.remove(f_name)
    return DataLoader(dataset=dataset, batch_size=len(file_names),
                                  shuffle=False, pin_memory=True)


def diagnose(files):
    chexnet = CheXNet(mode='test', checkpoint='model.pth')
    result = chexnet.test_one([files])
    return result


class HeatMapGenerator:
    def __init__(self):
        self.nn_architecture = 'DENSE-NET-121'
        self.pre_checkpoint = './model.pth'
        self.num_classes = NUM_CLASSES
        self.trans_crop = TRANS_CROP
        self.nn_is_pre_trained = True

        # initialize network model with pre-trained weights
        self.network_model = None
        if self.nn_architecture == 'DENSE-NET-121':
            self.network_model = DenseNet121(self.num_classes, self.nn_is_pre_trained)
        else:
            raise ValueError

        self.network_model = torch.nn.DataParallel(self.network_model)
        model_checkpoint = torch.load(self.pre_checkpoint, map_location='cpu')
        self.network_model.load_state_dict(model_checkpoint['state_dict'])

        # only need convolution layers' weights to generate heat map
        self.cam_model = self.network_model.module.dense_net_121.features
        self.cam_model.eval()  # set network as evaluation model without BN & Dropout
        self.network_model.eval()  # set network as evaluation model without BN & Dropout

        # CPU runtime, annotate below line to disable cuda
        # self.network_model.cuda()

        # select the last convolution layers' weights as weights of CAM
        self.weights = list(self.cam_model.parameters())[-2]

        # initialize the images transform sequence
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transform_list = list()
        transform_list.append(transforms.Resize(self.trans_crop))
        transform_list.append(transforms.ToTensor())
        transform_list.append(normalize)
        self.transform_sequence = transforms.Compose(transform_list)

    def generator(self, path_to_raw_image, path_to_output_image, trans_crop=None):
        if trans_crop is None:
            trans_crop = self.trans_crop

        # load image, transform, convert
        image_data = Image.open(path_to_raw_image).convert('RGB')
        image_data = self.transform_sequence(image_data)
        image_data = image_data.unsqueeze_(0)

        var_image = torch.autograd.Variable(image_data)

        with torch.autograd.no_grad():
            var_output = self.cam_model(var_image)
            var_prediction = self.network_model(var_image)

        # output predicted result
        predict_probability = np.max(var_prediction.data.numpy())
        predict_label = DISEASES[np.argmax(var_prediction.data.numpy())]
        print(predict_label, predict_probability)

        # start generating heat map
        heat_map = None
        for i in range(len(self.weights)):
            tmp_map = var_output[0, i, :, :]
            if i == 0:
                heat_map = self.weights[i] * tmp_map
            else:
                heat_map += self.weights[i] * tmp_map

        heat_map = heat_map.data.numpy()

        raw_img = cv2.imread(path_to_raw_image, 1)
        raw_img = cv2.resize(raw_img, (trans_crop, trans_crop))

        cam = heat_map / np.max(heat_map)
        cam = cv2.resize(cam, (trans_crop, trans_crop))
        heat_map = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_COOL)
        output_img = heat_map * 0.229 + raw_img
        cv2.imwrite(path_to_output_image, output_img)

