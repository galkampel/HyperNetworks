
import torch
from model.nmp_edge import NMPEdge
import os
import numpy as np
import argparse


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(description="Test parameters")
    parser.add_argument("--data_folder", type=str, default="data", choices=["data"])
    parser.add_argument("--data_filename", type=str)
    parser.add_argument("--target", type=int, default=7, choices=range(12))
    # parser.add_argument("--gpu_device", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--model_folder", type=str, default="checkpoint")
    parser.add_argument("--model_filename", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="NMPEdge")
    return parser.parse_args(arg_list)


class TestPretrained:
    def __init__(self, model_name, target):
        self.model_name = model_name
        self.target = target
        self.model = None
        self.device = None

    def init_model(self):
        model = None
        if self.model_name == "NMPEdge":
            model = NMPEdge()
        return model

    def predict(self, data_loader):
        maes = []
        with torch.no_grad():
            self.model.eval()
            for data_batch in data_loader:
                data_batch = data_batch.to(self.device)
                pred = self.model(data_batch.z, data_batch.pos, data_batch.batch)
                maes.append((pred.view(-1) - data_batch.y[:, self.target]).abs().cpu().numpy())
        self.model.train()
        mae = np.concatenate(maes).mean()
        return mae

    def load_testloader(self, data_folder, data_filename):
        path_testloader = os.path.join(data_folder, data_filename)
        test_loader = torch.load(path_testloader)
        return test_loader

    def load_model(self, folder_name, filename):
        self.model = self.init_model()
        path_pretrained_model = os.path.join(folder_name, f'{filename}.pth')
        pretrained_params = torch.load(path_pretrained_model)
        self.model.load_state_dict(pretrained_params['model_state_dict'])
        self.device = pretrained_params['device']
        self.model = self.model.to(self.device)
        # self.optimizer.load_state_dict(pretrained_params['optimizer_state_dict']) # remove?
        # self.start_iter = pretrained_params['iteration'] # remove?
        # self.is_best_model = pretrained_params['is_best_model'] # remove?


def test_pretrained_model(args):
    test_pretrained = TestPretrained(args.model_name, args.target)
    test_pretrained.load_model(args.model_folder, args.model_filename)
    test_loader = test_pretrained.load_testloader(args.data_folder, args.data_filename)
    mae = test_pretrained.predict(test_loader)
    print(f"{args.model_name}\tuse_hypernetworks={'hypernet' in args.model_filename}\ttarget={args.target}")
    print(f'Test set MAE = {mae}')


if __name__ == "__main__":
    arguments = get_arguments()
    test_pretrained_model(arguments)
