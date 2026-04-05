
import torch


def print_sep():
    print("=" * 50)

def choose_device():
    """ Проверяет доступность GPU """
    if torch.cuda.is_available():
        print(f"> GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda:0")
    else:
        print("> CPU")
        device = torch.device("cpu")
    return device

