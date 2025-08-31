import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm


def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            inputs = data[0].to(device)
            feature = feature_extractor(inputs).cpu()
            all_features.append(feature)
    return torch.cat(all_features, dim=0)


def collect_concept_feature(data_loader: DataLoader, concept_extractor: nn.Module,
                             device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `concept_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        concept_extractor (torch.nn.Module): A concept extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{C}|`).
    """
    concept_extractor.eval()
    all_c_pred = []
    all_concepts = []
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            inputs = data[0].to(device)
            concepts = data[2].to(device)
            _, c_pred = concept_extractor(inputs)
            c_pred = c_pred.cpu()
            concepts = concepts.cpu()
            all_c_pred.append(c_pred)
            all_concepts.append(concepts)
    return torch.cat(all_c_pred, dim=0), torch.cat(all_concepts, dim=0)