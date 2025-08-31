"""
https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/alignment/dann.py
"""
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.domain_adversarial_network import WarmStartGradientReverseLayer

from tllib.utils.metric import binary_accuracy, accuracy
import random

__all__ = ['DomainAdversarialLoss', 'Classifier', 'ImageClassifier']


class DomainAdversarialLoss(nn.Module):
    """
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(f_j^t)].

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.

    Examples::

        >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional = None, sigmoid=True):
        super(DomainAdversarialLoss, self).__init__() 
        # self.grl = WarmStartGradientReverseLayer(alpha=0.1, lo=0., hi=0.5, max_iters=3000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        ## change f to concept
        
        # f = self.grl(torch.cat((f_s, f_t), dim=0))
        f = torch.cat((f_s, f_t), dim=0)  # Concatenate features without gradient reversal
        d = self.domain_discriminator(f)
        if self.sigmoid:
            d_s, d_t = d.chunk(2, dim=0)
            d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
            d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
            
            self.domain_discriminator_accuracy = 0.5 * (
                        binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

            if w_s is None:
                w_s = torch.ones_like(d_label_s)
            if w_t is None:
                w_t = torch.ones_like(d_label_t)
            return 0.5 * (
                F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) +
                F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_t), reduction=self.reduction)
            )
        else:
            d_label = torch.cat((
                torch.ones((f_s.size(0),)).to(f_s.device),
                torch.zeros((f_t.size(0),)).to(f_t.device),
            )).long()
            if w_s is None:
                w_s = torch.ones((f_s.size(0),)).to(f_s.device)
            if w_t is None:
                w_t = torch.ones((f_t.size(0),)).to(f_t.device)
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)

class Classifier(nn.Module):
    """A generic Classifier class for domain adaptation.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None, 
                 binary=False, concept_emb_dim=512):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.bool = binary
        self.bottleneck_dim = bottleneck_dim  # number of concepts
        self.concept_emb_dim = concept_emb_dim # dimension of concept embedding
        self._features_dim = backbone.out_features
        # self.using_bottleneck = True
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            print("Using no bottleneck layer")
            # self.using_bottleneck = False
            self.bottleneck = nn.Identity()
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            print(f"Using bottleneck layer with concept dimension {bottleneck_dim}")

        if head is None:
            # if bottleneck is None, using features, else using concepts to make prediction
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head
        self.finetune = finetune
            
    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        # mapping to feature space (bs, 2048)
        f = self.pool_layer(self.backbone(x))
        print("feature space shape: ", f.shape) 
        bottleneck_output = self.bottleneck(f)
        print("concept space shape: ", bottleneck_output.shape)
        predictions = self.head(bottleneck_output)
        print("prediction shape: ", predictions.shape)
        if self.training:
            return predictions, bottleneck_output
        else:
            return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params
    

class ImageClassifier(Classifier):
    """An Image Classifier with concept embedding and classification support."""

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 112, 
                 concept_emb_dim: int = 512, **kwargs):
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck_dim=bottleneck_dim, **kwargs)
        """
        Args:
            backbone (nn.Module): Backbone model to extract feature space.
            num_classes (int): Number of output classes.
            bottleneck_dim (int): Number of concepts.
            concept_emb_dim (int): Dimension of each concept embedding.
        """
        self.backbone = backbone
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        self.concept_emb_dim = concept_emb_dim

        self.embedding_generator = nn.ModuleList([
            nn.Linear(backbone.out_features, concept_emb_dim * 2)
            for _ in range(bottleneck_dim)
        ])


        self.p_int_layer = nn.ModuleList([
            nn.Linear(concept_emb_dim * 2, 1) for _ in range(bottleneck_dim)
        ])

        # Head for final classification
        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim * concept_emb_dim, num_classes),
        )

        # # Softmax to ensure `p_int` probabilities
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            final_preds: Final predictions of the model for classification. [batch_size, num_classes]
            flattened_concept_embs: Flattened mixed embeddings for all concepts. [batch_size, bottleneck_dim * concept_emb_dim]
            c_pred: Predicted probabilities (p_int values) for each concept. [batch_size, bottleneck_dim]
        """
        batch_size = x.size(0)

        # Extract feature space
        feature_space = self.pool_layer(self.backbone(x))  # Shape: [batch_size, backbone_out_features]

        # Generate positive and negative embeddings
        embeddings = torch.stack([fc(feature_space) for fc in self.embedding_generator], dim=1)  # [batch_size, bottleneck_dim, concept_emb_dim * 2]
        positive_embs, negative_embs = torch.split(embeddings, self.concept_emb_dim, dim=-1)
        p_int_logits = torch.stack([layer(embeddings[:, i]) for i, layer in enumerate(self.p_int_layer)], dim=1)  # [batch_size, bottleneck_dim, 1]
        
        p_int_pos = self.sigmoid(p_int_logits).squeeze(-1)  # Apply sigmoid to get probabilities [batch_size, bottleneck_dim]

        # Compute `p_int_neg` as `1 - p_int_pos`
        p_int_neg = 1 - p_int_pos  # [batch_size, bottleneck_dim]

        # Mix embeddings based on `p_int_pos` and `p_int_neg`
        mixed_embs = (
            p_int_pos.unsqueeze(-1) * positive_embs +
            p_int_neg.unsqueeze(-1) * negative_embs
        )  # [batch_size, bottleneck_dim, concept_emb_dim]

        # Flatten mixed embeddings for final classification
        flattened_concept_embs = mixed_embs.view(mixed_embs.size(0), -1)  # [batch_size, bottleneck_dim * concept_emb_dim]

        # Final prediction
        final_preds = self.head(flattened_concept_embs)  # [batch_size, num_classes]

        # `c_pred` is equal to the `p_int_pos` values for each concept
        c_pred = p_int_pos  # [batch_size, bottleneck_dim]

        if self.training:
            return final_preds, flattened_concept_embs, c_pred
        else:
            return final_preds, c_pred
    

    def get_parameters(self, base_lr=1.0) -> list:
        """A parameter list which decides optimization hyper-parameters."""
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr},
            {"params": self.embedding_generator.parameters(), "lr": base_lr},
            {"params": self.p_int_layer.parameters(), "lr": base_lr},
            {"params": self.head.parameters(), "lr": base_lr},
        ]
        return params
    