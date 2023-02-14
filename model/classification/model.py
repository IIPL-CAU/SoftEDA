# Standard Library Modules
import os
import sys
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
# Huggingface Modules
from transformers import AutoConfig, AutoModel
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import get_huggingface_model_name

class ClassificationModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ClassificationModel, self).__init__()
        self.args = args

        if args.model_type == 'cnn':
            each_out_size = args.embed_size // 3

            self.embed = nn.Embedding(args.vocab_size, args.embed_size)
            self.conv = nn.ModuleList(
                [nn.Conv1d(in_channels=args.embed_size, out_channels=each_out_size,
                           kernel_size=kernel_size, stride=1, padding='same', bias=False)
                           for kernel_size in [3, 4, 5]]
            )

            self.classifier = nn.Sequential(
                nn.Linear(each_out_size * 3, args.hidden_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU(),
                nn.Linear(args.hidden_size, args.num_classes)
            )
        elif args.model_type == 'lstm':
            self.embed = nn.Embedding(args.vocab_size, args.embed_size)
            self.lstm = nn.LSTM(input_size=args.embed_size, hidden_size=args.hidden_size,
                                num_layers=2, batch_first=True, bidirectional=True)

            self.classifier = nn.Sequential(
                nn.Linear(args.hidden_size * 2 * 2, args.hidden_size), # *2 for bidirectional, *2 for num_layers
                nn.Dropout(args.dropout_rate),
                nn.GELU(),
                nn.Linear(args.hidden_size, args.num_classes)
            )
        else: # Huggingface models
            # Define model
            huggingface_model_name = get_huggingface_model_name(self.args.model_type)
            self.config = AutoConfig.from_pretrained(huggingface_model_name)
            if args.model_ispretrained:
                self.model = AutoModel.from_pretrained(huggingface_model_name)
            else:
                self.model = AutoModel.from_config(self.config)

            self.embed_size = self.model.config.hidden_size
            self.hidden_size = self.model.config.hidden_size
            self.num_classes = self.args.num_classes

            # Define classifier - As we need to train the model using soft labels, we need to define a custom classifier rather than using BERTforSequenceClassification
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Dropout(self.args.dropout_rate),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.num_classes),
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        if self.args.model_type == 'cnn':
            embed = self.embed(input_ids)
            embed = embed.transpose(1, 2) # (batch_size, embed_size, seq_len)

            conv_output = [conv(embed) for conv in self.conv] # [(batch_size, each_out_size, seq_len), ...]
            # Apply global max pooling to each conv output
            conv_output = [torch.max(conv, dim=2)[0] for conv in conv_output] # [(batch_size, each_out_size), ...]
            conv_output = torch.cat(conv_output, dim=1) # (batch_size, each_out_size * 3)

            classification_logits = self.classifier(conv_output) # (batch_size, num_classes)
        elif self.args.model_type == 'lstm':
            embed = self.embed(input_ids) # (batch_size, seq_len, embed_size)

            lstm_output, (lstm_hidden, lstm_cell) = self.lstm(embed) # (batch_size, seq_len, hidden_size * 2), (2*2, batch_size, hidden_size), (2*2, batch_size, hidden_size)

            # Flatten lstm_hidden
            lstm_hidden = lstm_hidden.transpose(0, 1).contiguous() # (batch_size, 2*2, hidden_size)
            lstm_hidden = lstm_hidden.view(lstm_hidden.size(0), -1) # (batch_size, 2*2*hidden_size)

            classification_logits = self.classifier(lstm_hidden) # (batch_size, num_classes)
        else: # Huggingface models
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
            cls_output = model_output.last_hidden_state[:, 0, :] # (batch_size, hidden_size)
            classification_logits = self.classifier(cls_output) # (batch_size, num_classes)

        return classification_logits
