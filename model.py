import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
            
        # Declare the embedding layer
        self.embedding_generator = nn.Embedding(vocab_size, embed_size)
        
        # Declaring the LSTM layer
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2)
        
        # Declaring the fully connected linear output layer
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
    
    def forward(self, features, raw_captions):
        # Extract features and embedding
        captions = raw_captions[:,:-1]
        embedding = self.embedding_generator(captions)
        features_unsqueezed = features.unsqueeze(dim = 1)
        
        # Concatenating features and caption embeddings
        torch_embeddings = torch.cat((features_unsqueezed, embedding), dim = 1)
        
        # Extracting lstm output
        lstm_output, hidden = self.lstm(torch_embeddings)
        
        # Extract prediction from lstm output using the linear layer
        prediction = self.linear(lstm_output)
        
        return prediction
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_sentence = []
        for _ in range(max_len):
            # Extracts the lstm outpt
            lstm_output, states = self.lstm(inputs, states)

            # Extracting the highest score prediction
            tensor_output = self.linear(lstm_output)
            highest_score_prediction = torch.argmax(tensor_output, dim=-1)
            
            # Parsing highest score prediction from torch to scalar
            prediction = int(highest_score_prediction.cpu().detach().numpy()[0][0])
            
            # Appends prediction to sentence
            predicted_sentence.append(prediction)
            
            # Loop until you find the end of the sentence, defined as 1 in the word dictionary 
            # (this was added as my samples had <end> several times)
            if prediction == 1:
                break
            
            # Generates the following input
            inputs = self.embedding_generator(highest_score_prediction)
        
        
        return predicted_sentence
 