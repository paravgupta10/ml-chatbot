import os
import json
import random
import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F

from datetime import datetime

nltk.download('punkt_tab')
nltk.download('wordnet')

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1=nn.Linear(input_size,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,output_size)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.5)


    def forward(self, x):

        x=self.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.relu(self.fc2(x))
        x=self.dropout(x)
        x=self.fc3(x)

        return x
    
class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        
        self.model = None
        self.intents_path=intents_path
        self.documents=[]
        self.vocabulary=[]
        self.intents=[]
        self.intents_responses={}
        self.function_mappings=function_mappings

        self.X=None
        self.y=None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer=nltk.WordNetLemmatizer()
        words=nltk.word_tokenize(text)
        words=[lemmatizer.lemmatize(word.lower()) for word in words] 

        return words
    
    def bag_of_words(self,words):
        return [1 if word in words else 0 for word in self.vocabulary]
    
    def parse_intents(self):
        lemmatizer=nltk.WordNetLemmatizer()


        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data=json.load(f)


            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']]=intent['responses']

                for pattern in intent['patterns']:
                    pattern_words=self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags=[]
        indices=[]

        for document in self.documents:
            words=document[0]
            bag=self.bag_of_words(words)

            intent_index=self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X=np.array(bags)
        self.y=np.array(indices)

    def train_model(self, epochs, batch_size, learning_rate):
        X_tensor=torch.tensor(self.X, dtype=torch.float32)
        y_tensor=torch.tensor(self.y, dtype=torch.long)

        dataset=TensorDataset(X_tensor, y_tensor)
        dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model=ChatbotModel(self.X.shape[1],len(self.intents))
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            running_loss=0.0

            for batch_X,batch_y in dataloader:
                optimizer.zero_grad()
                outputs=self.model(batch_X)
                loss=criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss

            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}')

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions=json.load(f)

            self.model=ChatbotModel(dimensions['input_size'], dimensions['output_size'])
            self.model.load_state_dict(torch.load(model_path))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor(bag, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)
            probabilities = F.softmax(predictions, dim=0)
            confidence, predicted_class_index = torch.max(probabilities, dim=0)

        if confidence.item() < 0.75:
            return "I'm not sure I understand. Could you rephrase?"

        predicted_intent = self.intents[predicted_class_index.item()]

        # Handle function mapping return
        if self.function_mappings and predicted_intent in self.function_mappings:
            response = self.function_mappings[predicted_intent]()
            return response if response else random.choice(
                self.intents_responses.get(predicted_intent, ["I'm not sure how to respond."])
            )

        # If no function mapping, return normal response
        return random.choice(self.intents_responses.get(predicted_intent, ["I'm not sure how to respond."]))



def get_news():
    return "Top headline: AI continues to revolutionize industries."

def get_weather():
    return "It's sunny with a high of 30°C."

def get_time():
    return f"Current time: {datetime.now().strftime('%H:%M:%S')}"

def get_stocks():
    stocks = ['APPL', 'META', 'NVDA', 'GS', 'MSFT']
    return f"Your stocks: {', '.join(random.sample(stocks, 3))}"


if __name__ == '__main__':
    assistant = ChatbotAssistant(
        'intents.json',
        function_mappings={
            'stocks': get_stocks,
            'time': get_time,
            'weather': get_weather,
            'news': get_news
        }
    )
    #assistant.parse_intents()
    #assistant.prepare_data()
    #assistant.train_model(batch_size=8, learning_rate=0.001, epochs=100)
    #assistant.save_model('chatbot_model.pth', 'dimensions.json')


    assistant = ChatbotAssistant('intents.json', function_mappings = {'stocks': get_stocks})
    assistant.parse_intents()
    assistant.load_model('chatbot_model.pth', 'dimensions.json')

    while True:
        message = input('Enter your message:')

        if message == '/quit':
            break

        print(assistant.process_message(message))

