import torch
from utils.dataset import IntentLoader
from utils.model import IntentClassifier
import nltk
from nltk.stem import WordNetLemmatizer
import random

class ChatbotModel:
    def _load_model_cpu(self, fp, n_classes, n_words):
        cpu_device =  torch.device("cpu")
        model = IntentClassifier(n_words, n_classes)
        model.load_state_dict(torch.load(fp, map_location=cpu_device))
        model.eval()
        return model
        

    def __init__(self, model_path='models/chatbot_model', intents_path="data/intents.json"):
        self.il = IntentLoader(intents_path)
        self.il.fit()
        self.model = self._load_model_cpu(model_path, len(self.il.classes), len(self.il.words))


    def clean_sentence(self, sentence):
        s_words = nltk.word_tokenize(sentence)
        s_words = [self.il.lemmatizer.lemmatize(word.lower()) for word in s_words]
        return s_words


    def bow(self, sentence, verbose=False):
        s_words = self.clean_sentence(sentence)
        bag = [0]*len(self.il.words)
        for s in s_words:
            for i, word in enumerate(self.il.words):
                if word == s: 
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if verbose:
                        print (f"found in bag: {word}")
        return torch.Tensor(bag) 

    @torch.no_grad()
    def predict_class(self, sentence):
        # filter below  threshold predictions
        p = self.bow(sentence)
        output = self.model(p)
        sort_preds, sort_idx = output.sort(descending=True)
        sort_classes = [self.il.classes[i] for i in sort_idx.tolist()]
        res_dict = {label:p for label, p in zip(sort_classes, sort_preds.tolist())}
        return res_dict, self.il.classes[output.argmax().item()]

    def generate_response(self, sentence, as_dict=False):
        response = "Sorry I didn't udnerstand your request."
        _, best_tag =self.predict_class(sentence)
        for intent in self.il.intents['intents']:
            if best_tag==intent['tag']:
                response = random.choice(intent['responses'])
                break
        if as_dict:
            return {"response": response}
        return response

    def chat(self):
        while True:
            message = input("Please enter your message: ")
            response = self.generate_response(message)
            print(response)
