# chatbot-api

This repository is about creating your own chatbot using an `intents.json` file!

## Train your model

You can directly use the `train.py` script in the `train` folder to train your model, as long as you have an `intents.json` file to provide.

You can define :
* `-fp` / `--filepath`: the path of your `intents.json` file.
* `-sp` / `--savepath`: your model folder. If it already exists, a new version of the model will be supplied. If not, it will be created and a model initiated
* `-bs` / `--batch-size`: your batch size
* `-ep` / `--epochs`: the number of epochs

```bash
train.py -fp my/intents/intents.json -sp my/model/folder -bs 16 -ep 100
```

The model is an IntentClassifier following that structure :

```
Sequential(
  (0): Linear(in_features=n_words, out_features=128, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.5, inplace=False)
  (3): Linear(in_features=128, out_features=64, bias=True)
  (4): ReLU()
  (5): Dropout(p=0.5, inplace=False)
  (6): Linear(in_features=64, out_features=n_tags, bias=True)
)
```

After training the model, you can see the results inside a graphs folder (where you launched the script). The results show the evolution of the training loss value through epochs.

## App

Tha app part is located in `flaskapp`. It needs two essential components to work:

* Your `intents.json` file used during training, in `utils/data`
* Your trained **model**, in `utils/model`, named `chatbot_model`

You can launch the app **locally** using the following command:

```bash
python app.py
```

The app accepts a POST request with a JSON content and replies with a JSON as a response :

```json
// POST
{
  "data": "Your message"
}

// Response
{
  "response": "ChatBot's response"
}
```

### Docker

Build the image and launch the container. Don't forget to replace the link where you're hosting your `intents.json` and your `chatbot_model` (Google Drive or Dropbox should be enough).

## Web

You can build a simple chatbot app with a web application (in progress). First you'll need to install the packages then launch the app. Open then `http://localhost:1234/` in your browser.

If you intend to build this app, run the build command and copy the `dist/` folder to your production server.

```bash
npm install #Install packages
npm start # Start the app and visit http://localhost:1234/

npm run build #If you want a production bundle (copy dist/ in your server)
```

### Issues

Trouble connecting the response on the web API and paste it in a test div.