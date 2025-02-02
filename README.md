# Name: Swaraj Bhanja | Student Id: st125052

# Welcome to Machine Translation Language!

This is a web-based end-to-end application named Machine Translation Language. It leverages the power of deep learning and web development to provide a website that provides translation based on the input sentence.

# About the Deep Learning Model

The brains of this solution is the deep learning model trained for this purpose. The DL model was trained based on the HuggingFace **JParaCrawl-Filtered English-Japanese Parallel** corpus. The complete **JParaCrawl-Filtered English-Japanese Parallel** dataset contains pairs of English and Japanese sentences. The subset selects only the first 30,000 rows focusing on achieving a balance between manageability and effective training. Additionally, it removes three specific columns: "id", "model1_accepted", and "model2_accepted". These are mostly metadata fields that aren't required for the intended task. The final dataset contains only the English and Japanese text pairs, making it cleaner and more efficient for further processing. The **Transformer** technique was used to train the model using this dataset.

# Loading the Dataset

**English** is defined as the source language and **Japanese** is defined as the target language. Next, an empty list to store each entry as a tuple is created. Each entry is a tuple containing an English sentence and its corresponding Japanese translation from the dataset. Instead of using a pre-defined dataset structure, a custom dataset class which mimics a PyTorch dataset is created.

The dataset is then split into three parts: training, validation, and testing. The split ratio is 70% for training, 20% for validation and 10% for testing.

This structure provides a convenient way to access the different splits by their respective keys ('train', 'validation', and 'test'), facilitating further use in model training and evaluation.

# Tokenization

Tokenizers and vocabulary mappings for both English (source language) and Japanese (target language) are initialized. SpaCy, a natural language processing library, is used to tokenize text by splitting sentences into individual words or subwords. The tokenizers are assigned to the token_transform dictionary, where English text is processed using the en_core_web_sm tokenizer and Japanese text is processed using ja_core_news_sm.

The function yield_tokens generates tokenized text sequences. It takes in a dataset (data) and a target language (language), determines whether it should process English or Japanese, and then applies the appropriate tokenizer to extract tokens from each sentence. This function acts as a generator, meaning it efficiently processes large amounts of text without storing everything in memory.

Next, special indices are defined for handling out-of-vocabulary words and structuring sequences: UNK_IDX (unknown words) is 0, PAD_IDX (padding for equal-length sequences) is 1, SOS_IDX (start-of-sequence token) is 2, and EOS_IDX (end-of-sequence token) is 3. The list special_symbols contains these symbols as strings, which will later be added to the vocabulary to ensure the model understands the structure of tokenized sentences. These special symbols help in handling sequence boundaries and missing words effectively during training and inference.

# Numericalization

A vocabulary for both the source language (English) and the target language (Japanese) using the tokenized training data is created. It is basically a mapping of words (tokens) to numerical indices, allowing the model to process text as numbers rather than raw words.

For each language (source language and target language), the function build_vocab_from_iterator constructs a vocabulary by iterating through tokenized sentences from the training dataset. The min_freq=2 parameter ensures that only words appearing at least twice in the training data are included in the vocabulary—words occurring less frequently are replaced with the unknown token (`<unk>`). 

The special symbols (`<unk>`, `<pad>`, `<sos>`, `<eos>`) are also explicitly added, and special_first=True ensures that these symbols are placed at the beginning of the vocabulary.

After building the vocabulary for both languages, the set_default_index(UNK_IDX) method ensures that any word not found in the vocabulary will be mapped to `<unk>` (index 0) instead of raising an error. This prevents issues during model inference when encountering unseen words. Essentially, this step converts the raw tokenized text into a well-structured dictionary that the model can use to look up numerical representations of words.

# Preparing Data Using Batch Loader

The tokenized dataset is split into train, test and validation sets using a batch loader approach with the batch size as 12. This is essential since the DL model must use the train and validation datasets to ensure that it learns effectively instead of memorizing.

# About the various classes

A Seq2Seq Transformer model architecture for machine translation, consisting of an encoder, a decoder, multi-head attention mechanisms, position-wise feedforward layers, and masking techniques is implemented. It follows the Transformer architecture, which is widely used for tasks like neural machine translation.

The Encoder processes input sentences (e.g., English) by embedding words and adding positional encodings. Each EncoderLayer consists of a multi-head self-attention mechanism, which allows the model to focus on different words in a sentence simultaneously, and a feedforward layer that helps capture complex relationships between words. These layers are stacked multiple times to enhance learning.

The Decoder takes the encoder’s processed output and generates the translated sequence. Similar to the encoder, it has multi-head self-attention, but it also includes an encoder-decoder attention mechanism, which helps it focus on relevant parts of the source sentence while generating the translation. It uses a masking mechanism to prevent it from looking at future words when predicting the next word, ensuring an autoregressive output generation process.

The Multi-Head Attention Layer allows the model to attend to multiple parts of the input at once, improving translation quality by considering multiple interpretations of a word in context. The Positionwise Feedforward Layer applies transformations independently to each position in the sequence, allowing non-linear feature extraction.

The Seq2SeqTransformer class ensures that input padding tokens are ignored through source and target masks, prevents the decoder from peeking at future words, and sequentially processes the input through the encoder and decoder. Finally, the model outputs a translated sentence and the attention scores, which indicate which words in the source sentence were most relevant when generating each word in the target sentence.

# Training

This function trains the Transformer model for one epoch using teacher forcing, which means the correct previous word is provided as input while generating the next word during training. The function takes in the model, a data loader that provides batches of source (src) and target (trg) sentences, an optimizer for updating the model’s weights, a loss function (criterion) to measure performance, a gradient clipping value (clip) to prevent exploding gradients, and the total number of batches (loader_length) for computing the average loss.

The model is set to training mode (model.train()), ensuring that dropout and other training-specific behaviors are active. The total loss for the epoch is initialized to zero. For each batch, the source (src) and target (trg) sentences are moved to the GPU (or relevant device). Teacher forcing is applied, meaning the input to the model excludes the end-of-sequence (`<eos>`) token (trg[:, :-1]), while the target excludes the start-of-sequence (`<sos>`) token (trg[:, 1:]). This aligns the predicted sequence correctly with the expected output.

The model predicts the output sentence (output, _ = model(src, trg[:, :-1])). The output is reshaped to match the expected format for computing the loss: it is flattened so that each token's prediction is compared with the correct target token. The loss is computed using the criterion and backpropagation (loss.backward()) is performed to adjust the model’s weights. Gradient clipping (torch.nn.utils.clip_grad_norm_) prevents gradients from becoming too large, which stabilizes training. Finally, the optimizer updates the model parameters (optimizer.step()). The total loss for the epoch is accumulated and returned as the average loss per batch, helping to track model improvement over time.

## Testing

First, the input and target sentences are transformed into their respective tokenized representations using text_transform, which converts raw text into numerical indices using the previously built vocabulary. The source text (src_text) is tokenized from English, while the target text (trg_text) is tokenized from Japanese. Since the model expects batched inputs, both tensors are reshaped to (1, sequence_length) to simulate a batch of size 1.

The model is then set to evaluation mode (model.eval()), ensuring that dropout and other training behaviors are disabled. To prevent gradient computations (which are unnecessary for inference), the torch.no_grad() context manager is used. The model is run without teacher forcing, meaning it generates predictions based only on the input without being fed the correct target sequence.

The output is processed by removing the batch dimension (squeeze(0)) and ignoring the first token (which is typically `<sos>`). The model's predicted probabilities for each token are converted into actual word indices using argmax(1), selecting the most likely word at each step. Finally, the numerical indices are mapped back to Japanese characters/words using the vocabulary mapping (get_itos()), and the translated sentence is printed as a concatenated string.

This function effectively demonstrates the model's ability to translate an English sentence into Japanese by generating a prediction without using the correct target sequence as input, simulating real-world usage.

## Pickling The Model
The Transformer DL model was chosen for deployment.
> The pickled model was saved using a .pkl extension to be used later in a web-based application

# Website Creation
The model was then hosted over the Internet with Flask as the backend, HTML, CSS, JS as the front end, and Docker as the container. The end-user is presented with a UI wherein a search input box is present. Once the user types in the first set of words, they click on the `Translate Content` button. The input text is sent to the JS handler which makes an API call to the Flask backend. The Flask backend has the GET route which intercepts the HTTP request. The input text is then fed to the model to generate the predicted translation token. These predicted tokens are then returned back to the JS handler as a list by the Flask backend. The JS handler then appends each token in the received list into the result container's inner HTML and finally makes it visible for the output to be shown. 

A Vanilla architecture was chosen due to time constraints. In a more professional scenario, the ideal approach would be used frameworks like React, Angular and Vue for Frontend and ASP.NET with Flask or Django for Backend.

The following describes the key points of the hosting discussion.
> **1. DigitalOcean (Hosting Provider)**
> 
>> - **Role:** Hosting and Server Management
>> - **Droplet:** Hosts the website on a virtual server, where all files, databases, and applications reside.
>> - **Dockerized Container:** The website is hosted in a Dockerized container running on the droplet. The container is built over a Ubuntu Linux 24.10 image.
>> - **Ports and Flask App:** The Dockerized container is configured to host the website on port 8000. It forwards requests to port 5000, where the Flask app serves the backend and static files. This flask app contains the pickled model, which is used for prediction.
>> - **IP Address:** The droplet’s public IP address directs traffic to the server.
>
>  **In Summary:** DigitalOcean is responsible for hosting the website within a Dockerized container, ensuring it is online and accessible via its IP address.
> 
>  **2. GoDaddy (Domain Registrar)**
>
>> - **Role:** Domain Registration and Management
>> - **Domain Purchase:** Registers and manages the domain name.
>> - **DNS Management:** Initially provided DNS setup, allowing the domain to be pointed to the DigitalOcean droplet’s IP address.
> 
> **In Summary:** GoDaddy ensures the domain name is registered and correctly points to the website’s hosting server.
>
>  **3. Cloudflare (DNS and Security/Performance Optimization)**
>
>> - **Role:** DNS Management, Security, and Performance Optimization
>> - **DNS Management:** Resolves the domain to the correct IP address, directing traffic to the DigitalOcean droplet.
>> - **CDN and Security:** Caches website content globally, enhances performance, and provides security features like DDoS protection and SSL encryption.
> 
> **In Summary:** Cloudflare improves the website’s speed, security, and reliability.
>
> **How It Works Together:**
> 
>> - **Domain Resolution:** The domain is registered with GoDaddy, which points it to Cloudflare's DNS servers. Cloudflare resolves the domain to the DigitalOcean droplet's IP address.
>> - **Content Delivery:** Cloudflare may serve cached content or forward requests to DigitalOcean, which processes and serves the website content to users.
> 
> **Advantages of This Setup:**
>
>> - **Security:** Cloudflare provides DDoS protection, SSL/TLS encryption, and a web application firewall.
>> - **Performance:** Cloudflare’s CDN reduces load times by caching content globally, while DigitalOcean offers scalable hosting resources.
>> - **Reliability:** The combination of GoDaddy, Cloudflare, and DigitalOcean ensures the website is always accessible, with optimized DNS resolution and robust hosting.



# Demo
https://github.com/user-attachments/assets/52425f0f-caa3-4309-a8b1-1f9cb01dd3c4



# Access The Final Website
You can access the website [here](https://aitmltask.online). 

# Limitations
Note that currently, the solution supports slightly accurate translation on the given input. The model may generate gibberish for certain inputs and is a known limitation.


# How to Run the Language Model Docker Container Locally
### Step 1: Clone the Repository
> - First, clone the repository to your local machine.
### Step 2: Install Docker
> - If you don't have Docker installed, you can download and install it from the [Docker](https://www.docker.com) website.
### Step 3: Build and Run the Docker Container
Once Docker is installed, navigate to the app folder in the project directory. Delete the docker-compose-deployment.yml file and run the following commands to build and run the Docker container:
> - `docker compose up -d`

### Important Notes
> - The above commands will serve the Docker container on port 5000 and forward the requests to the Flask application running on port 5000 in the containerized environment.
> - Ensure Ports Are Free: Make sure that port 5000 is not already in use on your machine before running the container.
> - Changing Flask's Port: If you wish to change the port Flask runs on (currently set to 5000), you must update the port in the app.py file. After making the change, remember to rebuild the Docker image in the next step. Execute the following command to stop the process: `docker compose down`. Then goto Docker Desktop and delete the container and image from docker. 
