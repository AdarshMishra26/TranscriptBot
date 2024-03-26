# TranscriptBot

This is a Flask application that utilizes a chatbot to interact with YouTube video transcripts. Users can input a YouTube video URL, and the chatbot will generate responses based on the content of the video transcript.

## Features

- Fetches YouTube video transcripts using the YouTubeTranscriptApi library.
- Implements a chatbot capable of generating responses to user queries based on the provided transcript.
- Utilizes natural language processing (NLP) techniques such as tokenization, TF-IDF vectorization, and cosine similarity for generating relevant responses.
- Integrates a Generative AI model provided by Google GenAI for enhanced response generation.
- Provides a simple web interface for users to interact with the chatbot.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/AdarshMishra26/TranscriptBot.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up the configuration:

    - Create a `config.json` file in the project root directory with the following structure:

    ```json
    {
        "param": {
            "API_Key": "your_google_generativeai_api_key"
        }
    }
    ```

    Replace `"your_google_generativeai_api_key"` with your Google GenerativeAI API key.

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Access the application in your web browser at `http://localhost:5000`.

3. Enter a YouTube video URL and start chatting with the chatbot.


## Video Link 
- For checking the working of the project you can visit at
  ```bash
    https://drive.google.com/file/d/1AfrwrjXphI8zMTE80i8TA-49pno2KE9O/view?usp=drivesdk
    ```

## Development

- For local development, you can run the chatbot directly from the command line using the `TranscriptBot` class in `transcript_bot.py`. Use the `run_chatbot()` method to interact with the chatbot via the command line.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvement, please open an issue or submit a pull request.


