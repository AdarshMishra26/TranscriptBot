from flask import Flask, render_template, request, jsonify
from transcript_bot import TranscriptBot

app = Flask(__name__)
bot = TranscriptBot()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_url = request.form['video_url']
        video_id = bot.get_video_id(video_url)
        transcript = bot.fetch_and_fit_transcript(video_id)
        bot.save_transcript_to_file(video_id)
        return render_template('chat.html')
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    transcript_file = open(f"{bot.get_video_id(request.form['video_url'])}_transcript.txt", 'r', encoding='utf-8').read()
    response = bot.generate_response(user_input, transcript_file)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
