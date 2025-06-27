from flask import Flask, request, send_file, render_template
from music21 import note, stream, instrument
import random
import tempfile
import numpy as np
import torch
import torch.nn as nn

app = Flask(__name__)

# Define default NOTE_VOCAB and mappings globally
NOTE_VOCAB = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_TO_INDEX = {note: i for i, note in enumerate(NOTE_VOCAB)}
INDEX_TO_NOTE = {i: note for note, i in NOTE_TO_INDEX.items()}
SEQUENCE_LENGTH = 20  # Default sequence length

# Define the RNN model architecture (must match your trained model)
class MusicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MusicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output[:, -1, :])  # Take the output of the last time step
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

# Load the pre-trained PyTorch model
model = None  # Initialize model outside the try block
try:
    # Define model parameters (must match training)
    INPUT_SIZE = len(NOTE_VOCAB)
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = len(NOTE_VOCAB)
    NUM_LAYERS = 2

    loaded_model = MusicRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS) # MusicRNN is now defined
    loaded_model.load_state_dict(torch.load('D:\working_place\Music_Generator\models', map_location=torch.device('cpu'))) # Load to CPU if no GPU
    loaded_model.eval()
    model = loaded_model

    # If the loaded model has a specific vocabulary size, you might want to update
    # NOTE_VOCAB and the mappings here if that information is also saved.
    # For simplicity in this example, we'll stick to the default.

except Exception as e:
    print(f"Error loading PyTorch model: {e}")
    print("Falling back to random music generation.")

@app.route('/')
def index():
    return render_template('index.html')

def prepare_sequence(notes, sequence_length, note_to_index):
    sequence_in = notes[-sequence_length:]
    encoded_sequence = [note_to_index.get(n, 0) for n in sequence_in]
    while len(encoded_sequence) < sequence_length:
        encoded_sequence.insert(0, 0)
    return torch.tensor(encoded_sequence, dtype=torch.long).unsqueeze(0)

def generate_notes_from_model(model, seed_notes, num_notes, sequence_length, index_to_note, note_vocab):
    generated_notes = list(seed_notes)
    hidden = None

    with torch.no_grad():
        for _ in range(num_notes):
            input_sequence = prepare_sequence(generated_notes, sequence_length, NOTE_TO_INDEX)
            if model:
                if hidden is None:
                    hidden = model.init_hidden(1)

                output, hidden = model(input_sequence, hidden)
                probabilities = torch.softmax(output, dim=1)
                predicted_index = torch.argmax(probabilities).item()
                predicted_note = index_to_note.get(predicted_index, random.choice(NOTE_VOCAB))
                generated_notes.append(predicted_note)
            else:
                generated_notes.append(random.choice(NOTE_VOCAB))
    return generated_notes

@app.route('/generate', methods=['POST'])
def generate():
    try:
        input_data = request.form.get('data', '')
        input_type = request.form.get('type', 'text')
        instr_name = request.form.get('instrument', 'Piano')
        style = request.form.get('style', 'Classical')
        duration = int(request.form.get('duration', 10))

        melody = stream.Stream()

        # Add instrument
        if instr_name == "Violin":
            melody.append(instrument.Violin())
        elif instr_name == "Flute":
            melody.append(instrument.Flute())
        else:
            melody.append(instrument.Piano())

        # Generate notes using the RNN
        if input_type == 'notes':
            seed_notes = input_data.split()
            if not seed_notes:
                seed_notes = random.choices(NOTE_VOCAB, k=SEQUENCE_LENGTH)
        else:
            seed_notes = random.choices(NOTE_VOCAB, k=SEQUENCE_LENGTH)

        num_notes_to_generate = duration * 2
        generated_notes = generate_notes_from_model(model, seed_notes, num_notes_to_generate, SEQUENCE_LENGTH, INDEX_TO_NOTE, NOTE_VOCAB)

        for pitch in generated_notes:
            n = note.Note(pitch, quarterLength=0.5)
            melody.append(n)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_file:
            melody.write('midi', fp=tmp_file.name)
            tmp_file_path = tmp_file.name
        return send_file(tmp_file_path, mimetype='audio/midi', as_attachment=False, download_name='music.mid')

    except Exception as e:
        return f"Error generating music: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')