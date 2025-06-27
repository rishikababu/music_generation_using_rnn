import os
import pickle
from music21 import converter, instrument, note, chord

def extract_notes(midi_folder='data/midi'):
    notes = []
    for file in os.listdir(midi_folder):
        if file.endswith(".mid"):
            midi_path = os.path.join(midi_folder, file)
            midi_stream = converter.parse(midi_path)
            notes_in_file = []

            for element in midi_stream.flat.notes:
                if isinstance(element, note.Note):
                    notes_in_file.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes_in_file.append('.'.join(str(n) for n in element.normalOrder))

            notes.append(notes_in_file)
    return notes

def save_notes(notes, output_file='data/processed/notes.pkl'):
    with open(output_file, 'wb') as f:
        pickle.dump(notes, f)

# Extract notes from MIDI files and save as pickle
notes = extract_notes()
save_notes(notes)
