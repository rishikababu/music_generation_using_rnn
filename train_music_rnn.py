import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1.  Define the Model
class MusicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(MusicRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc1(output[:, -1, :])  # Take last time step
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output, hidden

    def init_hidden(self, batch_size):
        # Consistent with Keras/TensorFlow defaults
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size
        return (torch.zeros(num_layers, batch_size, hidden_size),
                torch.zeros(num_layers, batch_size, hidden_size))

def train_music_rnn():
    # Load notes
    with open('data/processed/notes.pkl', 'rb') as f:
        notes = pickle.load(f)

    # Flatten the notes list (since it's a list of lists)
    flat_notes = [note for sublist in notes for note in sublist]  # Flatten the list of lists

    # Now, we can safely create a set from the flattened list
    pitchnames = sorted(set(flat_notes))  # Get unique pitch names
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    n_vocab = len(pitchnames)

    # Prepare sequences
    sequence_length = 100
    network_input = []
    network_output = []

    # Ensure we are using the flattened notes list for sequences
    for i in range(0, len(flat_notes) - sequence_length):
        seq_in = flat_notes[i:i + sequence_length]
        seq_out = flat_notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    # Convert to PyTorch tensors
    network_input = torch.tensor(network_input, dtype=torch.long)  # Use torch.long for indices
    network_output = torch.tensor(network_output, dtype=torch.long)  # Use torch.long for indices

    # Define model parameters
    input_size = n_vocab
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    dropout_rate = 0.3
    output_size = n_vocab  # Output size is the vocabulary size

    # Create model
    model = MusicRNN(input_size, hidden_size, output_size, num_layers, dropout_rate)
    optimizer = optim.RMSprop(model.parameters())
    loss_fn = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification

    # Train model
    epochs = 10
    batch_size = 64
    model.train()  # Set the model to training mode

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(network_input), batch_size):
            batch_input = network_input[i:i + batch_size]
            batch_output = network_output[i:i + batch_size]

            # Zero the gradients
            optimizer.zero_grad()

            # Initialize hidden state
            hidden = model.init_hidden(batch_input.size(0))

            # Forward pass
            output, hidden = model(batch_input, hidden)

            # Compute loss
            loss = loss_fn(output, batch_output)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(network_input)}')  # Average loss for the epoch

    # Save the PyTorch model
    torch.save(model, 'music_rnn_model.pth')
    print("PyTorch model saved to music_rnn_model.pth")


if __name__ == "__main__":
    train_music_rnn()
