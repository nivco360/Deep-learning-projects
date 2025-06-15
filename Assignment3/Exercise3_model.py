import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb


class Generate_model(nn.Module):

    def __init__(self, input_size, hidden_size, vocabulary_size):
        super(Generate_model, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers=3,
                          dropout=0.2,
                          batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocabulary_size)
        )

    def forward(self, sequence_input, hidden_state, hidden_bool=False):

        sequence_input = sequence_input.to(torch.float32)
        output, hidden_state = self.gru(sequence_input, hidden_state)
        linear_output = self.linear(output)

        return (linear_output, hidden_state) if hidden_bool else linear_output


def train_model(model, train_loader, optimizer, fn_loss, device):
    model.train()
    epoch_train_loss = 0.0

    for lyrics, words in train_loader:
        lyrics, words = lyrics.to(device), words.to(device)
        hidden_state = None
        current_song_loss = 0.0
        sequence_length = lyrics.size(1)

        # teacher forcing
        optimizer.zero_grad()  # reset gradients
        for word_idx in range(sequence_length - 1):
            current_input = lyrics[:, word_idx,:]
            word_predictions, hidden_state = model(current_input, hidden_state, hidden_bool=True)
            current_song_loss += fn_loss(word_predictions, words[:, word_idx + 1])                     

        # backward
        current_song_loss.backward()
        optimizer.step()
        epoch_train_loss += current_song_loss.item() / sequence_length

    return epoch_train_loss / len(train_loader)


def val_model(model, train_loader, val_loader, fn_loss, device):
    model.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():
        for lyrics, words in val_loader:
            lyrics, words = lyrics.to(device), words.to(device)
            hidden_state = None
            sequence_length = lyrics.size(1)

            current_input = lyrics[:, 0]

            for word_idx in range(1, sequence_length):

                word_predictions, hidden_state = model(current_input, hidden_state, hidden_bool=True)  
                epoch_val_loss += fn_loss(word_predictions, words[:, word_idx]).item() / sequence_length

                word_distribution = F.softmax(word_predictions, dim=-1)
                sampled_word_idx = torch.multinomial(word_distribution, num_samples=1).item()

                sampled_word_vector = train_loader.dataset._get_lyric_features(
                    train_loader.dataset.index_to_word[sampled_word_idx])[0].to(device)
                
                if sampled_word_vector.size(0) == 0:  # If empty, replace with a zero vector, no embedding for the word
                    sampled_word_vector = torch.zeros((lyrics.size(0), 300), device=device)

                midi_features = lyrics[:, word_idx, 300:]
                current_input = torch.cat([sampled_word_vector, midi_features], dim=1)

    return epoch_val_loss / len(val_loader)


def lyrics_generator(model, train_set, midi_features, first_word,
                     seed_sequence, device, song_len, max_wordLine):
    model.eval()
    output_sequence = []  # Track generated sequence
    hidden_state = None
    generated_words = [first_word]
    input_sequence = seed_sequence

    with torch.no_grad():
        while len(output_sequence) < song_len:  
            input_sequence = input_sequence.to(device)
            word_predictions, hidden_state = model(input_sequence, hidden_state, hidden_bool=True)
            hidden_state = hidden_state.detach()

            word_distribution = F.softmax(word_predictions, dim=-1)
            sampled_word_idx = torch.multinomial(word_distribution, num_samples=1).item()

            generated_words.append(train_set.index_to_word[sampled_word_idx])

            word_vector = train_set._get_lyric_features(train_set.index_to_word[sampled_word_idx])[0]

            if word_vector.size(0) == 0:  
                word_vector = torch.zeros((1, 300), device=device)  # If empty, replace with a zero vector, no embedding for the word

            word_vector = word_vector.to(device)
            midi_features = midi_features.to(device)
            next_word = torch.cat([word_vector, midi_features], dim=1)

            # Update sequence
            input_sequence = torch.tensor(next_word, dtype=torch.float32).to(device)
            output_sequence.append(next_word)

            # Reset at line end
            if len(output_sequence) % max_wordLine == 0:
                input_sequence = seed_sequence
                hidden_state = None

    # Format output
    formatted_song = ""
    for i, word in enumerate(generated_words):
        formatted_song += word + " "

    return formatted_song


def experiment(model, train_set, val_set, test_set, epochs, optimizer, criteria, device, first_words, song_len, max_wordline, method):

    #  dataloader creating
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

    # train and val part
    print("----- Start training part -----\n")
    avg_train_loss_list = []
    avg_val_loss_list = []

    # Early stopping criteria
    best_val_loss = float("inf")
    patience = 3  # Number of epochs to wait before stopping
    tolerance = 0.001  # Minimum delta for improvement
    no_improvement_count = 0

    print("----- Start training part -----\n")

    for epoch in range(epochs):
        print(f"------ Processing epoch No.{epoch + 1} -----")
        train_loss = train_model(model, train_loader, optimizer, criteria, device)
        val_loss = val_model(model, train_loader, val_loader, criteria, device)

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss},step = epoch+1)

        # store losses
        avg_train_loss_list.append(train_loss)
        avg_val_loss_list.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}\n")

        # Early stopping logic
        if val_loss < best_val_loss - tolerance:
            best_val_loss = val_loss
            no_improvement_count = 0  # Reset count
        else:
            no_improvement_count += 1
            print(f"No improvement in validation loss for {no_improvement_count} epochs.")

        if no_improvement_count >= patience:
            print("Early stopping triggered.")
            break

    # test - generate lyrics part
    generated_random_lyrics_dict = {}  
    generated_real_lyrics_dict = {}  
    seed_sequences= []
    real_first_words = []

    for lyrics, words in test_loader:
        real_first_word = words[:, 0].item() 
        real_first_words.append(real_first_word)
        seed_sequences.append(lyrics[:, 0, :])

    for j in range(len(seed_sequences)):
        # Get song info
        artist, song = test_loader.dataset.get_artist_song(j)
        midi_features = test_loader.dataset._get_midi_features(artist, song)

        # 7a
        real_first_word = test_loader.dataset.index_to_word[real_first_words[j]]  # Get the real first word as text
        generated_real_lyrics_dict[artist,song] = lyrics_generator(
                                                            model, train_set, midi_features, real_first_word, seed_sequences[j],
                                                            device, song_len, max_wordline
                                                            )   

        # 7b
        # Generate lyrics with different random first words
        for i,first_word in enumerate(first_words):
            generated_random_lyrics_dict[artist, first_word] = lyrics_generator(model, train_set, midi_features,
                                                                                first_word, seed_sequences[j],
                                                                                device, song_len, max_wordline)

    return avg_train_loss_list, avg_val_loss_list,generated_real_lyrics_dict, generated_random_lyrics_dict
