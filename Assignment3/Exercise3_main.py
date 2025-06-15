from Exercise3_pre_process import LyricsMIDIDataset, create_vocabulary
from Exercise3_model import Generate_model, experiment
import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb  # We had troubles with tensorboard package, so we used wandb package instead
import pickle
import warnings
import itertools
import random
warnings.filterwarnings('ignore')


random.seed(42)
torch.manual_seed(42)

# set the paths
csv_file_path_training = 'd:/Niv/Deep Learning/Exercise 3/Archive/lyrics_train_set.csv'
midi_folder_path = 'c:/Users/cognitive.BGU-USERS/Downloads/midi_files'
csv_file_path_test = 'd:/Niv/Deep Learning/Exercise 3/Archive/lyrics_test_set.csv'

# load the csv file
training_df = pd.read_csv(csv_file_path_training, header=None, usecols=[0, 1, 2], names=['artist', 'song', 'lyrics'])
test_set = pd.read_csv(csv_file_path_test, header=None, usecols=[0, 1, 2], names=['artist', 'song', 'lyrics'])
training_set, validation_set = train_test_split(training_df, test_size=0.1)   

# define methods names
method_melody = 'melody'
method_melody_rhythm = 'melody_rhythm'
methods = ['melody', 'melody_rhythm']

combined_df = pd.concat([training_df, test_set])
combined_df = combined_df.reset_index(drop=True, inplace=False)
vocab, vocab_size = create_vocabulary(combined_df)

# First method

# # creating datasets of train, val and test
# train_dataset_melody = LyricsMIDIDataset(training_set, midi_folder_path, method_melody_rhythm, vocab, vocab_size)
# val_dataset_melody = LyricsMIDIDataset(validation_set, midi_folder_path, method_melody_rhythm, vocab, vocab_size)
# test_dataset_melody = LyricsMIDIDataset(test_set, midi_folder_path, method_melody_rhythm, vocab, vocab_size)

# # Save the train dataset
# with open('train_dataset_melody.pkl', 'wb') as file:
#     pickle.dump(train_dataset_melody, file)

# # Save the validation dataset
# with open('val_dataset_melody.pkl', 'wb') as file:
#     pickle.dump(val_dataset_melody, file)

# # Save the test dataset
# with open('test_dataset_melody.pkl', 'wb') as file:
#     pickle.dump(test_dataset_melody, file)


# Load the train dataset
with open('train_dataset_melody.pkl', 'rb') as file:
    train_dataset_melody = pickle.load(file)

# Load the validation dataset
with open('val_dataset_melody.pkl', 'rb') as file:
    val_dataset_melody = pickle.load(file)

# Load the test dataset
with open('test_dataset_melody.pkl', 'rb') as file:
    test_dataset_melody = pickle.load(file)


# ## In order to pre compute the features and just load it in the method 'get item'
# train_dataset_melody.precompute_features("precomputed_features_melody")
# val_dataset_melody.precompute_features("precomputed_features_melody")
# test_dataset_melody.precompute_features("precomputed_features_melody")


# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device}')
input_size = 300 + train_dataset_melody.midi_feature_size   # 300 from word2vec
epochs = 20
criteria = nn.CrossEntropyLoss()
hidden_sizes = [64, 128, 256]
learning_rates = [0.0001, 0.0003, 0.001]
optimizers = ["Adam", "SGD"]
results = []
first_words = ['joy', 'baby', 'stranger']   
song_length = 75
max_word_in_line = 5

for hidden_size, lr, optimizer_name in itertools.product(hidden_sizes, learning_rates, optimizers):
    # Initialize model
    model = Generate_model(input_size, hidden_size, vocab_size)
    model.to(device)

    # Initialize optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr)

    # Define experiment name
    exp_name = f"Melody_Hidden-{hidden_size}_LR-{lr}_Opt-{optimizer_name}"

    # Initialize WandB
    wandb.init(
        project="lyrics-generation",
        name=exp_name,
        config={
            "hidden_size": hidden_size,
            "learning_rate": lr,
            "optimizer": optimizer_name,
            "epochs": epochs,
            "batch_size": 1,
            "feature_size": input_size,
            "architecture": "melody"
        }
    )
    avg_train_loss_list, avg_val_loss_list, generated_real_lyrics_dict, generated_random_lyrics_dict = experiment(
        model, train_dataset_melody, val_dataset_melody, test_dataset_melody,
        epochs, optimizer, criteria, device, first_words, song_length, max_word_in_line, method_melody
    )
    wandb.finish()

    # Store results for later analysis
    avg_val_loss = sum(avg_val_loss_list) / len(avg_val_loss_list)
    results.append({
        "experiment": exp_name,
        "hidden_size": hidden_size,
        "learning_rate": lr,
        "optimizer": optimizer_name,
        "avg_val_loss": avg_val_loss,
        "best_val_loss": min(avg_val_loss_list),
        "generated_real_lyrics_dict": generated_real_lyrics_dict,
        "generated_random_lyrics_dict": generated_random_lyrics_dict
    })

# Sort and display the best result
results_melody = sorted(results, key=lambda x: x["avg_val_loss"])

with open('results_melody.pkl', 'wb') as file:
    pickle.dump(results_melody, file)
print(f"Best Configuration: {results[0]}")


# Second method
# # creating datasets of train, val and test
# train_dataset_melody_rhythm = LyricsMIDIDataset(training_set, midi_folder_path, method_melody_rhythm, vocab, vocab_size)
# val_dataset_melody_rhythm = LyricsMIDIDataset(validation_set, midi_folder_path, method_melody_rhythm, vocab, vocab_size)
# test_dataset_melody_rhythm = LyricsMIDIDataset(test_set, midi_folder_path, method_melody_rhythm, vocab, vocab_size)


# # Save the train dataset
# with open('train_dataset_melody_rhytm.pkl', 'wb') as file:
#     pickle.dump(train_dataset_melody_rhythm, file)

# # Save the validation dataset
# with open('val_dataset_melody_rhytm.pkl', 'wb') as file:
#     pickle.dump(val_dataset_melody_rhythm, file)

# # Save the test dataset
# with open('test_dataset_melody_rhytm.pkl', 'wb') as file:
#     pickle.dump(test_dataset_melody_rhythm, file)


# Load the train dataset
with open('train_dataset_melody_rhytm.pkl', 'rb') as file:
    train_dataset_melody_rhythm = pickle.load(file)

# Load the validation dataset
with open('val_dataset_melody_rhytm.pkl', 'rb') as file:
    val_dataset_melody_rhythm = pickle.load(file)

# Load the test dataset
with open('test_dataset_melody_rhytm.pkl', 'rb') as file:
    test_dataset_melody_rhythm = pickle.load(file)


# #In order to pre compute the features and just load it in the method 'get item'
# train_dataset_melody_rhythm.precompute_features("precomputed_features_melody_rhythm")
# val_dataset_melody_rhythm.precompute_features("precomputed_features_melody_rhythm")
# test_dataset_melody_rhythm.precompute_features("precomputed_features_melody_rhythm")


input_size = 300 + train_dataset_melody_rhythm.midi_feature_size   # 300 from word2vec  
for hidden_size, lr, optimizer_name in itertools.product(hidden_sizes, learning_rates, optimizers):
    # Initialize model
    model = Generate_model(input_size, hidden_size, vocab_size)
    model.to(device)

    # Initialize optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr)

    # Define experiment name
    exp_name = f"Melody_Rhythm_Hidden-{hidden_size}_LR-{lr}_Opt-{optimizer_name}"

    # Initialize WandB
    wandb.init(
        project="lyrics-generation",
        name=exp_name,
        config={
            "hidden_size": hidden_size,
            "learning_rate": lr,
            "optimizer": optimizer_name,
            "epochs": epochs,
            "batch_size": 1,
            "feature_size": input_size,
            "architecture": "melody_rhythm"
        }
    )
    avg_train_loss_list, avg_val_loss_list, generated_real_lyrics_dict_rhythm, generated_random_lyrics_dict_rhythm = experiment(
        model, train_dataset_melody, val_dataset_melody, test_dataset_melody,
        epochs, optimizer, criteria, device, first_words, song_length, max_word_in_line, method_melody_rhythm
    )
    wandb.finish()

    # Store results for later analysis
    avg_val_loss = sum(avg_val_loss_list) / len(avg_val_loss_list)
    results.append({
        "experiment": exp_name,
        "hidden_size": hidden_size,
        "learning_rate": lr,
        "optimizer": optimizer_name,
        "avg_val_loss": avg_val_loss,
        "best_val_loss": min(avg_val_loss_list),
        "generated_real_lyrics_dict_rhythm": generated_real_lyrics_dict_rhythm,
        "generated_random_lyrics_dict_rhythm": generated_random_lyrics_dict_rhythm
    })

# Sort and display the best result
results_melody_rhythm = sorted(results, key=lambda x: x["avg_val_loss"])

with open('results_melody_rhythm.pkl', 'wb') as file:
    pickle.dump(results_melody_rhythm, file)
print(f"Best Configuration: {results_melody_rhythm[0]}")
