import os
import re
import torch
import pickle
import numpy as np
import pretty_midi
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
import gensim.downloader as api


class LyricsMIDIDataset(Dataset):
    """
    A PyTorch Dataset for combining lyrics and MIDI features for training RNN models.

    Parameters:
        songs_df (pd.DataFrame): DataFrame containing 'Artist', 'Song Name', and 'Lyrics'.
        midi_folder (str): Path to the folder containing MIDI files.
        method (str): Feature extraction method. Options are 'melody' or 'melody_rhythm'.
        vocab (set): Vocabulary set created from lyrics.
        vocab_size (int): Size of the vocabulary.
    """
    def __init__(self, songs_df, midi_folder, method="melody", vocab=None, vocab_size=None):
        self.songs_df = songs_df.copy()
        self.midi_folder = midi_folder
        self.method = method.lower()
        self._file_cache = None

        if vocab is None or vocab_size is None:
            self.vocab, self.vocab_size = create_vocabulary(self.songs_df)
        else:
            self.vocab = vocab
            self.vocab_size = vocab_size
      
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}

        self.word2vec_model = self._load_word2vec_model()

        self.songs_df["lyrics"] = self.songs_df["lyrics"].apply(clean_lyrics)

        self.midi_feature_size = 1293 if self.method == "melody" else 1303  # 'melody_rhythm' = 1303

    def _load_word2vec_model(self):
        """
        Load or download the Word2Vec model.
        """
        model_path = "word2vec-google-news-300.gensim"
        if os.path.exists(model_path):
            return KeyedVectors.load(model_path, mmap="r")
        else:
            return api.load("word2vec-google-news-300")
        
    def _build_file_cache(self):
        """
        Caches the list of files in the MIDI folder.
        """
        self._file_cache = os.listdir(self.midi_folder)

    def _get_midi_file_path(self, artist, song_name):
        """
        Retrieves the MIDI file path for a given artist and song name.
        """
        
        if self._file_cache is None:
            self._build_file_cache()

        artist = artist.strip()
        artist = re.sub(r"[^a-zA-Z0-9' ]", "", artist) 
        artist = re.sub(r"\s+", "_", artist).lower()  

        song_name = song_name.strip() 
        song_name = re.sub(r"[^a-zA-Z0-9' ]", "", song_name)  
        song_name = re.sub(r"\s+", "_", song_name).lower()   

        file_pattern = f"{artist}_-_{song_name}.*\.mid".lower()

        regex = re.compile(file_pattern)

        for file in self._file_cache:
            if regex.match(file.lower()):
                return os.path.join(self.midi_folder, file)
        raise FileNotFoundError(f"MIDI file for {artist} - {song_name} not found.")

    def _extract_melody_features(self, midi):
        """
        Extracts melody features from a MIDI file efficiently.
        """
        chroma = midi.get_chroma().mean(axis=-1)
        piano_roll = midi.get_piano_roll().mean(axis=-1)
        tempo = midi.estimate_tempo()

        features = np.zeros((128, 9), dtype=np.float32)

        for instrument in midi.instruments:
            if not instrument.notes:  # Skip instruments with no notes
                continue

            program = instrument.program
            pitches = np.array([note.pitch for note in instrument.notes])
            velocities = np.array([note.velocity for note in instrument.notes])

            features[program, 0] = len(instrument.notes)  # Number of notes
            features[program, 1] = len(instrument.pitch_bends)  # Pitch bends
            features[program, 2] = len(instrument.control_changes)  # Control changes
            features[program, 3] = pitches.max() if pitches.size > 0 else 0  # Max pitch
            features[program, 4] = pitches.min() if pitches.size > 0 else 0  # Min pitch
            features[program, 5] = pitches.mean() if pitches.size > 0 else 0  # Mean pitch
            features[program, 6] = velocities.max() if velocities.size > 0 else 0  # Max velocity
            features[program, 7] = velocities.min() if velocities.size > 0 else 0  # Min velocity
            features[program, 8] = velocities.mean() if velocities.size > 0 else 0  # Mean velocity

        instrument_features = features.flatten()
        combined_features = np.concatenate([instrument_features, chroma, piano_roll, [tempo]])

        return torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0)

    def _extract_rhythm_features(self, midi):
        """
        Efficiently extracts rhythm features from a MIDI file.
        """
        instruments = midi.instruments

        # Note durations and densities
        note_durations = [note.end - note.start for instrument in instruments for note in instrument.notes]
        if note_durations:
            avg_duration = np.mean(note_durations)
            std_duration = np.std(note_durations)
            note_count = len(note_durations)
        else:
            avg_duration = std_duration = note_count = 0

        total_duration = midi.get_end_time()
        note_density = note_count / total_duration if total_duration > 0 else 0

        # Tempo statistics
        tempo_changes = midi.get_tempo_changes()[1]
        if len(tempo_changes) > 0:
            mean_tempo = np.mean(tempo_changes)
            std_tempo = np.std(tempo_changes)
            min_tempo = np.min(tempo_changes)
            max_tempo = np.max(tempo_changes)
        else:
            mean_tempo = std_tempo = min_tempo = max_tempo = 0

        # Beat and downbeat statistics
        beats = midi.get_beats()
        downbeats = midi.get_downbeats()

        if len(beats) > 1:
            beat_duration = np.diff(beats).mean()
        else:
            beat_duration = 0

        if beat_duration > 0:
            average_notes_per_beat = note_count / len(beats)
        else:
            average_notes_per_beat = 0

        if downbeats.any():
            average_notes_per_downbeat = note_count / len(downbeats)
        else:
            average_notes_per_downbeat = 0

        features = np.array([
            avg_duration, std_duration, total_duration, note_density,
            mean_tempo, std_tempo, min_tempo, max_tempo,
            average_notes_per_beat, average_notes_per_downbeat
        ], dtype=np.float32)

        return torch.tensor(features).unsqueeze(0)

    def precompute_features(self, output_dir):
        """""
        Precomputes and saves the features for all songs in the dataset.
        """

        os.makedirs(output_dir, exist_ok=True)
        for idx in range(len(self)):
            artist, song_name = self.get_artist_song(idx)
            midi_features = self._get_midi_features(artist, song_name)
            lyric_vectors, labels = self._get_lyric_features(self.songs_df.iloc[idx]['lyrics'])

            song_id = f"{artist}_{song_name}".replace(" ", "_")
            feature_path = os.path.join(output_dir, f"{song_id}.pkl")
            with open(feature_path, "wb") as f:
                pickle.dump({
                    "midi_features": midi_features,
                    "lyric_vectors": lyric_vectors,
                    "labels": labels
                }, f)

    def _get_midi_features(self, artist, song_name):
        """
        Retrieves MIDI features for a given song.
        """
        midi_path = self._get_midi_file_path(artist, song_name)
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            if self.method == "melody":
                return self._extract_melody_features(midi)
            elif self.method == "melody_rhythm":
                melody_features = self._extract_melody_features(midi)
                rhythm_features = self._extract_rhythm_features(midi)
                return torch.cat([melody_features, rhythm_features], dim=1)
        except Exception:
            return torch.zeros((1, self.midi_feature_size), dtype=torch.float32)
        
    def get_artist_song(self, idx):
        """
        this function will return the artist and the song name of a given index
        params:
            idx: the index
        return:
            the artist and the song name
        """
        return self.songs_df.iloc[idx, 0], self.songs_df.iloc[idx, 1]
    
    def __len__(self):
        return len(self.songs_df)

    def __getitem__(self, idx):
        """
        Returns input features and labels for a given song.
        """
        artist, song_name = self.get_artist_song(idx)
        method = self.method

        song_id = f"{artist}_{song_name}".replace(" ", "_")
        feature_path = os.path.join(f"precomputed_features_{method}", f"{song_id}.pkl")
        with open(feature_path, "rb") as f:
            data = pickle.load(f)
        midi_features = data["midi_features"]
        lyric_vectors = data["lyric_vectors"]
        labels = data["labels"]
        midi_features = midi_features.repeat(lyric_vectors.size(0), 1)
        inputs = torch.cat([lyric_vectors, midi_features], dim=-1)
        return inputs, labels

    def _get_lyric_features(self, lyrics):
        """
        Converts lyrics into word embeddings and corresponding labels.
        """
        words = lyrics.split()
        vectors = []
        labels = []

        for word in words:
            if word in self.word2vec_model:
                vectors.append(self.word2vec_model[word]) 
                labels.append(self.word_to_index.get(word, -1))  
            else:
                vectors.append(np.zeros((300,)))  # Zero vector for unknown words
                labels.append(-1)  # Placeholder label for unknown words
                # print(f"No embedding found for word: {word}")

        vectors = torch.tensor(np.stack(vectors), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        valid_indices = labels != -1
        vectors = vectors[valid_indices]
        labels = labels[valid_indices]

        return vectors, labels


def clean_lyrics(lyrics):
    """
    Cleans lyrics by removing special characters and applying substitutions.
    """
    replacements = {
        "'re": " are", "this'will": "this will", "somethin'": "something",
        "doin'": "doing", "'m": " am", "'cause": " because", "'s": " is", "'ll": " will"
    }
    for pattern, replacement in replacements.items():
        lyrics = re.sub(pattern, replacement, lyrics)
    filters = '!"#$%()*+,&-.\'/:;<=>?@[\\]^_`{|}~\t\n'
    lyrics = re.sub(f"[{filters}]", "", lyrics)
    return lyrics


def create_vocabulary(songs_df):
    """
    Creates a vocabulary from lyrics.
    """
    vocab = set()
    for lyrics in songs_df["lyrics"]:
        vocab.update(clean_lyrics(lyrics).split())
    return sorted(vocab), len(vocab)

