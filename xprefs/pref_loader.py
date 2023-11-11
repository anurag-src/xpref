import os
import pandas as pd
from configs.constants import EMBODIMENTS

class PreferenceLoader:
    """
    Given a CSV file containing preferences, create a data structure containing the preference labels
    """

    def __init__(self, config, train=True):
        self.config = config
        self.training_data = train
        self.removed_embodiments = self.remove_embodiments()
        self.preferences = self.load_preferences()

    def load_preferences(self):
        data_root = self.config.data.demonstrations_root
        if self.training_data:
            directory = os.path.join(data_root, "train")
        else:
            directory = os.path.join(data_root, "valid")

        if self.config.data.preference_type == "cross_embodiment":
            file_name = "cross_embedding_prefs.csv"
            preferences_file = os.path.join(directory, file_name)
            df = pd.read_csv(preferences_file)
        elif self.config.data.preference_type == "same_embodiment":
            file_name = "same_embedding_prefs.csv"
            preferences_file = os.path.join(directory, file_name)
            df = pd.read_csv(preferences_file)
        else:
            preferences_file_A = os.path.join(directory, "cross_embedding_prefs.csv")
            dfA = pd.read_csv(preferences_file_A)
            preferences_file_B = os.path.join(directory, "same_embedding_prefs.csv")
            dfB = pd.read_csv(preferences_file_B)
            df = pd.concat([dfA, dfB])

        # Remove the withheld embodiment(s) from preference data
        for removal in self.removed_embodiments:
            df = df.loc[df["o1_embod"] != removal]
            df = df.loc[df["o2_embod"] != removal]

        # Truncate if necessary
        limit = self.config.data.truncate_training_preferences if self.training_data else self.config.data.truncate_testing_preferences
        df = df[:limit]
        return df

    def remove_embodiments(self):
        remove_embodiments = []
        for e in EMBODIMENTS:
            if e not in self.config.data.train_embodiments:
                remove_embodiments.append(e)
        return remove_embodiments