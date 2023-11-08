import unittest

class TestUnitTestingModule(unittest.TestCase):
    def test_python_strings(self):
        """
        A test that asserts that testing works independent of our code. Tests simple string methods.
        """
        self.assertEqual('foo'.upper(), 'FOO')
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

class TestLoadingXPrefsConfig(unittest.TestCase):
    def test_config_initialization(self):
        """
        Test Loading the XPrefs Configuration File
        """
        import ml_collections
        from base_configs.xprefs import get_config
        config = get_config()
        self.assertTrue(isinstance(config, ml_collections.ConfigDict))
        self.assertTrue("data" in config)
        self.assertTrue("experiments" in config)

class TestPreferencesLoading(unittest.TestCase):

    def test_preference_loading_initialization(self):
        from xprefs.pref_loader import PreferenceLoader
        from base_configs.xprefs import get_config
        config = get_config()
        try:
            prefs = PreferenceLoader(config)
        except Exception as e:
            print(e)
            self.assertTrue(False, "Loading the Preferences threw the above exception, when it was expected to execute smoothly")

        self.assertTrue(len(prefs.preferences) != 0)

    def test_training_vs_testing_prefs(self):
        """
        Confirm that the model is loading different preference data for training vs. validation
        """
        from xprefs.pref_loader import PreferenceLoader
        from base_configs.xprefs import get_config
        config = get_config()
        prefs_training = PreferenceLoader(config, train=True)
        prefs_testing = PreferenceLoader(config, train=False)
        self.assertTrue(prefs_testing != prefs_training)
        self.assertTrue(len(prefs_training.preferences) != 0)
        self.assertTrue(len(prefs_testing.preferences) != 0)

    def test_removing_embodiments_no_removal(self):
        """
        Should the config indicate that we only want to train on all embodiments, ensure no removal
        """
        from xprefs.pref_loader import PreferenceLoader
        from base_configs.xprefs import get_config
        config = get_config()
        config.data.train_embodiments = ["longstick", "shortstick", "gripper"]
        prefs_no_change = PreferenceLoader(config)

        config.data.train_embodiments = ["longstick", "shortstick", "gripper", "mediumstick"]
        prefs = PreferenceLoader(config)

        self.assertTrue(len(prefs.remove_embodiments()) == 0)
        self.assertTrue(len(prefs_no_change.preferences) < len(prefs.preferences))
        self.assertTrue(len(prefs.preferences) != 0)
        self.assertTrue(len(prefs_no_change.preferences) != 0)

    def test_removing_embodiments_all_but_one(self):
        """
        Should the config indicate that we only want to train on 2 embodiments, ensure removal of 2 embodiments
        """
        from xprefs.pref_loader import PreferenceLoader
        from base_configs.xprefs import get_config
        config = get_config()
        config.data.train_embodiments = ["longstick", "shortstick", "gripper"]
        prefs_no_change = PreferenceLoader(config)

        config.data.train_embodiments = ["longstick", "shortstick"]
        prefs = PreferenceLoader(config)


        self.assertTrue(len(prefs.remove_embodiments()) == 2)
        self.assertTrue(len(prefs_no_change.preferences) > len(prefs.preferences))
        self.assertTrue(len(prefs.preferences) != 0)
        self.assertTrue(len(prefs_no_change.preferences) != 0)

    def test_removal_in_pandas_easy(self):
        """
        Test that embodiment removal works for 1 embodiment in pandas
        """
        from xprefs.pref_loader import PreferenceLoader
        from base_configs.xprefs import get_config
        config = get_config()
        config.data.train_embodiments = ["longstick", "shortstick", "gripper"]

        prefs = PreferenceLoader(config)
        df = prefs.preferences

        embodiments = list(df["o1_embod"]) + list(df["o2_embod"])
        self.assertTrue(len(df) != 0)
        self.assertTrue("mediumstick" not in embodiments)
        self.assertTrue("longstick" in embodiments)
        self.assertTrue("shortstick" in embodiments)
        self.assertTrue("gripper" in embodiments)

        prefs = PreferenceLoader(config, train=False)
        df = prefs.preferences
        self.assertTrue(len(df) != 0)
        self.assertTrue("mediumstick" not in embodiments)
        self.assertTrue("longstick" in embodiments)
        self.assertTrue("shortstick" in embodiments)
        self.assertTrue("gripper" in embodiments)

    def test_removal_in_pandas_hard(self):
        """
        Test that embodiment removal works for 2 embodiments in pandas
        """
        from xprefs.pref_loader import PreferenceLoader
        from base_configs.xprefs import get_config
        config = get_config()
        config.data.train_embodiments = ["longstick", "shortstick"]

        prefs = PreferenceLoader(config)
        df = prefs.preferences
        embodiments = list(df["o1_embod"]) + list(df["o2_embod"])
        self.assertTrue(len(df) != 0)
        self.assertTrue("mediumstick" not in embodiments)
        self.assertTrue("shortstick" in embodiments)
        self.assertTrue("gripper" not in embodiments)
        self.assertTrue("longstick" in embodiments, "Longstick not found in training dataset")

        prefs = PreferenceLoader(config, train=False)
        df = prefs.preferences
        self.assertTrue(len(df) != 0)
        self.assertTrue("mediumstick" not in embodiments)
        self.assertTrue("shortstick" in embodiments)
        self.assertTrue("gripper" not in embodiments)
        self.assertTrue("longstick" in embodiments, "Longstick not found in testing dataset")

class TestTrajectoryLoading(unittest.TestCase):

    def setUp(self):
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    def test_preference_loading_initialization(self):
        from xprefs.trajectory_loader import TrajectoryLoader
        from base_configs.xprefs import get_config
        config = get_config()
        try:
            traj = TrajectoryLoader.full_dataset_from_config(config, train=True)
        except Exception as e:
            print(e)
            self.assertTrue(False, "Loading the Trajectories threw the above exception, when it was expected to execute smoothly")

        self.assertTrue(len(traj.trajectories) != 0)

    def test_video_sample_stride_1(self):
        from xprefs.trajectory_loader import TrajectoryLoader
        from base_configs.xprefs import get_config
        config = get_config()
        config.sampler.stride = 1
        config.data.train_embodiments = ["longstick", "shortstick", "gripper", "mediumstick"]
        traj = TrajectoryLoader.full_dataset_from_config(config, train=True)

        embodiment = "longstick"
        idx = list(traj.trajectories[embodiment].keys())[0]

        video = traj.get_item(embodiment, idx)
        self.assertIsNotNone(video)
        self.assertTrue(len(video["frame_idxs"]) == 50)  # Longstick trajectories should be 50 frames long
        self.assertTrue(video["frame_idxs"][1] - video["frame_idxs"][0] == config.sampler.stride) # Test Stride

    def test_video_sample_stride_2(self):
        from xprefs.trajectory_loader import TrajectoryLoader
        from base_configs.xprefs import get_config
        config = get_config()
        config.data.train_embodiments = ["longstick", "shortstick", "gripper", "mediumstick"]
        config.sampler.stride = 2
        traj = TrajectoryLoader.full_dataset_from_config(config, train=True)

        embodiment = "longstick"
        idx = list(traj.trajectories[embodiment].keys())[0]

        video = traj.get_item(embodiment, idx)
        self.assertIsNotNone(video)
        self.assertTrue(len(video["frame_idxs"]) == 25)  # Longstick trajectories should be 50 frames long
        self.assertTrue(video["frame_idxs"][1] - video["frame_idxs"][0] == config.sampler.stride) # Test Stride



if __name__ == '__main__':
    unittest.main()