from src import config, data_prep


def test_diabetes_test_frame_loads():
    _, test_df = data_prep.load_diabetes_frames()
    assert len(test_df) > 0
    assert config.CLASSIFICATION_TARGET in test_df.columns
    assert config.REGRESSION_TARGET in test_df.columns


def test_classification_xy_excludes_only_its_own_target():
    _, test_df = data_prep.load_diabetes_frames()
    x, y = data_prep.classification_xy(test_df)
    assert config.CLASSIFICATION_TARGET not in x.columns
    assert config.REGRESSION_TARGET in x.columns  # legitimate feature for this task
    assert set(y.unique()) <= {0, 1}
    assert len(x) == len(y)


def test_regression_xy_excludes_both_targets():
    _, test_df = data_prep.load_diabetes_frames()
    x, y = data_prep.regression_xy(test_df)
    assert config.REGRESSION_TARGET not in x.columns
    assert config.CLASSIFICATION_TARGET not in x.columns  # would leak a future event
    assert (y >= 0).all()
    assert len(x) == len(y)
