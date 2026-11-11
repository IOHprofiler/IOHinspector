import os
import warnings

import pytest

from iohinspector import DataManager, plot_ecdf, turbo_align
from iohinspector.convert import convert
from iohinspector.io import ReaderError

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "test_data"))
COCO_DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "test_coco_data"))
FOLDERS = [os.path.join(DATA_DIR, x) for x in sorted(os.listdir(DATA_DIR))]


@pytest.fixture(scope="module")
def manager():
    m = DataManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.add_folders(FOLDERS)
    return m


def one(path=FOLDERS[0]):
    m = DataManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.add_folder(path)
    return m


def test_add_folder_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataManager().add_folder("nonexistent_folder")
    with pytest.raises(ReaderError, match="no reader recognises"):
        DataManager().add_folder(tmp_path)


def test_add_folder_ioh_and_coco():
    for m in (one(), one(COCO_DATA_DIR)):
        assert len(m.functions) == 1
        assert len(m.algorithms) == 1
        assert len(m.datasets) == 1


def test_add_a_json_file():
    json_file = os.path.join(FOLDERS[0], "IOHprofiler_f1_Sphere.json")
    m = one(json_file)
    df = m.load(monotonic=False)
    assert m.n_runs == 5
    assert len(df) == 27
    assert df["run_id"].n_unique() == 5


def test_add_twice_needs_another_dataset_id():
    m = one()
    with pytest.raises(ValueError, match="already loaded"):
        m.add_folder(FOLDERS[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.add_folder(FOLDERS[0], dataset_id="again")
    assert m.n_runs == 10
    assert m.overview["run_id"].n_unique() == 10


def test_add_dataset_equals_add_folder(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        convert(FOLDERS[0], tmp_path / "ds")
        stored = DataManager()
        stored.add_dataset(tmp_path / "ds")
    read = one()
    assert stored.overview.equals(read.overview)
    assert stored.load(include_meta_data=True).equals(read.load(include_meta_data=True))


def test_overview_and_properties(manager):
    overview = manager.overview
    assert overview.columns[:3] == ["run_id", "algorithm_name", "algorithm_info"]
    assert overview.height == manager.n_runs
    assert manager.functions == (1, 2)
    assert manager.algorithms == ("algorithm_A", "algorithm_B")
    assert manager.dimensions == (2,)
    assert "algorithm_name" in manager.descriptor_keys
    assert manager.any


def test_select(manager):
    selection = manager.select(instances=[1], function_ids=[1])
    assert selection.any
    # The logs hold only improvements plus each run's final evaluation, so the
    # best-so-far rows are all 84. iohinspector 0.0.8 gave 69: it dropped the
    # final row of each of the 15 runs (defect fixed in WP I05).
    for monotonic in (False, True):
        df = selection.load(monotonic=monotonic)
        assert df.shape == (84, 3)
        assert df["run_id"].n_unique() == 15
    df = selection.load(monotonic=True, include_meta_data=True)
    assert df.shape == (84, manager.overview.width + 2)

    nothing = manager.select(function_ids=[0])
    assert not nothing.any
    assert len(nothing.load()) == 0

    both = selection + manager.select(instances=[1], function_ids=[2])
    assert len(both.load()) == 150  # 0.0.8: 125, without the final rows of 25 runs


def test_select_by_any_column(manager):
    a = manager.select(algorithms=["algorithm_A"])
    assert a.algorithms == ("algorithm_A",)
    assert manager.select(algorithm_info=["algorithm_info"]).n_runs == manager.n_runs
    assert manager.select(suite=["nope"]).n_runs == 0
    with pytest.raises(ValueError, match="no runs column or descriptor"):
        manager.select(colour=["red"])


def test_select_run_ids(manager):
    ids = manager.overview["run_id"][:3].to_list()
    selection = manager.select(run_ids=ids)
    assert selection.n_runs == 3
    assert set(selection.load()["run_id"]) == set(ids)


def test_load_columns(manager):
    df = manager.select(function_ids=[1]).load(include_columns=["function_id"])
    assert df.columns == ["run_id", "function_id", "evaluations", "raw_y"]
    assert manager.load(x_values=[1])["evaluations"].unique().to_list() == [1]
    assert len(DataManager().load()) == 0


def test_align(manager):
    selection = manager.select(
        function_ids=[1], algorithms=["algorithm_A", "algorithm_B"]
    )
    df = selection.load(monotonic=True, include_meta_data=True)
    evals = [1, 5, 10, 20, 50, 100]
    df = turbo_align(df, evals)
    assert set(df["evaluations"].unique()) == set(evals)
    assert df["run_id"].n_unique() * len(evals) == df.shape[0]


def test_plot_ecdf(manager):
    selection = manager.select(
        function_ids=[1], algorithms=["algorithm_A", "algorithm_B"]
    )
    df = selection.load(monotonic=True, include_meta_data=True)
    _, dt = plot_ecdf(df)
    # 0.0.8: 66 rows; the final rows extend the budget range (WP I05)
    assert dt.height == 68


def test_coco_folder():
    df = one(COCO_DATA_DIR).load(True, True)
    assert {"raw_y", "evaluations", "function_id", "algorithm_info"} <= set(df.columns)
    assert df["algorithm_name"].unique().to_list() == ["BFGS-scipy-2019"]
    assert df["suite"].unique().to_list() == ["bbob"]
    assert df["function_id"].unique().to_list() == [1]
    assert df["dimension"].unique().to_list() == [2]


@pytest.mark.parametrize(
    "call, error, hint",
    [
        (lambda m: m.add_json("x.json"), AttributeError, "add_folder"),
        (lambda m: m.data_sets, AttributeError, "datasets"),
        (lambda m: m.select_indexes([0]), AttributeError, "run_ids"),
        (lambda m: m.select(data_ids=[1]), TypeError, "run_ids"),
        (lambda m: m.select(experiment_attributes=[]), TypeError, "select"),
    ],
)
def test_removed_names_fail_with_a_hint(call, error, hint):
    with pytest.raises(error, match=f"removed in iohinspector 0.1.0: .*{hint}"):
        call(DataManager())


def test_removed_module_names_fail_with_a_hint():
    import iohinspector

    with pytest.raises(ImportError, match="removed in iohinspector 0.1.0"):
        iohinspector.Dataset
    with pytest.raises(ImportError, match="removed in iohinspector 0.1.0"):
        from iohinspector.data import Scenario  # noqa: F401
    with pytest.raises(AttributeError, match="has no attribute"):
        iohinspector.nothing_like_this
