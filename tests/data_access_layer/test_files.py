from pathlib import Path

from data_access_layer import files


def test_check_is_local_file():
    assert files.check_is_local_path("tests/data_access_layer/test_files.py")
    assert files.check_is_local_path("./tests/data_access_layer/test_files.py")
    assert not files.check_is_local_path("s3://tests/data_access_layer/test_files.py")
    assert not files.check_is_local_path("https://localhost:500/data_access_layer/test_files.py")


def test_create_dir_if_not_exists(tmp_path: Path):
    path = tmp_path / "foo.txt"
    tmp_path.rmdir()
    assert not tmp_path.exists()
    assert not path.exists()
    files.create_dir_if_not_exists(str(path))
    assert tmp_path.exists()
    files.create_dir_if_not_exists(str(path))
    assert tmp_path.exists()
