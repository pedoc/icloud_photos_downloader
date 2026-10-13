import inspect
import os
from typing import Any
from unittest import TestCase
from unittest.mock import Mock, PropertyMock, patch

import pytest

from pyicloud_ipd.exceptions import (
    PyiCloudConnectionErrorException,
    PyiCloudServiceNotActivatedException,
)
from pyicloud_ipd.services.photos import PhotoLibrary, PhotosService
from tests.helpers import (
    path_from_project_root,
    run_icloudpd_test,
    run_main_env,
)


def indexing_response(state: Any) -> Mock:
    return Mock(json=Mock(return_value={"records": [{"fields": {"state": {"value": state}}}]}))


def photos_service(session: Mock) -> PhotosService:
    return PhotosService("https://example.invalid", session, {})


@pytest.mark.parametrize("state", ["RUNNING", "INITIAL", "UNKNOWN", "", None])
def test_indexing_diagnostic(state: Any) -> None:
    session = Mock()
    session.post.return_value = indexing_response(state)
    with pytest.raises(PyiCloudServiceNotActivatedException) as error:
        PhotoLibrary(
            "https://example.invalid", {}, session, {"zoneName": "SharedSync-test"}, "shared"
        )
    message = str(error.value)
    assert f"state={state!r}" in message
    assert "zone='SharedSync-test'" in message
    assert "library_type='shared'" in message
    assert "retry" in message
    assert "iCloud website" in message
    assert session.post.call_count == 1


def test_finished_indexing_succeeds() -> None:
    session = Mock()
    session.post.return_value = indexing_response("FINISHED")
    assert photos_service(session).zone_id == {"zoneName": "PrimarySync"}


@pytest.mark.parametrize("library_type", ["private", "shared"])
def test_library_initialization_error_propagates(library_type: str) -> None:
    session = Mock()
    session.post.side_effect = [
        indexing_response("FINISHED"),
        Mock(
            json=Mock(
                return_value={
                    "zones": [
                        {"zoneID": {"zoneName": "Ready"}},
                        {"zoneID": {"zoneName": "NotReady"}},
                    ]
                }
            )
        ),
        indexing_response("FINISHED"),
        indexing_response("RUNNING"),
    ]
    photos = photos_service(session)
    with pytest.raises(PyiCloudServiceNotActivatedException, match="NotReady"):
        getattr(photos, f"{library_type}_libraries")
    assert getattr(photos, f"_{library_type}_libraries") is None


def test_deleted_library_does_not_check_indexing() -> None:
    session = Mock()
    session.post.side_effect = [
        indexing_response("FINISHED"),
        Mock(
            json=Mock(
                return_value={"zones": [{"zoneID": {"zoneName": "Deleted"}, "deleted": True}]}
            )
        ),
    ]
    assert photos_service(session).shared_libraries == {}
    assert session.post.call_count == 2


def test_library_connection_error_propagates() -> None:
    session = Mock()
    session.post.side_effect = [
        indexing_response("FINISHED"),
        PyiCloudConnectionErrorException("offline"),
    ]
    with pytest.raises(PyiCloudConnectionErrorException, match="offline"):
        _ = photos_service(session).private_libraries


@pytest.mark.parametrize(
    "mode", ["primary", "list-private", "list-shared", "selected-private", "selected-shared"]
)
def test_cli_indexing_failure_is_not_unknown_library(mode: str) -> None:
    session = Mock()
    session.post.return_value = indexing_response("FINISHED")
    photos = photos_service(session)
    icloud = Mock(photos=photos)
    args = ["--username", "test@example.invalid", "--password", "unused", "--list-albums"]
    if mode.startswith("list"):
        args[-1] = "--list-libraries"
    elif mode.startswith("selected"):
        args.extend(["--library", "NotReady"])
    if mode == "primary":
        target = patch.object(type(icloud), "photos", new_callable=PropertyMock, create=True)
    elif mode.endswith("shared"):
        photos._private_libraries = {"PrimarySync": Mock(albums={})}
        target = patch.object(PhotosService, "shared_libraries", new_callable=PropertyMock)
    else:
        target = patch.object(PhotosService, "private_libraries", new_callable=PropertyMock)
    with patch("icloudpd.base.authenticator", return_value=icloud), target as failing:
        failing.side_effect = PyiCloudServiceNotActivatedException(
            "indexing state=RUNNING zone=NotReady"
        )
        result = run_main_env({}, args)
    assert result.exit_code == 1
    assert result.exception is None
    assert "indexing state=RUNNING zone=NotReady" in result.output
    assert "Unknown library" not in result.output


@pytest.mark.parametrize("selected", [False, True])
def test_cli_private_library_is_not_blocked_by_shared(selected: bool) -> None:
    session = Mock()
    session.post.return_value = indexing_response("FINISHED")
    photos = photos_service(session)
    photos._private_libraries = {"PrimarySync": Mock(albums={})}
    args = ["--username", "test@example.invalid", "--password", "unused", "--list-albums"]
    if selected:
        args.extend(["--library", "PrimarySync"])
    with (
        patch("icloudpd.base.authenticator", return_value=Mock(photos=photos)),
        patch.object(PhotosService, "shared_libraries", new_callable=PropertyMock) as shared,
        patch.object(PhotosService, "albums", new_callable=PropertyMock, return_value={}),
    ):
        shared.side_effect = PyiCloudServiceNotActivatedException("shared indexing RUNNING")
        result = run_main_env({}, args)
    assert result.exit_code == 0
    assert result.exception is None
    shared.assert_not_called()


class ListingLibraryTestCase(TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        self.root_path = path_from_project_root(__file__)
        self.fixtures_path = os.path.join(self.root_path, "fixtures")

    def test_listing_library(self) -> None:
        base_dir = os.path.join(self.fixtures_path, inspect.stack()[0][3])
        _, result = run_icloudpd_test(
            self.assertEqual,
            self.root_path,
            base_dir,
            "listing_albums.yml",
            [],
            [],
            [
                "--username",
                "jdoe@gmail.com",
                "--password",
                "password1",
                "--list-libraries",
                "--no-progress-bar",
            ],
        )
        self.assertEqual(result.exit_code, 0, "exit code")
        albums = result.output.splitlines()

        self.assertIn("PrimarySync", albums)
        self.assertIn("SharedSync-00000000-1111-2222-3333-444444444444", albums)

    def test_listing_library_error(self) -> None:
        base_dir = os.path.join(self.fixtures_path, inspect.stack()[0][3])

        _, result = run_icloudpd_test(
            self.assertEqual,
            self.root_path,
            base_dir,
            "listing_albums.yml",
            [],
            [],
            [
                "--username",
                "jdoe@gmail.com",
                "--password",
                "password1",
                "--library",
                "doesnotexist",
                "--no-progress-bar",
            ],
        )
        self.assertEqual(result.exit_code, 1, "exit code")
        self.assertIn(
            "Unknown library: doesnotexist",
            result.output,
        )
