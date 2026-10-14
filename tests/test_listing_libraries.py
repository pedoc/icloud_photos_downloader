import inspect
import json
import logging
import os
from dataclasses import replace
from typing import Any
from unittest import TestCase
from unittest.mock import Mock, PropertyMock, patch

import pytest

from icloudpd.authentication import authenticator
from icloudpd.base import run_with_configs
from icloudpd.cli import format_help, parse
from icloudpd.mfa_provider import MFAProvider
from icloudpd.status import StatusExchange
from pyicloud_ipd.base import PyiCloudService
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


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["-u", "first", "-u", "second"], [False, False]),
        (["--skip-indexing-check", "-u", "first", "-u", "second"], [True, True]),
        (["-u", "first", "--skip-indexing-check", "-u", "second"], [True, False]),
        (["-u", "first", "-u", "second", "--skip-indexing-check"], [False, True]),
    ],
)
def test_skip_indexing_config_scope(args: list[str], expected: list[bool]) -> None:
    _, configs = parse(args)
    assert [config.skip_indexing_check for config in configs] == expected
    assert "--skip-indexing-check" in format_help()


@pytest.mark.parametrize("library_type", ["private", "shared"])
def test_skip_indexing_propagates_to_all_libraries(library_type: str) -> None:
    session = Mock()
    photos = PhotosService("https://example.invalid", session, {}, skip_indexing_check=True)
    session.post.assert_not_called()
    session.post.return_value = Mock(
        json=Mock(return_value={"zones": [{"zoneID": {"zoneName": "Ready"}}]})
    )
    libraries = getattr(photos, f"{library_type}_libraries")
    assert list(libraries) == ["Ready"]
    assert photos.skip_indexing_check is True
    assert libraries["Ready"].skip_indexing_check is True
    assert session.post.call_count == 1
    assert session.post.call_args.args[0].endswith(f"/{library_type}/zones/list")


@pytest.mark.parametrize("skip_check", [False, True])
def test_service_propagates_skip_indexing_to_photos(skip_check: bool) -> None:
    with patch.object(PyiCloudService, "authenticate"):
        service = PyiCloudService(
            "com", "test@example.invalid", lambda: None, skip_indexing_check=skip_check
        )
    service._webservices = {"ckdatabasews": {"url": "https://example.invalid"}}
    with patch("pyicloud_ipd.base.PhotosService") as factory:
        _ = service.photos
    assert factory.call_args.kwargs["skip_indexing_check"] is skip_check


@pytest.mark.parametrize("skip_check", [False, True])
def test_authenticator_propagates_skip_indexing(skip_check: bool) -> None:
    authenticated = Mock(requires_2fa=False, requires_2sa=False)
    with patch("icloudpd.authentication.PyiCloudService", return_value=authenticated) as factory:
        authenticator(
            logging.getLogger(__name__),
            "com",
            {},
            MFAProvider.CONSOLE,
            StatusExchange(),
            "test@example.invalid",
            lambda: None,
            skip_indexing_check=skip_check,
        )
    assert factory.call_args.kwargs["skip_indexing_check"] is skip_check


@pytest.mark.parametrize("skip_check", [False, True])
def test_cli_propagates_skip_indexing(skip_check: bool) -> None:
    args = ["-u", "test@example.invalid", "--auth-only"]
    if skip_check:
        args.append("--skip-indexing-check")
    with patch("icloudpd.base.authenticator", return_value=Mock()) as authenticate:
        result = run_main_env({}, args)
    assert result.exit_code == 0
    assert authenticate.call_args.kwargs["skip_indexing_check"] is skip_check


@pytest.mark.parametrize(
    "deletion_args",
    [
        ["--auto-delete"],
        ["--delete-after-download"],
        ["--keep-icloud-recent-days", "0"],
        ["--keep-icloud-recent-days", "7"],
        ["--keep-icloud-recent-days", "-1"],
    ],
)
@pytest.mark.parametrize("scope", ["common", "user", "no-user", "later-user"])
def test_skip_indexing_rejects_deletion_before_authentication(
    deletion_args: list[str], scope: str
) -> None:
    common = ["--skip-indexing-check", *deletion_args]
    if scope == "common":
        args = [*common, "-u", "test@example.invalid", "--auth-only"]
    elif scope == "no-user":
        args = common
    elif scope == "later-user":
        args = ["-u", "first", "--auth-only", "-u", "second", "--auth-only", *common]
    else:
        args = ["-u", "test@example.invalid", "--auth-only", *common]
    with (
        patch("icloudpd.base.authenticator") as authenticate,
        patch("icloudpd.cli.run_with_configs") as run,
    ):
        result = run_main_env({}, args)
    assert result.exit_code == 2
    assert "--skip-indexing-check cannot be combined" in result.output
    authenticate.assert_not_called()
    run.assert_not_called()


@pytest.mark.parametrize(
    "deletion",
    [
        {"auto_delete": True},
        {"delete_after_download": True},
        {"keep_icloud_recent_days": 0},
        {"keep_icloud_recent_days": 7},
    ],
)
def test_programmatic_config_rejects_deletion_before_side_effects(deletion: dict[str, Any]) -> None:
    global_config, configs = parse(["-u", "test@example.invalid", "--auth-only"])
    config = replace(configs[0], skip_indexing_check=True, **deletion)
    with (
        patch("icloudpd.base.create_logger") as create_logger,
        patch("icloudpd.base.authenticator") as authenticate,
    ):
        assert run_with_configs(global_config, [config]) == 2
    create_logger.assert_not_called()
    authenticate.assert_not_called()


def test_skip_indexing_warns_and_still_queries_photos(caplog: pytest.LogCaptureFixture) -> None:
    session = Mock()
    with caplog.at_level(logging.WARNING, logger="pyicloud_ipd.services.photos"):
        library = PhotoLibrary(
            "https://example.invalid", {}, session, {"zoneName": "PrimarySync"}, "private", True
        )
    session.post.assert_not_called()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "indexing may be incomplete" in caplog.text
    assert "photos may be omitted" in caplog.text
    assert "success does not guarantee a complete backup" in caplog.text
    session.post.return_value = Mock(json=Mock(return_value={"records": []}))
    assert list(library.all.photos) == []
    assert session.post.call_count == 1
    query = json.loads(session.post.call_args.kwargs["data"])
    assert query["query"]["recordType"] == library.WHOLE_COLLECTION["list_type"]


@pytest.mark.parametrize("operation", ["zones", "folders", "count", "photos"])
def test_skip_indexing_does_not_swallow_real_query_errors(operation: str) -> None:
    session = Mock()
    photos = PhotosService("https://example.invalid", session, {}, skip_indexing_check=True)
    session.post.side_effect = PyiCloudConnectionErrorException("offline")
    with pytest.raises(PyiCloudConnectionErrorException, match="offline"):
        if operation == "zones":
            _ = photos.private_libraries
        elif operation == "folders":
            _ = photos.albums
        elif operation == "count":
            _ = len(photos.all)
        else:
            _ = list(photos.all.photos)
    assert photos._private_libraries is None
    assert session.post.call_count == 1


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
