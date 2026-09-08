"""Calendar auth failures stay distinct from empty results and outages.

Exercise the real auth classifier, with credentials, discovery, API execution,
reauth persistence, and Chat alerts stubbed so these tests never use the network.
"""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import httplib2
import pytest
from test_reauth_tools import real_auth_modules  # shared, restoring isolation fixture

if TYPE_CHECKING:  # runtime bindings come from calendar_env, never collection imports
    import calendar_service
    import google_auth
    from connection_errors import ExternalAuthError, ExternalUnavailableError
    from googleapiclient.errors import HttpError


def _http_error(status):
    return HttpError(
        httplib2.Response({"status": str(status)}),
        b'{"error": {"message": "calendar request failed"}}',
    )


@pytest.fixture
def calendar_env(monkeypatch, real_auth_modules):
    # Imports must happen at execution time, inside the private module graph.
    # Collection-time sibling Google mocks are intentionally left untouched.
    from googleapiclient.errors import HttpError as RealHttpError

    for name in ("calendar_service", "google_auth"):
        monkeypatch.setitem(globals(), name, real_auth_modules[name])
    for name in ("ExternalAuthError", "ExternalUnavailableError"):
        monkeypatch.setitem(globals(), name, getattr(real_auth_modules["connection_errors"], name))
    monkeypatch.setitem(globals(), "HttpError", RealHttpError)
    credentials = MagicMock(return_value=object())
    build = MagicMock()
    execute = build.return_value.events.return_value.list.return_value.execute
    execute.return_value = {"items": []}
    mark = MagicMock()
    alert = MagicMock()
    monkeypatch.setattr(calendar_service, "get_credentials", credentials)
    monkeypatch.setattr(calendar_service, "build", build)
    monkeypatch.setattr(google_auth, "_mark_reauth_required", mark)
    monkeypatch.setattr(google_auth, "_send_throttled_reauth_alert", alert)
    return {
        "credentials": credentials, "build": build, "execute": execute,
        "mark": mark, "alert": alert,
    }


@pytest.mark.parametrize("fetch", [
    "fetch_todays_meetings", "fetch_meetings_for_date",
    "fetch_upcoming_meetings", "fetch_recently_ended_meetings",
], ids=["today", "date", "upcoming", "recently_ended"])
def test_credential_reauth_failure_is_typed_for_all_event_fetchers(calendar_env, fetch):
    failure = google_auth.ReauthRequiredError("Google credentials need re-auth")
    calendar_env["credentials"].side_effect = failure

    with pytest.raises(ExternalAuthError) as caught:
        args = ("2026-09-08",) if fetch == "fetch_meetings_for_date" else ()
        getattr(calendar_service, fetch)(*args)

    assert caught.value.connector == "google_workspace"
    assert "calendar_events" in str(caught.value)
    assert caught.value.__cause__ is failure
    calendar_env["credentials"].assert_called_once_with()
    calendar_env["build"].assert_not_called()
    calendar_env["mark"].assert_not_called()
    calendar_env["alert"].assert_not_called()


@pytest.mark.parametrize("boundary", ["credentials", "build", "execute"])
@pytest.mark.parametrize("status", [401, 403])
def test_http_auth_failure_is_typed_across_complete_boundary(calendar_env, boundary, status):
    failure = _http_error(status)
    calendar_env[boundary].side_effect = failure

    with pytest.raises(ExternalAuthError) as caught:
        calendar_service.fetch_upcoming_meetings()

    assert caught.value.connector == "google_workspace"
    assert caught.value.status == status
    assert caught.value.__cause__ is failure
    calendar_env[boundary].assert_called_once()
    calendar_env["mark"].assert_called_once_with(
        reason=f"http_{status}", source="calendar_events",
    )
    calendar_env["alert"].assert_called_once()


@pytest.mark.parametrize("boundary", ["credentials", "build", "execute"])
@pytest.mark.parametrize("failure", [
    lambda: RuntimeError("No valid Google credentials found"),
    lambda: ConnectionError("Calendar unreachable"),
    lambda: ExternalUnavailableError("google_workspace", "Calendar unavailable"),
    lambda: _http_error(429),
    lambda: _http_error(503),
], ids=["credential_unavailable", "connection", "typed_unavailable", "rate_limit", "http_unavailable"])
def test_non_auth_failure_propagates_unchanged(calendar_env, boundary, failure):
    failure = failure()
    calendar_env[boundary].side_effect = failure

    with pytest.raises(type(failure)) as caught:
        calendar_service.fetch_upcoming_meetings()

    assert caught.value is failure
    calendar_env[boundary].assert_called_once()
    calendar_env["mark"].assert_not_called()
    calendar_env["alert"].assert_not_called()


def test_legitimately_empty_calendar_returns_empty_list(calendar_env):
    start = datetime(2026, 9, 8, tzinfo=timezone.utc)
    end = start + timedelta(hours=4)

    assert calendar_service._fetch_events(start, end) == []

    calendar_env["build"].assert_called_once_with(
        "calendar", "v3", credentials=calendar_env["credentials"].return_value,
    )
    calendar_env["build"].return_value.events.return_value.list.assert_called_once_with(
        calendarId="primary", timeMin=start.isoformat(), timeMax=end.isoformat(),
        singleEvents=True, orderBy="startTime", maxResults=50,
    )
    calendar_env["execute"].assert_called_once_with()
    calendar_env["mark"].assert_not_called()
    calendar_env["alert"].assert_not_called()
