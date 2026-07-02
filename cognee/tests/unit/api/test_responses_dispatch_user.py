"""Regression test: the responses endpoint dispatches tools as the caller.

``/api/v1/responses`` authenticates the caller, but ``dispatch_function`` used
to unconditionally run every dispatched tool (search/cognify/add) as
``get_default_user()``, discarding the authenticated identity. Under
multi-tenant access control that let any authenticated user read and write the
default user's memory. ``dispatch_function`` now runs as the user threaded in
from the router, falling back to the default user only when none is supplied.
"""

import json
from types import SimpleNamespace

import pytest

from cognee.api.v1.responses import dispatch_function as df


def _search_tool_call(query="hello"):
    return {"function": {"name": "search", "arguments": json.dumps({"search_query": query})}}


@pytest.mark.asyncio
async def test_dispatch_runs_as_authenticated_user(monkeypatch):
    captured = {}

    async def fake_search(**kwargs):
        captured["user"] = kwargs.get("user")
        return ["ok"]

    default_used = {"called": False}

    async def fake_default_user():
        default_used["called"] = True
        return SimpleNamespace(id="default-user")

    monkeypatch.setattr(df, "search", fake_search)
    monkeypatch.setattr(df, "get_default_user", fake_default_user)

    authed_user = SimpleNamespace(id="authenticated-user")
    result = await df.dispatch_function(_search_tool_call(), authed_user)

    assert result == ["ok"]
    # The tool ran as the authenticated user, not the shared default user.
    assert captured["user"] is authed_user
    assert default_used["called"] is False


@pytest.mark.asyncio
async def test_dispatch_falls_back_to_default_user_when_none(monkeypatch):
    captured = {}

    async def fake_search(**kwargs):
        captured["user"] = kwargs.get("user")
        return ["ok"]

    default_user = SimpleNamespace(id="default-user")

    async def fake_default_user():
        return default_user

    monkeypatch.setattr(df, "search", fake_search)
    monkeypatch.setattr(df, "get_default_user", fake_default_user)

    await df.dispatch_function(_search_tool_call())

    assert captured["user"] is default_user
