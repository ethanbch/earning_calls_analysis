from __future__ import annotations

import logging

from playwright.sync_api import Page

from . import selectors
from .browser import click_first, fill_first, has_any


def _dismiss_dialogs(page: Page) -> None:
    click_first(
        page,
        [
            "button:has-text('Accept')",
            "button:has-text('Accept All')",
            "button:has-text('Got it')",
        ],
        timeout_ms=1000,
    )


def is_authenticated(page: Page) -> bool:
    _dismiss_dialogs(page)
    if has_any(page, ["text=Log In", "text=Sign In", "input[type='password']"]):
        return False
    # Treat any authenticated app surface as logged in, not just a specific search input.
    current_url = page.url.lower()
    if "app.koyfin.com" in current_url and "/login" not in current_url:
        return True
    return has_any(page, ["input[placeholder*='Search']", "aside", "nav", "text=Advanced Search"])


def login_and_save_state(
    page: Page,
    *,
    email: str,
    password: str,
    reuse_session: bool,
    logger: logging.Logger,
    manual_wait_seconds: int = 120,
) -> None:
    if reuse_session:
        page.goto(selectors.KOYFIN_TRANSCRIPTS_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(1000)
        if is_authenticated(page):
            logger.info("Reused existing authenticated session", extra={"event": "auth"})
            page.goto(selectors.KOYFIN_TRANSCRIPTS_URL, wait_until="domcontentloaded")
            return

    page.goto(selectors.KOYFIN_LOGIN_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(800)
    _dismiss_dialogs(page)

    if is_authenticated(page):
        page.goto(selectors.KOYFIN_TRANSCRIPTS_URL, wait_until="domcontentloaded")
        return

    if not fill_first(page, selectors.LOGIN_EMAIL, email):
        raise RuntimeError("Could not locate email input on login page")
    if not fill_first(page, selectors.LOGIN_PASSWORD, password):
        raise RuntimeError("Could not locate password input on login page")
    if not click_first(page, selectors.LOGIN_SUBMIT):
        raise RuntimeError("Could not submit login form")

    page.wait_for_timeout(2500)
    _dismiss_dialogs(page)
    if is_authenticated(page):
        page.goto(selectors.KOYFIN_TRANSCRIPTS_URL, wait_until="domcontentloaded")
        return

    # Headed fallback: allow manual completion (captcha/step-up/login interstitials).
    deadline_ms = max(5, manual_wait_seconds) * 1000
    logger.info(
        "Waiting for manual login completion",
        extra={"event": "auth_manual_wait", "seconds": str(manual_wait_seconds)},
    )
    waited = 0
    while waited < deadline_ms:
        page.wait_for_timeout(1000)
        waited += 1000
        _dismiss_dialogs(page)
        if is_authenticated(page):
            logger.info("Manual login detected as successful", extra={"event": "auth"})
            page.goto(selectors.KOYFIN_TRANSCRIPTS_URL, wait_until="domcontentloaded")
            return

    raise RuntimeError(
        "Login unsuccessful after manual wait. Complete login in headed browser, then retry with --reuse-session."
    )


def ensure_authenticated_context(
    page: Page,
    *,
    email: str,
    password: str,
    reuse_session: bool,
    logger: logging.Logger,
) -> None:
    if is_authenticated(page):
        page.goto(selectors.KOYFIN_TRANSCRIPTS_URL, wait_until="domcontentloaded")
        return
    login_and_save_state(
        page,
        email=email,
        password=password,
        reuse_session=reuse_session,
        logger=logger,
    )
    page.goto(selectors.KOYFIN_TRANSCRIPTS_URL, wait_until="domcontentloaded")
