from __future__ import annotations

KOYFIN_BASE_URL = "https://app.koyfin.com"
KOYFIN_LOGIN_URL = f"{KOYFIN_BASE_URL}/login"
KOYFIN_TRANSCRIPTS_URL = f"{KOYFIN_BASE_URL}/search/transcripts"

LOGIN_EMAIL = [
    "input[type='email']",
    "input[name='email']",
]
LOGIN_PASSWORD = [
    "input[type='password']",
    "input[name='password']",
]
LOGIN_SUBMIT = [
    "button[type='submit']",
    "button:has-text('Log In')",
    "button:has-text('Sign In')",
]

ADVANCED_SEARCH_NAV = [
    "role=link[name='Advanced Search']",
    "role=button[name='Advanced Search']",
    "text=Advanced Search",
]
TRANSCRIPTS_SEARCH_NAV = [
    "role=link[name='Transcripts Search']",
    "role=button[name='Transcripts Search']",
    "text=Transcripts Search",
]

TRANSCRIPTS_READY = [
    "button:has-text('Search Transcripts')",
    "text=Event Types",
    "text=Date Range",
]

# Primary + user-provided fallback selectors for date controls.
DATE_RANGE_BOX = [
    "div.time-range__root___mMPxy.finder-time-frame__finderTimeRange___nlkDl",
    "div[class*='time-range__root']",
    "text=Mar 22 2006 - Mar 22 2026",
    "div.time-range__root___mMPxy.finder-time-frame__finderTimeRange___nlkDl",
    "#root > div > div:nth-child(1) > section > div.base-container__main___eLLNG > div.base-container__content___amfQJ.base-container__rootWrap___p9R1X > div.base-container__contentWrap___DY7P1 > div.base-container__mainContentWrap___l7NWf > div > div > div.box__box___QniKz.box__box__autoScrollbar___XH_2f.box__roundedScrollBar___OM0Ay > div.finder-top-search__finderTopSearch__root___nRYMy > div.finder-time-frame__koy__buttonInputContainer___mYAII > div.time-range__root___mMPxy.finder-time-frame__finderTimeRange___nlkDl",
    "div:has-text('Date Range')",
]
DATE_BEGIN_INPUT = [
    "input[placeholder='MM/DD/YYYY'] >> nth=0",
    "input[placeholder*='MM/DD/YYYY'] >> nth=0",
    "body > div:nth-child(60) > div > div > div > div > div.range-date-picker__body___cFprH > div:nth-child(1) > div > div.date-picker__datePicker__inputContainer___G6kfK > div > div > input",
]
DATE_END_INPUT = [
    "input[placeholder='MM/DD/YYYY'] >> nth=1",
    "input[placeholder*='MM/DD/YYYY'] >> nth=1",
    "body > div:nth-child(60) > div > div > div > div > div.range-date-picker__body___cFprH > div:nth-child(2) > div > div.date-picker__datePicker__inputContainer___G6kfK > div > div > input",
]
DATE_APPLY = [
    "button:has-text('Apply Dates')",
    "button:has-text('Apply')",
    "button:has-text('Search Transcripts')",
    "body > div:nth-child(60) > div > div > div > div > div.range-date-picker__footer___UVLYu > button:nth-child(2) > label",
]

EARNINGS_HINTS = ["earnings call"]

# Content anchors provided by user + fallbacks.
ARTICLE_CONTAINER = [
    "div.common-news-article-content__commonNewsArticleContent__root___IpOjT",
    "div[class*='common-news-article-content__commonNewsArticleContent__root']",
    "div[class*='news-article-panel__newsArticlePanel__root']",
]
SLATE_CONTAINER = [
    "div.slate-editor__container___A7llH",
    "div[class*='slate-editor__container']",
]
SLATE_EDITOR = [
    "div.slate-editor__editor___TYaJ6",
    "div[data-slate-editor='true']",
    "div[class*='slate-editor__editor']",
]
PARTICIPANTS = [
    "div:has-text('Event Participants')",
    "div[class*='transcript-speakers__transcriptSpeakers__content']",
]
HEADER = [
    "div[class*='article-title__articleTitle']",
]
CLOSE_PANEL = [
    "button:has-text('Close')",
    "button[aria-label='Close']",
    "button:has-text('Back')",
]
NEXT_PAGE = [
    "button[aria-label*='Next page']",
    "button[aria-label*='Next']",
    "button:has-text('Next')",
    "button:has-text('›')",
]
