# Exploratory checks

- Initial standard-library venv creation lacked ensurepip; bootstrapped the official pip.pyz only inside the private venv. No system package changes.
- Initial transplant of all 26 F01 tests to the new Stream gave 25 PASS and 1 harness error: the serialization fault patched the old encoder, which the prototype deliberately replaces. The inherited test now injects the identical failure at canonical_fast and retains all original assertions. This was a test injection boundary mismatch, not a production failure or an omitted test.
- Initial smoke/calibration runs are exploratory and excluded from performance summaries.

- Additional ASCII probe found a real native-encoder mismatch for U+007F (DEL). Fixed by reference fallback, added all-ASCII regression before native optimization measurements. Completed scaling reference-path data retain the earlier SOURCE inventory; no live measured source was edited.

- Publication checks found CSV CRLF flagged by git diff --check; normalized only the derived CSV to LF, updated its SHA, and passed the check. Initial CLI push could not authenticate; the GitHub App publication preserves identical Git tree/bytes while source commit objects are separately retained in a verified bundle.
