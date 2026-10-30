# Interface refresh: 2026-10-31

Branch: ui/teaching-workspace-20261031
Baseline tag: ui-before-20261031

Scope: shared visual theme, course dashboard, navigation, authentication screens,
course material pages, source review, chat presentation and graph controls.
No model, database or graph mutation logic changes.

Validation: 133 existing offline tests passed. Browser inspection used the existing
cloud course; review/model snapshots and local runtime data are not committed.

To view the original UI locally:
    git switch local-baseline
Then restart start.cmd so Django reloads templates, and refresh the browser.
To return to the new UI:
    git switch ui/teaching-workspace-20261031
Restart the service and refresh again.

The GitHub branch stores the application under integrated_app/. Its first app
import commit is the UI baseline; the following commit contains the refresh.
Use git revert on the refresh commit for a history-preserving rollback.

Keep local.env, *.env secrets, runtime/ and .venv/ out of Git.
