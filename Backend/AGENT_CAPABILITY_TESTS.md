# Nexa agent capability test guide

This guide separates deterministic tests from live acceptance checks. The
deterministic suite does not send email, modify Google data, open applications,
or call external research providers.

## Run the automated suite

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Backend\tests -p "test_*.py"
```

The capability boundary matrix can be run by itself:

```powershell
.\.venv\Scripts\python.exe -m unittest Backend.tests.test_agent_capability_matrix
```

## What each test group covers

| Area | Test file |
|---|---|
| Workflow routing and deny-by-default tools | `test_agent_architecture.py`, `test_agent_capability_matrix.py` |
| Shared-session context, speakers, follow-ups, and privacy | `test_session_context.py` |
| Planner JSON validation and prompt window safety | `test_prompt_window.py` |
| Public capability reporting and tool selection | `test_capabilities.py` |
| Web-search fallback and research bounds | `test_search.py`, `test_web_research_bounds.py`, `test_deep_research.py` |
| Deep-research end-to-end contracts | `test_deep_research_e2e.py`, `test_deep_research_live.py` |
| Research PDF generation and path/user/session isolation | `test_research_pdf.py` |
| Gmail drafting, sending, and confirmation isolation | `test_email_manager.py`, `test_google_connectors.py` |
| Google identity and connection boundaries | `test_google_connection_identity.py` |
| MCP policy and connected-action approval | `test_mcp_policy.py` |
| Owner-profile RAG | `test_owner_profile_store.py` |
| Maps, weather, holidays, and currency connectors | `test_maps_connector.py`, `test_deep_research.py` |
| Chat API, feedback, and browser/session contracts | `test_api_sessions.py`, `test_message_feedback.py` |
| Model provider budgets and tracing | `test_llm_provider.py`, `test_langsmith_tracing.py` |

## Live acceptance checklist

Run these manually in the application after the automated suite is green.
Mark each item `OK`, `FAIL`, or `BLOCKED` and copy the visible error plus the
backend log timestamp for failures.

### Conversation and shared-session continuity

1. Ask: `Find three cafes near Indiranagar.`
2. Ask: `Which one has the best Wi-Fi?`
3. Ask: `Is it open tomorrow?`
4. Create a shared chat. User A says: `We are comparing laptops under 80000.`
5. User B says: `Focus on battery life.`
6. User A says: `Compare the second one with Dell.`

Expected: Nexa keeps the subject, understands the pronouns/ordinals, includes
the correct participant names, and asks for clarification instead of guessing
when two references are genuinely ambiguous.

### Public web research

1. Ask: `What is the latest price of ...?`
2. Ask: `Compare ... using cited sources.`
3. Ask: `Open the official website for ...`.

Expected: current-fact questions use research, sources are cited, and a
website is opened only when explicitly requested.

### Personal Google data

1. Connect Gmail and ask: `Summarize my latest three emails.`
2. Connect Drive and ask: `Find my project plan in Drive.`
3. Connect Calendar and ask: `What meetings do I have tomorrow?`.

Expected: read operations execute without mutation tools. Disconnect the
service and repeat; Nexa should ask you to connect it and must not pretend it
read anything.

### Side effects and approvals

1. Ask Nexa to draft an email.
2. Confirm the draft exists and nothing was sent.
3. Ask Nexa to send an email to an explicitly typed address.
4. Confirm the preview appears.
5. Cancel it and verify no message was delivered.
6. Ask Nexa to create/update a calendar event and cancel the approval.

Expected: external mutations pause for UI approval. Cancellation must leave the
external system unchanged. A private connected-app result must not appear to a
different participant in a shared chat.

### Windows actions

1. Ask: `What are my laptop specs?`
2. Ask: `What is my battery and Wi-Fi status?`
3. Ask: `Open Notepad.`
4. Ask: `Set brightness to 60 percent.`
5. Ask a normal question containing the word `application` and verify Nexa
   does not open an application merely because the word appeared.

Expected: inspection tools answer read-only requests; local mutations happen
only after explicit requests.

### Maps and live planning

1. Allow location and ask: `Find restaurants near me.`
2. Ask: `Give me directions from Bangalore to Mysore.`
3. Ask: `What is the weather in Delhi tomorrow?`
4. Ask: `Is 15 August a public holiday in India?`
5. Ask: `Convert 100 USD to INR.`

Expected: the correct specialized connector is used, location is used only for
the current request, and Nexa does not invent unavailable ratings, routes, or
live values.

### Documents and private knowledge

1. Attach a PDF and ask a question about it.
2. Ask for a saved-document follow-up.
3. Ask: `What projects are in your creator's resume?`
4. Ask an unrelated question and verify the resume tool is not used.

Expected: document answers stay grounded in the selected document and owner
profile answers stay grounded in the private resume index.

## Failure classification

- Wrong workflow/status: inspect `AgentArchitecture.py` routing and
  `test_agent_capability_matrix.py`.
- Wrong tool or tool leakage: inspect `tools_for_workflow()` and the capability
  matrix failure.
- Pronoun or shared-chat failure: inspect `SessionContext.py`, the visible
  message query, and `test_session_context.py`.
- Claims success without execution: inspect the tool result and finalizer path
  in `JarvisAgent.py`.
- Missing approval: inspect `MCPManager.py`, `EmailManager.py`, and the
  corresponding confirmation tests.
- Private data visible to the wrong participant: inspect MongoDB visibility
  filtering and run the shared-session checks again.
- Live provider failure: run the relevant connector tests first, then record
  provider/network status separately from agent behavior.
