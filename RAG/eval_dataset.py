"""
Evaluation Dataset for RAG Pipeline.

Contains question-answer pairs with expected contexts derived from the source documents.
Each entry includes:
  - input: The question to ask
  - expected_output: The ideal answer
  - expected_context: Relevant text that should be retrieved
  - source_file: Which file contains the answer

The dataset covers all 5 source documents:
  - company_handbook.txt (5 questions)
  - product_faq.txt (4 questions)
  - support_tickets.txt (3 questions)
  - api_refrence.txt (4 questions)
  - training_doc.txt (4 questions)
"""

from typing import TypedDict


class EvalItem(TypedDict):
    """Type definition for evaluation dataset items."""

    input: str
    expected_output: str
    expected_context: str
    source_file: str


# =============================================================================
# Evaluation Dataset
# =============================================================================

EVAL_DATASET: list[EvalItem] = [
    # =========================================================================
    # Company Handbook
    # =========================================================================
    {
        "input": "How many days of PTO do full-time employees accrue per month?",
        "expected_output": "Full-time employees accrue 1.5 days of PTO per month.",
        "expected_context": "Full-time employees accrue 1.5 days PTO per month.",
        "source_file": "company_handbook.txt",
    },
    {
        "input": "How many PTO days can be carried over to the next year?",
        "expected_output": "Up to 5 unused PTO days may be carried over into the next calendar year.",
        "expected_context": "Up to 5 unused PTO days may be carried over into the next calendar year.",
        "source_file": "company_handbook.txt",
    },
    {
        "input": "What are the core collaboration hours?",
        "expected_output": "Core collaboration hours are 10:00 AM to 3:00 PM local time.",
        "expected_context": "Core collaboration hours: 10:00 AM–3:00 PM local time.",
        "source_file": "company_handbook.txt",
    },
    {
        "input": "How many days per week can employees work remotely by default?",
        "expected_output": "Remote work is allowed up to 3 days per week by default.",
        "expected_context": "Remote work is allowed up to 3 days/week by default.",
        "source_file": "company_handbook.txt",
    },
    {
        "input": "What is the maximum meal reimbursement per day during travel?",
        "expected_output": "Meals are reimbursable up to $35 per day during travel.",
        "expected_context": "Meals are reimbursable up to $35/day during travel.",
        "source_file": "company_handbook.txt",
    },
    # =========================================================================
    # Product FAQ
    # =========================================================================
    {
        "input": "Does Acme Notes support Markdown?",
        "expected_output": "Yes, Acme Notes supports Markdown. You can toggle it in Settings → Editor.",
        "expected_context": "Yes. Toggle Markdown in Settings → Editor.",
        "source_file": "product_faq.txt",
    },
    {
        "input": "How do I reset my password in Acme Notes?",
        "expected_output": "Go to Settings → Account → Reset Password. You will receive a reset email.",
        "expected_context": "Go to Settings → Account → Reset Password. You will receive a reset email.",
        "source_file": "product_faq.txt",
    },
    {
        "input": "What are the limits of the Free plan in Acme Notes?",
        "expected_output": "The Free plan allows up to 50 notes, 3 folders, and 1 device.",
        "expected_context": "Free plan: up to 50 notes, 3 folders, and 1 device.",
        "source_file": "product_faq.txt",
    },
    {
        "input": "What app version is required for troubleshooting?",
        "expected_output": "The app version should be 2.4.0 or later for troubleshooting.",
        "expected_context": "Confirm the app version is 2.4.0 or later.",
        "source_file": "product_faq.txt",
    },
    # =========================================================================
    # Support Tickets
    # =========================================================================
    {
        "input": "What was the resolution for TICKET-1003?",
        "expected_output": (
            "TICKET-1003 was a known issue in version 2.3.9 where sync was stuck at 99% on iOS. "
            "The resolution was to upgrade to version 2.4.0."
        ),
        "expected_context": (
            'TICKET-1003\nDate: 2026-01-21\nCustomer: Fabrikam\n'
            'Issue: "Sync stuck at 99% on iOS."\nResolution: Known issue in 2.3.9; upgrade to 2.4.0.'
        ),
        "source_file": "support_tickets.txt",
    },
    {
        "input": "What caused the SSO login issue in TICKET-1002?",
        "expected_output": (
            "The SSO login issue in TICKET-1002 was caused by a misconfigured SAML audience. "
            "The customer updated their IdP settings to resolve it."
        ),
        "expected_context": "Misconfigured SAML audience; customer updated IdP settings.",
        "source_file": "support_tickets.txt",
    },
    {
        "input": "Why couldn't Adventure Works invite teammates to their workspace?",
        "expected_output": (
            "Adventure Works couldn't invite teammates because multi-user workspaces require a Team plan. "
            "They resolved it by upgrading their plan."
        ),
        "expected_context": "Team plan required for multi-user workspaces; customer upgraded plan.",
        "source_file": "support_tickets.txt",
    },
    # =========================================================================
    # API Reference
    # =========================================================================
    {
        "input": "What is the API rate limit for authenticated users?",
        "expected_output": "The rate limit is 60 requests per minute per user token for authenticated users.",
        "expected_context": "60 requests/minute per user token",
        "source_file": "api_refrence.txt",
    },
    {
        "input": "What is the base URL for the Acme Notes API?",
        "expected_output": "The base URL for the API is https://api.acme-notes.example/v1",
        "expected_context": "Base URL: https://api.acme-notes.example/v1",
        "source_file": "api_refrence.txt",
    },
    {
        "input": "What HTTP method is used to create a note in the API?",
        "expected_output": "The POST method is used to create a note via the /notes endpoint.",
        "expected_context": "POST /notes\nDescription: Create a note",
        "source_file": "api_refrence.txt",
    },
    {
        "input": "What response code indicates a note was deleted successfully?",
        "expected_output": "A 204 No Content response code indicates the note was deleted successfully.",
        "expected_context": "204 No Content: deleted successfully",
        "source_file": "api_refrence.txt",
    },
    # =========================================================================
    # Training Doc
    # =========================================================================
    {
        "input": "What are the escalation criteria for customer support?",
        "expected_output": (
            "Escalate to engineering if: the crash affects multiple customers and is reproducible, "
            "data loss is reported, or security concerns exist (account takeover, token leakage, etc.)."
        ),
        "expected_context": (
            "Escalate to engineering if:\n- Crash affects multiple customers and is reproducible\n"
            "- Data loss is reported\n- Security concerns exist (account takeover, token leakage, etc.)"
        ),
        "source_file": "training_doc.txt",
    },
    {
        "input": "What is the standard response pattern for customer support?",
        "expected_output": (
            "The standard response pattern is: 1) Acknowledge the issue, 2) Ask for minimal required details, "
            "3) Provide troubleshooting steps, 4) Offer escalation if unresolved."
        ),
        "expected_context": (
            "Standard Response Pattern\n1) Acknowledge the issue\n2) Ask for minimal required details\n"
            "3) Provide troubleshooting steps\n4) Offer escalation if unresolved"
        ),
        "source_file": "training_doc.txt",
    },
    {
        "input": "What minimal details are required when handling a support ticket?",
        "expected_output": (
            "The minimal required details are: User ID, Device type + OS version, App version, "
            "Steps to reproduce, and Screenshots (if UI-related)."
        ),
        "expected_context": (
            "Minimal Required Details\n- User ID\n- Device type + OS version\n- App version\n"
            "- Steps to reproduce\n- Screenshots (if UI-related)"
        ),
        "source_file": "training_doc.txt",
    },
    {
        "input": "What is the macro response for a sync issue?",
        "expected_output": (
            'The macro for sync issues is: "Please confirm you\'re on app version 2.4.0 or later, '
            'and try signing out/in. If the issue persists, share your user ID and iOS/Android version."'
        ),
        "expected_context": (
            "Macro: Sync Issue\n"
            '"Please confirm you\'re on app version 2.4.0 or later, and try signing out/in. '
            'If the issue persists, share your user ID and iOS/Android version."'
        ),
        "source_file": "training_doc.txt",
    },
]


# =============================================================================
# Helper Functions
# =============================================================================


def get_eval_dataset() -> list[EvalItem]:
    """Return the complete evaluation dataset."""
    return EVAL_DATASET


def get_dataset_by_source(source_file: str) -> list[EvalItem]:
    """Filter evaluation dataset by source file."""
    return [item for item in EVAL_DATASET if item["source_file"] == source_file]


def get_quick_dataset(n: int = 5) -> list[EvalItem]:
    """Return first n items for quick testing."""
    return EVAL_DATASET[:n]


# =============================================================================
# Summary
# =============================================================================

if __name__ == "__main__":
    print(f"Total evaluation samples: {len(EVAL_DATASET)}")

    # Count by source
    sources: dict[str, int] = {}
    for item in EVAL_DATASET:
        src = item["source_file"]
        sources[src] = sources.get(src, 0) + 1

    print("\nSamples per source file:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
