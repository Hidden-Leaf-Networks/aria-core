"""Phone agent — multi-provider telephony for voice-driven agents.

Provides:
- CallConfig: provider credentials and call settings (Plivo, Twilio, Vonage)
- CallRecord: immutable record of inbound/outbound calls
- PhoneAgent: initiate, handle, and manage calls per tenant

Implements ARIA-312 phone agent capabilities.
"""

from aria_core.phone.agent import CallConfig, CallRecord, PhoneAgent

__all__ = [
    "CallConfig",
    "CallRecord",
    "PhoneAgent",
]
