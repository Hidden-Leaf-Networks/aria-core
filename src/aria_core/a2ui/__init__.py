"""A2UI (Agent-to-UI) Generative UI protocol for Aria Core.

Implements the A2UI protocol (inspired by Google A2UI v0.9) enabling agents
to project dynamic UI components at runtime. Instead of returning text-only
responses, agents declare structured UI — forms, tables, charts, cards — and
the client renders them.

ARIA-292

Provides:
- A2UIComponent: Base model for all UI components
- Built-in types: TextBlock, FormField, ActionForm, DataTable, MetricCard,
  StatusBadge, ProgressBar, ActionButton, CardLayout, TabLayout
- A2UIRenderer: JSON serialization / deserialization
- A2UIBuilder: Fluent builder API for constructing component trees
"""

from aria_core.a2ui.protocol import (
    A2UIBuilder,
    A2UIComponent,
    A2UIRenderer,
    ActionButton,
    ActionForm,
    BadgeState,
    ButtonVariant,
    CardLayout,
    ComponentType,
    DataTable,
    FieldType,
    FormField,
    MetricCard,
    ProgressBar,
    StatusBadge,
    TabItem,
    TabLayout,
    TextBlock,
    Trend,
)

__all__ = [
    "A2UIBuilder",
    "A2UIComponent",
    "A2UIRenderer",
    "ActionButton",
    "ActionForm",
    "BadgeState",
    "ButtonVariant",
    "CardLayout",
    "ComponentType",
    "DataTable",
    "FieldType",
    "FormField",
    "MetricCard",
    "ProgressBar",
    "StatusBadge",
    "TabItem",
    "TabLayout",
    "TextBlock",
    "Trend",
]
