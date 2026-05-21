"""A2UI (Agent-to-UI) Generative UI protocol for Aria Core.

Implements the A2UI protocol (inspired by Google A2UI v0.9) enabling agents
to project dynamic UI components at runtime instead of text-only responses.

The agent declares what UI it wants (form, table, chart, card) and the client
renders it. Components are serializable Pydantic models that round-trip
through JSON.

ARIA-292
"""

from __future__ import annotations

import sys
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from aria_core.runtime.models import BaseModel

# Python 3.10 compat
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        def __new__(cls, value: str) -> StrEnum:
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def __str__(self) -> str:
            return self.value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ComponentType(StrEnum):
    """Built-in A2UI component types."""

    TEXT_BLOCK = "text_block"
    FORM_FIELD = "form_field"
    ACTION_FORM = "action_form"
    DATA_TABLE = "data_table"
    METRIC_CARD = "metric_card"
    STATUS_BADGE = "status_badge"
    PROGRESS_BAR = "progress_bar"
    ACTION_BUTTON = "action_button"
    CARD_LAYOUT = "card_layout"
    TAB_LAYOUT = "tab_layout"


class Trend(StrEnum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class BadgeState(StrEnum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


class ButtonVariant(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    DANGER = "danger"


class FieldType(StrEnum):
    TEXT = "text"
    NUMBER = "number"
    SELECT = "select"
    TOGGLE = "toggle"


# ---------------------------------------------------------------------------
# Component models
# ---------------------------------------------------------------------------

class A2UIComponent(BaseModel):
    """Base component — every UI element is one of these."""

    type: str
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    props: dict[str, Any] = Field(default_factory=dict)
    children: list[A2UIComponent] = Field(default_factory=list)


class TextBlock(A2UIComponent):
    """Text with optional formatting."""

    type: str = ComponentType.TEXT_BLOCK
    text: str = ""
    bold: bool = False
    code: bool = False
    heading_level: int | None = None  # 1-6 or None


class FormField(A2UIComponent):
    """Single input field."""

    type: str = ComponentType.FORM_FIELD
    label: str = ""
    field_type: FieldType = FieldType.TEXT
    options: list[str] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    required: bool = False


class ActionForm(A2UIComponent):
    """Group of FormFields with a submit button."""

    type: str = ComponentType.ACTION_FORM
    fields: list[FormField] = Field(default_factory=list)
    submit_label: str = "Submit"
    action_id: str = ""


class DataTable(A2UIComponent):
    """Tabular data display."""

    type: str = ComponentType.DATA_TABLE
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    sortable: bool = False


class MetricCard(A2UIComponent):
    """KPI / metric display."""

    type: str = ComponentType.METRIC_CARD
    label: str = ""
    value: str = ""
    trend: Trend = Trend.NEUTRAL


class StatusBadge(A2UIComponent):
    """Status indicator."""

    type: str = ComponentType.STATUS_BADGE
    label: str = ""
    state: BadgeState = BadgeState.INFO


class ProgressBar(A2UIComponent):
    """Progress indicator (0-100)."""

    type: str = ComponentType.PROGRESS_BAR
    value: int = Field(default=0, ge=0, le=100)
    label: str = ""


class ActionButton(A2UIComponent):
    """Clickable action trigger."""

    type: str = ComponentType.ACTION_BUTTON
    label: str = ""
    action_id: str = ""
    variant: ButtonVariant = ButtonVariant.PRIMARY


class CardLayout(A2UIComponent):
    """Card container with title, subtitle, body, and footer actions."""

    type: str = ComponentType.CARD_LAYOUT
    title: str = ""
    subtitle: str = ""
    body: list[A2UIComponent] = Field(default_factory=list)
    footer_actions: list[ActionButton] = Field(default_factory=list)


class TabLayout(A2UIComponent):
    """Tabbed container."""

    type: str = ComponentType.TAB_LAYOUT
    tabs: list[TabItem] = Field(default_factory=list)


class TabItem(BaseModel):
    """Single tab within a TabLayout."""

    label: str
    content: list[A2UIComponent] = Field(default_factory=list)


# Rebuild TabLayout now that TabItem is defined
TabLayout.model_rebuild()


# ---------------------------------------------------------------------------
# Type registry for deserialization
# ---------------------------------------------------------------------------

_COMPONENT_REGISTRY: dict[str, type[A2UIComponent]] = {
    ComponentType.TEXT_BLOCK: TextBlock,
    ComponentType.FORM_FIELD: FormField,
    ComponentType.ACTION_FORM: ActionForm,
    ComponentType.DATA_TABLE: DataTable,
    ComponentType.METRIC_CARD: MetricCard,
    ComponentType.STATUS_BADGE: StatusBadge,
    ComponentType.PROGRESS_BAR: ProgressBar,
    ComponentType.ACTION_BUTTON: ActionButton,
    ComponentType.CARD_LAYOUT: CardLayout,
    ComponentType.TAB_LAYOUT: TabLayout,
}


# ---------------------------------------------------------------------------
# Renderer — serialize / deserialize
# ---------------------------------------------------------------------------

class A2UIRenderer:
    """Serializes and deserializes A2UI component trees."""

    @staticmethod
    def _serialize_component(component: A2UIComponent) -> dict[str, Any]:
        """Serialize a single component, preserving actual subclass fields."""
        data = component.model_dump(mode="json")
        # Recursively serialize nested component lists to preserve subclass fields
        if isinstance(component, CardLayout):
            data["body"] = [A2UIRenderer._serialize_component(b) for b in component.body]
            data["footer_actions"] = [
                A2UIRenderer._serialize_component(a) for a in component.footer_actions
            ]
        if isinstance(component, ActionForm):
            data["fields"] = [A2UIRenderer._serialize_component(f) for f in component.fields]
        if isinstance(component, TabLayout):
            data["tabs"] = [
                {
                    "label": tab.label,
                    "content": [A2UIRenderer._serialize_component(c) for c in tab.content],
                }
                for tab in component.tabs
            ]
        if component.children:
            data["children"] = [
                A2UIRenderer._serialize_component(c) for c in component.children
            ]
        return data

    @staticmethod
    def render(components: list[A2UIComponent]) -> dict[str, Any]:
        """Serialize a list of components to a JSON-compatible dict."""
        return {
            "a2ui_version": "0.9",
            "components": [A2UIRenderer._serialize_component(c) for c in components],
        }

    @staticmethod
    def _deserialize_component(data: dict[str, Any]) -> A2UIComponent:
        """Deserialize a single component dict into its typed model."""
        comp_type = data.get("type", "")
        cls = _COMPONENT_REGISTRY.get(comp_type, A2UIComponent)

        # Recursively deserialize children
        if "children" in data and data["children"]:
            data = dict(data)
            data["children"] = [
                A2UIRenderer._deserialize_component(child) for child in data["children"]
            ]

        # Recursively deserialize nested component lists
        if cls is CardLayout:
            data = dict(data)
            if "body" in data and data["body"]:
                data["body"] = [
                    A2UIRenderer._deserialize_component(b) for b in data["body"]
                ]
            if "footer_actions" in data and data["footer_actions"]:
                data["footer_actions"] = [
                    A2UIRenderer._deserialize_component(a) for a in data["footer_actions"]
                ]

        if cls is ActionForm:
            data = dict(data)
            if "fields" in data and data["fields"]:
                data["fields"] = [
                    A2UIRenderer._deserialize_component(f) for f in data["fields"]
                ]

        if cls is TabLayout:
            data = dict(data)
            if "tabs" in data and data["tabs"]:
                tabs = []
                for tab in data["tabs"]:
                    tab = dict(tab)
                    if "content" in tab and tab["content"]:
                        tab["content"] = [
                            A2UIRenderer._deserialize_component(c) for c in tab["content"]
                        ]
                    tabs.append(TabItem(**tab))
                data["tabs"] = tabs

        return cls(**data)

    @staticmethod
    def from_json(data: dict[str, Any]) -> list[A2UIComponent]:
        """Deserialize a JSON dict back into typed component models."""
        raw_components = data.get("components", [])
        return [A2UIRenderer._deserialize_component(c) for c in raw_components]


# ---------------------------------------------------------------------------
# Builder — fluent API
# ---------------------------------------------------------------------------

class A2UIBuilder:
    """Fluent builder for constructing A2UI component lists."""

    def __init__(self) -> None:
        self._components: list[A2UIComponent] = []

    def text(self, content: str) -> A2UIBuilder:
        """Add a text block."""
        self._components.append(TextBlock(text=content))
        return self

    def heading(self, text: str, level: int = 1) -> A2UIBuilder:
        """Add a heading text block."""
        self._components.append(TextBlock(text=text, heading_level=level, bold=True))
        return self

    def code(self, text: str) -> A2UIBuilder:
        """Add a code text block."""
        self._components.append(TextBlock(text=text, code=True))
        return self

    def field(
        self,
        label: str,
        field_type: str = "text",
        options: list[str] | None = None,
        required: bool = False,
        validation: dict[str, Any] | None = None,
    ) -> FormField:
        """Create a standalone form field (also usable inside `.form()`)."""
        f = FormField(
            label=label,
            field_type=FieldType(field_type),
            options=options or [],
            required=required,
            validation=validation or {},
        )
        self._components.append(f)
        return f

    def form(
        self,
        fields: list[FormField],
        submit_label: str = "Submit",
        action_id: str = "",
    ) -> A2UIBuilder:
        """Add an action form."""
        self._components.append(
            ActionForm(fields=fields, submit_label=submit_label, action_id=action_id)
        )
        return self

    def table(
        self,
        columns: list[str],
        rows: list[list[Any]],
        sortable: bool = False,
    ) -> A2UIBuilder:
        """Add a data table."""
        self._components.append(DataTable(columns=columns, rows=rows, sortable=sortable))
        return self

    def metric(
        self,
        label: str,
        value: str,
        trend: str = "neutral",
    ) -> A2UIBuilder:
        """Add a metric card."""
        self._components.append(MetricCard(label=label, value=value, trend=Trend(trend)))
        return self

    def badge(self, label: str, state: str = "info") -> A2UIBuilder:
        """Add a status badge."""
        self._components.append(StatusBadge(label=label, state=BadgeState(state)))
        return self

    def progress(self, value: int, label: str = "") -> A2UIBuilder:
        """Add a progress bar."""
        self._components.append(ProgressBar(value=value, label=label))
        return self

    def button(
        self,
        label: str,
        action_id: str,
        variant: str = "primary",
    ) -> A2UIBuilder:
        """Add an action button."""
        self._components.append(
            ActionButton(label=label, action_id=action_id, variant=ButtonVariant(variant))
        )
        return self

    def card(
        self,
        title: str,
        subtitle: str = "",
        body: list[A2UIComponent] | None = None,
        footer_actions: list[ActionButton] | None = None,
    ) -> A2UIBuilder:
        """Add a card layout."""
        self._components.append(
            CardLayout(
                title=title,
                subtitle=subtitle,
                body=body or [],
                footer_actions=footer_actions or [],
            )
        )
        return self

    def tabs(self, items: list[tuple[str, list[A2UIComponent]]]) -> A2UIBuilder:
        """Add a tab layout. Each item is (label, content_components)."""
        tab_items = [TabItem(label=label, content=content) for label, content in items]
        self._components.append(TabLayout(tabs=tab_items))
        return self

    def build(self) -> list[A2UIComponent]:
        """Return the accumulated component list."""
        return list(self._components)
