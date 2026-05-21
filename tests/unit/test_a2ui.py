"""Tests for A2UI (Agent-to-UI) Generative UI protocol.

ARIA-292
"""

from __future__ import annotations

import pytest

from aria_core.a2ui import (
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


# ---------------------------------------------------------------------------
# Component model tests
# ---------------------------------------------------------------------------


class TestA2UIComponent:
    def test_base_component_defaults(self) -> None:
        comp = A2UIComponent(type="custom")
        assert comp.type == "custom"
        assert len(comp.id) == 12
        assert comp.props == {}
        assert comp.children == []

    def test_text_block(self) -> None:
        tb = TextBlock(text="Hello", bold=True, heading_level=2)
        assert tb.type == ComponentType.TEXT_BLOCK
        assert tb.text == "Hello"
        assert tb.bold is True
        assert tb.heading_level == 2

    def test_text_block_code(self) -> None:
        tb = TextBlock(text="print('hi')", code=True)
        assert tb.code is True
        assert tb.bold is False

    def test_form_field(self) -> None:
        ff = FormField(
            label="Name",
            field_type=FieldType.TEXT,
            required=True,
            validation={"min_length": 1},
        )
        assert ff.type == ComponentType.FORM_FIELD
        assert ff.label == "Name"
        assert ff.field_type == FieldType.TEXT
        assert ff.required is True

    def test_form_field_select(self) -> None:
        ff = FormField(
            label="Color",
            field_type=FieldType.SELECT,
            options=["Red", "Green", "Blue"],
        )
        assert ff.options == ["Red", "Green", "Blue"]

    def test_action_form(self) -> None:
        fields = [
            FormField(label="Email", field_type=FieldType.TEXT),
            FormField(label="Count", field_type=FieldType.NUMBER),
        ]
        form = ActionForm(fields=fields, submit_label="Go", action_id="submit_form")
        assert form.type == ComponentType.ACTION_FORM
        assert len(form.fields) == 2
        assert form.submit_label == "Go"
        assert form.action_id == "submit_form"

    def test_data_table(self) -> None:
        dt = DataTable(
            columns=["Name", "Age"],
            rows=[["Alice", 30], ["Bob", 25]],
            sortable=True,
        )
        assert dt.type == ComponentType.DATA_TABLE
        assert len(dt.rows) == 2
        assert dt.sortable is True

    def test_metric_card(self) -> None:
        mc = MetricCard(label="Revenue", value="$1,200", trend=Trend.UP)
        assert mc.type == ComponentType.METRIC_CARD
        assert mc.trend == Trend.UP

    def test_status_badge(self) -> None:
        sb = StatusBadge(label="Deployed", state=BadgeState.SUCCESS)
        assert sb.type == ComponentType.STATUS_BADGE
        assert sb.state == BadgeState.SUCCESS

    def test_progress_bar(self) -> None:
        pb = ProgressBar(value=75, label="Upload")
        assert pb.type == ComponentType.PROGRESS_BAR
        assert pb.value == 75

    def test_progress_bar_validation(self) -> None:
        with pytest.raises(Exception):
            ProgressBar(value=150)  # > 100

    def test_action_button(self) -> None:
        btn = ActionButton(label="Delete", action_id="delete_item", variant=ButtonVariant.DANGER)
        assert btn.type == ComponentType.ACTION_BUTTON
        assert btn.variant == ButtonVariant.DANGER

    def test_card_layout(self) -> None:
        body = [TextBlock(text="Card body content")]
        footer = [ActionButton(label="OK", action_id="ok")]
        card = CardLayout(title="My Card", subtitle="Sub", body=body, footer_actions=footer)
        assert card.type == ComponentType.CARD_LAYOUT
        assert card.title == "My Card"
        assert len(card.body) == 1
        assert len(card.footer_actions) == 1

    def test_tab_layout(self) -> None:
        tabs = TabLayout(
            tabs=[
                TabItem(label="Tab 1", content=[TextBlock(text="Content 1")]),
                TabItem(label="Tab 2", content=[MetricCard(label="M", value="42")]),
            ]
        )
        assert tabs.type == ComponentType.TAB_LAYOUT
        assert len(tabs.tabs) == 2
        assert tabs.tabs[0].label == "Tab 1"


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------


class TestA2UIRenderer:
    def test_render_produces_valid_structure(self) -> None:
        components = [TextBlock(text="Hello"), MetricCard(label="X", value="1")]
        result = A2UIRenderer.render(components)
        assert result["a2ui_version"] == "0.9"
        assert len(result["components"]) == 2

    def test_round_trip_simple(self) -> None:
        original = [
            TextBlock(text="Hi", bold=True),
            StatusBadge(label="OK", state=BadgeState.SUCCESS),
            ProgressBar(value=50, label="Half"),
        ]
        serialized = A2UIRenderer.render(original)
        restored = A2UIRenderer.from_json(serialized)
        assert len(restored) == 3
        assert isinstance(restored[0], TextBlock)
        assert restored[0].text == "Hi"
        assert restored[0].bold is True
        assert isinstance(restored[1], StatusBadge)
        assert restored[1].state == BadgeState.SUCCESS
        assert isinstance(restored[2], ProgressBar)
        assert restored[2].value == 50

    def test_round_trip_nested_card(self) -> None:
        card = CardLayout(
            title="Dashboard",
            subtitle="Overview",
            body=[
                MetricCard(label="Users", value="1.2k", trend=Trend.UP),
                DataTable(columns=["Name"], rows=[["Alice"]]),
            ],
            footer_actions=[ActionButton(label="Refresh", action_id="refresh")],
        )
        serialized = A2UIRenderer.render([card])
        restored = A2UIRenderer.from_json(serialized)
        assert len(restored) == 1
        rc = restored[0]
        assert isinstance(rc, CardLayout)
        assert rc.title == "Dashboard"
        assert len(rc.body) == 2
        assert isinstance(rc.body[0], MetricCard)
        assert rc.body[0].trend == Trend.UP
        assert isinstance(rc.body[1], DataTable)
        assert len(rc.footer_actions) == 1

    def test_round_trip_action_form(self) -> None:
        form = ActionForm(
            fields=[
                FormField(label="Name", field_type=FieldType.TEXT, required=True),
                FormField(label="Role", field_type=FieldType.SELECT, options=["Admin", "User"]),
            ],
            submit_label="Create",
            action_id="create_user",
        )
        serialized = A2UIRenderer.render([form])
        restored = A2UIRenderer.from_json(serialized)
        rf = restored[0]
        assert isinstance(rf, ActionForm)
        assert len(rf.fields) == 2
        assert isinstance(rf.fields[0], FormField)
        assert rf.fields[1].options == ["Admin", "User"]

    def test_round_trip_tab_layout(self) -> None:
        tabs = TabLayout(
            tabs=[
                TabItem(label="Metrics", content=[MetricCard(label="V", value="99")]),
                TabItem(label="Status", content=[StatusBadge(label="Live", state=BadgeState.SUCCESS)]),
            ]
        )
        serialized = A2UIRenderer.render([tabs])
        restored = A2UIRenderer.from_json(serialized)
        rt = restored[0]
        assert isinstance(rt, TabLayout)
        assert len(rt.tabs) == 2
        assert isinstance(rt.tabs[0].content[0], MetricCard)
        assert isinstance(rt.tabs[1].content[0], StatusBadge)

    def test_from_json_empty(self) -> None:
        result = A2UIRenderer.from_json({"components": []})
        assert result == []

    def test_from_json_unknown_type_falls_back(self) -> None:
        data = {"components": [{"type": "unknown_widget", "id": "x", "props": {}, "children": []}]}
        result = A2UIRenderer.from_json(data)
        assert len(result) == 1
        assert isinstance(result[0], A2UIComponent)
        assert result[0].type == "unknown_widget"


# ---------------------------------------------------------------------------
# Builder tests
# ---------------------------------------------------------------------------


class TestA2UIBuilder:
    def test_fluent_chain(self) -> None:
        components = (
            A2UIBuilder()
            .text("Hello")
            .heading("Title", level=1)
            .code("x = 1")
            .build()
        )
        assert len(components) == 3
        assert isinstance(components[0], TextBlock)
        assert components[0].text == "Hello"
        assert isinstance(components[1], TextBlock)
        assert components[1].heading_level == 1
        assert components[1].bold is True
        assert isinstance(components[2], TextBlock)
        assert components[2].code is True

    def test_builder_table(self) -> None:
        components = (
            A2UIBuilder()
            .table(["A", "B"], [[1, 2], [3, 4]], sortable=True)
            .build()
        )
        assert len(components) == 1
        assert isinstance(components[0], DataTable)
        assert components[0].sortable is True

    def test_builder_metric(self) -> None:
        components = A2UIBuilder().metric("Sales", "$500", "up").build()
        assert isinstance(components[0], MetricCard)
        assert components[0].trend == Trend.UP

    def test_builder_button(self) -> None:
        components = A2UIBuilder().button("Remove", "rm", "danger").build()
        assert isinstance(components[0], ActionButton)
        assert components[0].variant == ButtonVariant.DANGER

    def test_builder_badge(self) -> None:
        components = A2UIBuilder().badge("OK", "success").build()
        assert isinstance(components[0], StatusBadge)
        assert components[0].state == BadgeState.SUCCESS

    def test_builder_progress(self) -> None:
        components = A2UIBuilder().progress(42, "Loading").build()
        assert isinstance(components[0], ProgressBar)
        assert components[0].value == 42

    def test_builder_card(self) -> None:
        body = [TextBlock(text="inside")]
        actions = [ActionButton(label="Go", action_id="go")]
        components = A2UIBuilder().card("Title", "Sub", body=body, footer_actions=actions).build()
        assert isinstance(components[0], CardLayout)
        assert components[0].title == "Title"
        assert len(components[0].body) == 1

    def test_builder_tabs(self) -> None:
        components = (
            A2UIBuilder()
            .tabs([
                ("Tab A", [TextBlock(text="A content")]),
                ("Tab B", [TextBlock(text="B content")]),
            ])
            .build()
        )
        assert isinstance(components[0], TabLayout)
        assert len(components[0].tabs) == 2

    def test_builder_form(self) -> None:
        fields = [
            FormField(label="Name", field_type=FieldType.TEXT),
            FormField(label="Age", field_type=FieldType.NUMBER),
        ]
        components = (
            A2UIBuilder()
            .form(fields, submit_label="Save", action_id="save_user")
            .build()
        )
        assert isinstance(components[0], ActionForm)
        assert components[0].submit_label == "Save"
        assert len(components[0].fields) == 2

    def test_builder_field(self) -> None:
        builder = A2UIBuilder()
        f = builder.field("Email", "text", required=True)
        components = builder.build()
        assert isinstance(f, FormField)
        assert isinstance(components[0], FormField)
        assert components[0].label == "Email"

    def test_full_dashboard_round_trip(self) -> None:
        """End-to-end: build a dashboard, serialize, deserialize, verify."""
        components = (
            A2UIBuilder()
            .heading("Dashboard", level=1)
            .metric("Revenue", "$12k", "up")
            .metric("Churn", "2.1%", "down")
            .table(["Customer", "Plan"], [["Acme", "Pro"], ["Beta", "Free"]])
            .badge("System", "success")
            .progress(88, "Quota")
            .button("Export", "export_csv", "secondary")
            .build()
        )

        serialized = A2UIRenderer.render(components)
        restored = A2UIRenderer.from_json(serialized)

        assert len(restored) == 7
        assert isinstance(restored[0], TextBlock)
        assert restored[0].heading_level == 1
        assert isinstance(restored[1], MetricCard)
        assert restored[1].trend == Trend.UP
        assert isinstance(restored[3], DataTable)
        assert len(restored[3].rows) == 2
        assert isinstance(restored[4], StatusBadge)
        assert isinstance(restored[5], ProgressBar)
        assert isinstance(restored[6], ActionButton)


# ---------------------------------------------------------------------------
# Integration: A2UI in AgentResult metadata
# ---------------------------------------------------------------------------


class TestAgentResultIntegration:
    def test_a2ui_in_agent_metadata(self) -> None:
        """Agents can attach A2UI components via context.metadata."""
        from aria_core.runtime.models import AgentContext, AgentResult, AgentStateEnum

        components = A2UIBuilder().text("Agent says hello").build()
        payload = A2UIRenderer.render(components)

        ctx = AgentContext(metadata={"response": "Hello", "a2ui": payload})
        result = AgentResult(state=AgentStateEnum.COMPLETE, context=ctx)

        assert result.response == "Hello"
        assert "a2ui" in result.context.metadata
        restored = A2UIRenderer.from_json(result.context.metadata["a2ui"])
        assert len(restored) == 1
        assert isinstance(restored[0], TextBlock)
        assert restored[0].text == "Agent says hello"
