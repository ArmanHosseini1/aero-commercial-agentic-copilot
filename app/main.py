from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from aero_agentic_copilot.agents.orchestrator import AeroCommercialOrchestrator
from aero_agentic_copilot.agents.schemas import AgentStepTrace, BriefingResponse
from aero_agentic_copilot.config import settings


ROOT_DIR = Path(__file__).resolve().parents[1]
SURVEYS_PATH = ROOT_DIR / "data" / "customer_surveys.csv"
PRESET_QUERIES = (
    "Prepare an Executive Briefing before our meeting with Horizon Air (fleet status, top survey pain points, and commercial levers)",
    "Analyze Cabin Customisation pain points across all customer surveys and check our catalog lead-time rules",
    "Which Tier-1 airlines have high-risk contract milestones and low NPS scores?",
)


st.set_page_config(
    page_title="Aero Commercial Agentic Copilot",
    page_icon="✈️",
    layout="wide",
)


@st.cache_resource
def get_orchestrator() -> AeroCommercialOrchestrator:
    return AeroCommercialOrchestrator()


@st.cache_data
def load_surveys(csv_path: str = str(SURVEYS_PATH)) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, parse_dates=["date"])
    required_columns = {
        "survey_id",
        "airline_name",
        "date",
        "domain",
        "rating",
        "verbatim_text",
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "Survey file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if frame.empty:
        raise ValueError(f"No survey rows found in {csv_path}.")
    frame["rating"] = pd.to_numeric(frame["rating"], errors="raise").astype(int)
    return frame


def render_trace(trace: AgentStepTrace) -> None:
    st.markdown(
        f"**{trace.agent_name}** · `{trace.tool_used}` · "
        f"`{trace.latency_ms:.1f} ms`"
    )
    if trace.agent_name == "Customer360Agent":
        st.caption("Executed SQL")
        st.code(trace.query_or_input, language="sql")
    elif trace.agent_name in ("SurveyIntelligenceAgent", "ContractsRagAgent"):
        st.caption("Search query and filters")
        st.code(trace.query_or_input, language="text")
    else:
        st.caption("Input")
        st.code(trace.query_or_input, language="text")

    st.markdown(f"**Result:** {trace.result_summary}")
    st.markdown("**Cited sources:**")
    if trace.sources:
        for source in trace.sources:
            st.markdown(f"- `{source}`")
    else:
        st.caption("No source citations.")


def render_briefing(briefing: BriefingResponse) -> None:
    st.subheader("Executive Summary")
    st.write(briefing.executive_summary)

    findings_column, actions_column = st.columns(2, gap="large")
    with findings_column:
        st.subheader("Key Findings")
        if briefing.key_findings:
            for finding in briefing.key_findings:
                st.markdown(f"- {finding}")
        else:
            st.info("No supported findings were returned.")

    with actions_column:
        st.subheader("Recommended Actions")
        if briefing.recommended_actions:
            for action in briefing.recommended_actions:
                st.markdown(f"- {action}")
        else:
            st.info("No recommendations were returned.")

    st.subheader("Citations")
    if briefing.citations:
        for citation in briefing.citations:
            st.markdown(f"- `{citation}`")
    else:
        st.info("No citations were returned.")

    with st.expander("Agent Orchestration Trace & Governance"):
        for index, trace in enumerate(briefing.traces):
            if index:
                st.divider()
            render_trace(trace)


def render_copilot_tab() -> None:
    st.markdown(
        "Ask a commercial question below, or click an example query to test "
        "the multi-agent workflow."
    )
    st.markdown("**Example queries (click to run):**")
    for index, preset in enumerate(PRESET_QUERIES):
        if st.button(
            f"Example {index + 1} — {preset}",
            key=f"preset_query_{index}",
            type="secondary",
            width="stretch",
        ):
            st.session_state["commercial_query"] = preset
            st.session_state["pending_commercial_query"] = preset

    with st.form("custom_commercial_query"):
        st.text_input(
            "Custom commercial query",
            key="commercial_query",
            placeholder="Ask about an account, survey trend, delivery risk, or contract rule",
        )
        submitted = st.form_submit_button(
            "Generate executive briefing",
            type="primary",
            width="stretch",
        )
        if submitted:
            st.session_state["pending_commercial_query"] = st.session_state[
                "commercial_query"
            ]

    pending_query = st.session_state.pop("pending_commercial_query", None)
    if pending_query is not None:
        if not pending_query.strip():
            st.warning("Enter a commercial question before submitting.")
        else:
            with st.spinner("Routing your question and preparing the briefing..."):
                briefing = get_orchestrator().run(pending_query)
            st.session_state["last_briefing"] = briefing
            st.session_state["last_briefing_query"] = pending_query

    briefing = st.session_state.get("last_briefing")
    if briefing is not None:
        last_query = st.session_state.get("last_briefing_query")
        if last_query:
            st.caption(f"Briefing for: {last_query}")
        render_briefing(briefing)


def render_survey_tab() -> None:
    surveys = load_surveys()
    total_surveys = len(surveys)
    average_rating = surveys["rating"].mean()
    critical_percentage = surveys["rating"].le(2).mean() * 100

    kpi_columns = st.columns(3)
    kpi_columns[0].metric("Total Surveys", f"{total_surveys:,}")
    kpi_columns[1].metric("Overall Avg Rating", f"{average_rating:.2f} / 5")
    kpi_columns[2].metric("% Critical Ratings ≤ 2", f"{critical_percentage:.1f}%")

    st.subheader("Survey Rating Trends")
    domain_ratings = (
        surveys.groupby("domain", as_index=False)["rating"]
        .mean()
        .sort_values("rating", ascending=True)
    )
    rating_distribution = (
        surveys.groupby(["airline_name", "rating"], as_index=False)
        .size()
        .rename(columns={"size": "responses"})
    )
    rating_distribution["rating"] = rating_distribution["rating"].astype(str)
    airline_order = sorted(surveys["airline_name"].unique().tolist())

    chart_columns = st.columns(2, gap="large")
    with chart_columns[0]:
        domain_chart = px.bar(
            domain_ratings,
            x="rating",
            y="domain",
            orientation="h",
            color="rating",
            color_continuous_scale=["#be5a49", "#d9ad54", "#2d8b83"],
            range_color=(1, 5),
            labels={"rating": "Average rating", "domain": "Survey domain"},
            title="Average Rating by Domain",
        )
        domain_chart.update_yaxes(
            categoryorder="array",
            categoryarray=domain_ratings["domain"].tolist(),
            autorange="reversed",
        )
        domain_chart.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=12, r=12, t=54, b=12),
        )
        st.plotly_chart(domain_chart, width="stretch")

    with chart_columns[1]:
        airline_chart = px.bar(
            rating_distribution,
            x="airline_name",
            y="responses",
            color="rating",
            barmode="group",
            category_orders={
                "airline_name": airline_order,
                "rating": ["1", "2", "3", "4", "5"],
            },
            color_discrete_sequence=[
                "#b84d4d",
                "#d17b4b",
                "#d1a947",
                "#4a8f8a",
                "#274c77",
            ],
            labels={
                "airline_name": "Airline",
                "responses": "Survey responses",
                "rating": "Rating",
            },
            title="Rating Distribution by Airline",
        )
        airline_chart.update_layout(
            margin=dict(l=12, r=12, t=54, b=12),
            legend_title_text="Rating",
        )
        st.plotly_chart(airline_chart, width="stretch")

    st.subheader("Verbatim Explorer")
    filter_columns = st.columns(3)
    airline_options = ["All airlines", *airline_order]
    domain_options = ["All domains", *sorted(surveys["domain"].unique().tolist())]
    with filter_columns[0]:
        selected_airline = st.selectbox("Airline", airline_options)
    with filter_columns[1]:
        selected_domain = st.selectbox("Domain", domain_options)
    with filter_columns[2]:
        rating_range = st.slider(
            "Rating range",
            min_value=1,
            max_value=5,
            value=(1, 5),
            step=1,
        )

    filtered_surveys = surveys[
        surveys["rating"].between(rating_range[0], rating_range[1])
    ]
    if selected_airline != "All airlines":
        filtered_surveys = filtered_surveys[
            filtered_surveys["airline_name"] == selected_airline
        ]
    if selected_domain != "All domains":
        filtered_surveys = filtered_surveys[
            filtered_surveys["domain"] == selected_domain
        ]
    filtered_surveys = filtered_surveys.sort_values("date", ascending=False)
    st.caption(f"{len(filtered_surveys):,} matching survey response(s)")
    st.dataframe(
        filtered_surveys[
            ["date", "survey_id", "airline_name", "domain", "rating", "verbatim_text"]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "rating": st.column_config.NumberColumn("Rating", format="%d / 5"),
            "verbatim_text": st.column_config.TextColumn("Verbatim feedback", width="large"),
        },
    )


def main() -> None:
    st.markdown(
        """
        <style>
        :root {
            --aero-navy: #142a43;
            --aero-slate: #526579;
            --aero-mist: #edf2f6;
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f8fafc 0%, #edf2f6 100%);
        }
        [data-testid="stHeader"] {
            background: rgba(248, 250, 252, 0.88);
        }
        [data-testid="stSidebar"] {
            background: #142a43;
        }
        [data-testid="stSidebar"] * {
            color: #f3f6f9;
        }
        [data-testid="stSidebar"] code {
            background-color: #243f5e !important;
            color: #ffffff !important;
            padding: 0.15rem 0.45rem;
            border-radius: 0.3rem;
        }
        [data-testid="stMetric"] {
            padding: 1rem 1.1rem;
            border: 1px solid #dce4ec;
            border-radius: 0.8rem;
            background: #ffffff;
            box-shadow: 0 4px 14px rgba(20, 42, 67, 0.06);
        }
        h1, h2, h3 {
            color: var(--aero-navy);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Aero Commercial Agentic Copilot")
    st.caption("Commercial intelligence for confident, evidence-backed decisions.")

    with st.sidebar:
        st.markdown("### System Configuration")
        st.markdown(f"**Gemini model:** `{settings.GEMINI_MODEL}`")
        st.markdown(f"**Qdrant mode:** `{settings.QDRANT_MODE}`")
        st.divider()
        st.markdown("### Three-Agent Architecture")
        st.markdown(
            "- **Customer 360 SQL** — read-only CRM and account analysis\n"
            "- **Survey Intelligence** — voice-of-customer retrieval and trends\n"
            "- **Contracts RAG** — commercial clauses and guidance"
        )
        st.caption(
            "Agents share one cached orchestrator and Qdrant client across reruns."
        )

    copilot_tab, survey_tab = st.tabs(
        ["Agentic Copilot & Executive Briefing", "Voice of Customer (Survey Analytics)"]
    )
    with copilot_tab:
        render_copilot_tab()
    with survey_tab:
        render_survey_tab()


if __name__ == "__main__":
    main()