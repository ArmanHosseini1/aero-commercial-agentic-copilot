import json
from contextlib import ExitStack
from time import perf_counter

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from aero_agentic_copilot.agents._agent_clients import GeminiClient
from aero_agentic_copilot.agents.contracts_rag_agent import ContractsRagAgent
from aero_agentic_copilot.agents.customer360_agent import Customer360Agent
from aero_agentic_copilot.agents.schemas import (
    AgentOutput,
    AgentStepTrace,
    BriefingResponse,
)
from aero_agentic_copilot.agents.survey_agent import SurveyIntelligenceAgent
from aero_agentic_copilot.config import settings
from aero_agentic_copilot.llm_utils import call_with_retry
from aero_agentic_copilot.rag.ingestion import create_qdrant_client


class AgentRoutingPlan(BaseModel):
    use_customer360: bool
    use_surveys: bool
    use_contracts: bool
    customer360_query: str | None = None
    survey_query: str | None = None
    airline_name: str | None = None
    domain: str | None = None
    max_rating: int | None = Field(default=None, ge=1, le=5)
    contracts_query: str | None = None


class BriefingDraft(BaseModel):
    executive_summary: str
    key_findings: list[str]
    recommended_actions: list[str]
    citations: list[str]


class AeroCommercialOrchestrator:
    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        genai_client: GeminiClient | None = None,
    ) -> None:
        if genai_client is None and not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is required to use the orchestrator.")

        with ExitStack() as resources:
            if genai_client is None:
                genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)
                resources.callback(genai_client.close)
            if qdrant_client is None:
                qdrant_client = create_qdrant_client()
                resources.callback(qdrant_client.close)

            self.genai_client = genai_client
            self.qdrant_client = qdrant_client
            self.customer360_agent = Customer360Agent(client=genai_client)
            self.survey_agent = SurveyIntelligenceAgent(
                qdrant_client=qdrant_client,
                client=genai_client,
            )
            self.contracts_agent = ContractsRagAgent(
                qdrant_client=qdrant_client,
                client=genai_client,
            )
            self._resources = resources.pop_all()

        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._resources.close()

    def __enter__(self) -> "AeroCommercialOrchestrator":
        if self._closed:
            raise RuntimeError("The orchestrator is closed.")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _route_query(self, query: str) -> AgentRoutingPlan:
        prompt = f"""Choose the specialist agents needed to answer the commercial query.
Create a focused sub-query for every selected agent. Use only a filter that is explicit
or unambiguously implied by the user's question. Set unused sub-queries and filters to
null. Set all use_* values to false only when the available agents cannot help.

Commercial query:
{query}
"""
        response = call_with_retry(
            lambda: self.genai_client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AgentRoutingPlan.model_json_schema(),
                    temperature=0,
                ),
            )
        )
        if not response.text or not response.text.strip():
            raise ValueError("Gemini returned an empty agent-routing plan.")
        plan = AgentRoutingPlan.model_validate_json(response.text)
        if plan.use_customer360 and not (
            plan.customer360_query and plan.customer360_query.strip()
        ):
            raise ValueError("A customer360 sub-query is required when that agent is selected.")
        if plan.use_surveys and not (plan.survey_query and plan.survey_query.strip()):
            raise ValueError("A survey sub-query is required when that agent is selected.")
        if plan.use_contracts and not (
            plan.contracts_query and plan.contracts_query.strip()
        ):
            raise ValueError("A contracts sub-query is required when that agent is selected.")
        return plan

    @staticmethod
    def _required_subquery(query: str | None, agent_name: str) -> str:
        if query is None or not query.strip():
            raise ValueError(f"A {agent_name} sub-query is required.")
        return query

    def _synthesize(
        self,
        query: str,
        outputs: list[tuple[str, AgentOutput]],
    ) -> BriefingDraft:
        available_citations = list(
            dict.fromkeys(source for _, output in outputs for source in output.sources)
        )
        evidence = [
            {
                "agent": agent_name,
                "summary": output.summary,
                "data": output.data,
                "sources": output.sources,
            }
            for agent_name, output in outputs
        ]
        prompt = f"""Synthesize a concise executive briefing for this commercial query:
{query}

ANTI-HALLUCINATION REQUIREMENT: cite ONLY facts, numbers, survey_ids, and document
sections returned by the agents below. Do not introduce outside knowledge or infer
unsupported facts. Cite factual key findings using only the exact strings in the
available citations list. Clearly state when requested data is missing or no agent
returned supporting evidence. Recommended actions must be framed as recommendations,
not as established facts. Return JSON with executive_summary, key_findings,
recommended_actions, and citations.

Available citations:
{json.dumps(available_citations, ensure_ascii=False)}

Agent outputs:
{json.dumps(evidence, ensure_ascii=False)}
"""
        response = call_with_retry(
            lambda: self.genai_client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BriefingDraft.model_json_schema(),
                    temperature=0,
                ),
            )
        )
        if not response.text or not response.text.strip():
            raise ValueError("Gemini returned an empty briefing.")
        draft = BriefingDraft.model_validate_json(response.text)
        unknown_citations = set(draft.citations) - set(available_citations)
        if unknown_citations:
            raise ValueError(
                "Gemini returned citations that were not supplied by an agent: "
                + ", ".join(sorted(unknown_citations))
            )
        return draft

    def run(self, query: str) -> BriefingResponse:
        if self._closed:
            raise RuntimeError("The orchestrator is closed.")
        if not query.strip():
            raise ValueError("Commercial query cannot be empty.")

        routing_started_at = perf_counter()
        plan = self._route_query(query)
        routing_latency_ms = (perf_counter() - routing_started_at) * 1000
        outputs: list[tuple[str, AgentOutput]] = []
        if plan.use_customer360:
            outputs.append(
                (
                    "Customer360Agent",
                    self.customer360_agent.run(
                        self._required_subquery(plan.customer360_query, "customer360")
                    ),
                )
            )
        if plan.use_surveys:
            outputs.append(
                (
                    "SurveyIntelligenceAgent",
                    self.survey_agent.run(
                        self._required_subquery(plan.survey_query, "survey"),
                        airline_name=plan.airline_name,
                        domain=plan.domain,
                        max_rating=plan.max_rating,
                    ),
                )
            )
        if plan.use_contracts:
            outputs.append(
                (
                    "ContractsRagAgent",
                    self.contracts_agent.run(
                        self._required_subquery(plan.contracts_query, "contracts")
                    ),
                )
            )

        synthesis_started_at = perf_counter()
        draft = self._synthesize(query, outputs)
        traces = [self._routing_trace(query, plan, routing_latency_ms)]
        traces.extend(output.trace for _, output in outputs)
        traces.append(
            AgentStepTrace(
                agent_name="AeroCommercialOrchestrator",
                tool_used="Gemini structured briefing synthesis",
                query_or_input=query,
                result_summary=(
                    f"Produced {len(draft.key_findings)} finding(s) and "
                    f"{len(draft.recommended_actions)} recommended action(s)."
                ),
                sources=draft.citations,
                latency_ms=(perf_counter() - synthesis_started_at) * 1000,
            )
        )
        return BriefingResponse(
            executive_summary=draft.executive_summary,
            key_findings=draft.key_findings,
            recommended_actions=draft.recommended_actions,
            citations=draft.citations,
            traces=traces,
        )

    @staticmethod
    def _routing_trace(
        query: str,
        plan: AgentRoutingPlan,
        latency_ms: float,
    ) -> AgentStepTrace:

        selected_agents = [
            name
            for enabled, name in (
                (plan.use_customer360, "Customer360Agent"),
                (plan.use_surveys, "SurveyIntelligenceAgent"),
                (plan.use_contracts, "ContractsRagAgent"),
            )
            if enabled
        ]
        return AgentStepTrace(
            agent_name="AeroCommercialOrchestrator",
            tool_used="Gemini structured agent routing",
            query_or_input=query,
            result_summary=(
                "Selected agents: " + ", ".join(selected_agents)
                if selected_agents
                else "No specialist agent was selected."
            ),
            sources=[],
            latency_ms=latency_ms,
        )
