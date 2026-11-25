from pydantic import BaseModel


class AgentStepTrace(BaseModel):
    agent_name: str
    tool_used: str
    query_or_input: str
    result_summary: str
    sources: list[str]
    latency_ms: float


class AgentOutput(BaseModel):
    summary: str
    data: dict | list
    sources: list[str]
    trace: AgentStepTrace


class BriefingResponse(BaseModel):
    executive_summary: str
    key_findings: list[str]
    recommended_actions: list[str]
    citations: list[str]
    traces: list[AgentStepTrace]
