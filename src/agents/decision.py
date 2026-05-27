"""
Decision module selects the next action to take based on the current state of the agent.
It receives one goal, the relevant memory hits, the recent history, optionally the raw bytes of an artifact and a list of available MCP tools.


It returns a DecisionOutput object containing the ToolCall required or the final answer.
Decision does not pick more than one tool and does not narrate.

Decision routes through the gateway with auto_route="decision". The router pool classifies the call and picks a tier. 
Most Decision calls land on the LARGE-tier Gemini model. Smaller Decision calls (planning a single tool dispatch from a short context) 
land on TINY-tier workers. The router decision is visible in the gateway's response under router_decision, and on the dashboard at port 8101.
"""

class Decision:
    def __init__(self):
        """
        Decision prompt instructions:
        The first is the choice itself: respond with exactly one of two outputs. Answer or call a tool. 
        The model is not asked to do both.

        The second is a rule about artifact handles. The model is told that strings beginning with art: are internal artifact handles. 
        They reference the artifact store. The MCP tools accept real file paths and URLs as their arguments and reject the art: prefix at dispatch time. 
        When a goal requires the bytes of an artifact, those bytes appear in the prompt under ATTACHED ARTIFACTS:. The model reads them there. 
        This rule exists because TINY-tier models occasionally hallucinate that an artifact handle is something to pass to read_file or fetch_url. 
        The Action layer also blocks this at dispatch time; the prompt instruction reduces wasted iterations.

        The third is a rule about substantive answers. When the goal asks for an extraction, a list, a comparison, or a selection, 
        the answer must be substantive: at least three sentences or a list of items. This rule exists to prevent the model from 
        returning a meta-answer ("the page has been fetched, how would you like to proceed?") instead of doing the actual work the goal requires.
        """
        self.DECISION_SYSTEM_PROMPT = """
        TODO
        """

    def next_step(
        goal: Goal,
        hits: List[MemoryItem],
        history: List[dict],
        attached: Optional[list[tuple[str, bytes]]] = None,
        tools: Optional[List[Tool]] = None,
    ) -> DecisionOutput:
        pass