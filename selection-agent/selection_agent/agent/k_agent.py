import os
from logging import INFO
from typing import List

from flwr.common import log
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama.chat_models import ChatOllama

from .prompts import K_SYSTEM_PROMPT, K_SYSTEM_PROMPT_COT, K_SYSTEM_PROMPT_FS
from .schemas import AgentSchema, ClientSchema


class KAgent:
    name: str = 'agent-selection'

    def __init__(self, llm_model_name: str, k_agent_selection_method: List[str], prompt_type: str):
        self.round_summaries = {}
        # lazy import to avoid circular imports
        from .tools import (
            get_clients_stats,
            get_cids,
            get_info_by_cid,
            get_round_summary,
        )
        self.sel_name = k_agent_selection_method
        self.prompt_type = prompt_type
        # guardrail settings: retry cap (override with env var for ablations)
        self.max_tries = max(1, int(os.environ.get("KAGENT_MAX_TRIES", "3")))
        self.last_k = None

        llm = ChatOllama(
            model=llm_model_name,
            temperature=0,
            num_predict=1024,
            num_ctx=8192,
            keep_alive="5m",
        )
        self.agent = create_agent(
            model=llm,
            tools=[
                get_clients_stats,
                get_cids,
                get_info_by_cid,
                get_round_summary,
                self.get_selection_method(k_agent_selection_method),
            ],
            state_schema=AgentSchema,
            system_prompt=self.get_prompt_type(prompt_type),
        )

    def get_prompt_type(self, prompt_type: str):
        if prompt_type == 'chain-of-thought':
            return K_SYSTEM_PROMPT_COT
        if prompt_type == 'few-shot':
            return K_SYSTEM_PROMPT_FS
        if prompt_type == 'description-only':
            return K_SYSTEM_PROMPT

    def get_selection_method(self, selection: str):
        from .tools import (
            random_selection,
            oort_selection,
            poc_selection,
            round_robin_selection,
        )
        selection_dict = {
            "random_selection": random_selection,
            "oort_selection": oort_selection,
            "poc_selection": poc_selection,
            "round_robin_selection": round_robin_selection,
        }
        return selection_dict[selection]

    def sample_clients(self, clients: List[ClientSchema], sample_clients: int, **kargs):
        current_round = kargs['curr_rnd']
        total_rounds = kargs['total_rnd']

        instruction_text = (
            f"We are in round {current_round} of {total_rounds}. "
            f"N={len(clients)}. "
            f"Choose an integer K between 2 and {len(clients)}, then call the selection tool ({self.sel_name}) exactly once."
        )
        message_history = [HumanMessage(content=instruction_text)]
        last_error = None

        for attempt in range(1, self.max_tries + 1):
            log(INFO, f"Selection tries (K-Agent): {attempt}/{self.max_tries}")
            try:
                result = self.agent.invoke({
                    "messages": message_history,
                    "clients": clients,
                    "sample_size": sample_clients,
                    "round_summaries": self.round_summaries,
                })
                selected = result.get('selected_clients') if hasattr(result, 'get') else None
                if selected:
                    out = dict(result)
                    out['agent_tries'] = attempt
                    out['agent_fallback'] = False
                    self.last_k = len(selected)
                    log(INFO, "Agent selected clients successfully.")
                    return out
                last_error = "selection tool not called"
                log(INFO, "Valid response, but selection tool not called.")
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)[:200]}"
                log(INFO, f"Selection Agent Error Type: {type(e)}")
                log(INFO, last_error)

            # Fresh, short history for the next attempt: never carry the full previous state
            message_history = [
                HumanMessage(content=instruction_text),
                HumanMessage(content=(
                    f"Attempt {attempt + 1} of {self.max_tries}. Your previous reply failed ({last_error}). "
                    f"You MUST call the tool `{self.sel_name}` with your chosen K. Do not answer in plain text."
                )),
            ]

        # Guardrail fallback: the agent failed max_tries times
        from selection_agent.selections.random_selection import Random
        n = len(clients)
        if sample_clients and sample_clients > 0:
            k = min(sample_clients, n)
        else:
            k = min(n, self.last_k or max(2, n // 2))
        log(INFO, f"K-Agent FALLBACK after {self.max_tries} failed tries: K={k} ({last_error})")
        fb = Random().sample_clients(clients, k)
        if isinstance(fb, list):
            fb = {'selected_clients': fb}
        fb['selection_algorithm'] = 'fallback-random'
        fb['messages'] = []
        fb['agent_tries'] = self.max_tries
        fb['agent_fallback'] = True
        return fb
