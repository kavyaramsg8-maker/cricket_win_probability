"""Shared interface. All states describe the position BEFORE the next ball."""
from dataclasses import dataclass

@dataclass(frozen=True)
class State:
    balls_left: int
    wickets_lost: int
    runs_required: int
    def __post_init__(self):
        if self.balls_left < 0 or not 0 <= self.wickets_lost <= 10:
            raise ValueError('Invalid balls or wickets')
    @property
    def terminal(self):
        if self.runs_required <= 0: return 1.0
        if self.balls_left == 0 or self.wickets_lost == 10: return 0.0
        return None

@dataclass(frozen=True)
class Outcome:
    runs: int
    wicket: int
    legal: int
    def __post_init__(self):
        if self.runs < 0 or self.wicket not in (0,1) or self.legal not in (0,1):
            raise ValueError('Invalid delivery outcome')
        if not self.legal and self.runs == 0:
            raise ValueError('Illegal deliveries must award at least one extra')

def advance(state, outcome):
    if state.terminal is not None: return state
    return State(state.balls_left-outcome.legal, min(10,state.wickets_lost+outcome.wicket),state.runs_required-outcome.runs)
