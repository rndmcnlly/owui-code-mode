"""
title: Jig
author: adam
version: 0.1.0
description: A test toolkit for the Code Mode filter. Serves two purposes.
  (1) Smoke test: install this alongside the Code Mode filter on a fresh
  Open WebUI, enable both on a model, and you have a pile of harmless tools
  the model can exercise through run_python to confirm the wrapped-API path
  works end to end. (2) Signature fixture: each tool deliberately isolates
  one or more Python signature constructs (enums, arrays, tuples, unions,
  nested models, defaults, Optional, Field constraints) so you can observe
  exactly how Open WebUI's spec generator maps them into JSON tool specs,
  which is what the filter renders back into a typed-Python API. Every tool
  just echoes its arguments back as a string; nothing has side effects.
"""

from typing import List, Tuple, Dict, Optional, Union, Literal
from enum import Enum

from pydantic import BaseModel, Field


class Color(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Point(BaseModel):
    x: int
    y: int
    label: Optional[str] = None


class Tools:
    def __init__(self):
        pass

    # --- scalars: the baseline four ---------------------------------------
    def scalars(self, a: str, b: int, c: float, d: bool) -> str:
        """Four required scalar params, no defaults, no Field."""
        return f"{a} {b} {c} {d}"

    # --- scalar defaults of each type -------------------------------------
    def scalar_defaults(
        self,
        s: str = "hello",
        n: int = 7,
        f: float = 1.5,
        flag: bool = True,
    ) -> str:
        """Every scalar carries a literal default. Watch how 'default' lands."""
        return f"{s} {n} {f} {flag}"

    # --- Optional vs plain default ----------------------------------------
    def optionals(
        self,
        required: str,
        maybe: Optional[str] = None,
        maybe_int: Optional[int] = None,
    ) -> str:
        """Optional[...] with None default vs a plain required param."""
        return f"{required} {maybe} {maybe_int}"

    # --- arrays of varying item type --------------------------------------
    def arrays(
        self,
        strings: List[str],
        ints: List[int],
        floats: List[float],
        bare: list,
    ) -> str:
        """List[str], List[int], List[float], and an untyped bare list."""
        return f"{strings} {ints} {floats} {bare}"

    # --- nested array ------------------------------------------------------
    def nested_array(self, matrix: List[List[int]]) -> str:
        """A list of lists. Does items recurse?"""
        return str(matrix)

    # --- enum via typing.Literal ------------------------------------------
    def literal_enum(
        self,
        mode: Literal["fast", "slow", "auto"],
        level: Literal[1, 2, 3] = 1,
    ) -> str:
        """Literal[str...] required, Literal[int...] with default."""
        return f"{mode} {level}"

    # --- enum via enum.Enum subclass --------------------------------------
    def class_enum(self, color: Color, fallback: Color = Color.RED) -> str:
        """A str-Enum subclass param, required and with a default."""
        return f"{color} {fallback}"

    # --- Union -------------------------------------------------------------
    def union_param(self, value: Union[str, int], maybe: Union[int, float, None] = None) -> str:
        """Union[str, int] and a three-way Union with None."""
        return f"{value} {maybe}"

    # --- dict / mapping ----------------------------------------------------
    def mapping(self, payload: Dict[str, int], opts: dict = {}) -> str:
        """Typed dict and a bare dict with a mutable default."""
        return f"{payload} {opts}"

    # --- tuple -------------------------------------------------------------
    def tuple_param(self, pair: Tuple[int, int], triple: Tuple[str, int, bool]) -> str:
        """Fixed-length tuples with heterogeneous element types."""
        return f"{pair} {triple}"

    # --- nested pydantic model --------------------------------------------
    def model_param(self, point: Point, points: List[Point]) -> str:
        """A nested BaseModel and a list of them. Watch $ref / $defs."""
        return f"{point} {points}"

    # --- Field with description + constraints -----------------------------
    def field_described(
        self,
        name: str = Field(..., description="A required name described via Field."),
        count: int = Field(10, description="Optional count via Field.", ge=0, le=100),
        ratio: float = Field(0.5, description="Constrained float.", gt=0.0, lt=1.0),
    ) -> str:
        """Params documented via pydantic Field rather than the docstring."""
        return f"{name} {count} {ratio}"

    # --- docstring-only descriptions (Google style) -----------------------
    def docstring_described(self, alpha: str, beta: int) -> str:
        """Descriptions live only in the docstring, Google style.

        Args:
            alpha: The first thing, a string.
            beta: The second thing, an integer count.
        """
        return f"{alpha} {beta}"

    # --- no params ---------------------------------------------------------
    def no_params(self) -> str:
        """Zero arguments. Does an empty properties object appear?"""
        return "ok"

    # --- dunder-only (model sees no real args) ----------------------------
    def dunder_only(self, __user__=None, __chat_id__=None) -> str:
        """Only injected dunders. Should surface as a no-arg tool to the model."""
        return "ok"

    # --- mixed real + dunder ----------------------------------------------
    def mixed(self, query: str, limit: int = 5, __event_emitter__=None) -> str:
        """Real args plus an injected dunder; dunder should be stripped."""
        return f"{query} {limit}"
