from dataclasses import dataclass, field
from typing import Literal

from eval.prompts.pairwise import PAIRWISE_PROMPT
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class JudgeDecision(BaseModel):
    """The decision made by the LLM judge."""

    winner: Literal["A", "B", "Tie"] = Field(
        description="The winner of the comparison. 'A' if first is better, 'B' if second is better, or 'Tie'."
    )
    explanation: str = Field(description="A brief explanation of why this decision was made.")


@dataclass
class PairwiseJudge:
    """A judge that compares two LLM outputs using a stronger LLM.

    Attributes:
        model_name: The name of the OpenAI model to use as a judge.
        llm: The LLM used for judging.
        prompt_template: The prompt template used for judging.
    """

    model_name: str = "gpt-4o-mini"
    llm: BaseChatModel = field(init=False)
    prompt_template: ChatPromptTemplate = field(init=False)

    def __post_init__(self) -> None:
        self.llm = ChatOpenAI(model=self.model_name, temperature=0.0)

    def judge(self, question: str, response_a: str, response_b: str) -> JudgeDecision:
        """Compares two responses and determines the winner.

        Args:
            question: The question being answered.
            response_a: The output from the first model.
            response_b: The output from the second model.

        Returns:
            A JudgeDecision object containing the winner ('A', 'B', or 'Tie') and an explanation.
        """

        self.prompt_template = PAIRWISE_PROMPT
        inputs = {
            "question": question,
            "response_a": response_a,
            "response_b": response_b,
        }

        chain = self.prompt_template | self.llm.with_structured_output(JudgeDecision)

        result = chain.invoke(inputs)

        # For type checking
        if not isinstance(result, JudgeDecision):
            raise ValueError("The judge failed to return a valid decision structure.")

        return result

    async def judge_async(self, question: str, response_a: str, response_b: str) -> JudgeDecision:
        """Asynchronously compares two responses and determines the winner.

        Args:
            question: The question being answered.
            response_a: The output from the first model.
            response_b: The output from the second model.

        Returns:
            A JudgeDecision object containing the winner ('A', 'B', or 'Tie') and an explanation.
        """
        self.prompt_template = PAIRWISE_PROMPT
        inputs = {
            "question": question,
            "response_a": response_a,
            "response_b": response_b,
        }

        chain = self.prompt_template | self.llm.with_structured_output(JudgeDecision)
        result = await chain.ainvoke(inputs)

        if not isinstance(result, JudgeDecision):
            raise ValueError("The judge failed to return a valid decision structure.")

        return result
