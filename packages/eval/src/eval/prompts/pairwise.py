from langchain_core.prompts import ChatPromptTemplate

# PAIRWISE_PROMPT_WITH_REFERENCE = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "Please act as an impartial judge and evaluate the quality of the robotic actions proposed by two "
#             "VLM-based robot agents for a given visual scenario. Your evaluation should consider whether the "
#             "proposed action is appropriate, safe, and logically follows the scenario context. "
#             "You will be given the scenario ID/description, a reference (expected) action, agent A’s proposed action, "  # noqa
#             "and agent B’s proposed action. Your job is to evaluate which agent’s action is better aligned with the "  # noqa
#             "reference action and the scenario.\n\n"
#             "Common action examples include: 'drink water', 'pick up', 'sit down', 'hand waving', 'arm swings', etc.\n\n"  # noqa
#             "Begin your evaluation by comparing both agents' actions with the reference action. "
#             "Identify if the action matches the expected behavior. Avoid any position biases and ensure that the order in "  # noqa
#             "which the responses were presented does not influence your decision. Do not allow the length of the "  # noqa
#             "responses to influence your evaluation. Do not favor certain names of the agents. "
#             "Be as objective as possible. After providing your explanation, output your "
#             'final verdict by strictly following this format: "[[A]]" if agent A is better, "[[B]]" '
#             'if agent B is better, and "[[C]]" for a tie.',
#         ),
#         (
#             "human",
#             "[Scenario]\n{question}\n\n"
#             "[The Start of Agent A’s Action]\n{response_a}\n[The End of Agent A’s Action]\n\n"
#             "[The Start of Agent B’s Action]\n{response_b}\n[The End of Agent B’s Action]",
#         ),
#     ]
# )  # noqa


CRITERIA_PROMPT = (
    "comprehensiveness: How much detail does the answer provide to cover all the aspects and details of the "
    "question? A comprehensive answer should be thorough and complete, without being redundant or irrelevant. "
    "For example, if the question is 'What are the benefits and drawbacks of nuclear energy?', a comprehensive "
    "answer would provide both the positive and negative aspects of nuclear energy (e.g., efficiency, "
    "environmental impact, safety, cost, etc). A comprehensive answer should not leave out any important points "
    "or provide irrelevant information. An incomplete answer would only provide the benefits of nuclear energy "
    "without describing the drawbacks, or a redundant answer would repeat the same information multiple times.\n"
    "diversity: How varied and rich is the answer in providing different perspectives and insights on the "
    "question? A diverse answer should be multi-faceted, offering different viewpoints and angles. For example, "
    "if the question is 'What are the causes and effects of climate change?', a diverse answer would discuss "
    "different causes and effects (e.g., greenhouse gas emissions, deforestation, natural disasters, biodiversity "
    "loss, etc). A diverse answer should also provide different sources and evidence to support the answer. "
    "A single-source answer would only cite one source, or a biased answer would only provide one perspective or "
    "opinion.\n"
    "directness: How specifically and clearly does the answer address the question? A direct answer should provide "
    "a clear and concise response to the question. For example, if the question is 'What is the capital of France?', "
    "a direct answer would be 'Paris'. A direct answer should not provide irrelevant or unnecessary information. "
    "An indirect answer would be 'The capital of France is located on the river Seine.'\n"
    "empowerment: How well does the answer help the reader understand and make informed judgements about the topic "
    "without being misled or making fallacious assumptions. Evaluate each answer on the quality of explanation, "
    "reasoning, and sources behind the claims."
)

PAIRWISE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant responsible for grading two answers to a question provided by two different "
            "people.\nGiven a question and two responses (Response A and Response B), assess which response is better "
            "according to the following measure:\n{criteria}\nYour assessment should include two parts:\n"
            "- Winner: either \"A\" (if Response A is better), \"B\" (if Response B is better), or \"Tie\" if they are "
            "fundamentally similar and the differences are immaterial.\n- Explanation: a short explanation of why you "
            "chose the winner with respect to the measure described above.\nFormat your response as a JSON object with "
            "structure:\n"
            '{{"winner": "A|B|Tie", "explanation": "Response <winner> is better because <your explanation>."}}',
        ),
        (
            "human",
            "---Question---\n{question}\n"
            "---Response A---\n{response_a}\n"
            "---Response B---\n{response_b}\n"
            "Assess which response is better according to the following measure:\n{criteria}",
        ),
    ]
).partial(criteria=CRITERIA_PROMPT)

if __name__ == "__main__":
    print(
        PAIRWISE_PROMPT.format(
            question="What is the capital of France?", response_a="Paris", response_b="Berlin"
        )
    )
