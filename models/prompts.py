"""Prompt templates for each agent role."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Solver prompts
# ---------------------------------------------------------------------------

SOLVER_PROMPT = (
    "You are a math solver. Given the problem: {problem}. "
    "Topic: {topic}. "
    "Solve step by step and provide the final answer."
)

SOLVER_WITH_CONTEXT_PROMPT = (
    "You are a math solver. Given the problem: {problem}. "
    "Topic: {topic}. "
    "Relevant skills and context:\n{context}\n"
    "Solve step by step and provide the final answer."
)

# ---------------------------------------------------------------------------
# Teacher prompts
# ---------------------------------------------------------------------------

TEACHER_DECOMPOSE_PROMPT = (
    "You are a teacher. Decompose this problem into subtasks: {problem}. "
    "Return a JSON list of subtasks."
)

TEACHER_DECOMPOSE_WITH_CONTEXT_PROMPT = (
    "You are a teacher coordinating a team of student agents. "
    "Decompose this problem into {n_subtasks} subtasks that can be solved in parallel or sequence.\n"
    "Problem: {problem}\n"
    "Available context:\n{context}\n"
    "Return a JSON list of subtask objects, each with keys: "
    "'description' (string), 'depends_on' (list of subtask indices starting from 0).\n"
    "Example: [{\"description\": \"Set up the equation\", \"depends_on\": []}, "
    "{\"description\": \"Solve for x\", \"depends_on\": [0]}]"
)

TEACHER_GOLDEN_THOUGHT_PROMPT = (
    "The student is stuck on: {subtask}. "
    "Failed attempts: {failures}. "
    "Provide a correct reasoning hint."
)

TEACHER_SYNTHESIZE_PROMPT = (
    "You are a teacher synthesizing results from multiple student agents.\n"
    "Original problem: {problem}\n"
    "Subtask results:\n{results}\n"
    "Provide the final unified answer."
)

# ---------------------------------------------------------------------------
# Verifier prompts
# ---------------------------------------------------------------------------

VERIFIER_PROMPT = (
    "Verify if this answer is correct. "
    "Problem: {problem}. "
    "Answer: {answer}. "
    "Respond with CORRECT or INCORRECT and explain why."
)

VERIFIER_WITH_EXPECTED_PROMPT = (
    "Verify if this answer is correct.\n"
    "Problem: {problem}\n"
    "Candidate answer: {answer}\n"
    "Expected answer format: {expected_format}\n"
    "Respond with exactly CORRECT or INCORRECT on the first line, "
    "then provide a brief explanation."
)

# ---------------------------------------------------------------------------
# SubAgent prompts
# ---------------------------------------------------------------------------

SUBAGENT_SOLVE_PROMPT = (
    "You are a student agent solving a subtask.\n"
    "Subtask: {subtask}\n"
    "Context: {context}\n"
    "Think step by step and provide your answer."
)

SUBAGENT_SOLVE_WITH_HINT_PROMPT = (
    "You are a student agent solving a subtask. You have been given a hint.\n"
    "Subtask: {subtask}\n"
    "Hint from teacher: {hint}\n"
    "Context: {context}\n"
    "Use the hint to guide your reasoning. Think step by step and provide your answer."
)
