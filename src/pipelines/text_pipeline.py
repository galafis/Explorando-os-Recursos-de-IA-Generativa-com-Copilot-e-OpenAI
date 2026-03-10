"""
Text Generation Pipeline Framework

Provides a mock pipeline for text generation that simulates
the flow of prompt processing and response generation without
requiring real API calls.
"""

import re
import random
from typing import Callable, Dict, List, Optional


class PipelineStep:
    """A single step in the text processing pipeline."""

    def __init__(self, name: str, processor: Callable[[str], str]):
        self.name = name
        self.processor = processor

    def execute(self, text: str) -> str:
        """Execute this step on the given text."""
        return self.processor(text)


class TextPipeline:
    """
    A configurable text generation pipeline.

    Chains multiple processing steps and applies them
    sequentially to produce output text.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.steps: List[PipelineStep] = []
        self.history: List[Dict[str, str]] = []
        self._config: Dict[str, object] = {
            "max_length": 500,
            "temperature": 0.7,
            "top_k": 50,
        }

    def add_step(self, name: str, processor: Callable[[str], str]) -> "TextPipeline":
        """Add a processing step to the pipeline."""
        self.steps.append(PipelineStep(name, processor))
        return self

    def configure(self, **kwargs) -> "TextPipeline":
        """Update pipeline configuration."""
        self._config.update(kwargs)
        return self

    def run(self, prompt: str) -> Dict[str, object]:
        """
        Run the pipeline on the given prompt.

        Args:
            prompt: Input prompt text.

        Returns:
            Dictionary with 'input', 'output', 'steps_applied', and 'config'.
        """
        current_text = prompt
        steps_applied = []

        for step in self.steps:
            current_text = step.execute(current_text)
            steps_applied.append(step.name)

        result = {
            "input": prompt,
            "output": current_text,
            "steps_applied": steps_applied,
            "config": self._config.copy(),
        }

        self.history.append({"input": prompt, "output": current_text})
        return result

    def clear_history(self):
        """Clear the pipeline execution history."""
        self.history.clear()

    @property
    def step_count(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self.steps)


class MockTextGenerator:
    """
    Mock text generator that simulates language model behavior
    without requiring real API calls.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._response_templates = {
            "summarize": "The text discusses {topic}. Key points include the main arguments "
                         "presented and their supporting evidence. The conclusion reinforces "
                         "the central thesis.",
            "translate": "The translated text maintains the original meaning while adapting "
                         "to the target language conventions and cultural context.",
            "classify": "Based on the content analysis, the text belongs to the '{category}' "
                        "category with high confidence.",
            "qa": "Based on the provided context, the answer is: {answer}.",
            "default": "The generated response addresses the input prompt by providing "
                       "relevant and contextual information.",
        }

    def generate(self, prompt: str, task_type: str = "default",
                 max_length: int = 200, **kwargs) -> Dict[str, object]:
        """
        Generate a mock response for the given prompt.

        Args:
            prompt: Input prompt.
            task_type: Type of task (summarize, translate, classify, qa, default).
            max_length: Maximum response length.
            **kwargs: Additional parameters for template variables.

        Returns:
            Dictionary with 'response', 'task_type', 'prompt_length', 'response_length'.
        """
        template = self._response_templates.get(task_type, self._response_templates["default"])

        placeholders = re.findall(r'\{(\w+)\}', template)
        for ph in placeholders:
            if ph in kwargs:
                template = template.replace(f'{{{ph}}}', str(kwargs[ph]))
            else:
                template = template.replace(f'{{{ph}}}', f"[{ph}]")

        response = template[:max_length]

        return {
            "response": response,
            "task_type": task_type,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "tokens_used": len(prompt.split()) + len(response.split()),
        }


def create_summarization_pipeline(generator: Optional[MockTextGenerator] = None) -> TextPipeline:
    """Create a pipeline optimized for text summarization."""
    if generator is None:
        generator = MockTextGenerator(seed=42)

    pipeline = TextPipeline(name="summarization")

    pipeline.add_step("normalize_whitespace", lambda t: " ".join(t.split()))
    pipeline.add_step("truncate_input", lambda t: t[:2000] if len(t) > 2000 else t)
    pipeline.add_step("add_instruction", lambda t: f"Summarize the following text:\n\n{t}\n\nSummary:")
    pipeline.add_step("generate", lambda t: generator.generate(t, task_type="summarize")["response"])

    return pipeline


def create_classification_pipeline(categories: List[str],
                                   generator: Optional[MockTextGenerator] = None) -> TextPipeline:
    """Create a pipeline for text classification."""
    if generator is None:
        generator = MockTextGenerator(seed=42)

    cats = ", ".join(categories)

    pipeline = TextPipeline(name="classification")

    pipeline.add_step("normalize", lambda t: " ".join(t.split()))
    pipeline.add_step("add_instruction",
                      lambda t: f"Classify the following text into one of [{cats}]:\n\n{t}\n\nCategory:")
    pipeline.add_step("generate",
                      lambda t: generator.generate(t, task_type="classify",
                                                   category=categories[0])["response"])

    return pipeline
