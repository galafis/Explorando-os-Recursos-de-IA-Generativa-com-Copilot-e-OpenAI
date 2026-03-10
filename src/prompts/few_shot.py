"""
Few-Shot Prompt Builder

Constructs prompts with example demonstrations for in-context learning.
Supports different formatting styles and example selection strategies.
"""

from typing import Dict, List, Optional, Tuple


class Example:
    """Represents a single input-output example for few-shot prompting."""

    def __init__(self, input_text: str, output_text: str, label: Optional[str] = None):
        self.input_text = input_text
        self.output_text = output_text
        self.label = label

    def format(self, input_prefix: str = "Input", output_prefix: str = "Output",
               separator: str = ": ") -> str:
        """Format the example as a string."""
        parts = [f"{input_prefix}{separator}{self.input_text}"]
        if self.label:
            parts.append(f"Label{separator}{self.label}")
        parts.append(f"{output_prefix}{separator}{self.output_text}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"Example(input='{self.input_text[:30]}...', output='{self.output_text[:30]}...')"


class FewShotPromptBuilder:
    """
    Builds few-shot prompts with configurable formatting.

    Supports adding examples, setting system instructions,
    and formatting the final prompt in various styles.
    """

    def __init__(self, task_description: str = ""):
        self.task_description = task_description
        self.examples: List[Example] = []
        self.input_prefix = "Input"
        self.output_prefix = "Output"
        self.separator = ": "
        self.example_separator = "\n---\n"
        self.system_instruction = ""

    def set_prefixes(self, input_prefix: str, output_prefix: str) -> "FewShotPromptBuilder":
        """Set custom prefixes for input and output fields."""
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        return self

    def set_system_instruction(self, instruction: str) -> "FewShotPromptBuilder":
        """Set a system-level instruction for the prompt."""
        self.system_instruction = instruction
        return self

    def add_example(self, input_text: str, output_text: str,
                    label: Optional[str] = None) -> "FewShotPromptBuilder":
        """Add a single example to the prompt."""
        self.examples.append(Example(input_text, output_text, label))
        return self

    def add_examples(self, examples: List[Tuple[str, str]]) -> "FewShotPromptBuilder":
        """Add multiple examples as (input, output) tuples."""
        for inp, out in examples:
            self.examples.append(Example(inp, out))
        return self

    def clear_examples(self) -> "FewShotPromptBuilder":
        """Remove all examples."""
        self.examples.clear()
        return self

    def build(self, query: str, max_examples: Optional[int] = None) -> str:
        """
        Build the complete few-shot prompt.

        Args:
            query: The actual input to process.
            max_examples: Maximum number of examples to include.

        Returns:
            Complete formatted prompt string.
        """
        parts = []

        if self.system_instruction:
            parts.append(self.system_instruction)
            parts.append("")

        if self.task_description:
            parts.append(self.task_description)
            parts.append("")

        examples_to_use = self.examples
        if max_examples is not None:
            examples_to_use = examples_to_use[:max_examples]

        if examples_to_use:
            parts.append("Examples:")
            parts.append("")
            formatted = []
            for ex in examples_to_use:
                formatted.append(ex.format(self.input_prefix, self.output_prefix, self.separator))
            parts.append(self.example_separator.join(formatted))
            parts.append("")

        parts.append(f"{self.input_prefix}{self.separator}{query}")
        parts.append(f"{self.output_prefix}{self.separator}")

        return "\n".join(parts)

    def build_chat_format(self, query: str, max_examples: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Build the prompt in chat message format.

        Returns:
            List of message dictionaries with 'role' and 'content' keys.
        """
        messages = []

        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})

        if self.task_description:
            messages.append({"role": "system", "content": self.task_description})

        examples_to_use = self.examples
        if max_examples is not None:
            examples_to_use = examples_to_use[:max_examples]

        for ex in examples_to_use:
            messages.append({"role": "user", "content": ex.input_text})
            messages.append({"role": "assistant", "content": ex.output_text})

        messages.append({"role": "user", "content": query})

        return messages

    @property
    def example_count(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __repr__(self) -> str:
        return (f"FewShotPromptBuilder(task='{self.task_description[:40]}...', "
                f"examples={self.example_count})")
