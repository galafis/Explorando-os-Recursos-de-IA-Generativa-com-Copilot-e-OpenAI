"""
Prompt Template Engine

Manages prompt templates with variable injection, supporting
named placeholders and default values.
"""

import re
from typing import Dict, List, Optional


class PromptTemplate:
    """
    A template for generating prompts with variable placeholders.

    Placeholders use the format {{variable_name}} and can have
    default values specified as {{variable_name:default_value}}.
    """

    def __init__(self, template: str, name: str = "unnamed"):
        self.template = template
        self.name = name
        self._variables = self._extract_variables()

    def _extract_variables(self) -> Dict[str, Optional[str]]:
        """Extract variable names and default values from the template."""
        pattern = r'\{\{(\w+)(?::([^}]*))?\}\}'
        matches = re.findall(pattern, self.template)
        variables = {}
        for var_name, default_val in matches:
            variables[var_name] = default_val if default_val else None
        return variables

    @property
    def variables(self) -> List[str]:
        """Return list of variable names in the template."""
        return list(self._variables.keys())

    @property
    def required_variables(self) -> List[str]:
        """Return list of variables without default values."""
        return [k for k, v in self._variables.items() if v is None]

    def render(self, **kwargs) -> str:
        """
        Render the template by substituting variables.

        Args:
            **kwargs: Variable name-value pairs.

        Returns:
            Rendered string with all placeholders replaced.

        Raises:
            ValueError: If a required variable is missing.
        """
        result = self.template

        for var_name, default_val in self._variables.items():
            if var_name in kwargs:
                value = str(kwargs[var_name])
            elif default_val is not None:
                value = default_val
            else:
                raise ValueError(f"Missing required variable: '{var_name}' in template '{self.name}'")

            pattern = r'\{\{' + var_name + r'(?::[^}]*)?\}\}'
            result = re.sub(pattern, value, result)

        return result

    def validate(self, **kwargs) -> List[str]:
        """
        Validate that all required variables are provided.

        Returns:
            List of missing required variable names.
        """
        missing = []
        for var_name in self.required_variables:
            if var_name not in kwargs:
                missing.append(var_name)
        return missing

    def __repr__(self) -> str:
        return f"PromptTemplate(name='{self.name}', variables={self.variables})"


class TemplateRegistry:
    """
    Registry for managing multiple prompt templates.
    """

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}

    def register(self, name: str, template: str) -> PromptTemplate:
        """Register a new template with the given name."""
        pt = PromptTemplate(template, name=name)
        self._templates[name] = pt
        return pt

    def get(self, name: str) -> PromptTemplate:
        """Retrieve a template by name."""
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found in registry.")
        return self._templates[name]

    def list_templates(self) -> List[str]:
        """Return list of all registered template names."""
        return list(self._templates.keys())

    def render(self, name: str, **kwargs) -> str:
        """Render a registered template by name."""
        return self.get(name).render(**kwargs)

    def remove(self, name: str) -> bool:
        """Remove a template from the registry."""
        if name in self._templates:
            del self._templates[name]
            return True
        return False

    @property
    def count(self) -> int:
        """Return the number of registered templates."""
        return len(self._templates)


def create_default_registry() -> TemplateRegistry:
    """Create a registry with common prompt templates."""
    registry = TemplateRegistry()

    registry.register("summarize", (
        "Summarize the following {{content_type:text}} in {{language:English}}:\n\n"
        "{{content}}\n\n"
        "Provide a {{length:concise}} summary."
    ))

    registry.register("translate", (
        "Translate the following text from {{source_language}} to {{target_language}}:\n\n"
        "{{text}}\n\n"
        "Maintain the original tone and meaning."
    ))

    registry.register("classify", (
        "Classify the following {{item_type:text}} into one of these categories: "
        "{{categories}}.\n\n"
        "{{content}}\n\n"
        "Category:"
    ))

    registry.register("extract", (
        "Extract {{entity_type}} from the following text:\n\n"
        "{{text}}\n\n"
        "Format the results as a {{format:list}}."
    ))

    registry.register("qa", (
        "Context:\n{{context}}\n\n"
        "Question: {{question}}\n\n"
        "Answer the question based only on the provided context. "
        "If the answer is not in the context, say \"Not found in context.\""
    ))

    return registry
