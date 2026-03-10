"""
Tests for prompt templates, few-shot builder, and pipeline modules.
"""

import pytest
from src.prompts.template_engine import PromptTemplate, TemplateRegistry, create_default_registry
from src.prompts.few_shot import FewShotPromptBuilder
from src.pipelines.text_pipeline import TextPipeline, MockTextGenerator, create_summarization_pipeline


class TestPromptTemplate:
    def test_simple_render(self):
        pt = PromptTemplate("Hello {{name}}, welcome to {{place}}!")
        result = pt.render(name="Alice", place="Wonderland")
        assert result == "Hello Alice, welcome to Wonderland!"

    def test_default_values(self):
        pt = PromptTemplate("Translate to {{language:English}}: {{text}}")
        result = pt.render(text="Hello")
        assert "English" in result
        assert "Hello" in result

    def test_override_defaults(self):
        pt = PromptTemplate("Format: {{style:bold}}")
        result = pt.render(style="italic")
        assert "italic" in result

    def test_missing_required_variable(self):
        pt = PromptTemplate("Hello {{name}}!")
        with pytest.raises(ValueError):
            pt.render()

    def test_variables_property(self):
        pt = PromptTemplate("{{a}} and {{b:default}} and {{c}}")
        assert set(pt.variables) == {"a", "b", "c"}

    def test_required_variables(self):
        pt = PromptTemplate("{{a}} and {{b:default}}")
        assert pt.required_variables == ["a"]

    def test_validate(self):
        pt = PromptTemplate("{{x}} {{y}} {{z:default}}")
        missing = pt.validate(x="val")
        assert "y" in missing
        assert "x" not in missing


class TestTemplateRegistry:
    def test_register_and_get(self):
        reg = TemplateRegistry()
        reg.register("test", "Hello {{name}}")
        template = reg.get("test")
        assert template.name == "test"

    def test_render(self):
        reg = TemplateRegistry()
        reg.register("greet", "Hi {{name}}!")
        result = reg.render("greet", name="Bob")
        assert result == "Hi Bob!"

    def test_missing_template(self):
        reg = TemplateRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_list_templates(self):
        reg = TemplateRegistry()
        reg.register("a", "{{x}}")
        reg.register("b", "{{y}}")
        assert set(reg.list_templates()) == {"a", "b"}

    def test_remove(self):
        reg = TemplateRegistry()
        reg.register("temp", "{{x}}")
        assert reg.remove("temp") is True
        assert reg.count == 0

    def test_default_registry(self):
        reg = create_default_registry()
        assert reg.count >= 5
        assert "summarize" in reg.list_templates()


class TestFewShotPromptBuilder:
    def test_basic_build(self):
        builder = FewShotPromptBuilder("Classify sentiment.")
        builder.add_example("Great!", "Positive")
        builder.add_example("Bad.", "Negative")
        prompt = builder.build("Okay.")
        assert "Great!" in prompt
        assert "Classify sentiment." in prompt
        assert "Okay." in prompt

    def test_chat_format(self):
        builder = FewShotPromptBuilder()
        builder.set_system_instruction("You are helpful.")
        builder.add_example("Hi", "Hello")
        messages = builder.build_chat_format("How are you?")
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "How are you?"

    def test_max_examples(self):
        builder = FewShotPromptBuilder()
        builder.add_example("a", "1")
        builder.add_example("b", "2")
        builder.add_example("c", "3")
        prompt = builder.build("d", max_examples=2)
        assert "c" not in prompt

    def test_custom_prefixes(self):
        builder = FewShotPromptBuilder()
        builder.set_prefixes("Question", "Answer")
        builder.add_example("What?", "This.")
        prompt = builder.build("Why?")
        assert "Question: What?" in prompt
        assert "Answer: This." in prompt

    def test_clear_examples(self):
        builder = FewShotPromptBuilder()
        builder.add_example("a", "b")
        builder.clear_examples()
        assert builder.example_count == 0


class TestTextPipeline:
    def test_basic_pipeline(self):
        pipe = TextPipeline()
        pipe.add_step("upper", lambda t: t.upper())
        result = pipe.run("hello")
        assert result["output"] == "HELLO"
        assert result["steps_applied"] == ["upper"]

    def test_multi_step(self):
        pipe = TextPipeline()
        pipe.add_step("strip", lambda t: t.strip())
        pipe.add_step("upper", lambda t: t.upper())
        result = pipe.run("  hello  ")
        assert result["output"] == "HELLO"

    def test_history(self):
        pipe = TextPipeline()
        pipe.add_step("identity", lambda t: t)
        pipe.run("first")
        pipe.run("second")
        assert len(pipe.history) == 2

    def test_configure(self):
        pipe = TextPipeline()
        pipe.configure(max_length=1000)
        result = pipe.run("test")
        assert result["config"]["max_length"] == 1000


class TestMockGenerator:
    def test_default_generation(self):
        gen = MockTextGenerator(seed=42)
        result = gen.generate("Tell me about Python")
        assert "response" in result
        assert len(result["response"]) > 0

    def test_task_type(self):
        gen = MockTextGenerator()
        result = gen.generate("Summarize this", task_type="summarize", topic="Python")
        assert "Python" in result["response"]

    def test_max_length(self):
        gen = MockTextGenerator()
        result = gen.generate("test", max_length=50)
        assert len(result["response"]) <= 50
