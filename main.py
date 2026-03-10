"""
Generative Text Exploration Toolkit - Main Demo

Demonstrates prompt engineering, text generation pipelines,
evaluation metrics, and text similarity analysis.
"""

from src.prompts.template_engine import PromptTemplate, create_default_registry
from src.prompts.few_shot import FewShotPromptBuilder
from src.pipelines.text_pipeline import TextPipeline, MockTextGenerator, create_summarization_pipeline
from src.evaluation.text_metrics import bleu_score, rouge_n_score, rouge_l_score, evaluate_text
from src.embeddings.similarity import TFIDFSimilarity
from src.utils.tokenizer import SimpleTokenizer


def demo_prompt_templates():
    """Demonstrate prompt template engine."""
    print("\n" + "=" * 60)
    print("DEMO 1: PROMPT TEMPLATE ENGINE")
    print("=" * 60)

    registry = create_default_registry()
    print(f"\nRegistered templates: {registry.list_templates()}")

    summary_prompt = registry.render(
        "summarize",
        content="Machine learning is a subset of artificial intelligence that "
                "enables systems to learn from data without being explicitly programmed.",
        length="brief"
    )
    print(f"\n--- Summarization Prompt ---")
    print(summary_prompt)

    classify_prompt = registry.render(
        "classify",
        categories="positive, negative, neutral",
        content="The product quality exceeded my expectations and delivery was fast."
    )
    print(f"\n--- Classification Prompt ---")
    print(classify_prompt)

    custom_template = PromptTemplate(
        "Generate a {{format:paragraph}} about {{topic}} for a {{audience:general}} audience.\n"
        "Tone: {{tone:professional}}\nLength: {{length}}",
        name="content_generator"
    )
    print(f"\n--- Custom Template ---")
    print(f"Variables: {custom_template.variables}")
    print(f"Required: {custom_template.required_variables}")
    rendered = custom_template.render(topic="renewable energy", length="200 words")
    print(f"Rendered:\n{rendered}")


def demo_few_shot():
    """Demonstrate few-shot prompt building."""
    print("\n" + "=" * 60)
    print("DEMO 2: FEW-SHOT PROMPT BUILDER")
    print("=" * 60)

    builder = FewShotPromptBuilder("Classify the sentiment of the given text.")
    builder.set_prefixes("Text", "Sentiment")
    builder.set_system_instruction("You are a sentiment analysis system.")

    builder.add_example("This product is amazing!", "Positive")
    builder.add_example("Terrible experience, would not recommend.", "Negative")
    builder.add_example("The item arrived on time.", "Neutral")

    prompt = builder.build("I love this new feature, it works perfectly!")
    print(f"\n--- Few-Shot Prompt ---")
    print(prompt)

    messages = builder.build_chat_format("The delivery was delayed by two weeks.")
    print(f"\n--- Chat Format ({len(messages)} messages) ---")
    for msg in messages:
        print(f"  [{msg['role']}]: {msg['content'][:60]}...")


def demo_text_pipeline():
    """Demonstrate text generation pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 3: TEXT GENERATION PIPELINE")
    print("=" * 60)

    generator = MockTextGenerator(seed=42)

    pipeline = create_summarization_pipeline(generator)
    print(f"\nPipeline: {pipeline.name} ({pipeline.step_count} steps)")

    input_text = (
        "Natural language processing is a subfield of linguistics, computer science, "
        "and artificial intelligence concerned with the interactions between computers "
        "and human language. The goal is to enable computers to process and analyze "
        "large amounts of natural language data."
    )
    result = pipeline.run(input_text)
    print(f"\n--- Pipeline Result ---")
    print(f"Input length:  {len(result['input'])} chars")
    print(f"Output: {result['output']}")
    print(f"Steps applied: {result['steps_applied']}")

    result2 = generator.generate("Explain quantum computing", task_type="default")
    print(f"\n--- Direct Generation ---")
    print(f"Response: {result2['response']}")
    print(f"Tokens used: {result2['tokens_used']}")


def demo_evaluation():
    """Demonstrate text evaluation metrics."""
    print("\n" + "=" * 60)
    print("DEMO 4: TEXT EVALUATION METRICS")
    print("=" * 60)

    reference = "The cat sat on the mat in the room"
    hypothesis1 = "The cat sat on the mat"
    hypothesis2 = "A dog stood on the floor in the house"
    hypothesis3 = "The cat sat on the mat in the room"

    print(f"\nReference:    '{reference}'")
    print(f"Hypothesis 1: '{hypothesis1}' (partial match)")
    print(f"Hypothesis 2: '{hypothesis2}' (different content)")
    print(f"Hypothesis 3: '{hypothesis3}' (exact match)")

    for i, hyp in enumerate([hypothesis1, hypothesis2, hypothesis3], 1):
        results = evaluate_text(reference, hyp)
        print(f"\n--- Hypothesis {i} ---")
        print(f"  BLEU:    {results['bleu']['bleu']:.4f}")
        print(f"  ROUGE-1: P={results['rouge_1']['precision']:.3f} "
              f"R={results['rouge_1']['recall']:.3f} "
              f"F1={results['rouge_1']['f1']:.3f}")
        print(f"  ROUGE-2: P={results['rouge_2']['precision']:.3f} "
              f"R={results['rouge_2']['recall']:.3f} "
              f"F1={results['rouge_2']['f1']:.3f}")
        print(f"  ROUGE-L: P={results['rouge_l']['precision']:.3f} "
              f"R={results['rouge_l']['recall']:.3f} "
              f"F1={results['rouge_l']['f1']:.3f}")


def demo_similarity():
    """Demonstrate text similarity analysis."""
    print("\n" + "=" * 60)
    print("DEMO 5: TEXT SIMILARITY (TF-IDF)")
    print("=" * 60)

    documents = [
        "Machine learning algorithms learn patterns from data",
        "Deep learning is a subset of machine learning using neural networks",
        "Natural language processing handles text and speech data",
        "Computer vision processes and analyzes visual information",
        "Data science combines statistics and programming for insights",
    ]

    model = TFIDFSimilarity()
    model.fit(documents)

    print(f"\nCorpus: {model.document_count} documents, {model.vocabulary_size} unique terms")

    print("\n--- Similarity Matrix ---")
    matrix = model.similarity_matrix()
    for i, row in enumerate(matrix):
        vals = " | ".join(f"{v:.3f}" for v in row)
        print(f"  Doc {i}: [{vals}]")

    query = "learning algorithms for processing data"
    results = model.query(query, top_k=3)
    print(f"\n--- Query: '{query}' ---")
    for idx, score in results:
        print(f"  [{score:.3f}] Doc {idx}: {documents[idx][:50]}...")


def demo_tokenizer():
    """Demonstrate tokenizer utilities."""
    print("\n" + "=" * 60)
    print("DEMO 6: TOKENIZER")
    print("=" * 60)

    tokenizer = SimpleTokenizer(remove_stopwords=True, min_token_length=2)

    text = "Machine learning is transforming how we process and analyze large datasets."
    tokens = tokenizer.tokenize(text)
    print(f"\nText: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Count: {len(tokens)}")

    stats = tokenizer.text_statistics(text)
    print(f"\n--- Text Statistics ---")
    print(f"  Total tokens: {stats['token_count']}")
    print(f"  Unique tokens: {stats['unique_tokens']}")
    print(f"  Avg token length: {stats['avg_token_length']}")
    print(f"  Type-token ratio: {stats['type_token_ratio']}")


def main():
    """Run the full demo suite."""
    print("=" * 60)
    print("  GENERATIVE TEXT EXPLORATION TOOLKIT")
    print("=" * 60)

    demo_prompt_templates()
    demo_few_shot()
    demo_text_pipeline()
    demo_evaluation()
    demo_similarity()
    demo_tokenizer()

    print("\n" + "=" * 60)
    print("  ALL DEMOS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
