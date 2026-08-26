"""Prompts for the EvalTree pipeline (gerund-phrase annotation + summaries)."""

# Stage 1: leaf annotation
ANNOTATION_SYSTEM = "You are an expert at analyzing Large Language Model capabilities."
ANNOTATION_PROMPT = """Given an instruction and its corresponding response, summarize the specific skills or capabilities needed to fulfill the instruction effectively. 
The output should be a single gerund phrase (e.g., "Solving complex mathematical equations", "Generating creative storytelling").

Instruction: {instruction}
Response: {response}

Capability:"""

# Stage 2: cluster description (recursive summary)
SUMMARY_SYSTEM = "You are an expert at identifying common patterns in skills and capabilities."
SUMMARY_PROMPT = """Given a set of phrases that describe skills or capabilities, generate a high-level, generic gerund phrase that categorizes the collective set of skills or capabilities.
The summary should be broad and abstract, capturing the overarching domain rather than the specific details. Do NOT simply enumerate the descriptions or create a union of them using 'and'.
The output should be a single, broad gerund phrase (e.g., "Analyzing data" or "Writing code").

Descriptions:
{descriptions}

Summary Capability:"""

# Stage 3: in-pipeline entailment judging (child prompt vs parent summary)
ENTAILMENT_SYSTEM = "You are a logical reasoning evaluator."
ENTAILMENT_PROMPT = """
You will be given a prompt A (child) and a general prompt B (parent) that is a generalization of prompt A. Your task is to determine how well prompt B serves as a semantic category for prompt A.
Calculate a score between 0 to 1 for how well prompt B captures the core intent, domain, and primary task of prompt A. We are looking for semantic coverage, not strict detail preservation. If prompt B correctly identifies the type of task prompt A is asking for (e.g. "writing an analytical essay" or "coding a data processing script") it should score highly, even if it abstracts away specific tools or minor constraints.

Scoring guidance:
- 0.85-1.0 (Excellent Category): Prompt B is a highly accurate category that captures the core semantic intent of Prompt A. It may abstract away minor specifics (like specific libraries or datasets) but fully retains the domain and primary task.
- 0.6-0.85 (Good Generalization): Prompt B is a broad categorization. It captures the general type of task (e.g., "Write an analytical essay" or "Write a data processing script") but loses the specific topical domain of Prompt A.
- 0.3-0.6 (Vague or Gerund Phrase): Prompt B is extremely abstract (e.g., "Analyzing data" or "Writing content"). It is technically true but provides almost no structural or topical guidance for Prompt A.
- 0.0-0.3 (Mismatch): Prompt B hallucinated constraints that are NOT present in Prompt A, or it completely miscategorized the task.

Finally, output your score in the following format:
Score: [SCORE]

## Prompt A (child)
{prompt_a}

## Prompt B (parent)
{prompt_b}

Finally, output your score in the following format:
Score: [SCORE]

## Examples
Here are some examples:
High-Scoring Example
Prompt A:
Write a Python script that uses the Hugging Face Transformers library to fine-tune a BERT model for sentiment analysis on a custom dataset. Include data preprocessing using the datasets library, tokenizer usage, training loop with Trainer, and evaluation metrics such as accuracy and F1-score.
Prompt B:
Write a program that processes and manages structured data, implementing user-facing operations with organized output.
Score: 0.20
Reasoning for your understanding only:
Prompt B miscategorizes the task. Prompt A is about training a machine learning model, not managing user-facing CRUD operations. This is a mismatch.

High-Scoring Example 2
Prompt A:
Write a 2,000-word research paper about how social media influences political polarization in the United States. Discuss both positive and negative effects, provide examples from the past decade, and include at least five scholarly sources formatted in APA style.
Prompt B:
Write a structured piece of analytical writing in English that explores societal themes.
Score: 0.75
Reasoning for your understanding only:
Prompt B correctly captures the broad category (analytical writing exploring societal themes), but it loses the specific domain (political polarization and social media). Therefore, it gets a "Good Generalization" score.

Mid-Scoring Example
Prompt A:
Implement a convolutional neural network (CNN) from scratch using PyTorch to classify handwritten digits from the MNIST dataset. The implementation should include custom-defined layers, training and validation loops without using PyTorch's torchvision.models, and visualizations of training loss and accuracy curves using Matplotlib.
Prompt B:
Write a Python program to train a deep learning model for image classification.
Score: 0.95
Reasoning for your understanding only:
Prompt B captures the core domain, task, and language perfectly. It abstracts away MNIST, CNNs, and Matplotlib, but it is an excellent category for the child prompt.

Low-Scoring Example
Prompt A:
Develop a deep reinforcement learning agent using Python and TensorFlow 2. Implement the Deep Q-Network (DQN) algorithm to solve the CartPole-v1 environment from OpenAI Gym.
Prompt B:
Writing code.
Score: 0.40
Reasoning for your understanding only:
Prompt B is a gerund phrase that is technically true but provides almost zero structural or topical guidance. It is far too abstract.
"""
