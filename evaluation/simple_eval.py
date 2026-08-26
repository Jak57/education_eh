from typing import Iterable
import openpyxl
from openai import OpenAI
from tree import Level, Node, Tree, Stat
import re
import time
import random
import anthropic
# from anthropic import Anthropic 
# from anthropic import Anthropic
# from anthropic.exceptions import APIError, RateLimitError, OverloadedError # Import necessary exceptions


# Global variables and imports handled
level = 1
evalPrompt = """
Verifying Specific Common Summary of Two Similar Stories. 

You would be given two similar short stories and a description about the stories. 

First, determine the following:
1. Determine Score_A -  Given that story A happened, please check if the description also happened (story A implies descriptions) (i.e., The story implies the description and the description does NOT contain some text that are irrelevant or contradictory to the story). Determine a score between 0 to 1 for how much provided common_summary is a summary of story A (Score_A). Understand that the summary CANNOT contain any information that is not in story A. When judging the implication relations, the details should not be ignored. For example, if a description says A student went hiking tomorrow but a story is talking about a student went hiking yesterday, the story does NOT imply the summary even though their meanings are very similar. You should NOT select a description if the description contains typos (e.g., job -> jog or hate -> hat) or the readers might not understand the description. However, you can still select a description if the description contains some minor grammartical errors (e.g., tense difference) and/or does not include the important parts of the story (i.e., the description is not a very good summary). 
2. Determine Score_B -  Given that story B happened, please check if the description also happened (story B implies descriptions) (i.e., The story implies the description and the description does NOT contain some text that are irrelevant or contradictory to the story). Determine a score between 0 to 1 for how much provided common_summary is a summary of story B (Score_B). Understand that the summary CANNOT contain any information that is not in story B.
When judging the implication relations, the details should not be ignored. For example, if a description says A student went hiking tomorrow but a story is talking about a student went hiking yesterday, the story does NOT imply the summary even though their meanings are very similar. You should NOT select a description if the description contains typos (e.g., job -> jog or hate -> hat) or the readers might not understand the description. However, you can still select a description if the description contains some minor grammartical errors (e.g., tense difference) and/or does not include the important parts of the story (i.e., the description is not a very good summary). 

Finally, output 2 comma-separated values as follows:
1. Entail-1: Which is the average of score_A and score_B
2. Entail-2: Which is the min(score_A, score_B)

Format the output as: **entail-1, entail-2:** [score1], [score2]

Below this, you can also include score_A and score_B and the reasoning for the scores. 

Here are some examples:
Example 1: High Entailment
Prompt A:
Write a short story set in occupied France during World War II, where a teenage girl joins the local resistance. She begins secretly smuggling messages hidden in loaves of bread while pretending to be an innocent bakery assistant. Over time, her courage grows as she faces increasing danger from patrolling soldiers.
Prompt B:
Write a suspenseful narrative about a British schoolteacher who is recruited by Allied intelligence during WWII and sent behind enemy lines to intercept German communications. Though inexperienced, he learns to navigate espionage under constant threat.
Prompt C (Generalization):
Write a historical fiction story set during World War II, in which an ordinary civilian secretly participates in resistance or espionage activities, gradually gaining confidence while operating under life-threatening conditions.
Output:
Prompt C details:
	1. Historical fiction
	2. Set during WWII
	3. Civilian protagonist
	4. Secretly participates in resistance or espionage
	5. Gains confidence over time
	6. Operates under life-threatening danger
Score A calculation: 6/6 = 1.0
Score B calculation: 6/6 = 1.0
**entail-1, entail-2:** 1.0, 1.0
Reasoning:
All six details in Prompt C are directly present or strongly implied in both Prompt A and B.

Example 2: Low Entailment
Prompt A:
Write a detective story set in 1920s Chicago, where a private investigator is hired to find a missing jazz singer. As he digs deeper, he uncovers a web of corruption involving bootleggers, police officials, and a powerful music producer with ties to organized crime.
Prompt B:
Tell a noir-inspired mystery about a crime journalist in Prohibition-era New York who stumbles onto a murder cover-up linked to a speakeasy owner and a corrupt mayor. The deeper he investigates, the more dangerous the city becomes for him.
Prompt C (Poor Generalization):
Write a coming-of-age story about a young orphan who discovers a hidden talent for jazz trumpet and rises to fame in 1920s New Orleans, confronting prejudice and personal doubt along the way.
Output:
Prompt C details:
	1. Coming-of-age story
	2. Young orphan protagonist
	3. Learns jazz trumpet
	4. Rises to fame
	5. Set in 1920s New Orleans
	6. Themes of prejudice
	7. Themes of self-doubt
Score A calculation: 1/7 ≈ 0.14
Score B calculation: 1/7 ≈ 0.14
**entail-1, entail-2:** 0.14, 0.14
Reasoning:
Despite a shared time period and some mention of jazz in A, the themes, genre, and character arc of Prompt C are completely different and not implied in A or B.

Example 3: Medium Entailment
Prompt A:
Write a story about a lonely retiree who takes up creative writing and, through a local writing club, begins to reconnect with others. The story should explore how writing allows him to process his past and find renewed purpose.
Prompt B:
Write a short story about a single mother who starts writing fiction online as an outlet for stress. As her stories gain popularity, she gains confidence and begins to imagine a new future for herself.
Prompt C (Generalization):
Write a story about an adult facing emotional isolation who begins writing fiction as a form of healing and eventually builds meaningful new relationships and a sense of identity through a community of readers or writers.
Output:
Prompt C details:
	1. Adult protagonist
	2. Emotionally isolated
	3. Begins writing fiction
	4. Writing is a form of healing
	5. Builds new relationships
	6. Gains a sense of identity
	7. Community involves writers or readers
Score A calculation: 6.5/7 ≈ 0.93
Score B calculation: 5.5/7 ≈ 0.79
**entail-1, entail-2:** 0.86, 0.79
Reasoning:
Prompt A strongly implies most of the generalization but is slightly vague about the larger community aspect. Prompt B covers emotional growth and writing but is less explicit about isolation and building new relationships.
"""
evalPrompt2 = """
You will be given a prompt A and a general prompt B that is a generalization of prompt A. Your task is to determine how well prompt B serves as the generalization of prompt A.

Calculate a score between 0 to 1 for how much prompt A implies prompt C. Score close to 1 means all or most detail in prompt B exist or is implied in prompt A. Score close to 0 means all or most detail in prompt B does not exist or is not implied in prompt A. Base your score as much as possible to the ratio of how many details in prompt B are in or are implied in prompt A out of the total distinct details of B.

Finally, output your score in the following format:
Score: [SCORE]

Below this, give your reasoning for your score. 

Here are some examples:
High-Scoring Example
Prompt A:
Write a Python script that uses the Hugging Face Transformers library to fine-tune a BERT model for sentiment analysis on a custom dataset. Include data preprocessing using the datasets library, tokenizer usage, training loop with Trainer, and evaluation metrics such as accuracy and F1-score. Also, provide inline comments to explain each step of the process.
Prompt B:
Create a Python program that demonstrates how to fine-tune a pre-trained NLP model using the Hugging Face Transformers library on a labeled dataset for a text classification task. The program should show preprocessing, tokenization, training, and evaluation, and include explanations in the code.
Score: 0.95
Reasoning:
Prompt B accurately generalizes Prompt A. It captures all core elements: Python, Hugging Face Transformers, fine-tuning, text classification (which includes sentiment analysis), preprocessing, tokenization, training, and evaluation. It does omit the exact model name ("BERT") and specific metrics (like F1-score) but keeps the essential instructional structure and goal. The loss of only minor detail results in a very high score.

Mid-Scoring Example
Prompt A:
Implement a convolutional neural network (CNN) from scratch using PyTorch to classify handwritten digits from the MNIST dataset. The implementation should include custom-defined layers, training and validation loops without using PyTorch’s torchvision.models, and visualizations of training loss and accuracy curves using Matplotlib.
Prompt B:
Create a PyTorch-based image classification model for a standard dataset, avoiding pre-built architectures and training it from scratch.
Score: 0.65
Reasoning:
Prompt B captures several major elements from A: it specifies PyTorch, training from scratch, image classification, and a standard dataset (MNIST fits this). However, B omits key details: no mention of CNNs, handwritten digits, manual loop implementation, or visualization. It’s a reasonably good generalization but loses too many specifics for a higher score.

Low-Scoring Example
Prompt A:
Develop a deep reinforcement learning agent using Python and TensorFlow 2. Implement the Deep Q-Network (DQN) algorithm to solve the CartPole-v1 environment from OpenAI Gym. The agent should include experience replay, target network updates, and ε-greedy exploration strategy. Provide a training loop and visualize rewards over episodes.
Prompt B:
Write a program that uses AI to play a game.
Score: 0.25
Reasoning:
Prompt B is far too vague to meaningfully imply the details of Prompt A. While technically Prompt A fulfills the requirement in B (it is a program using AI to play a game), the inverse does not hold: Prompt B could refer to anything from rule-based agents in Tic-Tac-Toe to generative agents in Minecraft. It lacks any mention of reinforcement learning, specific algorithms (DQN), the use of Python or TensorFlow, or the CartPole environment. Therefore, the overlap is minimal, justifying a low score.
"""

# --------------------------------------------------------------------------- #
# ADDITIVE: faithfulness + informativeness judge prompts (split of evalPrompt2).
# Used by evaluate_common_summary alongside the existing entail/specificity
# scoring. {prompt_a}/{prompt_b} are appended at call time (see _judge_call).
# --------------------------------------------------------------------------- #
evalPrompt_faithfulness = """
You will be given a prompt A (child) and a general prompt B (parent) that is meant to be a generalization of prompt A. Your task is to determine how FAITHFUL prompt B is to prompt A.

Faithfulness measures whether everything prompt B asks for is actually grounded in prompt A. Calculate a score between 0 and 1 for what fraction of the distinct details in prompt B (the parent) are present in, or implied by, prompt A (the child).

Faithfulness measures ONE thing: of the details prompt B actually states, what fraction are supported by prompt A? It says nothing about how much of A that B leaves out.

Important:
- The ONLY reason to score below 1.0 is a detail B asserts that A does not (fully) support — a task, domain, constraint, or detail B invents. Nothing else lowers this score.
- Omission, abstraction, and genericness are NEVER penalized here. It is expected and correct for B to drop specifics from A (specific libraries, datasets, constraints) — even to keep only A's topic or domain and drop A's task or format. "B omits A's requirements" and "B is too vague" are INFORMATIVENESS concerns, not faithfulness ones. A parent that drops detail but invents nothing is fully faithful (1.0), no matter how sparse it is.
- Implicit support counts fully. A detail in B does NOT need to appear verbatim in A — if A clearly implies it, it is fully grounded; do not deduct just because A does not state B's wording explicitly. An abstract parent whose every detail is implied by A is fully faithful (1.0).
- Implication can be partial. If a detail in B is typically but not necessarily implied by A (it usually follows but is not guaranteed), give it partial credit. This — a weakly-supported or unsupported detail B chose to assert — is the only thing that pulls the score down.

Base your score on the ratio of B's distinct details that are supported by A, out of the total distinct details in B.

Scoring guidance:
- 0.85-1.0 (Fully Grounded): Every detail B states is present in or implied by A, however abstract or sparse B is. An abstract parent whose every detail is implied by A scores 1.0 — its abstraction does not cost anything.
- 0.6-0.85 (Mostly Grounded): Nearly everything B states is supported, but B asserts one detail A only probably/partially implies, or a minor shift in framing. Reserve this for a genuinely weakly-supported detail — NOT for B being abstract or omitting A's content.
- 0.3-0.6 (Partially Grounded): B mixes supported content with a noticeable detail or constraint that A does not support even implicitly. (Reserve this for genuinely absent or invented content — not for details that A merely implies rather than states.)
- 0.0-0.3 (Hallucinated / Mismatched): B introduces a task, domain, or constraint that is absent from or contradicts A (e.g., A trains an ML model, but B describes user-facing CRUD operations).

## Examples

Prompt A (child): Implement a convolutional neural network (CNN) from scratch using PyTorch to classify handwritten digits from MNIST, with custom layers, training/validation loops without torchvision.models, and loss/accuracy curves in Matplotlib.
Prompt B (parent): Write a Python program to train a deep learning model for image classification.
Score: 1.0
Reason: Every claim in B — Python, training a deep learning model, image classification — is fully present in A. B abstracts away CNNs, MNIST, and Matplotlib, but omission is never penalized, and B invents nothing, so faithfulness is perfect.

Prompt A (child): Develop a deep reinforcement learning agent. Implement the Deep Q-Network (DQN) algorithm to solve the CartPole-v1 environment from OpenAI Gym.
Prompt B (parent): Writing code.
Score: 1.0
Reason: B's only claim, "writing code," is fully entailed by implementing a DQN agent — you cannot implement the algorithm without it — so it is fully grounded. B is maximally vague, but vagueness lowers INFORMATIVENESS, not faithfulness; B asserts nothing A fails to support, so faithfulness is perfect.

Prompt A (child): Write a Python web scraper using requests and BeautifulSoup that collects book titles and prices from an online bookstore's catalog pages and writes them to a CSV file.
Prompt B (parent): Write a Python program that gathers information from a website and archives it for long-term record-keeping.
Score: 0.8
Reason: "Python," "gathers information from a website," and saving the data are all clearly implied by A, so they are fully grounded — abstraction costs nothing. The one dock is "for long-term record-keeping," a purpose A neither states nor clearly implies (writing to a CSV could serve any use); that single weakly-supported detail earns partial credit. The rest is grounded, so faithfulness stays high.

Prompt A (child): Write a 1,000-word op-ed arguing that city councils should expand protected bike lanes, citing reduced traffic fatalities and lower carbon emissions.
Prompt B (parent): Write an opinion piece that argues for a local policy change and backs it up with concrete benefits.
Score: 1.0
Reason: Every detail of B is implied by A: "opinion piece" covers an op-ed, "argues for a local policy change" is implied by advocating that city councils expand bike lanes, and "concrete benefits" is implied by citing fewer fatalities and lower emissions. Implicit support counts and B invents nothing, so the abstraction and dropped specifics cost nothing — faithfulness is perfect.

Prompt A (child): Design a REST API in Go with JWT authentication, rate limiting, a PostgreSQL backend, OpenAPI docs, and integration tests covering every endpoint.
Prompt B (parent): Build a backend web service.
Score: 1.0
Reason: B keeps almost nothing from A — no Go, JWT, rate limiting, PostgreSQL, docs, or tests — so it would score very low on INFORMATIVENESS. But faithfulness only checks that what B does keep ("build a backend web service") is supported by A, and it fully is. Omitting the rest never lowers faithfulness, so the score is 1.0.

Prompt A (child): Write a Python script that uses Hugging Face Transformers to fine-tune a BERT model for sentiment analysis, including preprocessing, tokenization, a Trainer training loop, and accuracy/F1 evaluation.
Prompt B (parent): Write a program that processes and manages structured data, implementing user-facing operations with organized output.
Score: 0.30
Reason: "Processes structured data" is loosely supported, but "implementing user-facing operations" is a hallucinated constraint — A trains a model and has no user-facing/CRUD component. The low score is for this invented task, NOT for B dropping A's specifics (BERT, sentiment analysis, F1) — that omission alone would not lower faithfulness.
"""

evalPrompt_informativeness = """
You will be given a prompt A (child) and a general prompt B (parent) that is meant to be a generalization of prompt A. Your task is to determine how INFORMATIVE prompt B is about prompt A.

Informativeness measures whether prompt B retains the important content of prompt A. Calculate a score between 0 and 1 for what fraction of the important contents of prompt A (the child) are present in, or implied by, prompt B (the parent).

Important:
- First identify the IMPORTANT contents of A: its core intent, domain, primary task, and key constraints. Ignore trivial or incidental details.
- We are looking for SEMANTIC COVERAGE, not strict detail preservation. If B correctly identifies the type of task and domain of A, it should score highly even if it abstracts away specific tools or minor constraints.
- Penalize VAGUENESS: a parent so abstract that it gives almost no structural or topical guidance about A (e.g., "writing code", "analyzing data") is uninformative, even if technically true.
- Do NOT reward B for content not in A; only the important content of A that B actually captures counts.

Base your score on the ratio of A's important contents that are captured by B, out of A's total important contents.

Scoring guidance:
- 0.85-1.0 (Excellent Coverage): B captures essentially all of A's important content — its domain, primary task, and key constraints — dropping only minor specifics (a particular library, dataset, or metric).
- 0.6-0.85 (Good Coverage): B captures A's core domain and primary task but loses several important specifics or constraints.
- 0.3-0.6 (Vague / Gerund Phrase): B is extremely abstract (e.g., "Analyzing data", "Writing content"). It captures the broad activity type but almost none of A's structural or topical content.
- 0.0-0.3 (Uninformative / Mismatched): B captures essentially none of A's important content, because it is far too generic or it miscategorizes the task.

## Examples

Prompt A (child): Implement a convolutional neural network (CNN) from scratch using PyTorch to classify handwritten digits from MNIST, with custom layers, training/validation loops without torchvision.models, and loss/accuracy curves in Matplotlib.
Prompt B (parent): Write a Python program to train a deep learning model for image classification.
Score: 0.75
Reason: B captures A's core content — Python, training a deep learning model, image classification — but loses several important specifics: the "from scratch" constraint, the MNIST dataset, and the visualization requirement.

Prompt A (child): Develop a deep reinforcement learning agent using Python and TensorFlow 2. Implement the Deep Q-Network (DQN) algorithm to solve the CartPole-v1 environment from OpenAI Gym.
Prompt B (parent): Writing code.
Score: 0.15
Reason: B conveys none of A's important content — not the reinforcement-learning domain, the DQN algorithm, TensorFlow, or CartPole. Technically true, but uninformative.

Prompt A (child): Write a Python script that uses Hugging Face Transformers to fine-tune a BERT model for sentiment analysis, including preprocessing, tokenization, a Trainer training loop, and accuracy/F1 evaluation.
Prompt B (parent): Write a program that processes and manages structured data, implementing user-facing operations with organized output.
Score: 0.15
Reason: B miscategorizes A. It captures none of A's important content — machine learning, fine-tuning BERT, sentiment analysis, the NLP domain.
"""

evalPrompt_specificity_faithfulness = """
You will be given a candidate prompt A and a general prompt B. Prompt B is a general prompt — a category or summary formed from a group of other prompts. Prompt A is a candidate prompt that we are testing to see whether it belongs under category B (A was not necessarily used to create B). Your task is to determine how FAITHFUL category B is to candidate A.

Faithfulness measures whether everything category B asks for is actually grounded in candidate A. Calculate a score between 0 and 1 for what fraction of the distinct details in prompt B (the category) are present in, or implied by, prompt A (the candidate). A high score means candidate A satisfies everything the category requires, so it plausibly belongs under B.

Faithfulness measures ONE thing: of the requirements category B actually states, what fraction are supported by candidate A? It says nothing about how much of A that B leaves out.

Important:
- The ONLY reason to score below 1.0 is a requirement B imposes that A does not (fully) meet — a task, domain, constraint, or detail B requires that A does not support. Nothing else lowers this score.
- Do NOT penalize the candidate for being more specific than B, for containing extra detail B omits, or for B being abstract — even if B keeps only A's topic or domain and drops A's task or format. "B omits A's requirements" and "B is too vague" are INFORMATIVENESS concerns, not faithfulness ones. A category that requires only things A supports is fully faithful (1.0), no matter how sparse it is.
- Implicit support counts fully. A requirement in B does NOT need to appear verbatim in A — if candidate A clearly implies it, it is fully satisfied; do not deduct just because A does not state B's wording explicitly. A candidate that implicitly satisfies every requirement of an abstract category is fully faithful (1.0).
- Implication can be partial. If a requirement in B is typically but not necessarily implied by A (it usually follows but is not guaranteed), give it partial credit. This — a requirement A only weakly supports or fails to meet — is the only thing that pulls the score down.

Base your score on the ratio of B's distinct details that are supported by A, out of the total distinct details in B.

Scoring guidance:
- 0.85-1.0 (Fully Grounded): Every requirement in category B is present in or implied by candidate A, however abstract or sparse B is. A candidate that implicitly satisfies every requirement of an abstract category scores 1.0 — B's abstraction does not cost anything.
- 0.6-0.85 (Mostly Grounded): Nearly every requirement B states is met, but B imposes one requirement A only probably/partially implies, or a minor shift in framing. Reserve this for a genuinely weakly-supported requirement — NOT for B being abstract or omitting A's content.
- 0.3-0.6 (Partially Grounded): B mixes supported content with a noticeable detail or requirement that A does not meet even implicitly. (Reserve this for genuinely absent or contradicted requirements — not for ones that A merely implies rather than states.)
- 0.0-0.3 (Hallucinated / Mismatched): B requires a task, domain, or constraint that is absent from or contradicts A (e.g., A trains an ML model, but B describes user-facing CRUD operations).

## Examples

Prompt A (candidate): Implement a convolutional neural network (CNN) from scratch using PyTorch to classify handwritten digits from MNIST, with custom layers, training/validation loops without torchvision.models, and loss/accuracy curves in Matplotlib.
Prompt B (category): Write a Python program to train a deep learning model for image classification.
Score: 1.0
Reason: Every requirement in B — Python, training a deep learning model, image classification — is fully present in A. B abstracts away CNNs, MNIST, and Matplotlib, but the candidate's extra detail is not penalized and B requires nothing A lacks, so faithfulness is perfect.

Prompt A (candidate): Develop a deep reinforcement learning agent. Implement the Deep Q-Network (DQN) algorithm to solve the CartPole-v1 environment from OpenAI Gym.
Prompt B (category): Writing code.
Score: 1.0
Reason: B's only requirement, "writing code," is fully met by implementing a DQN agent — you cannot implement the algorithm without it — so it is fully grounded. B is maximally vague, but vagueness lowers INFORMATIVENESS, not faithfulness; B requires nothing A fails to support, so faithfulness is perfect.

Prompt A (candidate): Write a Python web scraper using requests and BeautifulSoup that collects book titles and prices from an online bookstore's catalog pages and writes them to a CSV file.
Prompt B (category): Write a Python program that gathers information from a website and archives it for long-term record-keeping.
Score: 0.8
Reason: "Python," "gathers information from a website," and saving the data are all clearly implied by A, so the candidate satisfies them — abstraction costs nothing. The one dock is "for long-term record-keeping," a purpose A neither states nor clearly implies (writing to a CSV could serve any use); that single weakly-supported requirement earns partial credit. The rest is met, so faithfulness stays high.

Prompt A (candidate): Write a 1,000-word op-ed arguing that city councils should expand protected bike lanes, citing reduced traffic fatalities and lower carbon emissions.
Prompt B (category): Write an opinion piece that argues for a local policy change and backs it up with concrete benefits.
Score: 1.0
Reason: Every requirement in B is implied by A: an op-ed is an opinion piece, advocating that city councils expand bike lanes implies arguing for a local policy change, and citing fewer fatalities and lower emissions implies concrete benefits. Implicit support counts and B requires nothing A lacks, so the abstraction and dropped specifics cost nothing — faithfulness is perfect.

Prompt A (candidate): Design a REST API in Go with JWT authentication, rate limiting, a PostgreSQL backend, OpenAPI docs, and integration tests covering every endpoint.
Prompt B (category): Build a backend web service.
Score: 1.0
Reason: B keeps almost nothing from A — no Go, JWT, rate limiting, PostgreSQL, docs, or tests — so it would score very low on INFORMATIVENESS. But faithfulness only checks that the requirement B does impose ("build a backend web service") is supported by A, and it fully is. Omitting the rest never lowers faithfulness, so the score is 1.0.

Prompt A (candidate): Write a Python script that uses Hugging Face Transformers to fine-tune a BERT model for sentiment analysis, including preprocessing, tokenization, a Trainer training loop, and accuracy/F1 evaluation.
Prompt B (category): Write a program that processes and manages structured data, implementing user-facing operations with organized output.
Score: 0.30
Reason: "Processes structured data" is loosely supported, but "implementing user-facing operations" is a requirement the candidate does not meet — A trains a model and has no user-facing/CRUD component. The low score is for this unmet/invented requirement, NOT for B dropping A's specifics (BERT, sentiment analysis, F1) — that omission alone would not lower faithfulness.
"""

evalPrompt_specificity_informativeness = """
You will be given a candidate prompt A and a general prompt B. Prompt B is a general prompt — a category or summary formed from a group of other prompts. Prompt A is a candidate prompt that we are testing to see whether it belongs under category B (A was not necessarily used to create B). Your task is to determine how INFORMATIVE category B is about candidate A.

Informativeness measures whether category B retains the important content of candidate A. Calculate a score between 0 and 1 for what fraction of the important contents of prompt A (the candidate) are present in, or implied by, prompt B (the category). A high score means the category captures the candidate's core intent, so it is a good generalization of the candidate.

Important:
- First identify the IMPORTANT contents of A: its core intent, domain, primary task, and key constraints. Ignore trivial or incidental details.
- We are looking for SEMANTIC COVERAGE, not strict detail preservation. If B correctly identifies the type of task and domain of A, it should score highly even if it abstracts away specific tools or minor constraints.
- Penalize VAGUENESS: a category so abstract that it gives almost no structural or topical guidance about A (e.g., "writing code", "analyzing data") is uninformative, even if technically true.
- Do NOT reward B for content not in A; only the important content of A that B actually captures counts.

Base your score on the ratio of A's important contents that are captured by B, out of A's total important contents.

Scoring guidance:
- 0.85-1.0 (Excellent Coverage): B captures essentially all of A's important content — its domain, primary task, and key constraints — dropping only minor specifics.
- 0.6-0.85 (Good Coverage): B captures A's core domain and primary task but loses several important specifics or constraints.
- 0.3-0.6 (Vague / Gerund Phrase): B is extremely abstract. It captures the broad activity type but almost none of A's structural or topical content.
- 0.0-0.3 (Uninformative / Mismatched): B captures essentially none of A's important content, because it is far too generic or it miscategorizes the task.

## Examples

Prompt A (candidate): Implement a convolutional neural network (CNN) from scratch using PyTorch to classify handwritten digits from MNIST, with custom layers, training/validation loops without torchvision.models, and loss/accuracy curves in Matplotlib.
Prompt B (category): Write a Python program to train a deep learning model for image classification.
Score: 0.75
Reason: B captures A's core content — Python, training a deep learning model, image classification — but loses several important specifics: the "from scratch" constraint, the MNIST dataset, and the visualization requirement.

Prompt A (candidate): Develop a deep reinforcement learning agent using Python and TensorFlow 2. Implement the Deep Q-Network (DQN) algorithm to solve the CartPole-v1 environment from OpenAI Gym.
Prompt B (category): Writing code.
Score: 0.15
Reason: B conveys none of A's important content — not the reinforcement-learning domain, the DQN algorithm, TensorFlow, or CartPole.

Prompt A (candidate): Write a Python script that uses Hugging Face Transformers to fine-tune a BERT model for sentiment analysis, including preprocessing, tokenization, a Trainer training loop, and accuracy/F1 evaluation.
Prompt B (category): Write a program that processes and manages structured data, implementing user-facing operations with organized output.
Score: 0.15
Reason: B miscategorizes A and captures none of A's important content — machine learning, fine-tuning BERT, sentiment analysis, the NLP domain.
"""


def _judge_call(user_prompt: str, model_name: str = "openai", api_keys: dict = None) -> str:
    """Single LLM judge call for the faithfulness/informativeness evaluators.
    Mirrors the model dispatch used elsewhere in this module; returns the
    stripped text content."""
    system = "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."
    if model_name == "openai":
        client = OpenAI(api_key=api_keys["openai"])
        resp = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        )
        return resp.choices[0].message.content.strip()
    elif model_name == "claude":
        client = anthropic.Anthropic(api_key=api_keys["claude"])
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024, system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text.strip()
    elif model_name == "gemma":
        client = OpenAI(api_key=api_keys["gemma"]["api_key"], base_url=api_keys["gemma"]["base_url"])
        resp = client.chat.completions.create(
            model=api_keys["gemma"].get("model", "gemma-3-finetuned-merged-bf16"), max_tokens=1024,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        )
        return resp.choices[0].message.content.strip()
    elif model_name == "gemini":
        client = OpenAI(api_key=api_keys["gemini"]["api_key"], base_url=api_keys["gemini"]["base_url"])
        resp = client.chat.completions.create(
            model="gemini-2.0-flash", max_tokens=1024,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        )
        return resp.choices[0].message.content.strip()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def safe_extract_score(text):
    # Robust extraction: models often reason first and format the score with
    # markdown / prose (e.g. "**Score:** 0.6", "the score is 0.6"), which the
    # old strict "Score: 0.6" pattern missed and silently scored 0.0.
    text = text or ""
    # Tier 1: an explicit score line, e.g. "Score: 0.6", "**Score:** 0.6", "Score = 0.6".
    # First match wins so a "Reason:" line that mentions a number cannot override it.
    m = re.findall(r"(?im)^\s*\**\s*score\b[\s:=*]*?(\d+(?:\.\d+)?)", text)
    if m:
        return float(m[0])
    # Tier 2: any "score ... number" mention (e.g. "Final Score: 0.6", "the score is 0.6");
    # take the last so a trailing final score wins over earlier rubric mentions.
    m = re.findall(r"(?i)score[^\d\n]{0,20}?(\d+(?:\.\d+)?)", text)
    if m:
        return float(m[-1])
    # Tier 3: fall back to the last standalone number in [0, 1].
    m = re.findall(r"(?<![\d.])([01](?:\.\d+)?)(?![\d.])", text)
    if m:
        return float(m[-1])
    print(f"WARNING: Could not find a score in LLM output:\n{text[:120]}...")
    return 0.0

# def call_anthropic_with_retry(client: Anthropic, system_prompt: str, user_prompt: str, max_retries=5):
#     """Calls Anthropic API with exponential backoff on OverloadedError (529)."""
#     for attempt in range(max_retries):
#         try:
#             # 1. Attempt the API call
#             response = client.messages.create(
#                 model="claude-3-5-haiku-latest",
#                 max_tokens=1024,
#                 system=system_prompt,
#                 messages=[{"role": "user", "content": user_prompt}]
#             )
#             return response

#         except OverloadedError as e:
#             # 2. Handle the OverloadedError
#             if attempt < max_retries - 1:
#                 # Calculate sleep time: base * 2^attempt + random jitter
#                 sleep_time = (2 ** attempt) + random.uniform(0, 2)
#                 print(f"Anthropic Overloaded (529). Retrying in {sleep_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
#                 time.sleep(sleep_time)
#             else:
#                 # 3. Raise error if max retries are hit
#                 print(f"Failed after {max_retries} attempts due to OverloadedError.")
#                 raise e
        
#         # Optionally, handle other Anthropic errors if needed (e.g., RateLimitError 429)
#         except APIError as e:
#              print(f"Anthropic API Error: {e}")
#              raise e
             
#     # Should not be reached, but good for type checking
#     raise Exception("Exited retry loop unexpectedly.")


def evaluate_common_summary(tree: Tree, level_idx: int, eval_specificity=False, eval_entail_1: bool = True, eval_entail_2: bool = True, force: bool = False, model_name: str = "openai", api_keys: dict = None):
    global level

    level_obj = tree.levels[level_idx]
    
    # entail2s = []

    
    for idx, node in enumerate(level_obj.nodes):
        # Skip losers
       

        # might need to change to be more consistent
        partner_idx = node.match
        partner_node = level_obj.nodes[partner_idx] if partner_idx is not None else None
        
        if eval_specificity:
            # partner_idx = node.third
            # partner_node = level_obj.nodes[partner_idx] if partner_idx is not None else None
            prompt_a = node.prompt
            prompt_b = (level_obj.nodes[node.third].prompt if node.third is not None else None)
            score          = node.third_score
            common_summary = node.summary
            # NOTE: do NOT gate on `score` (third_score) — it is only metadata and is
            # cleared on leaves by propagate_attrs_up, which would otherwise make
            # re-scored leaves un-evaluatable. A present prompt_b already implies a pair.
            if not prompt_a or not prompt_b or not common_summary:
                node.evaluation_specificity = "Error: Missing input. No evaluation generated."
                node.specificity = None
                continue
        else: 
            
            if not node.alive:
                continue
            # If this node already has an evaluation, propagate it to partner (if missing) and skip API call
            # if node.evaluation:
            #     if partner_node and not partner_node.evaluation:
            #         partner_node.evaluation = node.evaluation
            #         partner_node.entail_1 = node.entail_1
            #         partner_node.entail_2 = node.ential_2
            #     continue

            # # If partner already evaluated, copy and skip
            # if partner_node and partner_node.evaluation:
            #     node.evaluation = partner_node.evaluation
            #     node.entail_1 = partner_node.entail_1
            #     node.entail_2 = partner_node.entail_2
            #     continue

            if not force and getattr(node, 'evalScore', None) is not None and partner_node is not None and getattr(partner_node, 'evalScore', None) is not None:
                continue

            prompt_a       = node.prompt
            
            prompt_b       = (level_obj.nodes[node.match].prompt if node.match is not None else None)
            score          = node.score
            common_summary = node.summary
            # NOTE: do NOT gate on `score` — it is only metadata and is cleared on
            # leaves by propagate_attrs_up, which would otherwise make re-scored
            # leaves un-evaluatable. A present prompt_b already implies a valid pair.
            if not prompt_a or not prompt_b or not common_summary:
                node.evaluation = "Error: Missing input. No evaluation generated."
                continue

        eval_input = f"""
            Here is prompt A, B, & C:
            Prompt A: {prompt_a} 
            Prompt B: {prompt_b} 
            Prompt C: {common_summary}
            """
        eval_input_a = f"""
        Now execute the above task with the following prompt A & B:
        prompt A: {prompt_a}
        prompt B: {common_summary}
        """
        
        eval_input_b = f"""
        Now execute the above task with the following prompt A & B:
        prompt A: {prompt_b}
        prompt B: {common_summary}
        """
        
        

        if model_name == "openai":
            client = OpenAI(
                api_key=api_keys["openai"]
            )
            responseA = client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."},
                    {"role": "user", "content": evalPrompt2 + eval_input_a}
                ]
            )
            responseB = client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."},
                    {"role": "user", "content": evalPrompt2 + eval_input_b}
                ]
            )
            evalA = responseA.choices[0].message.content.strip()
            evalB = responseB.choices[0].message.content.strip()

        elif model_name == "claude":
            client = anthropic.Anthropic(
                api_key=api_keys["claude"]
            )
            responseA = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system="You are a professional editor. Read the following prompt carefully and respond to the best of your ability.",
                messages=[{"role": "user", "content": evalPrompt2 + eval_input_a}]
            )
            responseB = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system="You are a professional editor. Read the following prompt carefully and respond to the best of your ability.",
                messages=[{"role": "user", "content": evalPrompt2 + eval_input_b}]
            )
            evalA = responseA.content[0].text.strip()
            evalB = responseB.content[0].text.strip()

        elif model_name == "gemma":
            client = OpenAI(
                api_key=api_keys["gemma"]["api_key"],
                base_url=api_keys["gemma"]["base_url"]
            )
            responseA = client.chat.completions.create(
                model=api_keys["gemma"].get("model", "gemma-3-finetuned-merged-bf16"),
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."},
                    {"role": "user", "content": evalPrompt2 + eval_input_a}
                ]
            )
            responseB = client.chat.completions.create(
                model=api_keys["gemma"].get("model", "gemma-3-finetuned-merged-bf16"),
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."},
                    {"role": "user", "content": evalPrompt2 + eval_input_b}
                ]
            )
            evalA = responseA.choices[0].message.content.strip()
            evalB = responseB.choices[0].message.content.strip()
        elif model_name == "gemini":
            client = OpenAI(
                api_key=api_keys["gemini"]["api_key"],
                base_url=api_keys["gemini"]["base_url"],
            )
            responseA = client.chat.completions.create(
                model="gemini-2.0-flash",
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."},
                    {"role": "user", "content": evalPrompt2 + eval_input_a},
                ],
            )
            responseB = client.chat.completions.create(
                model="gemini-2.0-flash",
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."},
                    {"role": "user", "content": evalPrompt2 + eval_input_b},
                ],
            )
            evalA = responseA.choices[0].message.content.strip()
            evalB = responseB.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        # evalA = responseA.content[0].text.strip() # for claude
        # evalB = responseB.content[0].text.strip()
        # scoreA = float(re.findall(r"Score: (\d+(?:\.\d+)?)", evalA)[0])
        # scoreB = float(re.findall(r"Score: (\d+(?:\.\d+)?)", evalB)[0])
        scoreA = safe_extract_score(evalA)
        scoreB = safe_extract_score(evalB)
        entail_1 = (scoreA + scoreB)/2
        entail_2 = min(scoreA, scoreB)
        
        if eval_specificity:
            # node.evaluation_specificity = evaluation
            # node.specificity = float(match[0][0])
            node.evaluation_specificity = "Eval A:\n" + evalA + "\n" + "Eval B:\n" + evalB
            node.specificity = entail_1
        else:
            # node.evaluation = evaluation
            # node.entail_1 = float(match[0][0])
            # node.entail_2 = float(match[0][1])
            node.evaluation = "Eval A:\n" + evalA + "\n" + "Eval B:\n" + evalB
            node.evalScore = scoreA
            if eval_entail_1:
                node.entail_1 = entail_1
            if eval_entail_2:
                node.entail_2 = entail_2
            
        # Also propagate to partner if they don't have evaluation yet
            if partner_node:
                if not partner_node.evaluation:
                    partner_node.evaluation = "Eval A:\n" + evalA + "\n" + "Eval B:\n" + evalB
                    if eval_entail_1:
                        partner_node.entail_1 = entail_1
                    if eval_entail_2:
                        partner_node.entail_2 = entail_2
                partner_node.evalScore = scoreB

        # ============================================================== #
        # ADDITIVE: faithfulness + informativeness (two-direction split).
        # Reuses prompt_a (node), prompt_b (match for entail / third for
        # specificity) and common_summary set above. Each dimension is the
        # average of the A->summary and B->summary judge calls. Stored on
        # new node fields; the entail_1/entail_2/specificity logic above is
        # left untouched.
        # ============================================================== #
        if eval_specificity:
            _faith_prompt = evalPrompt_specificity_faithfulness
            _info_prompt  = evalPrompt_specificity_informativeness
        else:
            _faith_prompt = evalPrompt_faithfulness
            _info_prompt  = evalPrompt_informativeness

        _fin_a = f"""

## Prompt A (child)
{prompt_a}

## Prompt B (parent)
{common_summary}

Output your score and a brief justification in the following format:
Score: [SCORE]
Reason: [one or two sentences explaining which details drove the score]
"""
        _fin_b = f"""

## Prompt A (child)
{prompt_b}

## Prompt B (parent)
{common_summary}

Output your score and a brief justification in the following format:
Score: [SCORE]
Reason: [one or two sentences explaining which details drove the score]
"""
        _fA = _judge_call(_faith_prompt + _fin_a, model_name, api_keys)
        _fB = _judge_call(_faith_prompt + _fin_b, model_name, api_keys)
        _iA = _judge_call(_info_prompt + _fin_a, model_name, api_keys)
        _iB = _judge_call(_info_prompt + _fin_b, model_name, api_keys)
        _fa, _fb = safe_extract_score(_fA), safe_extract_score(_fB)
        _ia, _ib = safe_extract_score(_iA), safe_extract_score(_iB)
        # 1 = average, 2 = minimum (mirrors entail_1 / entail_2)
        _faith1, _faith2 = (_fa + _fb) / 2, min(_fa, _fb)
        _info1,  _info2  = (_ia + _ib) / 2, min(_ia, _ib)
        _faith_eval = "Faithfulness A:\n" + _fA + "\nFaithfulness B:\n" + _fB
        _info_eval  = "Informativeness A:\n" + _iA + "\nInformativeness B:\n" + _iB

        if eval_specificity:
            node.specificity_faithfulness = _faith1
            node.specificity_informativeness = _info1
            node.evaluation_specificity_faithfulness = _faith_eval
            node.evaluation_specificity_informativeness = _info_eval
        else:
            node.faithfulness1 = _faith1
            node.faithfulness2 = _faith2
            node.informativeness1 = _info1
            node.informativeness2 = _info2
            node.evaluation_faithfulness = _faith_eval
            node.evaluation_informativeness = _info_eval
            if partner_node and partner_node.faithfulness1 is None:
                partner_node.faithfulness1 = _faith1
                partner_node.faithfulness2 = _faith2
                partner_node.informativeness1 = _info1
                partner_node.informativeness2 = _info2
                partner_node.evaluation_faithfulness = _faith_eval
                partner_node.evaluation_informativeness = _info_eval


    # tree.dump("10NNofASpecificity.json")
    # print(f"Successfully populated Evaluation for Level L{level} in the excel sheet.")
    # level += 1


def record_stat(tree: Tree, lvl_idx:int):
    print(f'Getting stat for level {lvl_idx}')
    level_obj = tree.levels[lvl_idx]
    specificity_average = 0
    specificity_range = [1,0]
    entail_1_average = 0
    entail_1_range = [1,0]
    entail_2_average = 0
    entail_2_range = [1,0]
    # --- new metrics (faithfulness/informativeness; entail-context averaged over
    # alive nodes, specificity-context averaged over all nodes, mirroring above) ---
    faithfulness_average = 0
    faithfulness_range = [1,0]
    informativeness_average = 0
    informativeness_range = [1,0]
    spec_faith_average = 0
    spec_faith_range = [1,0]
    spec_info_average = 0
    spec_info_range = [1,0]
    alive_nodes = 0
    for idx, node in enumerate(level_obj.nodes):
        entail_1 = node.entail_1
        entail_2 = node.entail_2
        specificity = node.specificity
        faithfulness = node.faithfulness1
        informativeness = node.informativeness1
        spec_faith = node.specificity_faithfulness
        spec_info = node.specificity_informativeness


        if node.alive == True:
            if entail_1:
                entail_1_average+=entail_1
                entail_1_range = [min(entail_1_range[0], entail_1), max(entail_1_range[1], entail_1)]

            if entail_2:
                entail_2_average+=entail_2
                entail_2_range = [min(entail_2_range[0], entail_2), max(entail_2_range[1], entail_2)]

            if faithfulness:
                faithfulness_average+=faithfulness
                faithfulness_range = [min(faithfulness_range[0], faithfulness), max(faithfulness_range[1], faithfulness)]

            if informativeness:
                informativeness_average+=informativeness
                informativeness_range = [min(informativeness_range[0], informativeness), max(informativeness_range[1], informativeness)]

            alive_nodes+=1
        if specificity:
            specificity_average= specificity_average+specificity
            specificity_range = [min(specificity_range[0], specificity), max(specificity_range[1], specificity)]
        if spec_faith:
            spec_faith_average += spec_faith
            spec_faith_range = [min(spec_faith_range[0], spec_faith), max(spec_faith_range[1], spec_faith)]
        if spec_info:
            spec_info_average += spec_info
            spec_info_range = [min(spec_info_range[0], spec_info), max(spec_info_range[1], spec_info)]

    entail_1_average = entail_1_average/alive_nodes
    entail_2_average = entail_2_average/alive_nodes
    faithfulness_average = faithfulness_average/alive_nodes
    informativeness_average = informativeness_average/alive_nodes
    specificity_average = specificity_average/len(level_obj.nodes)
    spec_faith_average = spec_faith_average/len(level_obj.nodes)
    spec_info_average = spec_info_average/len(level_obj.nodes)
    # print(f'sum:{entail_1_average},{entail_2_average},{specificity_average}')
    level_obj.stat.specificity_average = specificity_average
    level_obj.stat.entail_1_average = entail_1_average
    level_obj.stat.entail_2_average = entail_2_average
    level_obj.stat.entail_1_range = entail_1_range
    level_obj.stat.entail_2_range = entail_2_range
    level_obj.stat.specificity_range = specificity_range
    # --- new metrics ---
    level_obj.stat.faithfulness_average = faithfulness_average
    level_obj.stat.faithfulness_range = faithfulness_range
    level_obj.stat.informativeness_average = informativeness_average
    level_obj.stat.informativeness_range = informativeness_range
    level_obj.stat.specificity_faithfulness_average = spec_faith_average
    level_obj.stat.specificity_faithfulness_range = spec_faith_range
    level_obj.stat.specificity_informativeness_average = spec_info_average
    level_obj.stat.specificity_informativeness_range = spec_info_range
    # tree.dump("10NNofASpecificity.json")

    
def promote_to_next_level(tree: Tree, cur_lvl_idx: int) -> None:
    """
    Build the next level from each *alive* node (the pair-winner) in
    level `cur_lvl_idx`.
    """
    curr = tree.levels[cur_lvl_idx]
    next_nodes: list[Node] = []

    for n_a in curr.nodes:
        if not n_a.alive:                 # skip the loser of each pair
            continue

        # If this node was paired with a virtual dummy, do not create a parent here.
        # We mark it for reassignment and let the existing reassign() step attach it
        # under an appropriate parent at this level.
        if n_a.match is None:
            continue

        partner_idx = n_a.match
        n_b = curr.nodes[partner_idx] if partner_idx is not None else None

        prompt = n_a.summary or n_a.prompt        # fall back if no summary

        # --- create the *parent* (summary) node ---------------------------
        parent_node = Node(
            id       = tree.new_id(),             # globally unique
            prompt   = prompt,
            alive    = True,
            parent   = [],                        # will get its own parent later
            children = [n_a.id] + ([n_b.id] if n_b else [])
        )

        # --- link children → parent (single parent per child) -------------
        n_a.parent = [parent_node.id]
        if n_b:
            n_b.parent = [parent_node.id]

        next_nodes.append(parent_node)

    # append a new level only if we produced summary nodes
    if next_nodes:
        tree.levels.append(Level(
            level = cur_lvl_idx + 1,
            nodes = next_nodes,
            stat = Stat(
                None,
                [],
                None,
                [],
                []
                         )
        ))

from typing import Iterable

def propagate_attrs_up(tree: Tree, attrs: Iterable[str]) -> None:
    """
    For every attribute in *attrs*:

    1. Walk the hierarchy top-down.
       Each child copies its value up to each parent (overwriting).
    2. When level-0 is reached, those attributes are cleared on the leaves.
    """
    attr_list = tuple(attrs)
    # ADDITIVE: bubble the new entail-context scores up exactly like entail_1/entail_2.
    for _extra in ("faithfulness1", "faithfulness2", "informativeness1", "informativeness2"):
        if _extra not in attr_list:
            attr_list = attr_list + (_extra,)

    # build an id → node lookup once (works for the whole tree)
    id2node = {n.id: n for lvl in tree.levels for n in lvl.nodes}

    # ---- TOP-DOWN: highest level → level-1 ------------------------------
    for lvl_idx in range(len(tree.levels) - 1, 0, -1):
        child_lvl = tree.levels[lvl_idx - 1]
        for child in child_lvl.nodes:
            for pid in child.parent:
                parent = id2node[pid]
                # copy each requested attribute, overwriting parent
                for attr in attr_list:
                    val = getattr(child, attr)
                    setattr(parent, attr, val)

    # ---- CLEAR leaves (level-0 prompts) ---------------------------------
    for leaf in tree.levels[0].nodes:
        for attr in attr_list:
            setattr(leaf, attr, None)
def reassign(tree, level_idx, threshold, use_specificity_for_unmatched=False, model_name="openai", api_keys: dict = None):
    print(f"Adding additional edges between nodes and parents for level {level_idx} and {level_idx+1}")

    # Ensure the next level exists (promotion should have run already)
    if level_idx + 1 >= len(tree.levels):
        return

    curr = tree.levels[level_idx]
    next_lvl = tree.levels[level_idx+1]
    _model = None

    for n_a in curr.nodes:
        # If this node was unmatched (dummy-paired), attach to the nearest parent (no threshold)
        if not n_a.parent and n_a.unmatched:
            if not next_lvl.nodes:
                continue

            best_parent = None

            if use_specificity_for_unmatched:
                # Use LLM-based specificity: child prompt vs each parent (summary/prompt)
                best_score = -1.0
                for p in next_lvl.nodes:
                    txt = p.prompt or p.summary or ""
                    if not txt:
                        continue
                    eval_input = f"""
        Now execute the above task with the following prompt A & B:
        prompt A: {n_a.prompt}
        prompt B: {txt}
        """
                    if model_name == "openai":
                        client = OpenAI(
                            api_key=api_keys["openai"]
                        )
                        response = client.chat.completions.create(
                            model="gpt-5.4-mini",
                            messages=[
                                {"role": "system", "content": "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."},
                                {"role": "user", "content": evalPrompt2 + eval_input},
                            ],
                        )
                        content = response.choices[0].message.content.strip()
                    elif model_name == "claude":
                        client = anthropic.Anthropic(
                            api_key=api_keys["claude"]
                        )
                        response = client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=1024,
                            system="You are a professional editor. Read the following prompt carefully and respond to the best of your ability.",
                            messages=[
                                {"role": "user", "content": evalPrompt2 + eval_input},
                            ],
                        )
                        content = response.content[0].text.strip()
                    elif model_name == "gemma":
                        client = OpenAI(
                            api_key=api_keys["gemma"]["api_key"],
                            base_url=api_keys["gemma"]["base_url"]
                        )
                        response = client.chat.completions.create(
                            model=api_keys["gemma"].get("model", "gemma-3-finetuned-merged-bf16"),
                            max_tokens=1024,
                            messages=[
                                {"role": "system", "content": "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."},
                                {"role": "user", "content": evalPrompt2 + eval_input},
                            ],
                        )
                        content = response.choices[0].message.content.strip()
                    elif model_name == "gemini":
                        client = OpenAI(
                            api_key=api_keys["gemini"]["api_key"],
                            base_url=api_keys["gemini"]["base_url"],
                        )
                        response = client.chat.completions.create(
                            model="gemini-2.0-flash",
                            max_tokens=1024,
                            messages=[
                                {"role": "system", "content": "You are a professional editor. Read the following prompt carefully and respond to the best of your ability."},
                                {"role": "user", "content": evalPrompt2 + eval_input},
                            ],
                        )
                        content = response.choices[0].message.content.strip()
                    else:
                        raise ValueError(f"Unknown model_name: {model_name}")
                    m = re.findall(r"Score: (\d+(?:\.\d+)?)", content)
                    score = float(m[0]) if m else 0.0
                    if score > best_score:
                        best_score = score
                        best_parent = p
            else:
                # Default: cosine similarity using sentence embeddings
                if _model is None:
                    _model = SentenceTransformer('all-mpnet-base-v2')
                vec_child = _model.encode(n_a.prompt or "", convert_to_numpy=True)
                vec_child = vec_child / (np.linalg.norm(vec_child) or 1.0)
                best_sim = -1.0
                for p in next_lvl.nodes:
                    txt = p.prompt or p.summary or ""
                    if not txt:
                        continue
                    vec_p = _model.encode(txt, convert_to_numpy=True)
                    vec_p = vec_p / (np.linalg.norm(vec_p) or 1.0)
                    sim = float(np.dot(vec_child, vec_p))
                    if sim > best_sim:
                        best_sim = sim
                        best_parent = p

            if best_parent is not None:
                n_a.parent = [best_parent.id]
                if n_a.id not in best_parent.children:
                    best_parent.children.append(n_a.id)
                n_a.unmatched = False
            # After handling unmatched case (attached or not), skip specificity path for this node
            continue
        # Skip nodes that did not get a parent and aren't flagged for unmatched
        if not n_a.parent:
            continue
        
        # print(f'node: {n_a}')
        third_idx = n_a.third
        # print(f"third node index: {third_idx}")
        if third_idx is None:
            continue
        third = curr.nodes[third_idx]
        # print(f"third node: {third}")
        
        third_id = third.id
        
        # print(f"third id: {third_id}")
        
        specificity = n_a.specificity
        # print(f'specificity: {specificity}')
        
        # print(f'parents:{n_a.parent}')
        parent_id = n_a.parent[0]
        # print(f"parent_id:{parent_id}")
        
        # print(f'upper nodes: {tree.levels[level_idx+1].nodes}')
        
        parent = None
        for node in tree.levels[level_idx+1].nodes:
            if node.id == parent_id:
                parent = node
        # print(f"parent: {parent}")
        
        if specificity and specificity > threshold and parent:
            if parent_id not in third.parent:
                third.parent.append(parent_id)
            if third_id not in parent.children:
                parent.children.append(third_id) 
            
    # take pause here - too many questions:
    # what happens if node has more than one parents? Does it need to test for specificity more than once? do I need to see if third should be in either parents?
    # children can
    