import re
import anthropic
from tree import Tree
import time
from openai import OpenAI 

# openai_api_key = "DUMMY"
# openai_api_base = "http://<GPU_HOST>:8000/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )


# client = OpenAI(
#     api_key = os.environ.get("OPENAI_API_KEY", "")
# )


# Global initialization handled contextually
level = 1

def generate_summary(tree: Tree, level_idx: int, threshold = 0, model_name="openai", api_keys: dict = None):

    global level

    level_obj = tree.levels[level_idx]
    summarized = set()

    for idx, node in enumerate(level_obj.nodes):
        if not node.alive:
            continue
        
        story1 = node.prompt
        story2 = (level_obj.nodes[node.match].prompt
                   if node.match is not None else None)
        score  = node.score

        partner_idx = node.match
        partner_node = level_obj.nodes[partner_idx] if partner_idx is not None else None

        print(f"Level {level} │ Node {idx} │ Summary generation between prompts with entail_2 score: {node.entail_2}")

        if idx in summarized and node.entail_2 >= threshold:
            continue
        # --- do not overwrite existing summaries --------------------------
        # If this node already has a summary, copy it to the partner (if missing) and skip.
        # if node.summary:
        #     if partner_node and not partner_node.summary:
        #         partner_node.summary = node.summary
        #     continue

        # If the partner already has a summary, copy it to this node and skip.
        # if partner_node and partner_node.summary:
        #     node.summary = partner_node.summary
        #     continue

        if not story1 or not story2:
            # return "Error: One or both stories are missing. No summary generated."
            node.summary = "Error: One or both stories are missing. No summary generated."
            continue

        if not score:
            # return "Replicated row. No summary generated."
            node.summary = "Error: Replicated row. No summary generated."
            continue

        prompt = f"""
You will be given prompt A & B. Your task is to write a common prompt C where all details in C are broad enough to be implied by details in both prompt A & B, but are specific as possible so that details in another arbitrary prompt similar to A or B do not imply any details in prompt C. Make sure your prompt C is at least one paragraph and begin your prompt C with "(start)" and end it with "(end)." Here are the details:

First form a general main task alpha out of the main tasks asked of by both prompt A and prompt B.

Then either find shared details directly in prompts A AND B or generalize details that are implied by prompts A AND B. DO NOT include details that are only in or implied by one prompt and not the other. 
 
Finally create your prompt C by building around your general main task alpha with the shared or generalized details.

Here are some examples: 
Example 1
Prompt A:
Write a detailed tutorial on how to fine-tune a pre-trained BERT model for a sentiment classification task using PyTorch and the Hugging Face Transformers library. The tutorial should walk through preparing a labeled dataset, tokenizing the text, modifying the final layers of the model, setting up the training loop, and evaluating accuracy on a validation set. Include sample code and explain key decisions made during the fine-tuning process.
Prompt B:
Describe the full process for fine-tuning a transformer-based language model like RoBERTa to classify product reviews into positive and negative categories. Use the Hugging Face library in conjunction with PyTorch, and explain data preprocessing, setting up the optimizer and scheduler, training on a custom dataset, and validating the results. Provide a high-level overview of each step along with code snippets and recommendations for hyperparameter tuning.
Prompt C:
(start)
Write a comprehensive tutorial that explains how to fine-tune a pre-trained transformer-based language model for a text classification task using PyTorch and the Hugging Face Transformers library. The guide should cover the full workflow, including dataset preparation, tokenization, adjusting the model’s classification head, setting up the training configuration, and evaluating performance on a validation set. Code examples should be included to illustrate each stage of the pipeline, and the explanation should focus on generalizable practices for adapting transformer models to binary or multi-class classification problems in NLP.
(end)

Example 2
Prompt A:
Create a complete walkthrough for building a RESTful API using Python’s Flask framework that enables users to manage a list of tasks. The tutorial should cover setting up the Flask application, creating routes for CRUD operations, using JSON to communicate between client and server, and optionally persisting data using an SQLite database. Include code examples and explain key design choices.
Prompt B:
Write a tutorial that demonstrates how to implement a REST API using Node.js and Express to manage user data. The tutorial should guide readers through the setup process, define route handlers for creating, retrieving, updating, and deleting user entries, and show how to send and receive data using JSON. Optionally, include a section on connecting to a simple database for persistent storage.
Prompt C:
(start)
Write a step-by-step guide for developing a basic RESTful API using a modern backend web framework. The tutorial should show how to define routes that support CRUD (Create, Read, Update, Delete) operations on a structured data model and how to use JSON as the primary data exchange format. The guide should also introduce essential concepts such as routing, request/response handling, and setting up a simple data store for persistence. Code snippets should illustrate each concept, and the tutorial should be applicable to common backend programming environments.
(end)

Example 3
Prompt A:
Develop a set of prompt templates and design guidelines for improving the factual reliability of responses generated by GPT-style models in academic or research contexts. Include examples of how to phrase prompts to minimize hallucinations, how to include context effectively, and how to use few-shot learning to guide model behavior toward factual precision.
Prompt B:
Write an instructional guide for constructing prompts that reduce bias and improve answer neutrality in politically or ethically sensitive topics using large language models. The guide should explain how to word instructions clearly, use representative examples, and maintain balanced tone and framing when designing prompts to elicit objective responses.
Prompt C:
(start)
Write a guide on designing effective prompts for large language models that enhance the reliability, factuality, and neutrality of their responses across a range of contexts. The guide should explain general strategies for phrasing instructions, incorporating context, and structuring examples to reduce hallucination and mitigate bias. Include best practices for few-shot prompting and clear language usage to encourage accurate and balanced outputs from generative AI models.
(end)



Now carry out this task and output a prompt C with the following prompts A & B:
Prompt A: {story1}
Prompt B: {story2}   
        """
        prompt4 = f""""
        You will be given prompt A & B. Your task is to write a common prompt C where all details in C are broad enough to be implied by details in both prompt A & B, but are specific as possible so that details in another arbitrary prompt similar to A or B do not imply any details in prompt C. Make sure your prompt C is at least one paragraph and begin your prompt C with "(start)" and end it with "(end)." Here are the details:

        First form a general main task alpha out of the main tasks asked of by both prompt A and prompt B.
        Here are some examples:
        Prompt A
        Large language models (LLMs) have demonstrated remarkable performance across a variety of tasks, from code generation to question answering. However, fine-tuning these models for domain-specific applications remains a challenge, especially when training data is limited or expensive to obtain. Few-shot and instruction tuning have emerged as promising alternatives to full retraining. Write a technical explanation comparing few-shot learning and instruction tuning in the context of adapting LLMs, discussing their advantages, limitations, and typical use cases in real-world systems.
        Main sentence from Prompt A:
        "Write a technical explanation comparing few-shot learning and instruction tuning in the context of adapting LLMs, discussing their advantages, limitations, and typical use cases in real-world systems."

        Prompt B
        As organizations increasingly integrate machine learning models into production systems, the need for robust and scalable model deployment becomes more critical. Different strategies—such as model serving with REST APIs, containerization using Docker, and orchestration with Kubernetes—are commonly used. Write a detailed comparison of these deployment strategies, highlighting their architectural differences, performance considerations, and scenarios where each is most appropriate.
        Main sentence from Prompt B:
        "Write a detailed comparison of these deployment strategies, highlighting their architectural differences, performance considerations, and scenarios where each is most appropriate."

        Generalized Main Task Alpha
        "Write a detailed technical comparison of two or more approaches used in AI or programming, evaluating their trade-offs, practical use cases, and implementation considerations within real-world systems."

        Explanation:
        How prompt A's main task implies Main Task Alpha
        • "comparing few-shot learning and instruction tuning" → Satisfies "comparison of two or more approaches"
        • "advantages, limitations" → Equivalent to "evaluating trade-offs"
        • "typical use cases in real-world systems" → Matches "practical use cases" and "implementation considerations"

        How prompt B's main task implies Main task Alpha:
        • "comparison of deployment strategies" → Direct match for "comparison of two or more approaches"
        • "architectural differences, performance considerations" → These are "trade-offs" and "implementation considerations"
        • "scenarios where each is most appropriate" → These are "practical use cases in real-world systems"

        Also notice how specific main task alpha is in this example where instead of 'write a piece,' the example uses 'write detailed technical comparison' and include as many details shared by both prompt A and B as explained above.


        Next, either find shared details directly in prompts A AND B or generalize details that are implied by prompts A AND B. Pool these details in a set beta. DO NOT include details that are only in or implied by one prompt and not the other. 

        Here is an example using the same prompts as above:

        Beta:
        1. Comparison of Two or More Technical Approaches
        Explanation:
            • Prompt A: compares few-shot learning vs instruction tuning
            • Prompt B: compares deployment strategies like REST APIs, Docker, and Kubernetes
            • ✅ Generalized: Task involves comparing multiple methods/strategies

        2. Technical Depth and Specificity
        Explanation:
            • Prompt A: focuses on fine-tuning LLMs, learning paradigms, and practical adaptation
            • Prompt B: discusses deployment architecture, performance, and scalability
            • ✅ Generalized: Requires detailed technical understanding of AI or programming topics

        3. Evaluation of Trade-Offs
        Explanation:
            • Prompt A: asks for advantages and limitations of each adaptation method
            • Prompt B: asks for architectural differences and performance considerations
            • ✅ Generalized: Must involve critical evaluation, not just description

        4. Connection to Real-World Use Cases
        Explanation:
            • Prompt A: references real-world systems and practical model adaptation
            • Prompt B: highlights scenarios where each strategy is most appropriate
            • ✅ Generalized: The comparison must be grounded in real-world application contexts

        5. Focus on AI or Programming Domain
        Explanation:
            • Prompt A: AI-focused (language models and learning strategies)
            • Prompt B: Programming/ML Ops-focused (deployment tools and practices)
            • ✅ Generalized: Topic belongs to the AI or programming space

        6. Structured Output: Comparison + Evaluation + Application
        Explanation:
            • Both prompts implicitly require:
                ○ A structured, side-by-side comparison
                ○ An analysis of relative strengths/weaknesses
                ○ Discussion of when and why to use each approach


        Finally create your prompt C by building around your general main task alpha with the shared details you gathered in beta.

        Here is an example prompt C using the same prompts A & B as above:
        Prompts C:
        (start)
        Modern AI applications often require developers to decide how best to adapt or deploy models in a system. Two common approaches are using pre-trained models as-is through APIs versus customizing them through techniques like transfer learning or configuration. Each approach offers different trade-offs in terms of flexibility, complexity, resource usage, and suitability for specific tasks.
        Write a detailed technical comparison of using pre-trained models via APIs versus adapting them through customization techniques. Your analysis should cover their respective strengths and weaknesses, typical implementation scenarios, and factors influencing the choice between them in real-world software systems.
        (end)

        Explanation:
        Prompt C closely follows the criteria by directly reflecting the generalized main task of writing a detailed technical comparison with evaluation and real-world context. It only includes elements clearly implied by both Prompts A and B—such as comparing two AI/programming approaches, analyzing trade-offs, and discussing implementation scenarios—while avoiding details exclusive to either prompt. This ensures Prompt C is specific, technically deep, coherent, and fully grounded in the shared scope of A and B without adding extraneous concepts.

        Now carry out this task and output a prompt C with the following prompts A & B:
        Prompt A: {story1}
        Prompt B: {story2}   
        """
        
        prompt5 = f"""
        You will be given prompt A & B. Your task is to write a common prompt C where all details in C are broad enough to be implied by details in both prompt A & B, but are specific as possible so that details in another arbitrary prompt similar to A or B do not imply any details in prompt C. Make sure your prompt C is at least one paragraph and begin your prompt C with "(start)" and end it with "(end)." Here are the details:

Either find shared details directly in prompts A AND B or generalize details that are implied by prompts A AND B. DO NOT include details that are only in or implied by one prompt and not the other. Find as many common details you can.
 
Finally create your prompt C by compiling these details coherently. Make sure your prompt C is implied by both prompt A & B.

Here are some examples: 

Example 1

Prompt A:
In recent years, the adoption of machine learning in backend systems has dramatically improved predictive performance in services such as fraud detection, recommendation engines, and anomaly monitoring. These applications often require tight integration between model inference and existing APIs, necessitating collaboration between data scientists and software engineers. A successful project involves evaluating trade-offs in model performance versus latency, and ensuring the model can be deployed using containerized workflows such as Docker and Kubernetes in cloud environments.
Prompt B:
Modern AI applications increasingly rely on integrating machine learning models into production-grade software systems. This process often involves deploying models using containers, managing orchestration with tools like Kubernetes, and working closely with software engineers to meet performance constraints, especially latency. Common use cases include recommender systems, fraud detection, and real-time monitoring. These projects require careful attention to the system architecture to ensure that machine learning inference does not degrade the user experience.
Prompt C:
(start)
Describe a scenario where a machine learning model is integrated into a production software system to support real-time applications such as fraud detection or recommendation engines. The deployment should involve collaboration between data scientists and software engineers to ensure the model meets latency requirements. Emphasize the use of containerized environments and orchestration tools such as Docker and Kubernetes. Discuss how production constraints influence design choices in both the model and the system architecture. (end)

Example 2

Prompt A:
AI-powered code generation tools have become increasingly prevalent in integrated development environments (IDEs), helping developers autocomplete functions, generate boilerplate code, and even refactor existing codebases. These tools rely heavily on large language models trained on massive code repositories. While their integration boosts productivity, it requires attention to usability, such as intuitive UX, minimal latency, and compatibility with multiple programming languages. Developers must also address the risk of model hallucination and ensure suggestions align with secure and maintainable coding standards.
Prompt B:
Large language models are being embedded into modern IDEs to enhance developer productivity through capabilities such as code completion, test generation, and code refactoring. These tools are trained on large-scale code datasets and integrated into developer workflows with an emphasis on reducing latency and improving suggestion relevance. To be effective, they must support multiple languages and avoid generating insecure or incorrect code. Designing intuitive interfaces and monitoring user interactions are crucial to ensuring these tools provide real value in real-time software development.
Prompt C:
(start)
Design an AI-powered tool integrated into an IDE that enhances developer productivity by offering features like code completion, test generation, and code refactoring. The system should leverage large language models trained on diverse codebases and be optimized for low latency and multi-language support. Consider how to present suggestions through an intuitive interface and how to manage potential issues like incorrect or insecure code generation. Ensure the tool aligns with secure coding practices and fits seamlessly into the developer's workflow. (end)

Now carry out this task and output a prompt C with the following prompts A & B:
        Prompt A: {story1}
        Prompt B: {story2}   
        """
        prompt6 =f"""
        You will be given prompt A & B. Your task is to write a common prompt C where all details in C are implied by details in both prompt A & B. Make sure your prompt C is at least one paragraph and begin your prompt C with "(start)" and end it with "(end)." Here are the details:
 
First, read prompt A & B carefully and extract all details for each. Next, find the main task for each of the prompts. Then form a general common task alpha out of the two main tasks. Alpha should be implied by BOTH prompts A & B.
 
Then either find shared details directly in prompts A AND B or generalize details that are implied by prompts A AND B. DO NOT include details that are only in or implied by one prompt and not the other. Pool these details in a set Beta. 

Finally, build and output a coherent prompt C out of the common task alpha and common details beta. Overall Prompt C should be implied by both prompts A & B meaning all details in prompts C are in or implied by both prompts A and B.

Here are some examples:

Example 1
Prompt A:
Write a comprehensive blog post for intermediate Python programmers explaining how to use OpenAI’s API to build a chatbot. Include details on setting up the API key, making API calls, and handling user input/output in a terminal interface. Make sure to give practical examples that readers can follow.
Prompt B:
Create a detailed tutorial for software developers that shows how to build a command-line chatbot using OpenAI’s API and Python. The tutorial should walk through steps like authentication, sending messages, and receiving responses, with code snippets and best practices for clarity.
Step-by-step:
	• Alpha (common task):
Write a tutorial for programmers on how to build a command-line chatbot using OpenAI’s API and Python.
	• Beta (shared details):
		○ Audience: intermediate-level programmers (implied in both)
		○ Tool: Python
		○ API: OpenAI API
		○ Use case: command-line chatbot
		○ Key steps: setup/authentication, making API calls, handling input/output, providing code examples
Prompt C:
(start) Write a detailed tutorial for intermediate Python programmers on how to build a command-line chatbot using OpenAI’s API. The tutorial should cover setting up API credentials, making calls to the API, handling user input and output through the terminal, and include clear code examples for each step. (end)

Example 2
Prompt A:
Design an assignment for computer science students that teaches them how to implement a basic neural network from scratch in Python using only NumPy. The assignment should guide students through creating forward and backward propagation, loss calculation, and training with simple datasets like XOR or MNIST.
Prompt B:
Write a project brief aimed at undergraduate learners to help them understand how neural networks function by building one from the ground up using only NumPy. The brief should include requirements for implementing training loops, computing gradients manually, and testing on a toy dataset.
Step-by-step:
	• Alpha (common task):
Write a project prompt to guide students in building a neural network from scratch using NumPy.
	• Beta (shared details):
		○ Audience: undergraduate/computer science students
		○ Tools: Python + NumPy (only)
		○ Scope: build basic neural network manually
		○ Concepts: forward/backward propagation, loss, gradients
		○ Dataset: simple (e.g., XOR, toy datasets)
Prompt C:
(start) Write a project prompt for undergraduate computer science students that guides them in implementing a basic neural network from scratch using only NumPy. The project should cover forward and backward propagation, manual gradient computation, loss calculation, and training/testing on a simple dataset. (end)

Example 3
Prompt A:
Compose a tutorial for AI researchers interested in experimenting with reinforcement learning environments using OpenAI Gym and Python. The tutorial should walk through setting up the Gym environment, choosing a simple task like CartPole, implementing a basic policy gradient method, and visualizing training progress.
Prompt B:
Develop a practical guide for machine learning engineers to get started with reinforcement learning in Python using the OpenAI Gym toolkit. Focus on setting up environments, running basic simulations, and implementing a simple agent with a learning algorithm to solve a predefined control task.
Step-by-step:
	• Alpha (common task):
Write a tutorial for implementing a basic reinforcement learning agent in Python using OpenAI Gym.
	• Beta (shared details):
		○ Audience: ML engineers / AI researchers
		○ Tool: Python
		○ Library: OpenAI Gym
		○ Scope: set up environment, implement simple agent with learning algorithm
		○ Task: basic control task (e.g., CartPole)
		○ Visualization: training progress
Prompt C:
(start) Write a practical tutorial for AI practitioners on how to implement a basic reinforcement learning agent in Python using the OpenAI Gym toolkit. The tutorial should guide users through setting up a Gym environment, selecting a simple control task, applying a basic learning algorithm, and visualizing the agent’s training progress. (end)


        Now carry out this task and output a prompt C with the following prompts A & B:
        Prompt A: {story1}
        Prompt B: {story2}  
        """
        
        prompt7=f"""
        You will be given prompts A and B. Your task is to generate a prompt C where prompt C is implied by BOTH prompt A and prompt B. This means every detail in prompt C is directly mentioned or implied in BOTH prompt A and B. Your prompt C should be a concise 1-2 sentences. Begin your prompt C with "(start)" and end it with "(end)." Here are the step by step instruction:


First, find the one sentence in both prompt A and B that respectively represent the main ask of each prompt. Form a common main ask alpha that is a generalization of the two sentences you found. This means alpha should be strongly implied by both prompt A and B. Explain why your generated alpha satisfies all the criteria.

Here is an example prompt A & B with some good and bad alpha  example outputs:

Prompt A:
We are exploring how to integrate code generation models like Codex into our IDE to assist developers with boilerplate and common patterns. However, we're running into issues where the suggestions are inconsistent or irrelevant depending on the context. Could you provide some guidance on improving the contextual awareness and reliability of AI code suggestions within development environments?
Prompt B:
Our team is building an AI assistant that helps junior developers understand and write Python code by generating examples and explanations. While the assistant works decently, it often produces vague or overly complex answers. We’d like advice on how to improve the clarity and instructional quality of code generation to better serve educational use cases.

Good Alpha example:
Main Ask A:
Provide guidance on improving contextual accuracy and reliability in AI-powered code suggestions within development environments.
Main Ask B:
Suggest ways to improve the clarity and instructional quality of AI-generated Python code for educational use.
Common Main Ask Alpha:
Recommend methods to enhance the usefulness and reliability of AI-generated code in developer-facing tools.
Why it's Good:
• The alpha captures the shared concern: making AI-generated code more useful and dependable.
• It generalizes from "developer IDE context" and "educational context" to “developer-facing tools” overall.
• It stays strongly implied by both prompts without overreaching.

Bad Alpha example:
Main Ask A:
Explain how AI models like Codex can be integrated into an IDE.
Main Ask B:
Describe how to teach Python using AI tools.
Common Main Ask Alpha:
Compare AI tools for software development versus teaching.
Why It’s Bad:
	• The extracted main asks oversimplify the prompts and leave out key concerns like reliability and quality.
	• The alpha introduces a comparison that neither prompt suggested.
	• It is not implied by either A or B, and diverts from their focus on improving AI-generated code.

Another bad Alpha example:
Main Ask A:
Provide guidance on improving the contextual awareness and reliability of AI code suggestions in IDEs.
Main Ask B:
Suggest ways to improve the clarity and instructional quality of AI-generated Python code for educational purposes.
Common Main Ask Alpha:
Propose methods to improve the contextual accuracy of AI-generated code in developer environments.
Why it's bad:
	• The alpha ("improve contextual accuracy of AI-generated code") is clearly implied by Prompt A, which directly discusses "contextual awareness" in an IDE.
	• But Prompt B is not about context at all — it focuses on clarity and educational value, not how well the code fits a surrounding context.
	• So, this alpha fails to generalize both prompts and leans too heavily on Prompt A.
	• It ignores the instructional/educational dimension emphasized in Prompt B, making it an invalid generalization.
------------------------------------------

Next, find common details shared either directly in or strongly implied by both prompts A and B. Pool these details in a set beta. Do not put details in beta that is the conjunction of two similar but distinct detail in prompts A and B. You should be generalizing the two similar details rather than combining them. Explain why your generated beta satisfy all the criteria.

Using the same example prompts A and B as above, here are some good and bad beta example outputs:
Good Beta example
beta = 
	• AI is being used to generate code
	• The quality of generated code is inconsistent or suboptimal
	• There is a desire to improve the usefulness of generated code
	• The target users are developers (professional or novice)
	• Generated code should fit its intended use context (developer workflow or learner comprehension)

Why It’s Good:
	• Every detail is either clearly stated or strongly implied in both prompts
	• The items are specific enough to be meaningful, yet general enough to apply to both contexts
	• No detail relies only on one of the prompts
Bad beta example
beta = 
	• The focus is exclusively on Python
	• The goal is to generate user manuals automatically
	• Codex is being used in both cases
	• The prompts are about CI/CD pipelines
	• Developers are using AI to fix security issues

Why It’s Bad:
	• "Exclusively on Python" is only in Prompt B, not implied by Prompt A
	• "Generate user manuals" is nowhere in either prompt
	• "Codex in both cases" is wrong — Codex is mentioned only in Prompt A
	• "CI/CD pipelines" and "security issues" are not mentioned or implied in either
	• These details either don’t overlap or are made up, violating the instruction

-------------------------------
Finally build your prompt C by coherently assembling your details beta around your core main ask alpha. Prompt C should be strongly implied by both prompts A and B. Explain why prompt C is a good generalization of both prompt A and B.
 
Again using the same prompt A and B as above here is a good example prompt C output and bad example prompt C output.

Good prompt C example:
(start) How can we improve the clarity and contextual relevance of AI-generated code suggestions to better support users working with code? (end)

Why it's good:
Prompt C is a good generalization because it captures the shared goal of both prompts A and B: improving AI-generated code suggestions in terms of clarity and contextual relevance for users. It omits details unique to either prompt while preserving the core intent implied by both.

Bad prompt C example:
"(start) Redesign the AI code suggestion interface with interactive tutorials and real-time debugging to improve beginner onboarding. (end)"

Why it's bad:
This prompt C introduces specific features like tutorials and debugging that are only present in Prompt A, not strongly implied by both A and B. It fails to generalize and instead leans too heavily on one prompt, violating the requirement that C be supported by both.
-------------------------------

Now carry out this task and output a prompt C with the following prompts A & B:
Prompt A: {story1}
Prompt B: {story2}  

        """

        prompt8 = f"""
        You will be given prompts A and B. Your task is to generate a prompt C where prompt C is implied by BOTH prompt A and prompt B. This means every detail in prompt C is directly mentioned or implied in BOTH prompt A and B. Your prompt C should be a concise 1-2 sentences. Begin your prompt C with "(start)" and end it with "(end)." Here are the step by step instruction:

First, find the one sentence in both prompt A and B that respectively represent the main ask of each prompt. Form a common main ask alpha that is a generalization of the two sentences you found. This means alpha should be strongly implied by both prompt A and B. Explain why your generated alpha satisfies all the criteria.

Here is an example:
Prompt A (Creative Writing – Novel Chapter):
Write the first chapter of a speculative fiction novel set in the late 21st century, where climate engineering has become the dominant global response to the escalating climate crisis. Governments and corporations collaborate on ambitious geoengineering projects—such as atmospheric carbon scrubbing, cloud brightening, and orbital solar reflectors—to stabilize Earth’s climate. Your chapter should introduce a protagonist who lives in a region where one of these projects has caused unexpected environmental disruptions, such as altered weather patterns, ecological collapse, or health impacts. Through this character’s daily experience and internal reflections, begin to explore the ethical dilemmas and psychological toll of living in a world reshaped by human technological intervention in nature.
Prompt B (Academic – Research Paper):
Compose a research paper that critically examines the ecological, political, and ethical implications of large-scale geoengineering strategies designed to address global warming. Focus your paper on real or proposed methods such as stratospheric aerosol injection, ocean iron fertilization, and solar radiation management. Evaluate both the intended benefits and unintended side effects of these technologies, drawing on scientific literature, policy analysis, and environmental ethics. Your analysis should also consider how these interventions might impact vulnerable ecosystems and populations, and whether they reflect a technocratic mindset that overlooks deeper systemic solutions to climate change.

✅ Good Example – Main Ask Alpha:
Analyze the ethical and ecological consequences of climate engineering technologies, with emphasis on their unintended impacts on the environment and society.
Why this is good:
	• Covers core ideas shared by both prompts:
		○ Climate/geoengineering technologies
		○ Ethical concerns
		○ Unintended side effects
		○ Impact on both environment and human society
	• Avoids medium-specific language (not “write a novel” or “research paper”).
	• Includes no ideas not explicitly or implicitly in both.
	• Generalizes the shared ask without omitting essential details.

❌ Bad Example – Main Ask Alpha:
Write a story about how technology successfully restores the planet’s ecosystems after climate change.
Why this is bad:
	• Introduces a positive resolution (successfully restoring ecosystems) not implied or supported by either prompt.
	• Omits the core focus on ethical concerns and unintended consequences.
	• Does not reflect the critical and complex tone of either prompt.
	• Changes the overall stance from questioning/critical to optimistic/celebratory, which breaks alignment.
------------------------------------

Next, find common details shared either directly in or strongly implied by both prompts A and B. Pool these details in a set beta. Do not put details in beta that is the conjunction of two similar but distinct detail in prompts A and B. You should be generalizing the two similar details rather than combining them. Explain why your generated beta satisfy all the criteria.

Using the same prompts A and B as the previous example, here are some example betas:
✅ Good Beta (generalized shared details):
Beta (good):
	• Climate engineering or geoengineering technologies are central.
	• These technologies are used to address climate change.
	• They result in unintended environmental consequences.
	• There are ethical implications of using such technologies.
	• There is human experience or social impact involved.
Why this works:
	• Each detail is either directly stated or strongly implied in both prompts.
	• “Climate engineering” is general enough to cover solar shields (A) and aerosol injection (B).
	• “Used to address climate change” generalizes both motivations.
	• “Unintended environmental consequences” is explicit in both.
	• “Ethical implications” is mentioned in B and implied in A through character’s questioning.
	• “Human/social impact” is shown through the character in A and considered in the analysis in B.
Each point is a generalization, not a combination, and avoids medium-specific framing.

❌ Bad Beta (violates instructions):
Beta (bad):
	• Solar shield technology causes unpredictable weather patterns.
	• Stratospheric aerosol injection has political risks.
	• A character questions their government’s actions.
	• Scientific literature must be used to evaluate outcomes.
Why this fails:
	• These are conjunctions, not generalizations:
		○ “Solar shield” (A) and “aerosol injection” (B) are specific, non-overlapping; combining them violates the rule.
		○ “Weather patterns” (A) and “political risks” (B) are unrelated consequences.
		○ “Character questions government” is only in A, not implied in B.
		○ “Scientific literature” is medium-specific to B only.
	• This beta treats details as an aggregate, rather than distilling shared elements.
------------------------------------

Finally build your prompt C by coherently assembling your details beta around your core main ask alpha. Prompt C should be strongly implied by both prompts A and B. Explain why prompt C is a good generalization of both prompt A and B.

✅ Good Prompt C:
Prompt C:
(start) Write a piece that explores the ethical and ecological consequences of climate engineering technologies used to address climate change, focusing on their unintended impacts on the environment and society. Your response should examine how these technologies affect both ecosystems and human communities. (end)
Why this works:
	• Every element in Prompt C is directly implied or generalized from both A and B.
		○ “Climate engineering… used to address climate change” – from both prompts.
		○ “Ethical and ecological consequences” – explicitly asked in B, strongly implied in A.
		○ “Unintended impacts” – key theme in both.
		○ “Ecosystems and human communities” – combines ecological and social angles, matching both prompts.
	• It’s medium-neutral, so it covers both the novel (A) and the research paper (B).
	• It’s concise (1-2 sentences), as instructed.

❌ Bad Prompt C:
Prompt C:
(start) Write a research paper evaluating both the technical design and political ramifications of solar shield technology, including case studies and citations from recent literature. (end)
Why this fails:
	• This prompt is not implied by Prompt A:
		○ “Research paper” medium only applies to Prompt B.
		○ “Solar shield” is specific to A, not mentioned in B; the detail is not shared.
		○ “Case studies and citations” are required in B but not at all implied in A.
	• It violates generalization: it narrows down to specific form (research paper) and specific tech (solar shield).
	• It loses the shared ethical/social themes, and focuses instead on technical/political details, not a central shared ask.
------------------------------------

Now carry out this task and output a prompt C with the following prompts A & B:
Prompt A: {story1}
Prompt B: {story2}  
        """

        prompt9 = f"""
        Instructions
You will be given prompts A and B. Your task is to generate a prompt C where prompt C is implied by BOTH prompt A and prompt B. This means every detail in prompt C is directly mentioned or implied in BOTH prompt A and B. Begin your prompt C with "(start)" and end it with "(end)."
Follow these steps:

Step 1 – Main Ask Alpha
Find the one sentence in both prompt A and B that respectively represent the main ask of each prompt. Form a common main ask alpha that is a generalization of the two sentences. This should be strongly implied by both.

Step 2 – Shared Details Beta
Next, find common details shared either directly in or strongly implied by both prompts A and B. Pool these details into a set beta. Do not simply conjoin two distinct details; instead, generalize them into a shared element.

Step 3 – Final Prompt C
Finally, build your prompt C by coherently assembling your details beta around your core main ask alpha. Prompt C should be strongly implied by both prompts A and B.

Example 1 (shorter) – Climate Engineering
Prompt A (Creative Writing – Novel Chapter):
Write the first chapter of a speculative fiction novel set in the late 21st century, where climate engineering has become the dominant global response to the escalating climate crisis. Governments and corporations collaborate on ambitious geoengineering projects—such as atmospheric carbon scrubbing, cloud brightening, and orbital solar reflectors—to stabilize Earth’s climate. Your chapter should introduce a protagonist who lives in a region where one of these projects has caused unexpected environmental disruptions, such as altered weather patterns, ecological collapse, or health impacts. Through this character’s daily experience and internal reflections, begin to explore the ethical dilemmas and psychological toll of living in a world reshaped by human technological intervention in nature.
Prompt B (Academic – Research Paper):
Compose a research paper that critically examines the ecological, political, and ethical implications of large-scale geoengineering strategies designed to address global warming. Focus your paper on real or proposed methods such as stratospheric aerosol injection, ocean iron fertilization, and solar radiation management. Evaluate both the intended benefits and unintended side effects of these technologies, drawing on scientific literature, policy analysis, and environmental ethics. Your analysis should also consider how these interventions might impact vulnerable ecosystems and populations, and whether they reflect a technocratic mindset that overlooks deeper systemic solutions to climate change.

✅ Good Alpha
Analyze the ethical and ecological consequences of climate engineering technologies, with emphasis on their unintended impacts on the environment and society.
❌ Bad Alpha
Write a story about how technology successfully restores the planet’s ecosystems after climate change.
Why bad: introduces a happy ending not implied by either; drops the focus on ethics and unintended effects; shifts tone to celebratory instead of critical.

✅ Good Beta
	• Climate engineering or geoengineering technologies are central.
	• These technologies are used to address climate change.
	• They result in unintended environmental consequences.
	• There are ethical implications of using such technologies.
	• There is human experience or social impact involved.
❌ Bad Beta
	• Solar shield technology causes unpredictable weather patterns.
	• Stratospheric aerosol injection has political risks.
	• A character questions their government’s actions.
	• Scientific literature must be used to evaluate outcomes.
Why bad: mixes details that are specific to only one prompt; aggregates rather than generalizes.

✅ Good Prompt C
(start) Write a piece that explores the ethical and ecological consequences of climate engineering technologies used to address climate change, focusing on their unintended impacts on the environment and society. Your response should examine how these technologies affect both ecosystems and human communities. (end)
❌ Bad Prompt C
(start) Write a research paper evaluating both the technical design and political ramifications of solar shield technology, including case studies and citations from recent literature. (end)
Why bad: medium-specific to B, overly narrow (solar shields), excludes shared ethical/social focus.

Example 2 (longer) – Artificial Intelligence & Society
Prompt A (Philosophy – Reflective Essay):
Write a reflective essay on how the rise of artificial intelligence is reshaping human identity and meaning in the 21st century. Discuss how AI alters people’s sense of purpose in work, creativity, and relationships. Explore both hopeful and troubling perspectives: for instance, how automation might free people from routine labor, yet also risk eroding dignity and self-worth tied to human effort. Consider how art and creativity are redefined when machines generate images, music, and writing, raising questions about authenticity and value. Finally, examine how human connection and community may be transformed in a world where interactions are increasingly mediated by intelligent systems.
Prompt B (Sociology – Research Paper):
Compose a sociological research paper analyzing how artificial intelligence is transforming modern social structures. Examine the effects of AI on labor markets, professional roles, and collective understandings of creativity. Analyze how human identity and meaning are negotiated in a time when machines perform tasks once seen as uniquely human. Include both positive and negative possibilities, such as new opportunities for collaboration and innovation, but also risks of alienation and loss of community. Your paper should also address how interpersonal relationships and social bonds adapt to the integration of AI into communication, work, and cultural life.

✅ Good Alpha
Examine how artificial intelligence reshapes human identity, creativity, work, and relationships, considering both positive and negative implications.
❌ Bad Alpha
Write about how artificial intelligence makes society more innovative and collaborative by enhancing creativity and removing the need for human labor.
Why bad: one-sidedly optimistic, drops identity/relationships, and ignores risks.

✅ Good Beta
	• Artificial intelligence is the central technology under discussion.
	• Human identity and meaning are influenced by AI.
	• Work and labor structures are transformed.
	• Creativity and cultural production are impacted.
	• Social and interpersonal relationships are affected.
	• Community and collective bonds are reshaped.
	• Both optimistic opportunities and troubling risks must be considered.
	• These changes provoke reflection on the broader purpose and meaning of human life in a technological era.
❌ Bad Beta
	• AI threatens to eliminate most jobs in creative industries.
	• Human relationships are only strengthened by AI communication tools.
	• Sociological theories must be applied to analyze identity.
	• AI guarantees long-term improvements in human purpose.
Why bad: inserts details only in one prompt (creative job loss, theory requirement), shifts tone to overly positive or overly narrow, fails to generalize.

✅ Good Prompt C
(start) Write a piece that examines how artificial intelligence is transforming human life, with attention to its influence on identity, work, creativity, and relationships. Your response should also address how communities and collective bonds are reshaped, considering both the opportunities AI creates for new forms of collaboration and the risks it poses for alienation and disconnection. Finally, analyze how these developments affect humanity’s broader search for meaning and purpose in an age where machines increasingly share roles once thought uniquely human. (end)
❌ Bad Prompt C
(start) Write a philosophical essay about how AI-generated art challenges authenticity and value in culture, drawing only on creative case studies. (end)
Why bad: focuses on creativity alone (detail from A only), excludes work, identity, and relationships; medium-specific (essay with case studies).

Now carry out this task and output a prompt C with the following prompts A & B:
Prompt A: {story1}
Prompt B: {story2}  
        """
        prompt11 = f"""
        Instructions
You will be given prompts A and B. Your task is to generate a prompt C where prompt C is implied by BOTH prompt A and prompt B. This means every detail in prompt C is directly mentioned in BOTH prompt A and B. Your prompt C should be as concise as possible. Begin your prompt C with "(start)" and end it with "(end)." 
Follow these steps:

Step 1 – Main Ask Alpha
Find the one sentence in both prompt A and B that respectively represent the main ask of each prompt. Form a common main ask alpha that is a generalization of the two sentences. This should be strongly implied by both.

Step 2 – Shared Details Beta
Next, find common details shared directly in by both prompts A and B. Pool these details into a set beta. Do not simply conjoin two distinct details; instead, generalize them into a shared element. The pool should be relatively large if prompt A and B are very similar or relatively small if prompt A and B are very dissimilar. 

Step 3 – Final Prompt C
Finally, build your prompt C by coherently assembling your details beta around your core main ask alpha. Prompt C should be strongly implied by both prompts A and B. The length of prompt C should be proportional to the number of details found in Beta.

Example 1 (shorter) – Climate Engineering
Prompt A (Creative Writing – Novel Chapter):
Write the first chapter of a speculative fiction novel set in the late 21st century, where climate engineering has become the dominant global response to the escalating climate crisis. Governments and corporations collaborate on ambitious geoengineering projects—such as atmospheric carbon scrubbing, cloud brightening, and orbital solar reflectors—to stabilize Earth’s climate. Your chapter should introduce a protagonist who lives in a region where one of these projects has caused unexpected environmental disruptions, such as altered weather patterns, ecological collapse, or health impacts. Through this character’s daily experience and internal reflections, begin to explore the ethical dilemmas and psychological toll of living in a world reshaped by human technological intervention in nature.
Prompt B (Academic – Research Paper):
Compose a research paper that critically examines the ecological, political, and ethical implications of large-scale geoengineering strategies designed to address global warming. Focus your paper on real or proposed methods such as stratospheric aerosol injection, ocean iron fertilization, and solar radiation management. Evaluate both the intended benefits and unintended side effects of these technologies, drawing on scientific literature, policy analysis, and environmental ethics. Your analysis should also consider how these interventions might impact vulnerable ecosystems and populations, and whether they reflect a technocratic mindset that overlooks deeper systemic solutions to climate change.

✅ Good Alpha
Analyze the ethical and ecological consequences of climate engineering technologies, with emphasis on their unintended impacts on the environment and society.
❌ Bad Alpha
Write a story about how technology successfully restores the planet’s ecosystems after climate change.
Why bad: introduces a happy ending not implied by either; drops the focus on ethics and unintended effects; shifts tone to celebratory instead of critical.

✅ Good Beta
	• Climate engineering or geoengineering technologies are central.
	• These technologies are used to address climate change.
	• They result in unintended environmental consequences.
	• There are ethical implications of using such technologies.
	• There is human experience or social impact involved.
❌ Bad Beta
	• Solar shield technology causes unpredictable weather patterns.
	• Stratospheric aerosol injection has political risks.
	• A character questions their government’s actions.
	• Scientific literature must be used to evaluate outcomes.
Why bad: mixes details that are specific to only one prompt; aggregates rather than generalizes.

✅ Good Prompt C
(start) Write a piece that explores the ethical and ecological consequences of climate engineering technologies used to address climate change, focusing on their unintended impacts on the environment and society. Your response should examine how these technologies affect both ecosystems and human communities. (end)
❌ Bad Prompt C
(start) Write a research paper evaluating both the technical design and political ramifications of solar shield technology, including case studies and citations from recent literature. (end)
Why bad: medium-specific to B, overly narrow (solar shields), excludes shared ethical/social focus.

Example 2 (longer) – Artificial Intelligence & Society
Prompt A (Philosophy – Reflective Essay):
Write a reflective essay on how the rise of artificial intelligence is reshaping human identity and meaning in the 21st century. Discuss how AI alters people’s sense of purpose in work, creativity, and relationships. Explore both hopeful and troubling perspectives: for instance, how automation might free people from routine labor, yet also risk eroding dignity and self-worth tied to human effort. Consider how art and creativity are redefined when machines generate images, music, and writing, raising questions about authenticity and value. Finally, examine how human connection and community may be transformed in a world where interactions are increasingly mediated by intelligent systems.
Prompt B (Sociology – Research Paper):
Compose a sociological research paper analyzing how artificial intelligence is transforming modern social structures. Examine the effects of AI on labor markets, professional roles, and collective understandings of creativity. Analyze how human identity and meaning are negotiated in a time when machines perform tasks once seen as uniquely human. Include both positive and negative possibilities, such as new opportunities for collaboration and innovation, but also risks of alienation and loss of community. Your paper should also address how interpersonal relationships and social bonds adapt to the integration of AI into communication, work, and cultural life.

✅ Good Alpha
Examine how artificial intelligence reshapes human identity, creativity, work, and relationships, considering both positive and negative implications.
❌ Bad Alpha
Write about how artificial intelligence makes society more innovative and collaborative by enhancing creativity and removing the need for human labor.
Why bad: one-sidedly optimistic, drops identity/relationships, and ignores risks.

✅ Good Beta
	• Artificial intelligence is the central technology under discussion.
	• Human identity and meaning are influenced by AI.
	• Work and labor structures are transformed.
	• Creativity and cultural production are impacted.
	• Social and interpersonal relationships are affected.
	• Community and collective bonds are reshaped.
	• Both optimistic opportunities and troubling risks must be considered.
	• These changes provoke reflection on the broader purpose and meaning of human life in a technological era.
❌ Bad Beta
	• AI threatens to eliminate most jobs in creative industries.
	• Human relationships are only strengthened by AI communication tools.
	• Sociological theories must be applied to analyze identity.
	• AI guarantees long-term improvements in human purpose.
Why bad: inserts details only in one prompt (creative job loss, theory requirement), shifts tone to overly positive or overly narrow, fails to generalize.

✅ Good Prompt C
(start) Write a piece that examines how artificial intelligence is transforming human life, with attention to its influence on identity, work, creativity, and relationships. Your response should also address how communities and collective bonds are reshaped, considering both the opportunities AI creates for new forms of collaboration and the risks it poses for alienation and disconnection. Finally, analyze how these developments affect humanity’s broader search for meaning and purpose in an age where machines increasingly share roles once thought uniquely human. (end)
❌ Bad Prompt C
(start) Write a philosophical essay about how AI-generated art challenges authenticity and value in culture, drawing only on creative case studies. (end)
Why bad: focuses on creativity alone (detail from A only), excludes work, identity, and relationships; medium-specific (essay with case studies).

Now carry out this task and output a prompt C with the following prompts A & B:
Prompt A: {story1}
Prompt B: {story2} 
        """
        
        prompt12 = f"""
You will be given prompts A and B. Your task is to generate a prompt C where prompt C is implied by BOTH prompt A and prompt B. This means every detail in prompt C is directly mentioned in BOTH prompt A and B. Your prompt C should be as concise as possible with the following caveats: the more similar prompt A and B are too each other, the longer prompt C should be as it would include more details common to both prompt A and B.  Begin your prompt C with "(start)" and end it with "(end)." 


Examples:

1. (Dissimilar – Writing vs Poetry)
Prompt A:
"Write a detailed book review (800–1000 words) of George Orwell’s 1984. Focus on Orwell’s use of language, the historical context of the novel, and its relevance to modern surveillance culture. Please organize the review into an introduction, several analytical sections, and a concluding evaluation."
Prompt B:
"Compose a 12-stanza poem in free verse about a traveler crossing a desert, emphasizing themes of endurance, loneliness, and the harsh beauty of nature. Use vivid imagery and avoid rhyme schemes."
Prompt C:

(start) Write about themes. (end)

2. (Dissimilar – Coding in Different Languages)
Prompt A:
"Can you implement a Python script that scrapes product data from an e-commerce website (like titles, prices, and availability) using BeautifulSoup? Please also save the data into a CSV file with properly labeled columns."
Prompt B:
"I’d like a C++ program that simulates a basic banking system, allowing users to create accounts, deposit money, withdraw money, and view balances. The program should be menu-driven and use object-oriented design."
Prompt C:

(start) Write a program. (end)

3. (Moderately Similar – Writing)
Prompt A:
"Write a 2,000-word research paper about how social media influences political polarization in the United States. Discuss both positive and negative effects, provide examples from the past decade, and include at least five scholarly sources formatted in APA style."
Prompt B:
"Create a detailed argumentative essay about how modern technology (including social media, smartphones, and online forums) affects democracy. Focus on both risks and opportunities, use evidence from real-world examples, and cite at least three academic sources."
Prompt C:

(start) Write an essay on how social media affects politics and democracy, addressing both risks and benefits and including academic sources. (end)

4. (Moderately Similar – Coding)
Prompt A:
"Please create a Python script that takes a CSV file of sales transactions and generates summary statistics, including total revenue, average order value, and number of unique customers. Output the results to the terminal and also save them to a new CSV file."
Prompt B:
"Can you write a Python program that reads data from a JSON file containing customer purchases, calculates metrics like total sales and number of customers, and then produces a summary report saved to a text file?"
Prompt C:

(start) Write a Python script that reads purchase data, calculates total sales and customer counts, and outputs a summary report. (end)

5. (Very Similar – Writing)
Prompt A:
"Draft a 1,200-word persuasive essay arguing why renewable energy should replace fossil fuels as the dominant global energy source. The essay should include an introduction, three body sections (environmental benefits, economic advantages, and long-term sustainability), and a conclusion. Use real-world data and cite at least three credible sources."
Prompt B:
"Please write a well-structured essay, around 1,200 words, making the case for transitioning from fossil fuels to renewable energy. Discuss the environmental necessity, economic opportunities, and sustainable future benefits. Provide at least three reliable citations and organize the essay with intro, body, and conclusion."
Prompt C:

(start) Write a 1,200-word essay on why renewable energy should replace fossil fuels, discussing environmental benefits, economic advantages, and sustainability. Include at least three credible sources, with a structured introduction, body, and conclusion. (end)

6. (Very Similar – Coding)
Prompt A:
"Write a Python program that implements a REST API using Flask. The API should support basic CRUD operations for a task management app, including creating tasks, listing tasks, updating tasks, and deleting tasks. Store tasks in an in-memory dictionary for simplicity, and return JSON responses with appropriate HTTP status codes. Include comments and error handling."
Prompt B:
"I need a Python REST API built with Flask that can handle a simple to-do list. It should let users create tasks, retrieve all tasks, update them, and delete them. Use a dictionary to store tasks (no database needed), respond with JSON, and make sure to handle errors gracefully. Please comment the code clearly."
Prompt C:

(start) Write a Python REST API using Flask for a to-do app with CRUD operations, storing tasks in a dictionary, returning JSON responses, and including error handling and comments. (end)

Now carry out this task and output a prompt C with the following prompts A & B:
Prompt A: {story1}
Prompt B: {story2} 
        """
        prompt13 = f"""
You will be given prompts A and B. Your task is to generate a prompt C where prompt C is implied by BOTH prompt A and prompt B. This means every detail in prompt C is directly mentioned in BOTH prompt A and B. Your prompt C should be as concise as possible with the following caveats: the more similar prompt A and B are too each other, the longer prompt C should be as it would include more details common to both prompt A and B. The more broad prompt A and B are, the more concise and general your prompt C should be.  Begin your prompt C with "(start)" and end it with "(end)." 


Examples:

1. (Dissimilar – Writing vs Poetry)
Prompt A:
"Write a detailed book review (800–1000 words) of George Orwell’s 1984. Focus on Orwell’s use of language, the historical context of the novel, and its relevance to modern surveillance culture. Please organize the review into an introduction, several analytical sections, and a concluding evaluation."
Prompt B:
"Compose a 12-stanza poem in free verse about a traveler crossing a desert, emphasizing themes of endurance, loneliness, and the harsh beauty of nature. Use vivid imagery and avoid rhyme schemes."
Prompt C:

(start) Write about themes. (end)

2. (Dissimilar – Coding in Different Languages)
Prompt A:
"Can you implement a Python script that scrapes product data from an e-commerce website (like titles, prices, and availability) using BeautifulSoup? Please also save the data into a CSV file with properly labeled columns."
Prompt B:
"I’d like a C++ program that simulates a basic banking system, allowing users to create accounts, deposit money, withdraw money, and view balances. The program should be menu-driven and use object-oriented design."
Prompt C:

(start) Write a program. (end)

3. (Moderately Similar – Writing)
Prompt A:
"Write a 2,000-word research paper about how social media influences political polarization in the United States. Discuss both positive and negative effects, provide examples from the past decade, and include at least five scholarly sources formatted in APA style."
Prompt B:
"Create a detailed argumentative essay about how modern technology (including social media, smartphones, and online forums) affects democracy. Focus on both risks and opportunities, use evidence from real-world examples, and cite at least three academic sources."
Prompt C:

(start) Write an essay on how social media affects politics and democracy, addressing both risks and benefits and including academic sources. (end)

4. (Moderately Similar – Coding)
Prompt A:
"Please create a Python script that takes a CSV file of sales transactions and generates summary statistics, including total revenue, average order value, and number of unique customers. Output the results to the terminal and also save them to a new CSV file."
Prompt B:
"Can you write a Python program that reads data from a JSON file containing customer purchases, calculates metrics like total sales and number of customers, and then produces a summary report saved to a text file?"
Prompt C:

(start) Write a Python script that reads purchase data, calculates total sales and customer counts, and outputs a summary report. (end)

5. (Very Similar – Writing)
Prompt A:
"Draft a 1,200-word persuasive essay arguing why renewable energy should replace fossil fuels as the dominant global energy source. The essay should include an introduction, three body sections (environmental benefits, economic advantages, and long-term sustainability), and a conclusion. Use real-world data and cite at least three credible sources."
Prompt B:
"Please write a well-structured essay, around 1,200 words, making the case for transitioning from fossil fuels to renewable energy. Discuss the environmental necessity, economic opportunities, and sustainable future benefits. Provide at least three reliable citations and organize the essay with intro, body, and conclusion."
Prompt C:

(start) Write a 1,200-word essay on why renewable energy should replace fossil fuels, discussing environmental benefits, economic advantages, and sustainability. Include at least three credible sources, with a structured introduction, body, and conclusion. (end)

6. (Very Similar – Coding)
Prompt A:
"Write a Python program that implements a REST API using Flask. The API should support basic CRUD operations for a task management app, including creating tasks, listing tasks, updating tasks, and deleting tasks. Store tasks in an in-memory dictionary for simplicity, and return JSON responses with appropriate HTTP status codes. Include comments and error handling."
Prompt B:
"I need a Python REST API built with Flask that can handle a simple to-do list. It should let users create tasks, retrieve all tasks, update them, and delete them. Use a dictionary to store tasks (no database needed), respond with JSON, and make sure to handle errors gracefully. Please comment the code clearly."
Prompt C:

(start) Write a Python REST API using Flask for a to-do app with CRUD operations, storing tasks in a dictionary, returning JSON responses, and including error handling and comments. (end)

Example 7
Prompt A (generalized)
Write about how different programming paradigms influence the way developers design and structure their code. Discuss the benefits and drawbacks of these approaches and how they affect maintainability and performance.
Prompt B (generalized)
Explain how programming paradigms shape software development. Highlight the trade-offs each paradigm brings and how they influence scalability and readability.
Prompt C (very broad)
(start) Write about how programming paradigms affect software design and development. (end)

Example 8
Prompt A (generalized)
Discuss the impact of artificial intelligence on fields traditionally driven by humans. Include opportunities, risks, and ethical implications.
Prompt B (generalized)
Write about the role of AI in human-centered domains, focusing on how it changes processes, raises ethical questions, and creates new forms of collaboration.
Prompt C (very broad)
(start) Write about the influence of AI on human-driven fields and its ethical implications. (end)

Example 9
Prompt A (generalized)
Explain methods that improve software reliability, focusing on techniques for finding and fixing errors in code.
Prompt B (generalized)
Write about practices that enhance code quality, particularly approaches to identifying problems and ensuring correctness.
Prompt C (very broad)
(start) Write about techniques for improving code quality and reliability. (end)

Example 10
Prompt A (highly abstracted)
Write about how different approaches to problem solving affect outcomes in technology and society.
Prompt B (highly abstracted)
Explain how methodologies shape the way humans build systems and solve challenges.
Prompt C (extremely broad)
(start) Write about how approaches influence outcomes. (end)

Example 11
Prompt A (highly abstracted)
Discuss the role of emerging technologies in reshaping human activity, including benefits and challenges.
Prompt B (highly abstracted)
Write about how innovations impact the way people live and work, considering risks and opportunities.
Prompt C (extremely broad)
(start) Write about how technology influences human activity. (end)

Example 12
Prompt A (highly abstracted)
Explain how humans improve the systems they create to make them more effective and reliable.
Prompt B (highly abstracted)
Write about methods for refining processes to achieve better performance and consistency.
Prompt C (extremely broad)
(start) Write about how humans refine systems to improve them. (end)

Now carry out this task and output a prompt C with the following prompts A & B:
Prompt A: {story1}
Prompt B: {story2} 

        """

        ## Summary
        prompt15 = f"""
You will be given prompts A and B. Your task is to generate a prompt C where prompt C is implied by BOTH prompt A and prompt B. This means every detail in prompt C is directly mentioned in BOTH prompt A and B. Your prompt C should be as concise as possible with the following caveats: the more similar prompt A and B are too each other, the longer prompt C should be as it would include more details common to both prompt A and B. The more broad prompt A and B are, the more concise and general your prompt C should be. If your prompt A and B are so general, instead of summarizing prompt A and B, find a one sentence category that encapsulates the two prompts that includes the medium and the topic the user requests. Begin your prompt C with "(start)" and end it with "(end)."

1. (Dissimilar – Writing vs Poetry)
Prompt A:
"Write a detailed book review (800–1000 words) of George Orwell’s 1984. Focus on Orwell’s use of language, the historical context of the novel, and its relevance to modern surveillance culture. Please organize the review into an introduction, several analytical sections, and a concluding evaluation."
Prompt B:
"Compose a 12-stanza poem in free verse about a traveler crossing a desert, emphasizing themes of endurance, loneliness, and the harsh beauty of nature. Use vivid imagery and avoid rhyme schemes."
Prompt C:

(start) Write about themes. (end)

2. (Dissimilar – Coding in Different Languages)
Prompt A:
"Can you implement a Python script that scrapes product data from an e-commerce website (like titles, prices, and availability) using BeautifulSoup? Please also save the data into a CSV file with properly labeled columns."
Prompt B:
"I’d like a C++ program that simulates a basic banking system, allowing users to create accounts, deposit money, withdraw money, and view balances. The program should be menu-driven and use object-oriented design."
Prompt C:

(start) Write a program. (end)

3. (Moderately Similar – Writing)
Prompt A:
"Write a 2,000-word research paper about how social media influences political polarization in the United States. Discuss both positive and negative effects, provide examples from the past decade, and include at least five scholarly sources formatted in APA style."
Prompt B:
"Create a detailed argumentative essay about how modern technology (including social media, smartphones, and online forums) affects democracy. Focus on both risks and opportunities, use evidence from real-world examples, and cite at least three academic sources."
Prompt C:

(start) Write an essay on how social media affects politics and democracy, addressing both risks and benefits and including academic sources. (end)

4. (Moderately Similar – Coding)
Prompt A:
"Please create a Python script that takes a CSV file of sales transactions and generates summary statistics, including total revenue, average order value, and number of unique customers. Output the results to the terminal and also save them to a new CSV file."
Prompt B:
"Can you write a Python program that reads data from a JSON file containing customer purchases, calculates metrics like total sales and number of customers, and then produces a summary report saved to a text file?"
Prompt C:

(start) Write a Python script that reads purchase data, calculates total sales and customer counts, and outputs a summary report. (end)

5. (Very Similar – Writing)
Prompt A:
"Draft a 1,200-word persuasive essay arguing why renewable energy should replace fossil fuels as the dominant global energy source. The essay should include an introduction, three body sections (environmental benefits, economic advantages, and long-term sustainability), and a conclusion. Use real-world data and cite at least three credible sources."
Prompt B:
"Please write a well-structured essay, around 1,200 words, making the case for transitioning from fossil fuels to renewable energy. Discuss the environmental necessity, economic opportunities, and sustainable future benefits. Provide at least three reliable citations and organize the essay with intro, body, and conclusion."
Prompt C:

(start) Write a 1,200-word essay on why renewable energy should replace fossil fuels, discussing environmental benefits, economic advantages, and sustainability. Include at least three credible sources, with a structured introduction, body, and conclusion. (end)

6. (Very Similar – Coding)
Prompt A:
"Write a Python program that implements a REST API using Flask. The API should support basic CRUD operations for a task management app, including creating tasks, listing tasks, updating tasks, and deleting tasks. Store tasks in an in-memory dictionary for simplicity, and return JSON responses with appropriate HTTP status codes. Include comments and error handling."
Prompt B:
"I need a Python REST API built with Flask that can handle a simple to-do list. It should let users create tasks, retrieve all tasks, update them, and delete them. Use a dictionary to store tasks (no database needed), respond with JSON, and make sure to handle errors gracefully. Please comment the code clearly."
Prompt C:

(start) Write a Python REST API using Flask for a to-do app with CRUD operations, storing tasks in a dictionary, returning JSON responses, and including error handling and comments. (end)

Example 7
Prompt A (generalized)
Write about how different programming paradigms influence the way developers design and structure their code. Discuss the benefits and drawbacks of these approaches and how they affect maintainability and performance.
Prompt B (generalized)
Explain how programming paradigms shape software development. Highlight the trade-offs each paradigm brings and how they influence scalability and readability.
Prompt C (very broad)
(start) Write about how programming paradigms affect software design and development. (end)

Example 8
Prompt A (generalized)
Discuss the impact of artificial intelligence on fields traditionally driven by humans. Include opportunities, risks, and ethical implications.
Prompt B (generalized)
Write about the role of AI in human-centered domains, focusing on how it changes processes, raises ethical questions, and creates new forms of collaboration.
Prompt C (very broad)
(start) Write about the influence of AI on human-driven fields and its ethical implications. (end)

Example 9
Prompt A (generalized)
Explain methods that improve software reliability, focusing on techniques for finding and fixing errors in code.
Prompt B (generalized)
Write about practices that enhance code quality, particularly approaches to identifying problems and ensuring correctness.
Prompt C (very broad)
(start) Write about techniques for improving code quality and reliability. (end)

Example 10
Prompt A (highly abstracted)
Write about how different approaches to problem solving affect outcomes in technology and society.
Prompt B (highly abstracted)
Explain how methodologies shape the way humans build systems and solve challenges.
Prompt C (extremely broad)
(start) Write about how approaches influence outcomes. (end)

Example 11
Prompt A (highly abstracted)
Discuss the role of emerging technologies in reshaping human activity, including benefits and challenges.
Prompt B (highly abstracted)
Write about how innovations impact the way people live and work, considering risks and opportunities.
Prompt C (extremely broad)
(start) Write about how technology influences human activity. (end)

Example 12
Prompt A (highly abstracted)
Explain how humans improve the systems they create to make them more effective and reliable.
Prompt B (highly abstracted)
Write about methods for refining processes to achieve better performance and consistency.
Prompt C (extremely broad)
(start) Write about how humans refine systems to improve them. (end)

Example 13
Prompt A:
Write a poem that reflects on natural landscapes, using imagery and metaphor to highlight the relationship between humans and the environment.
Prompt B:
Compose a poem exploring the power of nature, focusing on themes of transformation, resilience, and human connection to the earth.
Prompt C:
(start)Poem on Nature(end)

Example 14
Prompt A:
Write an essay examining how revolutions throughout history have transformed societies, paying attention to causes, consequences, and cultural shifts.
Prompt B:
Compose a structured essay analyzing historical uprisings, their impact on leadership, governance, and social identity.
Prompt C:
(start)Essay on Revolutions(end)

Example 15
Prompt A:
Write a program that processes user input, applies logical conditions, and outputs meaningful results, ensuring clear structure and readability.
Prompt B:
Compose code that implements basic algorithms with input handling, decision-making, and structured output, emphasizing clarity and functionality.
Prompt C:
(start)Code on Input Processing(end)

Now carry out this task and output a prompt C with the following prompts A & B:
Prompt A: {story1}
Prompt B: {story2} 

        """

## Education
        prompt14 = f"""
You will be given two science exercise descriptions, A and B. Your task is to generate a common skill description C that represents the underlying scientific skill required by BOTH exercise A and exercise B. The goal is to identify the most specific scientific skill that is genuinely shared by the two exercises. Skill C should describe what a learner must know or be able to do in order to successfully solve or complete BOTH exercises. The shared skill may be explicitly stated in the exercises or implicitly required by them. You should infer the underlying scientific reasoning, concept, principle, law, formula, method, or problem-solving procedure when appropriate. Skill C represents the INTERSECTION of the skills required by exercise A and exercise B, NOT their union. It should contain only information that is genuinely applicable to BOTH exercises. Your skill C should be as concise as possible with the following caveats:

• The more similar the underlying skills required by exercise A and B are, the more specific skill C should be, preserving relevant shared scientific concepts, principles, laws, formulas, or procedures.
• The more broad the shared skill between exercise A and B is, the more concise and general skill C should be.
• Do not make skill C artificially longer simply because the exercises contain similar wording.
• If exercise A and B are very different but share a meaningful higher-level scientific reasoning process, describe that shared skill at the appropriate level of abstraction.
• If exercise A and B are so broad that no more specific common skill can be identified, generate a concise higher-level scientific skill category that describes the shared scientific ability or methodology rather than merely naming a scientific discipline.

IMPORTANT RULES:
1. Skill C MUST apply to BOTH exercise A and exercise B.
2. Skill C should represent the INTERSECTION of A and B, NOT the UNION. Do not combine scientific concepts, procedures, or requirements that are specific to only one exercise.
3. Skill C should describe an underlying LEARNER CAPABILITY rather than merely describing the topic of the exercises.
4. Skill C may contain a scientific concept, principle, law, formula, or procedure even if it is not explicitly named in the exercises, provided that the underlying concept or procedure is genuinely required by BOTH exercises.
5. Prefer the underlying scientific reasoning or problem-solving process over surface-level word overlap.
6. Skill C should be as SPECIFIC as possible while still applying to BOTH exercises.
7. Do not include exercise-specific numerical values, object names, substances, organisms, experimental settings, or other details that are not shared by both exercises.
8. Do not solve either exercise. Do not provide numerical answers or detailed solution steps.
9. Do not simply name a scientific discipline or broad topic such as "physics", "chemistry", "biology", "mechanics", or "thermodynamics" when a more meaningful skill can be identified.
10. Do not generate an extremely broad skill such as "solve scientific problems", "apply scientific knowledge", "perform calculations", or "analyze information" when a more specific shared skill can be identified.
11. Do not invent a relationship between the exercises. If they have only a weak or broad relationship, identify the narrowest genuinely shared scientific skill.
12. Skill C should be written as an action-oriented capability whenever possible. Prefer formulations such as:
* "Apply Newton's laws to analyze forces acting on a physical system."
* "Use conservation of energy to determine changes in a physical system."
* "Apply stoichiometric relationships to determine quantities of reactants and products."
* "Interpret experimental data to determine relationships between scientific variables."
    rather than:
* "Newton's laws"
* "Energy"
* "Stoichiometry"
* "Experimental data"
13. When exercise A and B share a specific scientific law, formula, concept, or reasoning procedure, preserve that specificity in skill C.
14. When exercise A and B are only broadly related, move to a higher level of abstraction, but retain a meaningful scientific capability.
15. The hierarchy should become progressively more general at higher levels. Therefore, a common skill generated from two highly specific skills should preserve their shared specificity, while a common skill generated from already-general skills may need to be broader.

Here are examples:

Example 1 — Very Similar Physics Exercises
Exercise A:
"Calculate the acceleration of an object when its mass and the net force acting on it are given."
Exercise B:
"Determine the net force acting on an object when its mass and acceleration are known."
Common Skill C:

(start) Apply Newton's second law to relate force, mass, and acceleration. (end)

Example 2 — Similar Physics Exercises
Exercise A:
"A ball is thrown vertically upward with a given initial velocity. Determine its velocity after a specified time."
Exercise B:
"An object is dropped from a height. Calculate its position after a given amount of time using its initial conditions."
Common Skill C:

(start) Apply kinematic equations to determine an object's motion from its initial conditions and time. (end)

Example 3 — Similar Chemistry Exercises
Exercise A:
"Calculate the amount of product formed from a given amount of reactant using a balanced chemical equation."
Exercise B:
"Determine the amount of reactant required to produce a specified quantity of product using the balanced reaction equation."
Common Skill C:

(start) Use stoichiometric relationships from balanced chemical equations to relate quantities of reactants and products. (end)

Example 4 — Similar Biology Exercises
Exercise A:
"Predict the possible phenotypes of offspring given the genotypes of two parents."
Exercise B:
"Determine the possible genotypes of offspring resulting from a genetic cross between two parents."
Common Skill C:

(start) Apply Mendelian inheritance principles to predict genetic outcomes from parental genotypes. (end)

Example 5 — Moderately Similar Physics Exercises
Exercise A:
"Calculate the acceleration of a block on an inclined plane by considering the forces acting on the block."
Exercise B:
"Determine the tension in a rope supporting an object by analyzing the forces acting on the system."
Common Skill C:

(start) Apply Newton's laws to analyze forces acting on a physical system. (end)

Example 6 — Shared Experimental Analysis
Exercise A:
"Analyze experimental measurements of temperature and pressure to determine how the two variables are related."
Exercise B:
"Examine experimental data showing changes in concentration and reaction rate to determine their relationship."
Common Skill C:

(start) Analyze experimental data to determine relationships between measured scientific variables. (end)

Example 7 — Shared Quantitative Calculation
Exercise A:
"Calculate the acceleration of an object using its mass and the net force acting on it."
Exercise B:
"Determine the concentration of a solution from the amount of solute and volume of the solution."
Common Skill C:

(start) Use quantitative relationships to calculate an unknown scientific quantity from known variables. (end)

Example 8 — Shared Variable-Effect Analysis
Exercise A:
"Use observations of an ecosystem to determine how changes in one population affect another population."
Exercise B:
"Analyze experimental measurements to determine how changing temperature affects the rate of a chemical reaction."
Common Skill C:

(start) Analyze evidence to determine how changes in one variable affect another variable. (end)

Example 9 — Broad Science Exercises
Exercise A:
"Explain why increasing temperature changes the rate of a chemical reaction."
Exercise B:
"Explain how increasing temperature affects the pressure of a gas."
Common Skill C:

(start) Explain how increasing temperature affects physical or chemical properties. (end)


Example 10 — Shared Scientific Explanation
Exercise A:
"Use experimental observations to support a conclusion about a scientific phenomenon."
Exercise B:
"Analyze measurements and evidence to determine an explanation for an observed phenomenon."
Common Skill C:

(start) Use scientific evidence to explain observed phenomena and draw conclusions. (end)

Example 11 — Shared Quantitative Reasoning
Exercise A:
"Calculate the density of a material from its mass and volume."
Exercise B:
"Calculate the pressure of a gas from its force and area."
Common Skill C:

(start) Use quantitative relationships to calculate scientific quantities from known variables. (end)

Example 12 — Shared Graph Analysis
Exercise A:
"Interpret a graph showing population growth over time and identify the trend."
Exercise B:
"Analyze a graph showing temperature changes over time and identify the trend."
Common Skill C:

(start) Interpret scientific graphs to identify trends in a variable over time. (end)

Example 13 — Very Similar Scientific Procedure
Exercise A:
"Determine the pH of a solution from its hydrogen ion concentration."
Exercise B:
"Calculate the hydrogen ion concentration of a solution from its pH."
Common Skill C:

(start) Use the logarithmic relationship between pH and hydrogen ion concentration to determine one from the other. (end)

Example 14 — Shared Newtonian Reasoning
Exercise A:
"Explain why an object continues moving at constant velocity when no net force acts on it."
Exercise B:
"Explain why an object changes its motion when an unbalanced force acts on it."
Common Skill C:

(start) Apply Newton's laws to explain how net force affects an object's motion. (end)

Example 15 — Shared Experimental Design
Exercise A:
"Design an experiment to determine how light intensity affects plant growth."
Exercise B:
"Design an experiment to determine how temperature affects bacterial growth."
Common Skill C:

(start) Design controlled experiments to investigate how an independent variable affects a measurable outcome. (end)

Now carry out this task and output a common skill C with the following exercises A and B:
Exercise A:
"{story1}"
Exercise B:
"{story2}"
Return ONLY the common skill description.
Begin your response with "(start)" and end it with "(end)."
            """


        # time.sleep(5)
        # chat_response = client.chat.completions.create(
        #     model="gemma-3-finetuned-merged-bf16",
        #     max_tokens=1024,
        #     messages=[
        #         {"role": "system", "content": "You are a professional editor who writes quality summary that generalizes two prompts A and B. No matter the content of prompts A & B, you must write a summary."},

        #         {"role": "user", "content": prompt14},
        #     ]
        # )
        # # print("Chat response:", chat_response)
        # raw = chat_response.choices[0].message.content
        # print("Chat response: ", raw)

        # time.sleep(5)
        # chat_response = client.chat.completions.create(
        #     model = "gpt-4o-mini",
        #     messages = [
        #         {
        #             "role": "system",
        #             "content": "You are a professional editor who writes quality summary that generalizes two prompts A and B. No matter the content of prompts A & B, you must write a summary."
        #         },
        #         {
        #             "role": "user",
        #             "content": prompt14
        #         }
        #     ]
        # )
        # print("Chat response:", chat_response)
        # raw = chat_response.choices[0].message.content
        # print("Chat response: ", raw)


        ## Summary
        # system_prompt = "You are a professional editor who writes quality summary that generalizes two prompts A and B. No matter the content of prompts A & B, you must write a summary."

        ## Education
        system_prompt = "You are a scientific skill taxonomy expert who identifies and describes the underlying scientific skills shared by two science exercises. Given exercises A and B, your task is to generate a concise common skill that generalizes the underlying scientific knowledge, reasoning, concepts, principles, laws, formulas, or procedures required by BOTH exercises. The common skill should represent the intersection of the skills required by A and B, not their union. Infer latent or implicit skills when they are genuinely required by both exercises, while avoiding exercise-specific details and unsupported relationships. The skill should be as specific as possible while remaining applicable to both exercises, and should be expressed as an action-oriented learner capability rather than merely a topic, scientific discipline, or summary of the exercises. No matter the content of exercises A and B, you must identify and write a common scientific skill."

        
        
        if model_name == "openai":
            client = OpenAI(
                api_key=api_keys["openai"]
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt14}
                ]
            )
            raw = response.choices[0].message.content.strip()
        elif model_name == "claude":
            client = anthropic.Anthropic(
                api_key=api_keys["claude"]
            )
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt14}
                ]
            )
            raw = message.content[0].text.strip()
        elif model_name == "gemma":
            client = OpenAI(
                api_key=api_keys["gemma"]["api_key"],
                base_url=api_keys["gemma"]["base_url"]
            )
            response = client.chat.completions.create(
                model=api_keys["gemma"].get("model", "gemma-3-finetuned-merged-bf16"),
                max_tokens=1024,
                stop=["<end_of_turn>"],  # IMPORTANT: never stop on "(end)" -- the
                                          # extraction regex needs it in the output
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt14}
                ]
            )
            raw = response.choices[0].message.content.strip()
        elif model_name == "gemini":
            client = OpenAI(
                api_key=api_keys["gemini"]["api_key"],
                base_url=api_keys["gemini"]["base_url"],
            )
            response = client.chat.completions.create(
                model="gemini-3.1-flash-lite",
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt14},
                ],
            )
            raw = response.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unknown model_name: {model_name}")



        
        # keywords = ['i apologize']
        
        # if not any((keyword in raw.lower() for keyword in keywords)):

            
        match = re.search(r"\(start\)(.*?)\(end\)", raw, re.I | re.S)
        summary = match.group(1).strip() if match else raw.strip()
        node.summary = summary
        # also propagate to partner, but only if partner is missing a summary
        # if partner_node and not partner_node.summary:
        #     partner_node.summary = summary
        if partner_node:
            partner_node.summary = summary
        summarized.add(idx)
        summarized.add(partner_idx)

    # tree.dump("similar_64_sonnet_sumv5_4o-mini_evalv1.json")         # keep JSON current
    # print(f"Succesfully populated Level L{level} in the excel sheet")
    level+=1

def get_entail_2_score(tree: Tree, level_idx: int):
    level = tree.levels[level_idx]
    checked = set()
    entail_2s = list()
    for node in level.nodes:
        if node.id not in checked and node.entail_2:
            entail_2s.append(node.entail_2)
            checked.add(node.id)
            checked.add(node.match)
    return entail_2s

