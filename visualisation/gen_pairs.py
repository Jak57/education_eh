import json

def create_training_data(input_json_path, output_jsonl_path):
    system_prompt = "You are a professional editor who writes quality summary that generalizes two prompts A and B. No matter the content of prompts A & B, you must write a summary."

    user_prompt_template = """
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

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_json_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_json_path}'")
        return

    id_to_prompt = {prompt['id']: prompt for prompt in prompts}

    parent_to_children = {}
    for prompt in prompts:
        if prompt.get('parentIds'):
            # Assumption: one parent ID per prompt
            parent_id = prompt['parentIds'][0]
            if parent_id not in parent_to_children:
                parent_to_children[parent_id] = []
            parent_to_children[parent_id].append(prompt)


    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        generated_count = 0
        for parent_id, children in parent_to_children.items():
            if len(children) == 2:
                parent_prompt_obj = id_to_prompt.get(parent_id)
                if not parent_prompt_obj:
                    print(f"Warning: Parent with ID '{parent_id}' not found. Skipping.")
                    continue

                prompt_a = children[0]
                prompt_b = children[1]

                story1 = prompt_a.get('full_text','couldnt extract')
                story2 = prompt_b.get('full_text','couldnt extract')

                assistant_response = parent_prompt_obj.get('full_text', 'couldnt extract')

                user_content = user_prompt_template.format(story1=story1, story2=story2)

                line_data = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_response}
                    ]
                }

                f_out.write(json.dumps(line_data) + '\n')
                generated_count += 1
    
    print(f"Successfully generated {generated_count} examples in '{output_jsonl_path}'")


if __name__ == '__main__':

    input_file = 'tree_data_64s_claude_gemma.json'

    output_file = 'training_data_64s_claude_distill.jsonl'

    create_training_data(input_file, output_file)