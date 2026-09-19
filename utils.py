import pandas as pd
import json

def load_xlsx(path):
    df = pd.read_excel(path)
    return df

def load_csv(path):
    df = pd.read_csv(path)
    return df

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_to_json(path, data):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"File saved at {path}")

## ------------------------------------------------- Prompt ------------------------------------------------- ##

def get_system_prompt():
    # system_prompt = "You are a scientific skill taxonomy expert who identifies and describes the underlying scientific skills shared by two science exercises. Given exercises A and B, your task is to generate a concise common skill that generalizes the underlying scientific knowledge, reasoning, concepts, principles, laws, formulas, or procedures required by BOTH exercises. The common skill should represent the intersection of the skills required by A and B, not their union. Infer latent or implicit skills when they are genuinely required by both exercises, while avoiding exercise-specific details and unsupported relationships. The skill should be as specific as possible while remaining applicable to both exercises, and should be expressed as an action-oriented learner capability rather than merely a topic, scientific discipline, or summary of the exercises. No matter the content of exercises A and B, you must identify and write a common scientific skill."
    system_prompt = """
    You are a Java programming skill taxonomy expert who identifies and describes the underlying programming skills and knowledge components shared by two Java programming exercises. Given exercises A and B, your task is to generate a concise common skill that generalizes the underlying Java programming knowledge, computational reasoning, concepts, algorithms, data structures, language constructs, or procedures required by BOTH exercises. The common skill should represent the intersection of the skills required by A and B, not their union. Infer latent or implicit programming skills when they are genuinely required by both exercises, while avoiding exercise-specific details and unsupported relationships. The skill should be as specific as possible while remaining applicable to both exercises, and should be expressed as an action-oriented learner capability rather than merely a programming topic, Java feature, API, programming language construct, or summary of the exercises. The skill should represent a transferable knowledge component that can apply to other Java programming exercises, rather than merely restating the specific exercises. No matter the content of exercises A and B, you must identify and write a common Java programming skill.
    """

    return system_prompt

def get_common_summary_prompt(story1, story2):
    # prompt14 = f"""
    # You will be given two science exercise descriptions, A and B. Your task is to generate a common skill description C that represents the underlying scientific skill required by BOTH exercise A and exercise B. The goal is to identify the most specific scientific skill that is genuinely shared by the two exercises. Skill C should describe what a learner must know or be able to do in order to successfully solve or complete BOTH exercises. The shared skill may be explicitly stated in the exercises or implicitly required by them. You should infer the underlying scientific reasoning, concept, principle, law, formula, method, or problem-solving procedure when appropriate. Skill C represents the INTERSECTION of the skills required by exercise A and exercise B, NOT their union. It should contain only information that is genuinely applicable to BOTH exercises. Your skill C should be as concise as possible with the following caveats:

    # • The more similar the underlying skills required by exercise A and B are, the more specific skill C should be, preserving relevant shared scientific concepts, principles, laws, formulas, or procedures.
    # • The more broad the shared skill between exercise A and B is, the more concise and general skill C should be.
    # • Do not make skill C artificially longer simply because the exercises contain similar wording.
    # • If exercise A and B are very different but share a meaningful higher-level scientific reasoning process, describe that shared skill at the appropriate level of abstraction.
    # • If exercise A and B are so broad that no more specific common skill can be identified, generate a concise higher-level scientific skill category that describes the shared scientific ability or methodology rather than merely naming a scientific discipline.

    # IMPORTANT RULES:
    # 1. Skill C MUST apply to BOTH exercise A and exercise B.
    # 2. Skill C should represent the INTERSECTION of A and B, NOT the UNION. Do not combine scientific concepts, procedures, or requirements that are specific to only one exercise.
    # 3. Skill C should describe an underlying LEARNER CAPABILITY rather than merely describing the topic of the exercises.
    # 4. Skill C may contain a scientific concept, principle, law, formula, or procedure even if it is not explicitly named in the exercises, provided that the underlying concept or procedure is genuinely required by BOTH exercises.
    # 5. Prefer the underlying scientific reasoning or problem-solving process over surface-level word overlap.
    # 6. Skill C should be as SPECIFIC as possible while still applying to BOTH exercises.
    # 7. Do not include exercise-specific numerical values, object names, substances, organisms, experimental settings, or other details that are not shared by both exercises.
    # 8. Do not solve either exercise. Do not provide numerical answers or detailed solution steps.
    # 9. Do not simply name a scientific discipline or broad topic such as "physics", "chemistry", "biology", "mechanics", or "thermodynamics" when a more meaningful skill can be identified.
    # 10. Do not generate an extremely broad skill such as "solve scientific problems", "apply scientific knowledge", "perform calculations", or "analyze information" when a more specific shared skill can be identified.
    # 11. Do not invent a relationship between the exercises. If they have only a weak or broad relationship, identify the narrowest genuinely shared scientific skill.
    # 12. Skill C should be written as an action-oriented capability whenever possible. Prefer formulations such as:
    # * "Apply Newton's laws to analyze forces acting on a physical system."
    # * "Use conservation of energy to determine changes in a physical system."
    # * "Apply stoichiometric relationships to determine quantities of reactants and products."
    # * "Interpret experimental data to determine relationships between scientific variables."
    #     rather than:
    # * "Newton's laws"
    # * "Energy"
    # * "Stoichiometry"
    # * "Experimental data"
    # 13. When exercise A and B share a specific scientific law, formula, concept, or reasoning procedure, preserve that specificity in skill C.
    # 14. When exercise A and B are only broadly related, move to a higher level of abstraction, but retain a meaningful scientific capability.
    # 15. The hierarchy should become progressively more general at higher levels. Therefore, a common skill generated from two highly specific skills should preserve their shared specificity, while a common skill generated from already-general skills may need to be broader.

    # Here are examples:

    # Example 1 — Very Similar Physics Exercises
    # Exercise A:
    # "Calculate the acceleration of an object when its mass and the net force acting on it are given."
    # Exercise B:
    # "Determine the net force acting on an object when its mass and acceleration are known."
    # Common Skill C:

    # (start) Apply Newton's second law to relate force, mass, and acceleration. (end)

    # Example 2 — Similar Physics Exercises
    # Exercise A:
    # "A ball is thrown vertically upward with a given initial velocity. Determine its velocity after a specified time."
    # Exercise B:
    # "An object is dropped from a height. Calculate its position after a given amount of time using its initial conditions."
    # Common Skill C:

    # (start) Apply kinematic equations to determine an object's motion from its initial conditions and time. (end)

    # Example 3 — Similar Chemistry Exercises
    # Exercise A:
    # "Calculate the amount of product formed from a given amount of reactant using a balanced chemical equation."
    # Exercise B:
    # "Determine the amount of reactant required to produce a specified quantity of product using the balanced reaction equation."
    # Common Skill C:

    # (start) Use stoichiometric relationships from balanced chemical equations to relate quantities of reactants and products. (end)

    # Example 4 — Similar Biology Exercises
    # Exercise A:
    # "Predict the possible phenotypes of offspring given the genotypes of two parents."
    # Exercise B:
    # "Determine the possible genotypes of offspring resulting from a genetic cross between two parents."
    # Common Skill C:

    # (start) Apply Mendelian inheritance principles to predict genetic outcomes from parental genotypes. (end)

    # Example 5 — Moderately Similar Physics Exercises
    # Exercise A:
    # "Calculate the acceleration of a block on an inclined plane by considering the forces acting on the block."
    # Exercise B:
    # "Determine the tension in a rope supporting an object by analyzing the forces acting on the system."
    # Common Skill C:

    # (start) Apply Newton's laws to analyze forces acting on a physical system. (end)

    # Example 6 — Shared Experimental Analysis
    # Exercise A:
    # "Analyze experimental measurements of temperature and pressure to determine how the two variables are related."
    # Exercise B:
    # "Examine experimental data showing changes in concentration and reaction rate to determine their relationship."
    # Common Skill C:

    # (start) Analyze experimental data to determine relationships between measured scientific variables. (end)

    # Example 7 — Shared Quantitative Calculation
    # Exercise A:
    # "Calculate the acceleration of an object using its mass and the net force acting on it."
    # Exercise B:
    # "Determine the concentration of a solution from the amount of solute and volume of the solution."
    # Common Skill C:

    # (start) Use quantitative relationships to calculate an unknown scientific quantity from known variables. (end)

    # Example 8 — Shared Variable-Effect Analysis
    # Exercise A:
    # "Use observations of an ecosystem to determine how changes in one population affect another population."
    # Exercise B:
    # "Analyze experimental measurements to determine how changing temperature affects the rate of a chemical reaction."
    # Common Skill C:

    # (start) Analyze evidence to determine how changes in one variable affect another variable. (end)

    # Example 9 — Broad Science Exercises
    # Exercise A:
    # "Explain why increasing temperature changes the rate of a chemical reaction."
    # Exercise B:
    # "Explain how increasing temperature affects the pressure of a gas."
    # Common Skill C:

    # (start) Explain how increasing temperature affects physical or chemical properties. (end)


    # Example 10 — Shared Scientific Explanation
    # Exercise A:
    # "Use experimental observations to support a conclusion about a scientific phenomenon."
    # Exercise B:
    # "Analyze measurements and evidence to determine an explanation for an observed phenomenon."
    # Common Skill C:

    # (start) Use scientific evidence to explain observed phenomena and draw conclusions. (end)

    # Example 11 — Shared Quantitative Reasoning
    # Exercise A:
    # "Calculate the density of a material from its mass and volume."
    # Exercise B:
    # "Calculate the pressure of a gas from its force and area."
    # Common Skill C:

    # (start) Use quantitative relationships to calculate scientific quantities from known variables. (end)

    # Example 12 — Shared Graph Analysis
    # Exercise A:
    # "Interpret a graph showing population growth over time and identify the trend."
    # Exercise B:
    # "Analyze a graph showing temperature changes over time and identify the trend."
    # Common Skill C:

    # (start) Interpret scientific graphs to identify trends in a variable over time. (end)

    # Example 13 — Very Similar Scientific Procedure
    # Exercise A:
    # "Determine the pH of a solution from its hydrogen ion concentration."
    # Exercise B:
    # "Calculate the hydrogen ion concentration of a solution from its pH."
    # Common Skill C:

    # (start) Use the logarithmic relationship between pH and hydrogen ion concentration to determine one from the other. (end)

    # Example 14 — Shared Newtonian Reasoning
    # Exercise A:
    # "Explain why an object continues moving at constant velocity when no net force acts on it."
    # Exercise B:
    # "Explain why an object changes its motion when an unbalanced force acts on it."
    # Common Skill C:

    # (start) Apply Newton's laws to explain how net force affects an object's motion. (end)

    # Example 15 — Shared Experimental Design
    # Exercise A:
    # "Design an experiment to determine how light intensity affects plant growth."
    # Exercise B:
    # "Design an experiment to determine how temperature affects bacterial growth."
    # Common Skill C:

    # (start) Design controlled experiments to investigate how an independent variable affects a measurable outcome. (end)

    # Now carry out this task and output a common skill C with the following exercises A and B:
    # Exercise A:
    # "{story1}"
    # Exercise B:
    # "{story2}"
    # Return ONLY the common skill description.
    # Begin your response with "(start)" and end it with "(end)."
    # """

    prompt14 = f"""
You will be given two Java programming exercise descriptions, A and B. Your task is to generate a common skill or knowledge component (KC) description C that represents the underlying programming skill or knowledge component required by BOTH exercise A and exercise B. The goal is to identify the most specific programming skill or KC that is genuinely shared by the two exercises. Skill C should describe what a learner must know or be able to do in order to successfully solve or complete BOTH exercises. The shared skill may be explicitly stated in the exercises or implicitly required by them. You should infer the underlying programming concepts, algorithms, data structures, control-flow patterns, Java language constructs, computational reasoning, or problem-solving procedures when appropriate. Skill C represents the INTERSECTION of the skills required by exercise A and exercise B, NOT their union. It should contain only information that is genuinely applicable to BOTH exercises. Your skill C should be as concise as possible with the following caveats:

• The more similar the underlying programming skills required by exercise A and B are, the more specific skill C should be, preserving relevant shared programming concepts, algorithms, data structures, Java constructs, or procedures.
• The more broad the shared skill between exercise A and B is, the more concise and general skill C should be.
• Do not make skill C artificially longer simply because the exercises contain similar wording.
• If exercise A and B are very different but share a meaningful higher-level programming reasoning or problem-solving process, describe that shared skill at the appropriate level of abstraction.
• If exercise A and B are so broad that no more specific common skill can be identified, generate a concise higher-level programming skill or knowledge component that describes the shared programming ability or methodology rather than merely naming a programming topic, Java feature, or programming language construct.

IMPORTANT RULES:

1. Skill C MUST apply to BOTH exercise A and exercise B.
2. Skill C should represent the INTERSECTION of A and B, NOT the UNION. Do not combine programming concepts, algorithms, data structures, Java features, or requirements that are specific to only one exercise.
3. Skill C should describe an underlying LEARNER CAPABILITY or KNOWLEDGE COMPONENT rather than merely describing the topic of the exercises.
4. Skill C may contain a programming concept, algorithm, data structure, Java language construct, or problem-solving procedure even if it is not explicitly named in the exercises, provided that the underlying concept or procedure is genuinely required by BOTH exercises.
5. Prefer the underlying programming reasoning or problem-solving process over surface-level word overlap.
6. Skill C should be as SPECIFIC as possible while still applying to BOTH exercises.
7. Do not include exercise-specific variable names, values, class names, method names, input examples, output examples, or other implementation details that are not shared by both exercises.
8. Do not solve either exercise. Do not provide code, outputs, or detailed solution steps.
9. Do not simply name a programming topic or Java feature such as "Java", "arrays", "loops", "inheritance", "classes", "objects", or "recursion" when a more meaningful learner capability can be identified.
10. Do not generate an extremely broad skill such as "solve programming problems", "write Java programs", "apply programming knowledge", "perform calculations", or "analyze data" when a more specific shared skill can be identified.
11. Do not invent a relationship between the exercises. If they have only a weak or broad relationship, identify the narrowest genuinely shared programming skill or KC.
12. Skill C should be written as an action-oriented capability whenever possible. Prefer formulations such as:
    * "Traverse an array while maintaining a running maximum."
    * "Use recursion to decompose a problem into smaller instances of the same problem."
    * "Use a mapping structure to count the frequency of elements."
    * "Implement a Java interface by providing concrete method implementations."
    * "Iterate through a collection while maintaining an accumulated result."
    rather than:
    * "Arrays"
    * "Recursion"
    * "HashMap"
    * "Java interfaces"
    * "Loops"
13. When exercise A and B share a specific programming algorithm, data structure, Java construct, or reasoning procedure, preserve that specificity in skill C.
14. When exercise A and B are only broadly related, move to a higher level of abstraction, but retain a meaningful programming capability or knowledge component.
15. The hierarchy should become progressively more general at higher levels. Therefore, a common skill generated from two highly specific programming skills should preserve their shared specificity, while a common skill generated from already-general skills may need to be broader.
16. Distinguish between a shared programming TOPIC and a shared programming SKILL. Merely using the same programming construct does not necessarily mean that the construct itself is the most appropriate common skill. Identify what the learner actually needs to know or do with that construct.
17. Prefer a skill that represents transferable programming knowledge. The generated skill should be applicable to other Java programming exercises that require the same underlying capability, rather than merely restating the two given exercises.
18. Java-specific knowledge should be included when the Java feature or language mechanism itself is genuinely part of the shared knowledge component. Otherwise, prefer the underlying transferable programming skill rather than unnecessarily forcing the word "Java" into the skill.

Here are examples:

Example 1 — Very Similar Array Exercises

Exercise A:
"Write a program to find the largest integer in an array."

Exercise B:
"Given an array of student scores, determine the highest score."

Common Skill C:

(start) Traverse an array while maintaining a running maximum. (end)


Example 2 — Shared Frequency Counting

Exercise A:
"Count how many times each word appears in a list of words."

Exercise B:
"Given a sequence of integers, count the frequency of each integer."

Common Skill C:

(start) Use a mapping structure to count the frequency of elements. (end)


Example 3 — Shared Recursive Reasoning

Exercise A:
"Write a recursive method to calculate the factorial of a number."

Exercise B:
"Write a recursive method to calculate the sum of integers from 1 to n."

Common Skill C:

(start) Use recursion to decompose a problem into a smaller instance of the same problem. (end)


Example 4 — Shared Java Interface Knowledge

Exercise A:
"Create a class that implements an interface containing methods for calculating an object's area."

Exercise B:
"Implement an interface by providing concrete methods for processing employee information."

Common Skill C:

(start) Implement a Java interface by providing concrete method implementations. (end)


Example 5 — Shared Iterative Accumulation

Exercise A:
"Calculate the sum of all positive numbers in an integer array."

Exercise B:
"Calculate the total cost of all items in a list."

Common Skill C:

(start) Iterate through a collection while maintaining an accumulated result. (end)


Example 6 — Shared Object-Oriented Behavior

Exercise A:
"Create a superclass containing a method and override that method in a subclass."

Exercise B:
"Define a base class method and provide a specialized implementation in a derived class."

Common Skill C:

(start) Override inherited methods to provide specialized subclass behavior. (end)


Example 7 — Moderately Similar Collection Processing

Exercise A:
"Find all duplicate values in an array."

Exercise B:
"Determine which words occur more than once in a list."

Common Skill C:

(start) Track previously encountered elements to identify repeated values. (end)


Example 8 — Shared String Processing

Exercise A:
"Determine whether a string is a palindrome."

Exercise B:
"Check whether a word reads the same forward and backward."

Common Skill C:

(start) Compare characters symmetrically from opposite ends of a string to determine whether it is a palindrome. (end)


Example 9 — Shared Input Validation

Exercise A:
"Read an integer from the user and continue prompting until a valid positive value is entered."

Exercise B:
"Ask the user for a valid menu choice and repeat the input request when the choice is invalid."

Common Skill C:

(start) Repeatedly validate user input until it satisfies specified conditions. (end)


Example 10 — Broad Programming Reasoning

Exercise A:
"Process a sequence of values and determine whether any value satisfies a given condition."

Exercise B:
"Examine a collection of objects and determine whether at least one object meets a specified requirement."

Common Skill C:

(start) Traverse a collection to determine whether at least one element satisfies a specified condition. (end)


Example 11 — Shared Higher-Level Problem Solving

Exercise A:
"Process an array to calculate a statistical result."

Exercise B:
"Process a list of values to determine an aggregate result."

Common Skill C:

(start) Traverse a collection and compute an aggregate result from its elements. (end)


Example 12 — Different Surface Tasks, Shared Control Flow

Exercise A:
"Read numbers until the user enters zero, then display the total."

Exercise B:
"Process input values until a sentinel value is encountered, then report the accumulated result."

Common Skill C:

(start) Use sentinel-controlled iteration to process input until a terminating value is encountered. (end)


Example 13 — Shared Java Exception Handling

Exercise A:
"Write a program that handles invalid integer input using a try-catch block."

Exercise B:
"Handle an exception that occurs when accessing an invalid array index."

Common Skill C:

(start) Use Java exception handling to detect and handle runtime errors. (end)


Example 14 — Shared Sorting Knowledge

Exercise A:
"Sort an array of integers in ascending order."

Exercise B:
"Arrange a list of values from smallest to largest."

Common Skill C:

(start) Order a collection of comparable elements according to their values. (end)


Example 15 — Very Broad Java Programming Exercises

Exercise A:
"Write a Java program that reads input, processes the data, and produces an output."

Exercise B:
"Create a Java application that accepts user data, performs some computation, and displays the result."

Common Skill C:

(start) Design a program that accepts input, processes information, and produces an output. (end)


Now carry out this task and output a common skill C with the following exercises A and B:

Exercise A:
"{story1}"

Exercise B:
"{story2}"

Return ONLY the common skill description.
Begin your response with "(start)" and end it with "(end)."
"""
    
    return prompt14

def get_faithfulness_prompt():
    FAITHFULNESS_PROMPT = """
You will be given a Java programming exercise A (child) and a programming skill or knowledge component B (parent) that is intended to be a generalization of A. Your task is to determine how FAITHFUL the programming skill B is to the Java programming exercise A.

### Definition of Faithfulness

Faithfulness measures whether the programming capability or knowledge component described by B is actually tested, required, demonstrated, or genuinely implied by A.

In other words:

"Does the learner genuinely need to use or demonstrate the programming skill described by B in order to solve or complete exercise A?"

Faithfulness is a PRECISION measure. It evaluates whether B introduces programming capabilities or knowledge components that are not supported by A. A high-faithfulness skill describes a capability that the learner genuinely needs in order to implement, solve, analyze, debug, or complete A.

### Important principles

1. B must be grounded in A.

   Every important programming capability or knowledge component claimed by B must be supported by what the learner is required to do in A.

2. DO NOT require explicit wording.

   A does not need to explicitly name the programming concept, algorithm, data structure, Java construct, or programming technique. If the skill is genuinely required to solve A, it counts as supported.

   For example:

   A: "Given an array of integers, write a program to find the largest value."

   B: "Traverse an array while maintaining a running maximum."

   B is faithful even though A never explicitly says "maintain a running maximum."

3. DO NOT penalize abstraction.

   B is expected to be more general than A. Removing specific variable names, values, class names, method names, input examples, output examples, or other exercise-specific implementation details does not reduce faithfulness.

4. DO NOT penalize omission.

   B does not need to capture every skill or detail in A. Missing information is an informativeness issue, not a faithfulness issue.

   For example:

   A: "Read an array, calculate its sum, and then determine whether the sum is greater than a threshold."

   B: "Iterate through an array while maintaining an accumulated sum."

   B may omit the comparison with the threshold, but the capability it does claim is fully supported by A.

5. DO NOT infer skills from topic alone.

   Sharing a programming topic, language feature, or data structure is not sufficient.

   For example:

   A: "Find the largest value in an integer array."

   B: "Implement object-oriented inheritance."

   Both are Java programming tasks, but A does not require inheritance. Therefore B is not faithful.

6. Implicit programming skills are valid when genuinely required.

   A skill may be considered supported even when it is not explicitly stated, provided that it is necessary or strongly implied by the exercise.

7. Do not infer unnecessarily advanced skills.

   Only infer the level of programming knowledge needed to perform A.

   For example:

   A: "Calculate the sum of all elements in an array."

   B: "Apply advanced graph algorithms and dynamic programming techniques to optimize computational complexity."

   B is not faithful because A does not require those advanced methods.

8. Evaluate the capability, not just the programming topic.

   A skill should describe what the learner can DO with programming knowledge.

   "Arrays" is a programming topic or data structure.

   "Traverse an array while maintaining an accumulated result" is a programming skill.

9. Do not confuse the skill with the solution.

   B should describe a general capability or knowledge component rather than the particular answer, output, or implementation of A.

10. Do not penalize a skill simply because it is broad.

    A broad skill can still be faithful if everything it claims is supported by A. However, broadness or vagueness should affect informativeness or specificity, not faithfulness.

### What counts as a programming skill or knowledge component?

A programming skill or KC may involve the ability to:

- apply programming control-flow structures,
- use conditional logic,
- use iteration to process data,
- use recursion to solve a problem,
- manipulate arrays, lists, maps, sets, or other data structures,
- traverse and process collections,
- perform aggregation or accumulation,
- search or filter data,
- sort or order data,
- manipulate and analyze strings,
- parse or transform input,
- use algorithms or algorithmic procedures,
- reason about computational states,
- implement classes and objects,
- apply encapsulation,
- use inheritance or polymorphism,
- implement or use interfaces,
- override methods,
- use exception handling,
- work with Java language constructs,
- use Java APIs when they are genuinely required,
- read input and produce output,
- validate input,
- decompose a problem into smaller computational steps,
- debug or reason about program behavior,
- or apply a programming concept or procedure to solve a class of problems.

The specific skill must be grounded in what exercise A actually requires.

### Topic versus Knowledge Component

Do not treat a shared programming topic as automatically being a shared knowledge component.

For example:

A: "Find the maximum value in an array."

B: "Use arrays."

Although arrays are involved, "use arrays" is not necessarily the most meaningful capability being tested. The relevant capability may instead be:

"Traverse an array while maintaining a running maximum."

Similarly:

A: "Reverse an array."

B: "Find the maximum value in an array."

Both exercises use arrays, but they do not necessarily share the same specific knowledge component. Do not assign a more specific KC merely because the same data structure or Java feature appears in both.

### Multiple capabilities

If B contains multiple distinct capabilities, evaluate each one separately.

For example:

A: "Calculate the maximum value in an integer array."

B: "Traverse an array while maintaining a running maximum and implement the solution using inheritance."

The array traversal and running-maximum capability are supported by A, but inheritance is not.

Therefore, B is only partially faithful.

### Faithfulness versus Informativeness

Keep these two metrics strictly separate.

FAITHFULNESS asks:

"Are the capabilities or knowledge components claimed by B actually supported by A?"

INFORMATIVENESS asks:

"Does B capture the important programming capabilities or knowledge components that A tests?"

For example:

A: "Calculate the maximum value in an integer array."

B: "Solve programming problems."

B is broad and not very informative, but it can still be faithful because A is indeed a programming problem.

DO NOT lower faithfulness merely because B is vague, overly general, or incomplete.

### Faithfulness versus Specificity

Keep specificity separate from faithfulness.

A skill may be faithful even if it is very general.

For example:

A: "Calculate the maximum value in an integer array."

B: "Process data using a programming algorithm."

B is very general and may have low specificity, but its general claim is still compatible with A.

Faithfulness should decrease only when B claims programming capabilities that are not supported by A.

### Java-specific knowledge

Java-specific knowledge should be considered when the Java feature itself is genuinely required by A.

For example:

A:
"Create a class that implements the Comparable interface and define compareTo() to order objects."

B:
"Implement Java interfaces and provide the required method implementations."

B is faithful because implementing the interface is genuinely required.

However, do not force Java-specific terminology when the underlying capability is more general.

For example:

A:
"Calculate the sum of all values in an integer array."

B:
"Use Java arrays to compute a sum."

This may be faithful, but the more meaningful underlying KC may be:

"Traverse an array while maintaining an accumulated result."

The evaluation should focus on whether the capability claimed by B is genuinely required by A, rather than rewarding or penalizing the use of the word "Java."

### Scoring

Score from 0 to 1.

Consider the distinct programming capabilities or knowledge components claimed by B and determine how well each is supported by A.

- Fully supported capability → full credit.
- Partially implied or not necessarily required → partial credit.
- Unsupported capability → little or no credit.

Use the following guidelines:

0.85-1.00 — Fully Faithful

All or almost all capabilities claimed by B are genuinely supported or strongly implied by A. B introduces no meaningful unsupported programming capability.

0.60-0.85 — Mostly Faithful

Most capabilities in B are supported, but one capability or nuance is only partially implied or slightly extends beyond what A requires.

0.30-0.60 — Partially Faithful

B contains both supported and unsupported capabilities. At least one meaningful programming capability in B is not supported by A.

0.00-0.30 — Not Faithful

A substantial part of B is unsupported by A, or B describes a substantially different programming capability.

### Examples

Example 1:

A:
"Write a Java program to find the largest integer in an array."

B:
"Traverse an array while maintaining a running maximum."

Score: 1.0

Reason:
The learner must examine the array elements and maintain the largest value encountered. The capability described by B is therefore genuinely required by A.

---

Example 2:

A:
"Write a Java program to calculate the sum of all integers in an array."

B:
"Iterate through a collection while maintaining an accumulated result."

Score: 1.0

Reason:
The learner must iterate through the array and maintain an accumulated sum. B correctly captures an underlying programming capability required by A.

---

Example 3:

A:
"Write a recursive Java method to calculate the factorial of a number."

B:
"Use recursion to decompose a problem into smaller instances of the same problem."

Score: 1.0

Reason:
The exercise explicitly requires a recursive solution. The recursive problem-decomposition capability is therefore genuinely supported.

---

Example 4:

A:
"Find the maximum value in an integer array."

B:
"Use arrays and implement object-oriented inheritance."

Score: 0.5

Reason:
Working with an array is supported by A, but inheritance is not required or implied. The unsupported inheritance capability reduces faithfulness.

---

Example 5:

A:
"Given a list of integers, determine how many times each integer occurs."

B:
"Use a mapping structure to count the frequency of elements."

Score: 1.0

Reason:
The exercise requires tracking the occurrence count of elements. B correctly describes the underlying frequency-counting capability.

---

Example 6:

A:
"Read integers until the user enters zero and then display their sum."

B:
"Use sentinel-controlled iteration to process input until a terminating value is encountered."

Score: 1.0

Reason:
The learner must repeatedly process input until a designated terminating value is encountered. The capability described by B is directly supported by A.

---

Example 7:

A:
"Create a Java class that implements an interface and provide implementations for all required methods."

B:
"Implement a Java interface by providing concrete method implementations."

Score: 1.0

Reason:
Implementing the interface and providing the required methods are directly required by A.

---

Example 8:

A:
"Create a superclass and override one of its methods in a subclass."

B:
"Apply inheritance and method overriding to provide specialized subclass behavior."

Score: 1.0

Reason:
The exercise requires inheritance between the classes and overriding a superclass method. Both capabilities in B are supported.

---

Example 9:

A:
"Calculate the sum of the values in an array."

B:
"Apply dynamic programming to solve optimization problems."

Score: 0.1

Reason:
The exercise does not require dynamic programming or optimization techniques. B introduces an unsupported programming capability.

---

Example 10:

A:
"Sort an array of integers in ascending order."

B:
"Order a collection of comparable elements according to their values."

Score: 1.0

Reason:
The exercise requires ordering elements according to their values. B expresses the generalized capability without relying on the specific array or integer details.

---

Example 11:

A:
"Read a string and determine whether it is a palindrome."

B:
"Process and compare characters in a string to determine a property of the string."

Score: 1.0

Reason:
The learner must process the characters of a string and compare them to determine whether the required property holds. B is broader but remains grounded in A.

---

Example 12:

A:
"Read a student's name and score and store each student's information in a HashMap using the name as the key."

B:
"Use a mapping structure to associate keys with corresponding values."

Score: 1.0

Reason:
The exercise requires associating each student name with a corresponding score. The mapping capability described by B is genuinely required.

---

Example 13:

A:
"Write a Java program that catches an exception caused by invalid integer input."

B:
"Use Java exception handling to detect and handle runtime errors."

Score: 1.0

Reason:
The exercise requires catching and handling an exception. B correctly generalizes this capability.

---

Example 14:

A:
"Analyze an array and determine whether any element satisfies a specified condition."

B:
"Traverse a collection to determine whether at least one element satisfies a condition."

Score: 1.0

Reason:
The learner must examine collection elements and determine whether at least one satisfies the condition. B captures the underlying capability.

---

Example 15:

A:
"Calculate the maximum value in an integer array."

B:
"Understand object-oriented programming."

Score: 0.2

Reason:
The exercise does not require object-oriented programming as a meaningful capability. Although the program may be written in Java, B does not describe a capability genuinely required by the exercise.

---

Example 16:

A:
"Calculate the sum of all elements in an array."

B:
"Traverse an array while maintaining a running sum and optimize the algorithm using dynamic programming."

Score: 0.5

Reason:
Traversing the array while maintaining a running sum is supported by A, but dynamic programming is not required. The unsupported capability reduces faithfulness.

### Final principle

The central question is:

"Does the learner genuinely need the capability or knowledge component described by B to perform A?"

Generalization is allowed.
Abstraction is allowed.
Omission is allowed.
Implicit programming skills are allowed when genuinely required.

Only unsupported programming capabilities or knowledge components should reduce the faithfulness score.

## Prompt A (child)
{prompt_a}

## Prompt B (parent)
{prompt_b}

Output your score in exactly the following format:

Score: [SCORE]
"""
    return FAITHFULNESS_PROMPT

def get_informativeness_prompt():
    INFORMATIVENESS_PROMPT = """

You will be given a Java programming exercise A (child) and a programming skill or knowledge component B (parent) that is intended to be a generalization of A. Your task is to determine how INFORMATIVE the programming skill B is about the Java programming exercise A.

### Definition of Informativeness

Informativeness measures how much of the IMPORTANT PROGRAMMING CAPABILITY or KNOWLEDGE COMPONENT tested or required by A is captured by B.

In other words:

"How much of what a learner needs to know or be able to do to successfully perform A is captured by the skill or knowledge component B?"

Informativeness is a RECALL measure. A highly informative parent identifies the important underlying programming skill(s), reasoning, algorithm, data structure, Java construct, or problem-solving procedure of the exercise rather than merely describing its broad topic or activity.

### What to consider as important content in A

First identify the important programming capabilities or knowledge components that A requires.

These may include:

• applying programming control-flow structures,

• using conditional logic,

• using iteration to process data,

• using recursion to solve a problem,

• manipulating arrays, lists, maps, sets, or other data structures,

• traversing and processing collections,

• performing aggregation or accumulation,

• searching or filtering data,

• sorting or ordering data,

• manipulating and analyzing strings,

• parsing or transforming input,

• applying an algorithm or algorithmic procedure,

• reasoning about computational states,

• implementing classes and objects,

• applying encapsulation,

• using inheritance or polymorphism,

• implementing or using interfaces,

• overriding methods,

• handling exceptions,

• using Java language constructs,

• using Java APIs when they are genuinely required,

• reading input and producing output,

• validating input,

• decomposing a problem into smaller computational steps,

• reasoning about program behavior,

• debugging or analyzing code behavior,

• or other meaningful programming reasoning or problem-solving capabilities.

Focus on the underlying programming capabilities and knowledge components rather than incidental details.

Do NOT treat the following as important capabilities unless they are essential to the programming task:

• specific numerical values,

• variable names,

• method names,

• class names,

• specific input values,

• specific output values,

• particular object names,

• incidental story context,

• wording of the question,

• minor formatting requirements,

• superficial implementation details.

### Important principles

1. Measure CAPABILITY COVERAGE.

Determine which important programming capabilities or knowledge components A requires and how many of them are captured by B.

For example:

A: "Write a Java program that finds the largest value in an integer array."

B: "Traverse an array while maintaining a running maximum."

B captures the central programming capability required by A, so informativeness is high.

2. Capture the UNDERLYING SKILL OR KNOWLEDGE COMPONENT, not merely the topic.

A parent such as:

"Understand Java."

may be technically related to A, but it provides very little information about what the learner actually needs to do.

Similarly:

A: "Find the largest value in an integer array."

B: "Arrays."

B is extremely uninformative because it identifies only the data structure, not the programming capability being demonstrated.

3. Semantic coverage matters, not exact wording.

B does not need to repeat A's wording.

For example:

A: "Determine the largest number stored in an integer array."

B: "Traverse an array while maintaining a running maximum."

B is highly informative even though the wording is different because it captures the same underlying programming capability.

4. Generalization is allowed and expected.

B is intended to be more general than A.

Do NOT penalize B merely because it removes:

• numerical values,

• variable names,

• class names,

• method names,

• particular input examples,

• particular output examples,

• specific objects,

• specific datasets,

• or other exercise-specific details.

The question is whether B preserves the important programming capability behind those details.

5. Do not require every detail of A to appear in B.

Only IMPORTANT PROGRAMMING CAPABILITIES should affect informativeness.

For example:

A: "Calculate the sum of 10 integers stored in an array."

B: "Traverse an array while maintaining an accumulated sum."

B does not mention the number 10 or the specific array values, but these are incidental details. Their omission should not reduce informativeness.

6. Implicit programming skills count.

A may not explicitly state the programming concept, algorithm, data structure, or reasoning procedure that must be used.

Infer the underlying capability when it is genuinely required by A.

For example:

A: "Write a recursive Java method to calculate factorial."

B: "Use recursion to decompose a problem into smaller instances of the same problem."

The recursive problem-decomposition skill is implicit in A, but it is still an important capability and B captures it.

7. Penalize VAGUENESS.

A parent that is technically related to A but too broad to identify the important programming capability is not informative.

For example:

A: "Find the largest value in an integer array."

B: "Solve programming problems."

B is related to A, but it does not identify the important array-processing and maximum-finding capability.

Therefore, it should receive substantially lower informativeness than:

"Traverse an array while maintaining a running maximum."

8. Topic overlap alone is insufficient.

A parent should not receive high informativeness merely because it mentions the same programming language, topic, data structure, or Java feature.

For example:

A: "Find the largest value in an integer array."

B: "Use Java arrays."

B identifies a relevant programming topic but fails to capture the important capability of traversing the array and maintaining a maximum.

Therefore, informativeness should be limited.

9. Multiple capabilities must be considered.

If A requires several important programming capabilities, determine whether B captures all, most, some, or very few of them.

For example:

A: "Read an array, calculate its sum, and determine whether the sum exceeds a threshold."

Important capabilities include:

• traversing an array,

• maintaining an accumulated sum,

• comparing the resulting value with a condition.

B: "Traverse an array while maintaining an accumulated sum."

B captures the first two important capabilities but misses the threshold comparison.

Therefore, B is informative but incomplete.

10. Do not reward unsupported content.

Only evaluate how much of A is captured by B.

Do not give additional informativeness credit because B contains technically correct programming capabilities that are not required by A.

For example:

A: "Calculate the sum of values in an array."

B: "Traverse an array, calculate its sum, and implement the solution using inheritance."

The inheritance component does not increase informativeness because A does not require inheritance.

11. Distinguish informativeness from faithfulness.

Keep these two metrics strictly separate.

FAITHFULNESS asks:

"Are the capabilities claimed by B actually supported by A?"

INFORMATIVENESS asks:

"Does B capture the important capabilities that A requires?"

A skill can therefore be:

• highly faithful but poorly informative,

• highly faithful and highly informative,

• partially faithful and partially informative,

• or poorly faithful and poorly informative.

For example:

A: "Find the largest value in an integer array."

B: "Solve programming problems."

B may be faithful because A is a programming problem, but it is poorly informative because it fails to identify the important underlying capability.

### Knowledge Component versus Programming Topic

The evaluator should distinguish between a programming topic and an actual knowledge component.

A KC should represent meaningful knowledge or capability that contributes to solving the exercise.

For example:

A:
"Find the maximum value in an array."

B:
"Arrays."

B identifies a topic but captures very little of the actual knowledge required.

In contrast:

B:
"Traverse an array while maintaining a running maximum."

captures the relevant knowledge component much more directly.

Similarly:

A:
"Create a subclass that overrides a superclass method."

B:
"Object-oriented programming."

B captures only a broad topic.

A more informative parent would be:

"Override inherited methods to provide specialized subclass behavior."

The goal is to measure whether B captures the underlying knowledge needed to perform A, not whether B merely mentions a related programming topic.

### Java-specific knowledge

Java-specific knowledge should be considered when it represents an important part of what A requires.

For example:

A:
"Create a class that implements the Comparable interface and define compareTo() to order objects."

B:
"Implement a Java interface and provide the required method implementation."

B captures an important Java-specific capability.

However, do not require Java-specific terminology when the underlying knowledge component is language-independent.

For example:

A:
"Calculate the sum of all elements in an integer array."

B:
"Traverse an array while maintaining an accumulated result."

This can be highly informative even though it does not explicitly mention Java.

### Scoring

Identify the important programming capabilities or knowledge components required by A. Then determine what proportion of those important capabilities are captured by B.

Use semantic and conceptual coverage rather than literal word matching.

Scoring guidance:

0.85-1.00 — Excellent Coverage

B captures essentially all of the important programming capabilities or knowledge components required by A. B may generalize away exercise-specific details, but it preserves the core programming reasoning, algorithm, data structure, Java construct, or problem-solving capability.

0.60-0.85 — Good Coverage

B captures the main programming capability of A but misses one or more important capabilities, relationships, reasoning steps, or distinctions.

0.30-0.60 — Partial / Vague Coverage

B captures a broad aspect of A, such as the programming domain, general activity, Java feature, or data structure, but fails to capture much of the specific underlying capability.

0.00-0.30 — Very Low Coverage / Mismatch

B captures little or none of the important programming capabilities required by A, or it describes a substantially different capability.

### Examples

Example 1:

Prompt A:
"Write a Java program to find the largest value in an integer array."

Prompt B:
"Traverse an array while maintaining a running maximum."

Score: 1.0

Reason:
B captures the central programming capability required by A. The specific values, variable names, and Java implementation details do not need to be preserved.

---

Example 2:

Prompt A:
"Calculate the sum of all values stored in an integer array."

Prompt B:
"Traverse an array while maintaining an accumulated result."

Score: 1.0

Reason:
B captures the main traversal and accumulation capability required by A.

---

Example 3:

Prompt A:
"Write a recursive Java method to calculate the factorial of a number."

Prompt B:
"Use recursion to decompose a problem into smaller instances of the same problem."

Score: 1.0

Reason:
B captures the underlying recursive problem-solving capability required by A.

---

Example 4:

Prompt A:
"Find the largest value in an integer array."

Prompt B:
"Use arrays."

Score: 0.35

Reason:
B identifies the data structure involved but does not capture the important capability of traversing the array and determining the maximum value.

---

Example 5:

Prompt A:
"Given a list of integers, determine how many times each integer occurs."

Prompt B:
"Use a mapping structure to count the frequency of elements."

Score: 1.0

Reason:
B captures the central frequency-counting knowledge component required by A.

---

Example 6:

Prompt A:
"Read integers until the user enters zero and then display their sum."

Prompt B:
"Use iteration to process input."

Score: 0.55

Reason:
B captures the general iterative processing capability but misses the important sentinel-controlled termination and accumulation aspects of A.

---

Example 7:

Prompt A:
"Create a Java class that implements an interface and provide implementations for all required methods."

Prompt B:
"Implement a Java interface by providing concrete method implementations."

Score: 1.0

Reason:
B captures the central Java-specific knowledge component required by A.

---

Example 8:

Prompt A:
"Create a superclass and override one of its methods in a subclass."

Prompt B:
"Use object-oriented programming."

Score: 0.25

Reason:
B identifies only a broad programming paradigm and does not capture the important inheritance and method-overriding capabilities required by A.

---

Example 9:

Prompt A:
"Calculate the sum of all elements in an array."

Prompt B:
"Apply dynamic programming to solve optimization problems."

Score: 0.05

Reason:
B does not capture the important capability required by A and describes a substantially different programming technique.

---

Example 10:

Prompt A:
"Sort an array of integers in ascending order."

Prompt B:
"Order a collection of comparable elements according to their values."

Score: 1.0

Reason:
B captures the underlying ordering capability while generalizing away the specific array and integer details.

---

Example 11:

Prompt A:
"Read a string and determine whether it is a palindrome."

Prompt B:
"Process strings."

Score: 0.35

Reason:
B captures the broad activity of string processing but does not capture the important character-comparison reasoning required to identify a palindrome.

---

Example 12:

Prompt A:
"Read a string and determine whether it is a palindrome."

Prompt B:
"Compare characters from corresponding positions to determine whether a string has a symmetric structure."

Score: 0.90

Reason:
B captures the main character-comparison reasoning underlying the task, although it is somewhat more general than the specific palindrome capability.

---

Example 13:

Prompt A:
"Calculate the maximum value in an array and then determine whether the maximum exceeds a given threshold."

Prompt B:
"Traverse an array while maintaining a running maximum."

Score: 0.75

Reason:
B captures the central maximum-finding capability but misses the additional important capability of comparing the maximum against a threshold.

---

Example 14:

Prompt A:
"Create a subclass that overrides a method inherited from its superclass."

Prompt B:
"Override inherited methods to provide specialized subclass behavior."

Score: 1.0

Reason:
B captures both the inheritance relationship and method-overriding capability required by A.

---

Example 15:

Prompt A:
"Read a list of student scores and calculate the average score."

Prompt B:
"Process a collection of numerical values."

Score: 0.40

Reason:
B captures the general collection-processing activity but misses the important aggregation and average-calculation capability.

### Final principle

The central question is:

"How much of the important programming capability or knowledge component required by A is captured by B?"

Do not measure literal similarity.

Do not require exercise-specific details to be preserved.

Do not reward mere topic overlap.

Focus on whether B captures the underlying programming knowledge, reasoning, algorithm, data structure, Java construct, method, or problem-solving capability that the learner needs to perform A.

Generalization is expected, but the important programming capability should be retained.

## Prompt A (child)

{prompt_a}

## Prompt B (parent)

{prompt_b}

Output your score in exactly the following format:

Score: [SCORE]

"""
    return INFORMATIVENESS_PROMPT


def get_eval_prompt():
    evalPrompt = """
Verifying Specific Common Knowledge Component of Two Similar Java Programming Exercises.

You would be given two similar Java programming exercises and a description of the common programming skill or Knowledge Component (KC) shared by the exercises.

The goal is to determine whether the proposed common skill is actually supported, required, demonstrated, or genuinely implied by each Java programming exercise.

The common skill should describe an underlying programming capability, reasoning process, algorithm, data structure usage, programming construct, Java language knowledge, or procedure that the learner needs to apply when solving the exercise.

First, determine the following:

1. Determine Score_A - Given that Exercise A needs to be solved, please check if the proposed common skill is also required or genuinely implied by Exercise A (Exercise A implies the common skill).

Determine a score between 0 to 1 for how strongly the provided common_skill is supported by Exercise A (Score_A).

The common skill CANNOT contain important programming capabilities, requirements, or relationships that are not supported by Exercise A.

When judging the implication relation, the details of the programming exercise should not be ignored.

For example, if Exercise A asks the learner to find the maximum value in an array, a common skill such as "Traverse an array while maintaining a running maximum" is strongly implied because the learner needs to examine the array elements and maintain the largest value encountered so far.

However, a skill such as "Use dynamic programming to optimize the computation" is NOT implied merely because the exercise involves computation. Dynamic programming is not required by the exercise.

Similarly, if an exercise asks the learner to override a method in a subclass, a skill such as "Override inherited methods to provide specialized subclass behavior" is strongly implied. A skill such as "Use Java multithreading" is not implied unless the exercise actually requires it.

The exact wording of the common skill does NOT need to appear in Exercise A. Evaluate the underlying semantic programming capability rather than lexical overlap.

The common skill may be more general than the specific exercise. Generalization is allowed and should not be penalized as long as the skill is genuinely supported by the exercise.

However, the proposed skill should not introduce unsupported programming capabilities merely because they are related to the topic of the exercise.

A programming topic alone is not necessarily a Knowledge Component. For example, an exercise involving an array does not automatically imply that "Use arrays" is the most meaningful skill. The relevant underlying capability might instead be "Traverse an array while maintaining a running maximum."

You should evaluate what the learner genuinely needs to know or do to perform the exercise, rather than the exact code implementation that the learner might choose.

Implicit or latent programming skills should be considered when they are genuinely required.

However, do NOT infer unnecessarily advanced algorithms, data structures, design patterns, programming paradigms, or Java features that are not actually required by the exercise.

2. Determine Score_B - Given that Exercise B needs to be solved, please check if the proposed common skill is also required or genuinely implied by Exercise B (Exercise B implies the common skill).

Determine a score between 0 to 1 for how strongly the provided common_skill is supported by Exercise B (Score_B).

Apply the same principles used for Exercise A.

The common skill CANNOT contain important programming capabilities, requirements, or relationships that are not supported by Exercise B.

When judging the implication relation, the details of the programming exercise should not be ignored.

For example, if Exercise B asks the learner to count the frequency of elements using a map, a skill such as "Use a mapping structure to count the frequency of elements" is strongly implied.

However, a skill such as "Use a balanced binary search tree to optimize frequency queries" should not receive a high score unless the exercise actually requires such a data structure.

The proposed common skill should capture an underlying learner capability rather than simply repeating the wording of the exercise.

The skill should also be transferable to other Java programming exercises. However, transferability does not mean that the skill must be extremely broad.

Important principles for evaluating both Score_A and Score_B:

1. The proposed common skill must be grounded in the programming exercise.
2. The exact wording of the skill does not need to appear in the exercise.
3. Evaluate semantic meaning rather than lexical overlap.
4. Generalization and abstraction are allowed and expected.
5. Do not penalize a skill simply because it is more general than the exercise.
6. Do not add unsupported programming capabilities to the skill.
7. Evaluate underlying programming knowledge, reasoning, algorithms, data structures, language constructs, or procedures rather than only surface-level topics.
8. A Java feature should count as part of the skill when the feature itself is genuinely required by the exercise.
9. Do not assume that every Java exercise requires knowledge of Java-specific features beyond what the exercise actually tests.
10. Implicit or latent programming skills should be counted when they are genuinely required.
11. Do not infer unnecessarily advanced programming knowledge.
12. Evaluate the learner capability required to perform the exercise, not a particular implementation chosen by the learner.
13. Do not solve the exercise. Only determine whether the proposed skill is supported by the exercise.
14. The proposed skill should represent a transferable Knowledge Component rather than merely restating the exercise.
15. A topic such as "arrays", "loops", "Java", or "object-oriented programming" should not automatically be treated as the underlying KC when a more meaningful capability is supported by the exercise.
16. The skill should represent the intersection of the underlying capabilities of the exercise, not a union of unrelated capabilities.
17. If the proposed skill contains multiple capabilities, evaluate whether each capability is actually supported by the exercise.
18. Unsupported additional capabilities should reduce the entailment score.
19. A broad but genuinely supported skill can still receive high entailment. Do not reduce entailment merely because another possible skill could be more specific.
20. Distinguish entailment from informativeness and specificity. Entailment asks whether the exercise supports the proposed skill; it does not ask whether the skill is the most specific or most informative possible description.

Examples of meaningful Java programming Knowledge Components include:

- Traverse an array while maintaining a running maximum.
- Traverse an array while maintaining an accumulated result.
- Search for an element satisfying a specified condition.
- Filter elements according to a condition.
- Count element frequencies using a mapping structure.
- Sort elements according to an ordering criterion.
- Use recursion to decompose a problem into smaller instances of the same problem.
- Compare corresponding or symmetric elements in a sequence.
- Parse and process structured input.
- Validate input against specified conditions.
- Use conditional logic to handle alternative cases.
- Use sentinel-controlled iteration to process an unknown number of inputs.
- Implement an interface by providing concrete method implementations.
- Override inherited methods to provide specialized subclass behavior.
- Use inheritance and polymorphism to specialize object behavior.
- Handle exceptional conditions using Java exception mechanisms.
- Decompose a programming problem into smaller computational steps.

Examples of descriptions that may be too superficial to fully characterize an underlying KC include:

- "Use Java."
- "Use arrays."
- "Use loops."
- "Use variables."
- "Use object-oriented programming."
- "Write a program."
- "Process data."

These descriptions may be technically related to the exercise, but they may not capture the meaningful underlying programming capability being demonstrated.

Nevertheless, remember that this evaluation is specifically about ENTAILMENT.

If a broad statement is genuinely supported by the exercise, it may still receive a high entailment score even if it is less informative or less specific.

For example:

Exercise A:
Write a Java program that finds the largest value in an integer array.

Common Skill C:
Use an array to process a collection of values.

This is broadly supported by Exercise A, so its entailment should not be extremely low merely because the skill is generic.

However:

Exercise A:
Write a Java program that finds the largest value in an integer array.

Common Skill C:
Use dynamic programming to optimize overlapping subproblems.

This is not supported by Exercise A and should receive a very low entailment score.

Finally, output 2 comma-separated values as follows:

1. Entail-1: Which is the average of score_A and score_B.
2. Entail-2: Which is the min(score_A, score_B).

Format the output as:

**entail-1, entail-2:** [score1], [score2]

Below this, you can also include score_A and score_B and the reasoning for the scores.

Scoring guidance:

1.00:
The proposed skill is directly required or clearly demonstrated by the exercise, with essentially all important components of the skill supported.

0.85–0.99:
The proposed skill is strongly supported, with only minor ambiguity or a small amount of abstraction.

0.60–0.84:
The proposed skill is substantially supported, but one component is only implicit, somewhat broader than necessary, or not fully demonstrated.

0.30–0.59:
The proposed skill is partially supported, but important parts of the skill are missing or only weakly implied.

0.01–0.29:
The exercise has some weak or indirect relationship to the proposed skill, but the skill is largely unsupported.

0.00:
The proposed skill is not required, demonstrated, or genuinely implied by the exercise.

When a proposed skill contains multiple distinct capabilities, evaluate the capabilities separately.

For example:

Exercise A:
Write a Java program that finds the maximum value in an array.

Common Skill C:
Traverse an array, maintain a running maximum, and sort the array before finding the maximum.

The first two capabilities are supported, but sorting is not required. Therefore, the score should be reduced because the proposed skill contains an unsupported additional capability.

Important distinction between entailment and other evaluation dimensions:

Entailment asks:
"Given that the learner must solve this exercise, is the proposed Knowledge Component genuinely supported or required?"

It does NOT ask:
"Is this the best possible Knowledge Component?"

It does NOT ask:
"Is this the most specific Knowledge Component?"

It does NOT ask:
"Does this KC capture all important capabilities of the exercise?"

It does NOT ask:
"Is this KC the most informative description?"

Those questions belong to other evaluation dimensions such as specificity, informativeness, and faithfulness.

Therefore, a broad but true skill may have high entailment even though it has lower specificity or informativeness.

Here are some examples:

Example 1: High Entailment

Exercise A:
Write a Java program that reads an array of integers and finds the largest value in the array.

Exercise B:
Write a Java program that reads an array of student scores and determines the highest score.

Common Skill C:
Traverse an array while maintaining a running maximum.

Output:
Common Skill C details:
    1. Traverse the array.
    2. Examine the elements.
    3. Maintain the maximum value encountered so far.

Score A calculation: 3/3 = 1.0
Score B calculation: 3/3 = 1.0
**entail-1, entail-2:** 1.0, 1.0

Reasoning:
All important components of the proposed skill are required by both exercises. Both exercises require traversal of an array while maintaining the largest value encountered.

Example 2: Low Entailment

Exercise A:
Write a Java program that calculates the sum of all integers in an array.

Exercise B:
Write a Java program that calculates the total cost of all items in a shopping cart.

Common Skill C:
Use inheritance and polymorphism to specialize object behavior.

Output:
Common Skill C details:
    1. Use inheritance.
    2. Use polymorphism.
    3. Specialize subclass behavior.

Score A calculation: 0/3 = 0.0
Score B calculation: 0/3 = 0.0
**entail-1, entail-2:** 0.0, 0.0

Reasoning:
Neither exercise requires inheritance, polymorphism, or subclass specialization. The proposed skill is not genuinely implied by either exercise.

Example 3: Medium Entailment

Exercise A:
Write a Java program that finds the largest value in an integer array.

Exercise B:
Write a Java program that finds all values in an integer array that are greater than a specified threshold.

Common Skill C:
Traverse an array while processing its elements according to a condition.

Output:
Common Skill C details:
    1. Traverse the array.
    2. Process the elements.
    3. Apply a condition while processing the elements.

Score A calculation: 0.65
Score B calculation: 1.0
**entail-1, entail-2:** 0.825, 0.65

Reasoning:
Exercise B clearly requires traversal and condition-based processing. Exercise A clearly requires traversal and processing of array elements, but it does not necessarily require the same explicit condition-based filtering operation. Therefore, the skill is only partially supported by Exercise A.

Example 4: High Entailment with Implicit Knowledge

Exercise A:
Create a superclass Animal with a method makeSound(), then create subclasses Dog and Cat that provide their own implementations of makeSound().

Exercise B:
Create a superclass Vehicle with a method move(), then create subclasses Car and Bicycle that override move() with specialized behavior.

Common Skill C:
Override inherited methods to provide specialized subclass behavior.

Output:
Common Skill C details:
    1. Work with an inherited method.
    2. Provide a subclass implementation.
    3. Specialize inherited behavior.

Score A calculation: 3/3 = 1.0
Score B calculation: 3/3 = 1.0
**entail-1, entail-2:** 1.0, 1.0

Reasoning:
Both exercises directly require subclasses to provide specialized implementations of inherited methods.

Example 5: Broad but Supported

Exercise A:
Write a Java program that reads integers until the user enters -1 and then prints their sum.

Exercise B:
Write a Java program that reads numbers until the user enters 0 and then prints their average.

Common Skill C:
Use iteration to process a sequence of input values.

Output:
Common Skill C details:
    1. Repeatedly process input.
    2. Use iteration.
    3. Process a sequence of values.

Score A calculation: 3/3 = 1.0
Score B calculation: 3/3 = 1.0
**entail-1, entail-2:** 1.0, 1.0

Reasoning:
Both exercises require repeated processing of input values using iteration. Although the proposed skill is broad, it is genuinely supported by both exercises.

Example 6: Unsupported Advanced Knowledge

Exercise A:
Write a Java program that sorts an integer array in ascending order.

Exercise B:
Write a Java program that sorts a list of student names alphabetically.

Common Skill C:
Use dynamic programming to optimize repeated subproblems.

Output:
Common Skill C details:
    1. Use dynamic programming.
    2. Identify overlapping subproblems.
    3. Optimize repeated subproblems.

Score A calculation: 0.0
Score B calculation: 0.0
**entail-1, entail-2:** 0.0, 0.0

Reasoning:
Both exercises require sorting, but neither exercise requires dynamic programming or optimization of overlapping subproblems.

Example 7: Java-Specific Knowledge

Exercise A:
Define a Java interface named Printable and implement it in a class Report.

Exercise B:
Define a Java interface named Drawable and implement it in a class Circle.

Common Skill C:
Implement a Java interface by providing concrete method implementations.

Output:
Common Skill C details:
    1. Work with an interface.
    2. Implement the interface in a class.
    3. Provide concrete method implementations.

Score A calculation: 1.0
Score B calculation: 1.0
**entail-1, entail-2:** 1.0, 1.0

Reasoning:
Both exercises explicitly require implementation of an interface and concrete implementations of its methods.

Example 8: Underlying Skill Rather Than Surface Topic

Exercise A:
Write a Java program that counts how many times each integer appears in an array.

Exercise B:
Write a Java program that counts how many times each word appears in a sentence.

Common Skill C:
Use a mapping structure to count the frequency of elements.

Output:
Common Skill C details:
    1. Associate an element with its count.
    2. Update the count when an element occurs.
    3. Maintain frequencies using a mapping structure.

Score A calculation: 1.0
Score B calculation: 1.0
**entail-1, entail-2:** 1.0, 1.0

Reasoning:
Although the data types differ, both exercises require the same underlying frequency-counting capability.

Example 9: Partially Supported

Exercise A:
Write a Java program that finds the maximum value in an array.

Exercise B:
Write a Java program that finds the maximum value in an array and then checks whether the maximum is greater than 100.

Common Skill C:
Traverse an array while maintaining a running maximum and then evaluate the result against a condition.

Output:
Common Skill C details:
    1. Traverse the array.
    2. Maintain a running maximum.
    3. Evaluate the resulting maximum against a condition.

Score A calculation: 2/3 ≈ 0.67
Score B calculation: 3/3 = 1.0
**entail-1, entail-2:** 0.835, 0.67

Reasoning:
Exercise A supports traversal and maintenance of a running maximum, but it does not require the additional conditional evaluation. Exercise B requires all three components.

Example 10: Topic-Level Entailment

Exercise A:
Write a Java program that stores employee records in an ArrayList and searches for an employee by ID.

Exercise B:
Write a Java program that stores student records in an ArrayList and sorts them by score.

Common Skill C:
Use an ArrayList to store a collection of objects.

Output:
Common Skill C details:
    1. Store multiple objects.
    2. Use an ArrayList as the collection structure.

Score A calculation: 1.0
Score B calculation: 1.0
**entail-1, entail-2:** 1.0, 1.0

Reasoning:
Both exercises explicitly use an ArrayList to store objects. Therefore, the proposed skill is entailed by both exercises. Although it may be less informative than the underlying searching or sorting skills, that should not reduce its entailment score.

Example 11: Unsupported Java Feature

Exercise A:
Write a Java program that reads integers and calculates their average.

Exercise B:
Write a Java program that reads integers and determines their maximum.

Common Skill C:
Use Java generics to create a reusable generic algorithm.

Output:
Common Skill C details:
    1. Define generic types.
    2. Create a reusable generic algorithm.
    3. Apply Java generics.

Score A calculation: 0.0
Score B calculation: 0.0
**entail-1, entail-2:** 0.0, 0.0

Reasoning:
Neither exercise requires Java generics. The fact that both programs could theoretically be implemented using generics does not mean that generics are entailed by the exercises.

Example 12: Common Programming Reasoning

Exercise A:
Write a Java program that calculates the sum of all positive values in an array.

Exercise B:
Write a Java program that calculates the product of all positive values in an array.

Common Skill C:
Traverse an array while conditionally selecting elements and maintaining an accumulated result.

Output:
Common Skill C details:
    1. Traverse the array.
    2. Determine whether each element satisfies a condition.
    3. Include qualifying elements in an accumulated result.

Score A calculation: 1.0
Score B calculation: 1.0
**entail-1, entail-2:** 1.0, 1.0

Reasoning:
Both exercises require traversal, conditional selection of positive elements, and accumulation of the selected values.

Now carry out this task and evaluate the following:

Exercise A:
"{story1}"

Exercise B:
"{story2}"

Common Skill C:
"{common_summary}"

Return the output in the requested format.
"""
    return evalPrompt


def get_eval_prompt2():
    evalPrompt2 = """
You will be given a Java programming exercise A and a programming skill or Knowledge Component (KC) B that is proposed as a generalization of the skill required to solve exercise A. Your task is to evaluate how well skill B represents and generalizes the underlying programming skill or capability needed to solve exercise A.

The relationship being evaluated is:

Java Exercise A → underlying programming skill/KC needed to solve A → generalized programming skill/KC B

B does not need to describe the exercise itself. Instead, B should capture a more general programming capability that represents the knowledge, reasoning, procedure, algorithm, data structure, Java language construct, computational operation, or programming technique needed to successfully solve or understand A.

Give a score between 0 and 1.

A score close to 1 means that the skill expressed by B is strongly supported by the underlying programming skill required to solve A.

A score close to 0 means that B does not accurately represent the skill required to solve A, or that B introduces important programming capabilities, concepts, procedures, or Java features that are not required by A.

Evaluate the underlying programming meaning and capability rather than surface-level word overlap.

Important principles:

1. UNDERLYING SKILL

   Focus on the programming capability required to solve or understand exercise A, rather than simply describing the programming topic or Java feature appearing in A.

   For example, if A asks the learner to find the largest value in an array, the underlying skill may be "Traverse an array while maintaining a running maximum" rather than simply "Use arrays."

2. GENERALIZATION

   Skill B should represent a more general form of the underlying skill required by A. Exercise-specific details such as variable names, particular input values, specific objects, class names, array contents, or particular problem contexts may be abstracted away.

   For example:

   Exercise A:
   Find the largest value in an integer array.

   Skill B:
   Traverse a sequence while maintaining a running maximum.

   B generalizes the underlying operation while removing the exercise-specific details.

3. IMPLICIT SKILLS

   The underlying skill does not need to be explicitly stated in exercise A. Infer a skill when it is genuinely required to solve the exercise.

   For example, if A asks a learner to create subclasses that provide their own implementations of an inherited method, the exercise may implicitly require understanding method overriding and specialized subclass behavior even if the term "polymorphism" is not explicitly used.

   However, do NOT infer advanced programming knowledge merely because it could theoretically be useful.

4. SEMANTIC MATCHING

   Do not rely on word overlap.

   A and B may use different terminology while expressing the same underlying programming capability.

   For example:

   Exercise A:
   Count how many times each word occurs in a sentence.

   Skill B:
   Maintain element frequencies using a mapping structure.

   These express the same underlying capability even though the wording is different.

5. FAITHFULNESS

   Every important capability expressed by B should be supported by the skill required to solve A.

   Reduce the score if B introduces programming concepts, algorithms, data structures, procedures, Java features, or capabilities that are not required by A.

   For example:

   Exercise A:
   Find the maximum value in an array.

   Skill B:
   Use dynamic programming to optimize overlapping subproblems.

   B introduces a programming technique that is not required by A, so the score should be very low.

6. ABSTRACTION IS EXPECTED

   B may omit exercise-specific values, objects, variable names, input examples, class names, or other details.

   Such omissions should not reduce the score as long as the generalized skill remains faithful to the underlying programming skill required by A.

7. DO NOT CONFUSE TOPIC WITH SKILL

   A programming topic, programming paradigm, data structure, or Java feature is not automatically a meaningful Knowledge Component.

   B should describe a meaningful capability that a learner can use to solve Java programming exercises.

   For example:

   Exercise A:
   Find the maximum value in an array.

   Skill B:
   Use arrays.

   B is related to A, but it does not capture the important underlying operation as well as:

   "Traverse an array while maintaining a running maximum."

   Therefore, the more meaningful KC should receive a higher score when it better represents the underlying skill.

8. DO NOT EVALUATE COMPLETENESS

   This metric evaluates whether B is a faithful generalization of the underlying skill.

   It should not strongly penalize B simply because it omits some aspects of the skill required by A.

   Coverage of important capabilities is evaluated separately by informativeness.

   For example, if A requires traversing an array and maintaining a running maximum, B may capture the central capability of maintaining a running maximum over a sequence without explicitly mentioning every implementation detail.

9. AVOID OVER-GENERALIZATION

   A skill such as "solve programming exercises," "write Java programs," or "apply programming knowledge" may technically encompass A, but it is too general to meaningfully represent the underlying skill.

   Reduce the score when B loses the essential programming capability being generalized.

   For example:

   Exercise A:
   Find the maximum value in an integer array.

   Skill B:
   Solve programming problems using computational reasoning.

   B is related to the exercise at a very broad level but does not meaningfully preserve the specific underlying capability.

10. KNOWLEDGE COMPONENT VERSUS PROGRAMMING TOPIC

   The goal is to evaluate B as a Knowledge Component or transferable programming skill.

   A KC should represent something a learner can apply across multiple programming exercises.

   Examples include:

   - Traverse a sequence while maintaining an accumulated result.
   - Traverse an array while maintaining a running maximum.
   - Search for an element satisfying a condition.
   - Filter elements according to a condition.
   - Count element frequencies using a mapping structure.
   - Sort elements according to an ordering criterion.
   - Use recursion to decompose a problem into smaller instances of the same problem.
   - Compare corresponding or symmetric elements in a sequence.
   - Parse and process structured input.
   - Validate input against specified conditions.
   - Use conditional logic to handle alternative cases.
   - Use sentinel-controlled iteration to process an unknown number of inputs.
   - Implement an interface by providing concrete method implementations.
   - Override inherited methods to provide specialized subclass behavior.
   - Use inheritance and polymorphism to specialize object behavior.
   - Handle exceptional conditions using Java exception mechanisms.
   - Decompose a programming problem into smaller computational steps.

   A topic such as "arrays," "loops," "Java," or "object-oriented programming" may be relevant, but it should not automatically be considered the underlying KC when a more meaningful capability can be identified.

11. JAVA-SPECIFIC KNOWLEDGE

   Java-specific knowledge should be considered when the Java feature itself is part of the knowledge being tested.

   For example:

   Exercise A:
   Implement an interface named Printable in a Java class.

   Skill B:
   Implement a Java interface by providing concrete method implementations.

   B is a strong generalization because the Java interface mechanism itself is part of the required knowledge.

   However, do not add "Java" to every programming skill simply because exercise A is written in Java.

12. DO NOT INFER IMPLEMENTATION-SPECIFIC KNOWLEDGE

   Do not assume that a learner must use a particular implementation when multiple valid implementations can solve the exercise.

   For example, if A asks the learner to determine whether an element exists in a collection, do not assume that the learner must use a HashSet unless the exercise specifically requires it.

   The generalized skill should capture the underlying capability rather than one arbitrary implementation.

13. DO NOT INFER UNNECESSARILY ADVANCED SKILLS

   Do not infer advanced algorithms, optimization techniques, design patterns, data structures, concurrency mechanisms, or language features unless they are genuinely required by A.

   For example, an exercise involving sorting does not automatically imply knowledge of quicksort, mergesort, heapsort, or algorithmic optimization unless the exercise requires one of them.

14. SEMANTIC GENERALIZATION, NOT EXERCISE RESTATEMENT

   B should generalize the underlying programming skill rather than merely restating exercise A using slightly different words.

   For example:

   Exercise A:
   Find the largest number in an integer array.

   Weak B:
   Find the largest value in an array.

   Better B:
   Traverse a sequence while maintaining a running maximum.

   The second formulation captures a transferable programming capability rather than simply restating the exercise.

15. DO NOT CONFUSE FAITHFULNESS WITH INFORMATIVENESS

   A skill may be a faithful generalization even if it does not capture every important detail of the exercise.

   For example:

   Exercise A:
   Read an array of integers and find the largest positive value.

   Skill B:
   Traverse a sequence while maintaining a running maximum.

   B may be faithful to an important underlying operation, even though it omits the positive-value filtering condition.

   The omission affects coverage/informativeness more than faithfulness.

16. AVOID UNSUPPORTED ADDITIONAL CAPABILITIES

   If B combines several programming capabilities, each capability must be supported by A.

   For example:

   Exercise A:
   Find the maximum value in an array.

   Skill B:
   Traverse an array, maintain a running maximum, and sort the array before finding the maximum.

   B contains an additional sorting capability that is not required by A. Therefore, the score should be reduced.

17. MOST IMPORTANTLY

   Ask:

   "What programming skill or Knowledge Component is required to solve exercise A, and does skill B faithfully represent a generalized version of that capability?"

Use the following scale as guidance:

1.0:
B is an accurate and well-grounded generalization of the underlying programming skill required to solve A. It captures the essential capability without introducing unsupported programming capabilities.

0.8-0.99:
B is a strong generalization of the required programming skill, with only minor over-generalization, abstraction, or differences.

0.6-0.79:
B captures a substantial part of the underlying programming skill but is somewhat broader, narrower, or less precise than an ideal generalization.

0.4-0.59:
B has some meaningful relationship to the underlying programming skill but misses important aspects or introduces some unsupported capabilities.

0.1-0.39:
B has only weak correspondence with the underlying programming skill, such as sharing only a broad programming topic, Java feature, or general programming activity.

0.0:
B does not represent the programming skill required to solve A or describes a fundamentally different programming capability.

Finally, output your score in the following format:

Score: [SCORE]

Below this, give your reasoning for your score.

Here are some examples:

High-Scoring Example 1 — Direct Generalization

Exercise A:
Write a Java program that reads an integer array and determines the largest value in the array.

Skill B:
Traverse a sequence while maintaining a running maximum.

Score: 1.0

Reasoning:
The underlying skill required to solve A is traversing the array while keeping track of the largest value encountered. B accurately generalizes this capability by removing the specific array values and exercise context while preserving the essential operation.

High-Scoring Example 2 — Generalizing an Implicit Skill

Exercise A:
Write a Java program that reads integers until the user enters -1 and then prints their sum.

Skill B:
Use sentinel-controlled iteration to process an unknown number of input values while maintaining an accumulated result.

Score: 1.0

Reasoning:
Solving A requires repeatedly reading input until a sentinel value is encountered and accumulating the relevant values. B accurately generalizes both the sentinel-controlled iteration and accumulation required by the exercise.

High-Scoring Example 3 — Different Exercise Details, Same Skill

Exercise A:
Write a Java program that calculates the total price of all products in a shopping cart.

Skill B:
Iterate through a collection while maintaining an accumulated result.

Score: 0.9

Reasoning:
The underlying skill required by A is iterating through the collection of products and accumulating their prices. B generalizes this capability while removing the specific shopping-cart context and product details.

High-Scoring Example 4 — Java-Specific Skill

Exercise A:
Create a Java interface named Printable and implement it in a class Report.

Skill B:
Implement a Java interface by providing concrete method implementations.

Score: 1.0

Reasoning:
The exercise directly requires implementing an interface and providing concrete implementations of its required methods. B accurately generalizes this Java-specific capability without retaining the particular interface or class names.

High-Scoring Example 5 — Object-Oriented Programming

Exercise A:
Create a superclass Animal with a method makeSound(), then create subclasses Dog and Cat that override makeSound() with their own implementations.

Skill B:
Override inherited methods to provide specialized subclass behavior.

Score: 1.0

Reasoning:
The underlying skill required by A is overriding an inherited method in subclasses to provide specialized behavior. B directly captures this transferable object-oriented programming capability.

High-Scoring Example 6 — Frequency Counting

Exercise A:
Write a Java program that counts how many times each integer occurs in an array.

Skill B:
Maintain element frequencies using a mapping structure.

Score: 1.0

Reasoning:
The exercise requires associating each element with its count and updating that count as elements are processed. B accurately generalizes the frequency-counting capability.

Mid-Scoring Example 1 — Broad Programming Skill

Exercise A:
Write a Java program that finds the largest value in an integer array.

Skill B:
Process data using iteration.

Score: 0.55

Reasoning:
The exercise does require processing array data through iteration, so B has a meaningful relationship to the underlying skill. However, B is substantially broader and does not preserve the important capability of maintaining a running maximum.

Mid-Scoring Example 2 — Partially Correct Generalization

Exercise A:
Write a Java program that finds the largest positive value in an integer array.

Skill B:
Traverse a sequence while maintaining a running maximum.

Score: 0.75

Reasoning:
B captures the important maximum-finding operation and traversal required by A. However, it omits the additional condition that only positive values should be considered. The omission should reduce the score somewhat, but it does not make B unrelated to the underlying skill.

Mid-Scoring Example 3 — Topic Instead of Meaningful KC

Exercise A:
Write a Java program that counts the frequency of each word in a sentence.

Skill B:
Use Java collections.

Score: 0.45

Reasoning:
Collections may be involved in solving A, so B has some relationship to the exercise. However, "use Java collections" is a broad topic rather than a meaningful representation of the underlying frequency-counting capability.

Low-Scoring Example 1 — Wrong Skill

Exercise A:
Write a Java program that finds the maximum value in an array.

Skill B:
Implement an interface using polymorphism.

Score: 0.0

Reasoning:
The underlying skill required by A concerns traversing an array and determining a maximum value. Interface implementation and polymorphism represent different programming capabilities that are not required by A.

Low-Scoring Example 2 — Unsupported Advanced Technique

Exercise A:
Write a Java program that calculates the sum of all values in an array.

Skill B:
Use dynamic programming to optimize overlapping subproblems.

Score: 0.0

Reasoning:
The exercise requires traversal and accumulation but does not require dynamic programming or optimization of overlapping subproblems. B introduces unsupported programming capabilities.

Low-Scoring Example 3 — Too General

Exercise A:
Write a Java program that determines whether a string is a palindrome.

Skill B:
Solve programming problems using computational reasoning.

Score: 0.2

Reasoning:
B is broad enough to encompass the exercise, but it does not meaningfully represent the underlying skill. It loses the important capability of comparing characters or corresponding positions to determine whether the string is symmetric.

Low-Scoring Example 4 — Wrong Java Feature

Exercise A:
Write a Java program that calculates the average of values stored in an array.

Skill B:
Use Java exception handling to recover from runtime errors.

Score: 0.0

Reasoning:
Exception handling is not required by the exercise. The proposed skill describes a different Java capability.

Low-Scoring Example 5 — Over-Generalization

Exercise A:
Write a Java program that recursively calculates the factorial of a number.

Skill B:
Write Java programs.

Score: 0.1

Reasoning:
B technically encompasses the exercise but is far too general to meaningfully represent the underlying skill. It loses the essential capability of using recursion to decompose a problem into smaller instances of the same problem.

Low-Scoring Example 6 — Unsupported Additional Capability

Exercise A:
Write a Java program that finds the maximum value in an array.

Skill B:
Traverse an array while maintaining a running maximum and sort the array before determining the maximum.

Score: 0.55

Reasoning:
The first part of B accurately represents the underlying maximum-finding capability, but sorting the array is not required by A. The unsupported additional capability should substantially reduce the score.

"""
    return evalPrompt2


def get_eval_prompt_faithfulness():
    evalPrompt_faithfulness = """
You will be given a Java programming exercise A and a programming skill or Knowledge Component (KC) B that is proposed to describe a capability tested by exercise A. Your task is to determine how FAITHFUL skill B is to exercise A.

Faithfulness measures PRECISION: whether exercise A actually tests, requires, or demonstrates the programming capability described by skill B.

In other words, ask:

"Does a learner need to possess or apply the capability described by B in order to successfully solve or complete exercise A?"

A skill is faithful when the exercise genuinely tests that skill, even if the skill is implicit in the exercise rather than explicitly stated.

Faithfulness measures ONE thing: whether the skill B is supported by what exercise A actually requires.

It does NOT measure how many of the important skills in A are captured by B. That is an INFORMATIVENESS concern.

Important:

* The ONLY reason to score below 1.0 is that skill B describes a capability that exercise A does not actually test, require, or meaningfully demonstrate.

* Implicit programming skills count. If solving A genuinely requires applying an algorithm, data structure, programming construct, computational procedure, Java language feature, or reasoning pattern, B can receive full credit for describing that capability even if A does not explicitly name it.

* Do NOT require B to use the same wording as A. Evaluate the underlying programming capability, not lexical overlap.

* Do NOT penalize B for being more abstract than the skill required by A. Generalization is expected when constructing a programming skill or KC hierarchy.

* Do NOT penalize B for omitting details of A. Faithfulness asks whether B is supported by A, not whether B captures everything A tests.

* Do NOT penalize B merely because it is broad or generic. A broad programming skill can still be fully faithful if the exercise genuinely tests that capability.

* However, a broad statement should not receive full credit if it contains additional programming capabilities that the exercise does not test.

* Topic overlap alone is NOT sufficient. Two programming concepts may belong to the same programming domain without one being a skill actually tested by the exercise.

* Do NOT infer a skill merely because it is commonly associated with the programming topic. The capability must be genuinely required to solve or complete A.

* If B contains multiple distinct capabilities, evaluate each capability separately. Reduce the score when some of those capabilities are not supported by A.

* Do not evaluate how informative, specific, or useful B is as a description of A. Those properties are evaluated separately.

* Do not evaluate whether B captures all important skills required by A. That is the purpose of INFORMATIVENESS.

* Distinguish between the programming topic or Java feature appearing in an exercise and the actual skill required to solve it. "Arrays," "loops," "Java," or "object-oriented programming" is not necessarily the Knowledge Component being tested.

* The exercise may test an implicit programming skill. For example, an exercise asking the learner to find the largest value in an array may implicitly test the ability to traverse the array while maintaining a running maximum, even if the exercise never explicitly states this procedure.

* Java-specific knowledge should be treated as a skill when the Java feature itself is genuinely part of what the exercise tests. Do not assume that every Java exercise tests the same Java-specific knowledge simply because it is written in Java.

* Do not assume a particular implementation unless exercise A requires it. If multiple valid implementations can solve A, evaluate the underlying capability rather than an arbitrary implementation choice.

* Do not infer unnecessarily advanced algorithms, optimization techniques, design patterns, data structures, concurrency mechanisms, or Java features merely because they could theoretically be used to solve A.

* A programming skill can be faithful even when it is broader than the exact operation used in A. Broadness alone should not reduce faithfulness.

* A skill can be faithful even if it does not capture all the important aspects of A. Missing coverage is an informativeness issue, not necessarily a faithfulness issue.

* A skill should receive a lower score when it contains unsupported additional requirements, even if another part of the skill is genuinely tested by A.

* Faithfulness concerns whether B is TRUE of the capability required by A. It does not concern whether B is the BEST or MOST SPECIFIC possible description of A.

Examples of meaningful Knowledge Components include:

- Traverse an array while maintaining a running maximum.
- Traverse a collection while maintaining an accumulated result.
- Search for an element satisfying a condition.
- Filter elements according to a condition.
- Count element frequencies using a mapping structure.
- Sort elements according to an ordering criterion.
- Use recursion to decompose a problem into smaller instances of the same problem.
- Compare corresponding or symmetric elements in a sequence.
- Parse and process structured input.
- Validate input against specified conditions.
- Use conditional logic to handle alternative cases.
- Use sentinel-controlled iteration to process an unknown number of inputs.
- Implement an interface by providing concrete method implementations.
- Override inherited methods to provide specialized subclass behavior.
- Use inheritance and polymorphism to specialize object behavior.
- Handle exceptional conditions using Java exception mechanisms.
- Decompose a programming problem into smaller computational steps.

Examples of programming topics that should NOT automatically be treated as meaningful KCs:

- Java
- Arrays
- Loops
- Variables
- Classes
- Objects
- Collections
- Object-oriented programming
- Data structures
- Algorithms

These topics may be relevant to an exercise, but faithfulness should be evaluated according to whether the proposed capability is actually tested by the exercise.

Scoring guidance:

* 0.85-1.0 (Highly Faithful / High Precision): Exercise A clearly tests or requires the capability described by B. Every important capability stated in B is grounded in what A requires. B may be abstract or generalized without reducing faithfulness.

* 0.6-0.85 (Mostly Faithful): B is largely supported by A, but one capability or aspect of B is only weakly supported, partially required, or somewhat broader than what A actually tests.

* 0.3-0.6 (Partially Faithful): B contains a meaningful capability that A tests, but also contains one or more additional capabilities that A does not clearly test or require.

* 0.0-0.3 (Not Faithful / Low Precision): B mainly describes a capability that A does not test, require, or meaningfully demonstrate. Shared programming topic, Java vocabulary, or general programming context alone is not sufficient.

## Examples

Exercise A:
Write a Java program that reads an integer array and finds the largest value in the array.

Skill B:
Traverse an array while maintaining a running maximum.

Score: 1.0

Reason:
To solve A, the learner must examine the elements of the array and maintain the largest value encountered. Although A does not explicitly state the procedure "maintain a running maximum," this capability is genuinely required. Therefore B is highly faithful.

---

Exercise A:
Write a Java program that reads an integer array and finds the largest value in the array.

Skill B:
Process data using iteration.

Score: 1.0

Reason:
A requires repeatedly processing the elements of the array. B describes a broad capability that is genuinely used by the exercise. Although B is less informative and less specific than "maintain a running maximum," its broadness does not reduce faithfulness.

---

Exercise A:
Write a Java program that reads an integer array and finds the largest value in the array.

Skill B:
Use arrays.

Score: 1.0

Reason:
The exercise genuinely requires working with an array. Therefore the capability described by B is supported by A. B is extremely broad and may be less informative than a more meaningful KC, but faithfulness concerns whether the capability is supported, not whether it is the best description.

---

Exercise A:
Write a Java program that reads an integer array and finds the largest value in the array.

Skill B:
Use dynamic programming to optimize overlapping subproblems.

Score: 0.05

Reason:
A does not require dynamic programming, overlapping-subproblem analysis, or optimization using dynamic programming. The fact that both are programming concepts does not make B faithful to A.

---

Exercise A:
Write a Java program that reads an integer array and calculates its sum.

Skill B:
Traverse a sequence while maintaining an accumulated result.

Score: 1.0

Reason:
The exercise requires processing the array elements and maintaining an accumulated sum. B describes this underlying capability at a generalized level and is therefore highly faithful.

---

Exercise A:
Write a Java program that reads an integer array and calculates its sum.

Skill B:
Traverse an array, maintain an accumulated result, and sort the array before calculating the sum.

Score: 0.5

Reason:
A requires traversal and accumulation, but it does not require sorting the array. B therefore contains genuinely supported capabilities as well as an unsupported additional capability. The unsupported sorting requirement reduces faithfulness.

---

Exercise A:
Write a Java program that counts how many times each integer occurs in an array.

Skill B:
Maintain element frequencies using a mapping structure.

Score: 1.0

Reason:
The exercise requires associating each element with a count and updating that count as elements are processed. A mapping structure is a natural implementation of this capability, and B accurately describes the frequency-counting skill being tested.

---

Exercise A:
Write a Java program that determines whether a string is a palindrome.

Skill B:
Compare corresponding or symmetric elements in a sequence to determine whether they satisfy a required relationship.

Score: 1.0

Reason:
Determining whether a string is a palindrome requires comparing corresponding or symmetric characters. B generalizes this capability from strings to sequences without introducing unsupported requirements.

---

Exercise A:
Write a Java program that determines whether a string is a palindrome.

Skill B:
Use recursion to solve string-processing problems.

Score: 0.4

Reason:
A can potentially be solved recursively, but recursion is not necessarily required by the exercise. String processing is relevant, but B introduces a recursion requirement that A does not necessarily test. Therefore faithfulness is only partial.

---

Exercise A:
Create a Java interface named Printable and implement it in a class Report.

Skill B:
Implement a Java interface by providing concrete method implementations.

Score: 1.0

Reason:
The exercise directly tests the ability to implement an interface in a class and provide the required concrete methods. B accurately describes the Java-specific capability being tested.

---

Exercise A:
Create a Java interface named Printable and implement it in a class Report.

Skill B:
Use object-oriented programming.

Score: 1.0

Reason:
The exercise genuinely requires object-oriented programming concepts involving interfaces and classes. B is extremely broad and therefore may be less informative or specific, but the broad capability is genuinely supported by A. Faithfulness remains high.

---

Exercise A:
Create a Java subclass that overrides a method inherited from a superclass.

Skill B:
Override inherited methods to provide specialized subclass behavior.

Score: 1.0

Reason:
The exercise directly requires overriding an inherited method and providing specialized behavior in the subclass. B accurately describes the capability tested by A.

---

Exercise A:
Create a Java subclass that overrides a method inherited from a superclass.

Skill B:
Use Java multithreading to coordinate concurrent objects.

Score: 0.0

Reason:
The exercise does not require multithreading, concurrency, or synchronization. The fact that both are Java programming concepts does not make B faithful to A.

---

Exercise A:
Write a Java program that reads numbers until the user enters -1 and then prints their sum.

Skill B:
Use iteration to process a sequence of input values.

Score: 1.0

Reason:
A requires repeatedly reading input until a sentinel value is encountered. Therefore iterative processing of a sequence of inputs is genuinely required. B is broader than the exact sentinel-controlled procedure but remains faithful.

---

Exercise A:
Write a Java program that reads numbers until the user enters -1 and then prints their sum.

Skill B:
Use sentinel-controlled iteration to process an unknown number of inputs.

Score: 1.0

Reason:
The exercise explicitly requires repeated input processing until a sentinel value is entered. B accurately describes this capability.

---

Exercise A:
Write a Java program that reads numbers until the user enters -1 and then prints their sum.

Skill B:
Use recursion to process an unknown number of inputs.

Score: 0.3

Reason:
The exercise can potentially be implemented recursively, but recursion is not required by A. The input-processing aspect is relevant, but the recursion requirement is unsupported. Therefore faithfulness is low to partial.

---

Exercise A:
Write a Java program that sorts an integer array in ascending order.

Skill B:
Sort a collection according to an ordering criterion.

Score: 1.0

Reason:
A directly requires arranging elements according to an ascending ordering criterion. B accurately generalizes this sorting capability from an integer array to collections more generally.

---

Exercise A:
Write a Java program that sorts an integer array in ascending order.

Skill B:
Use quicksort to sort the array.

Score: 0.6

Reason:
The exercise requires sorting, but it does not necessarily require quicksort. The sorting capability is supported, but the specific quicksort requirement is an unsupported implementation-specific addition. Therefore faithfulness is reduced.

---

Exercise A:
Write a Java program that calculates the average of values stored in an array.

Skill B:
Solve quantitative programming problems.

Score: 1.0

Reason:
A genuinely requires quantitative computation, so B is faithful. B is extremely broad and may have low informativeness and specificity, but those properties are evaluated separately.

---

Exercise A:
Write a Java program that calculates the average of values stored in an array.

Skill B:
Use Java exception handling to recover from invalid input.

Score: 0.1

Reason:
The exercise concerns calculating an average and does not necessarily require exception handling or invalid-input recovery. The proposed skill is therefore not a faithful description of the capability actually tested.

---

Exercise A:
Write a Java program that validates whether an input password satisfies minimum length and character requirements.

Skill B:
Validate input against specified conditions.

Score: 1.0

Reason:
The exercise directly requires checking input against specified conditions. B generalizes the validation capability while removing the password-specific details.

---

Exercise A:
Write a Java program that validates whether an input password satisfies minimum length and character requirements.

Skill B:
Use cryptographic hashing to securely store passwords.

Score: 0.0

Reason:
The exercise requires validation of password requirements but does not require cryptographic hashing or secure password storage. B describes a different capability.

---

Exercise A:
Write a Java program that recursively calculates the factorial of a number.

Skill B:
Use recursion to decompose a problem into smaller instances of the same problem.

Score: 1.0

Reason:
The exercise directly requires recursive decomposition: factorial is calculated through smaller factorial instances until a base case is reached. B accurately describes this underlying programming capability.

---

Exercise A:
Write a Java program that recursively calculates the factorial of a number.

Skill B:
Use recursion, dynamic programming, and memoization to optimize recursive computation.

Score: 0.4

Reason:
Recursion is genuinely required by A, but dynamic programming and memoization are not necessarily required. B therefore contains an unsupported additional set of capabilities, reducing faithfulness.

---

Exercise A:
Write a Java program that searches a list of student records for a student with a specified ID.

Skill B:
Search a collection for an element satisfying a specified condition.

Score: 1.0

Reason:
The exercise requires examining collection elements and identifying the element whose ID satisfies the search condition. B accurately generalizes this capability.

---

Exercise A:
Write a Java program that searches a list of student records for a student with a specified ID.

Skill B:
Use a HashMap to perform constant-time lookup.

Score: 0.5

Reason:
A requires searching for a matching student, but it does not necessarily require a HashMap or constant-time lookup. The search capability is supported, but the specific data structure and complexity requirement are not necessarily tested.

---

Exercise A:
Write a Java program that finds the maximum value in an array.

Skill B:
Use object-oriented design patterns to structure computational problems.

Score: 0.05

Reason:
The exercise does not require object-oriented design patterns. The fact that the program is written in Java does not imply that design patterns are being tested.

---

Exercise A:
Write a Java program that reads an array of integers and finds the largest positive value.

Skill B:
Traverse a sequence while maintaining a running maximum.

Score: 1.0

Reason:
A requires traversal and maintaining the maximum among the values considered. The additional positive-value filtering condition is not captured by B, but omission of that detail does not reduce faithfulness. It affects informativeness rather than whether B describes a genuinely required capability.

---

Exercise A:
Write a Java program that reads an array of integers and finds the largest positive value.

Skill B:
Filter positive values, traverse the sequence, maintain a running maximum, and sort the values before finding the maximum.

Score: 0.4

Reason:
Filtering positive values, traversal, and maintaining a running maximum are supported by A. However, sorting is not required by the exercise. The unsupported sorting capability reduces faithfulness.

---

Exercise A:
Write a Java program that processes a list of objects and prints those satisfying a given condition.

Skill B:
Filter elements according to a specified condition.

Score: 1.0

Reason:
The exercise directly requires selecting elements according to a condition. B accurately describes the underlying filtering capability.

---

Exercise A:
Write a Java program that processes a list of objects and prints those satisfying a given condition.

Skill B:
Use Java Streams and lambda expressions to filter elements.

Score: 0.6

Reason:
The underlying filtering capability is required, but Java Streams and lambda expressions are not necessarily required because the exercise can be implemented using other mechanisms such as a loop and conditional statement. B therefore introduces implementation-specific Java knowledge that is not necessarily tested by A.

---

Exercise A:
Write a Java program that reads student scores and calculates the average score.

Skill B:
Use loops to process data.

Score: 1.0

Reason:
The exercise genuinely requires processing multiple student scores, and iteration is a valid and natural way to perform this computation. B is broad and less informative than "iterate through a collection while maintaining an accumulated sum," but broadness does not reduce faithfulness.

---

Exercise A:
Write a Java program that reads student scores and calculates the average score.

Skill B:
Use inheritance and polymorphism to specialize student objects.

Score: 0.0

Reason:
The exercise does not require inheritance or polymorphism. These are unrelated programming capabilities in the context of the stated task.

---

Exercise A:
Write a Java program that reads two integers and prints their sum.

Skill B:
Perform arithmetic operations on numerical values.

Score: 1.0

Reason:
The exercise directly requires an arithmetic operation on numerical values. B generalizes the specific addition operation while preserving the underlying capability.

---

Exercise A:
Write a Java program that reads two integers and prints their sum.

Skill B:
Implement a recursive divide-and-conquer algorithm.

Score: 0.0

Reason:
The exercise requires only a basic arithmetic operation and does not require recursion or divide-and-conquer reasoning.

---

The final evaluation principle is:

"Does exercise A genuinely require, test, demonstrate, or meaningfully imply the programming capability described by skill B?"

If YES, the skill can be highly faithful even if it is broad, abstract, implicit, or incomplete.

If NO, the skill should receive a lower score even if it shares programming terminology, belongs to the same Java topic, or could theoretically be used to solve the exercise.

Remember:

Faithfulness = PRECISION, not COVERAGE.

Do not reduce the faithfulness score simply because B omits some important capability from A.

Do reduce the faithfulness score when B introduces capabilities that A does not actually require.
"""
    return evalPrompt_faithfulness


def get_eval_prompt_informativeness():
    evalPrompt_informativeness = """
You will be given a Java programming exercise A and a knowledge component or programming skill B that is proposed to describe or generalize the knowledge and skills required to solve exercise A. Your task is to determine how INFORMATIVE knowledge component B is about exercise A.

Informativeness measures RECALL: whether B captures the important programming knowledge, concepts, skills, reasoning patterns, and capabilities that exercise A actually requires.

In other words, ask:

"What are the important programming knowledge components and skills required to solve or correctly answer exercise A, and how much of those important components does B capture?"

A highly informative knowledge component should capture the important underlying programming capabilities required by the exercise, rather than merely describing its topic, programming language, or general type of activity.

Important:

* First identify the IMPORTANT PROGRAMMING KNOWLEDGE COMPONENTS and SKILLS tested by exercise A.

* Focus on the core knowledge and capabilities that a learner needs to successfully solve, understand, trace, debug, or complete A.

* Important components may include:

  * programming concepts
  * Java language features
  * syntax and language constructs
  * variables, constants, and data types
  * operators and expressions
  * conditional statements
  * loops and iteration
  * methods and method invocation
  * parameters and return values
  * arrays and strings
  * collections and data structures
  * classes and objects
  * encapsulation, inheritance, polymorphism, and abstraction
  * interfaces and abstract classes
  * constructors and object initialization
  * access modifiers
  * static and instance members
  * exception handling
  * generics
  * Java APIs and library usage
  * input/output
  * file handling
  * recursion
  * algorithms and algorithmic reasoning
  * searching, sorting, and data manipulation
  * program tracing and execution reasoning
  * debugging and error identification
  * understanding compiler/runtime behavior
  * identifying or predicting program output
  * translating requirements into code
  * decomposing a problem into programming steps
  * selecting appropriate data structures or programming constructs
  * reasoning about relationships between program components
  * testing and validating program behavior
  * other meaningful programming knowledge or capabilities.

* Consider the underlying knowledge component even when it is implicit in the wording of A.

* Evaluate SEMANTIC COVERAGE, not lexical overlap. B does not need to use the same words, syntax, class names, variable names, or examples as A.

* B may abstract away exercise-specific details such as variable names, numerical values, particular input values, class names, method names, or specific program contexts without losing informativeness.

* Do NOT require B to reproduce the exact exercise. We are evaluating whether B captures the underlying programming knowledge and capabilities required by the exercise.

* Do NOT reward B merely because it mentions the same programming topic or Java feature. Topic overlap is not sufficient if B fails to capture the actual knowledge or capability required by A.

* Penalize VAGUENESS when B is so broad that it captures little of the important programming knowledge required by A.

* Statements such as "write Java programs," "solve programming problems," "understand Java," or "use object-oriented programming" may be technically related to A but should receive low informativeness when they fail to identify the important knowledge component or capability being tested.

* If A requires multiple important knowledge components, B should capture the important components to receive a high score.

* Do NOT penalize B for omitting trivial, incidental, or exercise-specific details.

* Do NOT evaluate whether B contains unsupported knowledge or skills. That is the purpose of FAITHFULNESS.

* Do NOT evaluate whether B is the most specific possible generalization. Evaluate how much of the important knowledge and capability tested by A is retained by B.

* A knowledge component can be highly faithful but poorly informative. For example, a very broad component such as "use Java programming concepts" may genuinely apply to A but fail to capture the important programming knowledge required by the exercise.

* When judging informativeness, focus on the knowledge and capabilities that are CENTRAL to successfully solving A, not every piece of information appearing in the exercise.

* Distinguish between knowledge that is merely present in the exercise and knowledge that is actually required to solve it.

* If a programming construct is incidental rather than central to solving A, do not treat it as an important component that B must capture.

* When an exercise can be solved using multiple steps, consider the important knowledge components underlying those steps rather than simply counting the number of keywords or operations.

* Do not assume that every Java-specific feature appearing in the code is an important knowledge component. Determine whether understanding or applying that feature is necessary for solving the exercise.

* If A is primarily a code-tracing, debugging, output-prediction, or code-completion exercise, evaluate the knowledge required for that particular task rather than assuming that the learner must know how to write an entire program from scratch.

Base the score on the proportion of the IMPORTANT PROGRAMMING KNOWLEDGE COMPONENTS and CAPABILITIES required by A that are captured or implied by B.

Scoring guidance:

* 0.85-1.0 (Excellent Coverage / High Recall):
  B captures essentially all of the important programming knowledge components and skills required by A. It represents the central programming concepts, language constructs, relationships, reasoning, or procedures needed to solve the exercise, while appropriately abstracting away incidental details.

* 0.6-0.85 (Good Coverage):
  B captures the main knowledge component or skill tested by A but misses one or more important components or meaningful aspects of the reasoning required.

* 0.3-0.6 (Partial / Vague Coverage):
  B captures a broad aspect of what A requires but misses much of the important programming knowledge or capability. It may correctly identify the general programming activity or topic while remaining too vague.

* 0.0-0.3 (Very Low Coverage / Mismatched):
  B captures little or none of the important programming knowledge or skills tested by A, or describes an essentially different capability.

Examples:

---

Exercise A:
Write a Java program that reads an integer and prints whether the number is positive, negative, or zero.

Knowledge Component B:
Use conditional statements to classify a value based on mutually exclusive conditions.

Score: 1.0

Reason:
B captures the central programming capability required by A: using conditional logic to classify an input according to different conditions. The specific input type and output messages are incidental details.

---

Exercise A:
Write a Java program that reads an integer and prints whether the number is positive, negative, or zero.

Knowledge Component B:
Write Java programs that process user input.

Score: 0.55

Reason:
B captures the general input-processing aspect of the exercise but misses the central knowledge component of using conditional logic to classify the value.

---

Exercise A:
Write a Java program that reads an integer and prints whether the number is positive, negative, or zero.

Knowledge Component B:
Solve Java programming problems.

Score: 0.2

Reason:
B captures only the very broad activity of solving a programming problem. It does not capture the important conditional reasoning required by A.

---

Exercise A:
Write a Java program that reads an integer and prints whether the number is positive, negative, or zero.

Knowledge Component B:
Use inheritance to implement relationships between Java classes.

Score: 0.05

Reason:
B describes a Java programming concept, but inheritance is not a capability required to solve this exercise.

---

Exercise A:
Write a Java method that receives an array of integers and returns the largest value in the array.

Knowledge Component B:
Traverse an array while maintaining and updating a variable that represents the current maximum value.

Score: 1.0

Reason:
B captures the central algorithmic knowledge required by A: iterating through the array and maintaining the current maximum.

---

Exercise A:
Write a Java method that receives an array of integers and returns the largest value in the array.

Knowledge Component B:
Use arrays to process collections of integers.

Score: 0.55

Reason:
B captures the use of arrays and integer data but does not capture the important algorithmic capability of finding and maintaining the maximum value.

---

Exercise A:
Write a Java method that receives an array of integers and returns the largest value in the array.

Knowledge Component B:
Perform computations on numerical data.

Score: 0.3

Reason:
B is broadly related to the exercise but is too vague. It does not identify array traversal or the algorithm for determining the maximum value.

---

Exercise A:
Given a Java class with a private field and public getter and setter methods, explain how the field can be accessed from outside the class.

Knowledge Component B:
Apply encapsulation by controlling access to object state through public methods.

Score: 1.0

Reason:
B captures the central object-oriented knowledge component tested by A: encapsulation and controlled access to private state through public methods.

---

Exercise A:
Given a Java class with a private field and public getter and setter methods, explain how the field can be accessed from outside the class.

Knowledge Component B:
Understand object-oriented programming.

Score: 0.3

Reason:
B identifies the general topic but does not capture the specific knowledge component of encapsulation and controlled access to private fields.

---

Exercise A:
Predict the output of a Java program containing a for loop that increments a variable from 0 to 4 and prints the variable during each iteration.

Knowledge Component B:
Trace iterative control flow to determine how a loop variable changes across successive iterations and predict the resulting output.

Score: 1.0

Reason:
B captures the central capability required by A: tracing loop execution, tracking the loop variable, and determining the resulting output.

---

Exercise A:
Predict the output of a Java program containing a for loop that increments a variable from 0 to 4 and prints the variable during each iteration.

Knowledge Component B:
Understand Java loops.

Score: 0.45

Reason:
B identifies the relevant topic but does not sufficiently capture the program-tracing capability required to determine the exact output.

---

Exercise A:
Predict the output of a Java program containing a for loop that increments a variable from 0 to 4 and prints the variable during each iteration.

Knowledge Component B:
Predict program output by tracing execution.

Score: 0.9

Reason:
B captures the important program-tracing and output-prediction capability. It abstracts away the specific loop construct, which is appropriate if the exercise is being generalized at the level of execution tracing.

---

Exercise A:
Write a Java program that stores student names and scores in an ArrayList and prints the names of students whose scores are above 80.

Knowledge Component B:
Use a collection to store multiple records and iterate through the records while filtering elements according to a condition.

Score: 1.0

Reason:
B captures the important capabilities required by A: storing multiple records in a collection, iterating over them, accessing relevant data, and filtering based on a condition.

---

Exercise A:
Write a Java program that stores student names and scores in an ArrayList and prints the names of students whose scores are above 80.

Knowledge Component B:
Use ArrayLists in Java.

Score: 0.6

Reason:
B captures the important collection-related component but misses the additional capabilities of iterating through records, accessing their values, and filtering based on a condition.

---

Exercise A:
Write a Java program that catches an exception caused by invalid integer input and displays an appropriate error message.

Knowledge Component B:
Use exception handling to detect and handle errors that may occur during program execution.

Score: 1.0

Reason:
B captures the central knowledge component required by A: using exception handling to respond to runtime errors. The specific exception type and error message are incidental details.

---

Exercise A:
Write a Java program that catches an exception caused by invalid integer input and displays an appropriate error message.

Knowledge Component B:
Write robust Java programs.

Score: 0.3

Reason:
B captures a broad goal related to the exercise but does not identify the important knowledge component of exception handling.

---

Exercise A:
Create a Java superclass Shape with a method area(), then create subclasses Circle and Rectangle that override area() to calculate their respective areas.

Knowledge Component B:
Use inheritance and method overriding to implement polymorphic behavior across related classes.

Score: 1.0

Reason:
B captures the central object-oriented capabilities required by A: inheritance, method overriding, and polymorphic behavior.

---

Exercise A:
Create a Java superclass Shape with a method area(), then create subclasses Circle and Rectangle that override area() to calculate their respective areas.

Knowledge Component B:
Use object-oriented programming to create classes.

Score: 0.4

Reason:
B captures the broad object-oriented aspect of the exercise but misses the important specific capabilities of inheritance, overriding, and polymorphism.

---

Exercise A:
Write a recursive Java method that calculates the factorial of a positive integer.

Knowledge Component B:
Use recursion by defining a function in terms of a smaller instance of the same problem and terminating at a base case.

Score: 1.0

Reason:
B captures the central recursive reasoning required by A, including recursive decomposition and termination through a base case.

---

Exercise A:
Write a recursive Java method that calculates the factorial of a positive integer.

Knowledge Component B:
Perform mathematical calculations in Java.

Score: 0.25

Reason:
B captures only the general computational aspect. It does not capture the important recursion capability required by the exercise.

---

Exercise A:
Given a Java method that searches an array for a target value, determine the index returned when the target occurs multiple times.

Knowledge Component B:
Trace a search algorithm and determine its result based on the order in which elements are examined.

Score: 0.9

Reason:
B captures the important algorithm-tracing capability required by A and appropriately abstracts away the specific array and target values.

---

Exercise A:
Fix a Java program that produces a NullPointerException when accessing a method on an object that has not been initialized.

Knowledge Component B:
Identify and correct errors caused by dereferencing a null object reference.

Score: 1.0

Reason:
B captures the central debugging knowledge required by A: recognizing null references and correcting invalid dereferencing.

---

Exercise A:
Fix a Java program that produces a NullPointerException when accessing a method on an object that has not been initialized.

Knowledge Component B:
Debug Java programs.

Score: 0.35

Reason:
B identifies the broad activity of debugging but does not capture the specific knowledge component involving null references and invalid dereferencing.

---

Important distinction between INFORMATIVENESS and FAITHFULNESS:

INFORMATIVENESS asks:
"How much of the important knowledge and skills required by exercise A does B capture?"

FAITHFULNESS asks:
"Does B actually describe knowledge or skills that are required by exercise A?"

For example:

Exercise A:
Write a Java method that returns the largest element in an integer array.

B1:
Traverse an array and maintain the maximum value seen so far.
→ High informativeness and high faithfulness.

B2:
Process an array of integers.
→ Moderate or low informativeness but high faithfulness.

B3:
Sort an array before finding its largest element.
→ Potentially low faithfulness because sorting is not required, even though it could be used as one possible approach.

B4:
Understand Java programming.
→ Very low informativeness because it is too broad, even though it is generally related to the exercise.

Do not confuse these dimensions.

For INFORMATIVENESS, focus on RECALL/COVERAGE of the important knowledge components in A.

Do not penalize a candidate B for being more general than A if it still captures the important underlying programming capability.

Do not reward a candidate B simply because it contains Java terminology or shares words with A.

The goal is to determine whether B would be a useful knowledge component for constructing a hierarchy of programming knowledge from individual Java exercises.

Return a score between 0 and 1, where higher values indicate that B captures a larger proportion of the important programming knowledge and capabilities required by A.
"""

    return evalPrompt_informativeness


def get_eval_prompt_specificity_faithfulness():
    evalPrompt_specificity_faithfulness = """
You will be given a candidate Java programming exercise A and a general programming knowledge component or skill B. Knowledge component B represents a higher-level, more general, or abstract programming capability. Your task is to determine how FAITHFUL knowledge component B is to candidate exercise A.

Faithfulness measures PRECISION: whether the programming knowledge, capability, or skill represented by B is actually tested, required, or meaningfully demonstrated by exercise A.

In other words, ask:

"Does solving, understanding, tracing, debugging, or completing exercise A genuinely require the programming knowledge or capability described by B?"

A high score means that A is a valid instance or specialization of B.

A low score means that B claims programming capabilities that A does not actually test, require, or demonstrate.

Faithfulness measures ONLY whether the capabilities claimed by B are grounded in A. It does NOT measure how completely B describes everything that A tests. Completeness or coverage is evaluated separately by INFORMATIVENESS.

Important:

* The ONLY reason to score below 1.0 is that B contains a programming capability, knowledge requirement, or claim that A does not actually support.
* Do NOT penalize A for being more specific than B.
* Do NOT penalize B for omitting important programming skills, concepts, procedures, or details that are present in A. Such omissions are INFORMATIVENESS concerns, not FAITHFULNESS concerns.
* Do NOT penalize B simply because it is more abstract or general than A. Generalization is expected when constructing a hierarchical knowledge-component taxonomy.
* Implicit programming knowledge counts. If solving A genuinely requires a programming concept, Java language feature, algorithmic principle, reasoning pattern, or procedure represented by B, then B is supported even if A does not explicitly name that knowledge component.
* However, do NOT infer a programming skill merely because it is commonly associated with the topic. The capability represented by B must be genuinely required or demonstrated by A.
* Evaluate SEMANTIC SUPPORT rather than lexical overlap. B does not need to use the same terminology, variable names, method names, class names, or syntax as A.
* Distinguish between a PROGRAMMING TOPIC and a PROGRAMMING SKILL/KNOWLEDGE COMPONENT. For example:

  * "Object-oriented programming" is a broad topic.
  * "Use inheritance and method overriding to implement polymorphic behavior" is a programming skill.
  * "Collections" is a topic.
  * "Iterate through a collection and filter elements according to a condition" is a programming skill.
* Topic overlap alone is not sufficient for faithfulness.
* If B contains multiple distinct programming capabilities, evaluate each capability separately. If A supports only some of them, reduce the score accordingly.
* A broad knowledge component can still be fully faithful if every capability it explicitly claims is genuinely relevant to A.
* Do NOT require B to describe the entire exercise.
* Do NOT require B to mention exercise-specific details such as variable names, class names, method names, numerical values, input values, output strings, specific objects, or particular test cases.
* Do NOT evaluate whether B is the most informative or most specific possible description of A. That is a separate judgment.
* Do NOT solve exercise A. Determine only whether the programming capability represented by B is genuinely required to solve, understand, trace, debug, or complete A.
* Do NOT introduce capabilities into A merely because they could be used as an alternative solution method. A capability should count only when it is genuinely required or clearly demonstrated by the exercise.
* If A can be solved without a capability claimed by B, that capability is not fully supported unless the wording or structure of A otherwise clearly requires it.
* Do not assume that every Java feature appearing in A is an important required capability. Determine whether the learner must actually understand or apply that feature to successfully complete the exercise.
* For code-tracing or output-prediction exercises, evaluate the knowledge required to trace the provided program. Do not assume that the learner must know how to write the entire program.
* For debugging exercises, evaluate the knowledge required to identify or correct the specific error represented by A. Do not automatically assume that all general debugging knowledge is required.
* For code-completion exercises, evaluate the knowledge required to produce the missing code, not every concept appearing elsewhere in the provided program.
* For exercises involving APIs or libraries, consider whether knowledge of the particular API/library behavior is genuinely required by A.
* For exercises involving multiple programming concepts, evaluate whether B's individual claims are actually supported rather than assuming that the presence of one supported concept makes the entire B faithful.
* If B includes a broad capability that necessarily encompasses the capability required by A, the broad capability may be fully faithful even if it is less informative. Do not confuse breadth with lack of faithfulness.

Base the score on the proportion of the DISTINCT PROGRAMMING CAPABILITIES represented by B that are actually supported by exercise A.

Scoring guidance:

* 0.85-1.0 (Highly Faithful / High Precision):
  Nearly every important capability explicitly represented by B is clearly tested, required, or meaningfully demonstrated by A. B is a valid generalization or abstraction of A.

* 0.6-0.85 (Mostly Faithful):
  A supports most of the capabilities represented by B, but one capability or meaningful aspect is only partially or weakly supported.

* 0.3-0.6 (Partially Faithful):
  A supports some meaningful capabilities in B, but B also contains one or more important capabilities that A does not actually test or require.

* 0.0-0.3 (Not Faithful / Mismatched):
  B describes a substantially different programming capability, or most of the capabilities claimed by B are not supported by A. A topical relationship alone is insufficient.

## Examples

Exercise A:
Write a Java program that reads an integer and prints whether the number is positive, negative, or zero.

Knowledge Component B:
Use conditional statements to classify a value according to mutually exclusive conditions.

Score: 1.0

Reason:
A requires conditional reasoning to distinguish among positive, negative, and zero values. The capability described by B is directly required by the exercise.

---

Exercise A:
Write a Java program that reads an integer and prints whether the number is positive, negative, or zero.

Knowledge Component B:
Write Java programs that process input and produce output.

Score: 1.0

Reason:
The exercise requires processing an input value and producing an output based on that value. B is broader and less informative than the exercise, but every capability it claims is supported by A.

---

Exercise A:
Write a Java program that reads an integer and prints whether the number is positive, negative, or zero.

Knowledge Component B:
Use loops to repeatedly process input values.

Score: 0.2

Reason:
The exercise does not require repeated processing or iteration. Although loops can potentially be used in some programming solutions, they are not required by A. Therefore the central capability claimed by B is not supported.

---

Exercise A:
Write a Java program that reads an integer and prints whether the number is positive, negative, or zero.

Knowledge Component B:
Understand Java programming.

Score: 1.0

Reason:
The exercise genuinely requires knowledge of Java programming. B is extremely broad and therefore may have very low INFORMATIVENESS, but its stated capability is not contradicted by A. Faithfulness concerns precision of the capabilities claimed by B, not how much detail B captures.

---

Exercise A:
Write a Java method that receives an integer array and returns the largest element.

Knowledge Component B:
Traverse an array while maintaining the largest value encountered so far.

Score: 1.0

Reason:
The exercise requires processing the array and determining its maximum value. The capability described by B is directly required by a standard solution and represents the central algorithmic knowledge of the exercise.

---

Exercise A:
Write a Java method that receives an integer array and returns the largest element.

Knowledge Component B:
Sort an array and retrieve the largest element.

Score: 0.55

Reason:
Sorting is one possible way to obtain the largest element, but the exercise does not require sorting. The capability of retrieving the largest value is supported, but the additional claim that sorting is required is not. Therefore faithfulness is reduced.

---

Exercise A:
Write a Java method that receives an integer array and returns the largest element.

Knowledge Component B:
Use arrays, sorting algorithms, and binary search to solve numerical problems.

Score: 0.35

Reason:
A requires array processing, but it does not require sorting or binary search, and it does not generally require solving numerical problems. Only part of B is supported by A.

---

Exercise A:
Predict the output of a Java program containing a for loop that increments a variable from 0 to 4 and prints the variable during each iteration.

Knowledge Component B:
Trace iterative control flow to determine how a loop variable changes across successive iterations.

Score: 1.0

Reason:
A explicitly requires tracing the loop and tracking the changing loop variable. The capability described by B is directly demonstrated by the exercise.

---

Exercise A:
Predict the output of a Java program containing a for loop that increments a variable from 0 to 4 and prints the variable during each iteration.

Knowledge Component B:
Understand Java loops.

Score: 1.0

Reason:
Understanding the loop construct is genuinely required to determine the program's output. B is much less informative than a skill describing loop tracing, but its broad capability is still supported by A.

---

Exercise A:
Predict the output of a Java program containing a for loop that increments a variable from 0 to 4 and prints the variable during each iteration.

Knowledge Component B:
Use recursion to repeatedly decompose a problem into smaller subproblems.

Score: 0.1

Reason:
A uses iteration rather than recursion and does not require recursive decomposition. The fact that recursion is another programming technique does not make it faithful to A.

---

Exercise A:
Create a Java class with a private field and public getter and setter methods.

Knowledge Component B:
Apply encapsulation by restricting direct access to object state and providing controlled access through methods.

Score: 1.0

Reason:
A directly requires the use of a private field and public methods for controlled access. The knowledge component described by B is directly demonstrated by the exercise.

---

Exercise A:
Create a Java class with a private field and public getter and setter methods.

Knowledge Component B:
Use object-oriented programming to create classes with encapsulation, inheritance, and polymorphism.

Score: 0.5

Reason:
A supports the encapsulation component, but it does not require inheritance or polymorphism. Because B claims multiple capabilities that are not supported by A, its faithfulness is reduced.

---

Exercise A:
Create a Java superclass Shape and subclasses Circle and Rectangle that override an area() method.

Knowledge Component B:
Use inheritance and method overriding to implement specialized behavior in subclasses.

Score: 1.0

Reason:
A directly requires a superclass, subclasses, inheritance, and method overriding. The capability represented by B is fully supported.

---

Exercise A:
Create a Java superclass Shape and subclasses Circle and Rectangle that override an area() method.

Knowledge Component B:
Use inheritance, method overriding, interfaces, exception handling, and multithreading to implement Java applications.

Score: 0.35

Reason:
A supports inheritance and method overriding, but it does not require interfaces, exception handling, or multithreading. These unsupported capabilities substantially reduce faithfulness.

---

Exercise A:
Write a recursive Java method that calculates the factorial of a positive integer.

Knowledge Component B:
Use recursion with a base case and recursively solve a smaller instance of the same problem.

Score: 1.0

Reason:
The exercise explicitly requires recursive problem solving and a terminating base case. The capabilities described by B are directly supported.

---

Exercise A:
Write a recursive Java method that calculates the factorial of a positive integer.

Knowledge Component B:
Use recursion and dynamic programming to optimize repeated subproblems.

Score: 0.55

Reason:
A requires recursion, but factorial does not require dynamic programming or optimization of repeated subproblems. The unsupported dynamic-programming capability lowers faithfulness.

---

Exercise A:
Write a Java program that stores student records in an ArrayList and prints the records whose scores are above 80.

Knowledge Component B:
Use collections to store multiple elements and iterate through them to select elements satisfying a condition.

Score: 1.0

Reason:
A requires storing multiple records in a collection, traversing the collection, and selecting elements based on a condition. The capability described by B is directly supported.

---

Exercise A:
Write a Java program that stores student records in an ArrayList and prints the records whose scores are above 80.

Knowledge Component B:
Use ArrayList, HashMap, and TreeSet to organize and retrieve data efficiently.

Score: 0.45

Reason:
A supports the use of ArrayList, but it does not require HashMap, TreeSet, or reasoning about efficient retrieval. Only part of B is supported.

---

Exercise A:
Write a Java program that catches an invalid integer-input exception and displays an error message.

Knowledge Component B:
Use exception handling to detect and respond to runtime errors.

Score: 1.0

Reason:
The exercise directly requires catching an exception and responding to it. The general exception-handling capability represented by B is fully supported.

---

Exercise A:
Write a Java program that catches an invalid integer-input exception and displays an error message.

Knowledge Component B:
Use exception handling, logging, custom exception classes, and recovery strategies in large-scale applications.

Score: 0.4

Reason:
A supports exception handling, but it does not require logging, custom exception classes, or large-scale application recovery strategies. Those unsupported capabilities reduce faithfulness.

---

Exercise A:
Fix a Java program that throws a NullPointerException when a method is called on an uninitialized object.

Knowledge Component B:
Identify null references and correct invalid dereferencing of null objects.

Score: 1.0

Reason:
The exercise directly demonstrates the need to recognize a null reference and correct the invalid dereference. B precisely describes the relevant debugging knowledge.

---

Exercise A:
Fix a Java program that throws a NullPointerException when a method is called on an uninitialized object.

Knowledge Component B:
Debug Java programs by identifying syntax errors, runtime errors, logical errors, and performance problems.

Score: 0.4

Reason:
A supports debugging a runtime error caused by a null reference, but it does not require identifying syntax errors, general logical errors, or performance problems. B therefore contains several unsupported capabilities.

---

Exercise A:
Complete a Java method that takes a String and returns the number of vowels it contains.

Knowledge Component B:
Process strings by examining individual characters and applying a condition to each character.

Score: 1.0

Reason:
A requires examining characters in the string and determining whether each character satisfies the vowel condition. B accurately describes the underlying capability.

---

Exercise A:
Complete a Java method that takes a String and returns the number of vowels it contains.

Knowledge Component B:
Perform natural-language processing on textual data.

Score: 0.25

Reason:
Although the exercise operates on text, it does not require natural-language processing. The topical relationship between strings and textual data is insufficient for faithfulness.

---

Exercise A:
Write a Java method that uses binary search to determine whether a target value exists in a sorted array.

Knowledge Component B:
Apply binary search by repeatedly narrowing the search interval based on comparisons with the target.

Score: 1.0

Reason:
A explicitly requires binary search and therefore requires the capability described by B.

---

Exercise A:
Write a Java method that uses binary search to determine whether a target value exists in a sorted array.

Knowledge Component B:
Search for values in arrays using efficient search algorithms.

Score: 1.0

Reason:
A is a specific instance of searching an array using an efficient search algorithm. B is broader than A but does not claim any unsupported capability. It may be less informative, but it remains faithful.

---

Exercise A:
Write a Java method that uses binary search to determine whether a target value exists in a sorted array.

Knowledge Component B:
Apply binary search and sorting algorithms to organize and search data.

Score: 0.65

Reason:
A supports binary search, but it does not require the learner to perform sorting. The additional sorting capability is unsupported, reducing faithfulness.

---

IMPORTANT DISTINCTION BETWEEN FAITHFULNESS AND INFORMATIVENESS:

FAITHFULNESS asks:

"Are the programming capabilities claimed by B actually supported by exercise A?"

INFORMATIVENESS asks:

"How much of the important programming knowledge and capabilities required by A does B capture?"

These dimensions must be evaluated independently.

For example:

Exercise A:
Write a Java method that returns the largest element in an integer array.

B1:
Traverse an array while maintaining the maximum value encountered so far.

→ High faithfulness and high informativeness.

B2:
Process arrays of integers.

→ High faithfulness but lower informativeness.

B3:
Understand Java programming.

→ High faithfulness but very low informativeness because the description is extremely broad.

B4:
Use binary search to find the largest element.

→ Low faithfulness if binary search is not required by A, even though both involve searching or processing arrays.

B5:
Sort the array and return the last element.

→ Lower faithfulness because sorting is not required by the exercise, even though this may be one possible solution strategy.

B6:
Traverse an array, maintain the maximum value, and use inheritance and polymorphism.

→ Partial faithfulness because array traversal and maximum tracking are supported, while inheritance and polymorphism are not.

Remember:

FAITHFULNESS = PRECISION / SUPPORT

INFORMATIVENESS = RECALL / COVERAGE

A candidate knowledge component should receive a high FAITHFULNESS score when every capability it claims is genuinely grounded in exercise A, even if it omits many important details of A.

A candidate knowledge component should receive a high INFORMATIVENESS score when it captures most of the important knowledge components required by A.

Do not use informativeness considerations when assigning the faithfulness score.

The goal is to determine whether B is a valid higher-level or generalized knowledge component for exercise A in a hierarchy of Java programming knowledge.
"""
    return evalPrompt_specificity_faithfulness


def get_eval_prompt_specificity_informativeness():
    evalPrompt_specificity_informativeness = """
You will be given a candidate Java programming exercise A and a general programming knowledge component or skill B. Skill B represents a higher-level or more general knowledge component in a hierarchy of Java programming knowledge. Candidate exercise A is being evaluated to determine how well it is represented by knowledge component B.

Your task is to determine how INFORMATIVE knowledge component B is about candidate exercise A.

Informativeness measures RECALL: whether B captures the important programming knowledge, concepts, skills, reasoning patterns, and capabilities that exercise A actually tests or requires.

In other words, ask:

"What are the important programming knowledge components and capabilities required by exercise A, and how much of those capabilities does B capture?"

A highly informative knowledge component captures the CORE PROGRAMMING CAPABILITY of the exercise while appropriately abstracting away exercise-specific details.

Important:

* First identify the IMPORTANT PROGRAMMING KNOWLEDGE COMPONENTS and CAPABILITIES of A.

* Focus on the knowledge and capabilities a learner must have to successfully solve, understand, complete, trace, explain, or debug the exercise.

* Important programming capabilities may include:

  * Java syntax and language constructs
  * variables, constants, and data types
  * operators and expressions
  * type conversion and casting
  * conditional logic
  * loops and iteration
  * methods, parameters, and return values
  * arrays and strings
  * collections and data structures
  * classes and objects
  * constructors and object initialization
  * encapsulation
  * inheritance
  * polymorphism
  * abstraction
  * interfaces
  * access modifiers
  * static and instance members
  * exception handling
  * generics
  * Java APIs and library usage
  * input/output and file handling
  * recursion
  * algorithms and algorithmic reasoning
  * searching and sorting
  * data processing and filtering
  * program tracing
  * output prediction
  * debugging and error identification
  * reasoning about program execution
  * understanding compiler or runtime behavior
  * translating requirements into code
  * decomposing a programming problem into steps
  * selecting appropriate programming constructs
  * selecting or applying appropriate data structures
  * testing and validating program behavior
  * other meaningful programming knowledge or capabilities.

* Consider capabilities that are IMPLICIT in A. A capability does not need to be explicitly named in the exercise if it is genuinely required to solve or understand it.

* Evaluate SEMANTIC COVERAGE, not lexical overlap. B does not need to use the same terminology, Java syntax, variable names, method names, class names, or examples as A.

* B is expected to be more GENERAL than A because it represents a higher-level knowledge component in a hierarchy. Do NOT penalize B merely because it abstracts away exercise-specific details.

* Exercise-specific details such as variable names, class names, method names, numerical values, particular input values, output strings, specific objects, or particular test cases do not need to be preserved unless they represent an important programming capability.

* The goal is to identify whether B captures the UNDERLYING PROGRAMMING KNOWLEDGE or SKILL of A, not whether B reproduces the wording of A.

* Penalize VAGUENESS when B is so broad that it fails to communicate the important programming capability being tested by A.

* Statements such as "solve programming problems," "write Java programs," "use Java," "understand programming," or "understand object-oriented programming" may be related to A but should receive low informativeness when they fail to capture the important knowledge component that makes A meaningful.

* Topic overlap alone is not sufficient. For example, if A requires using inheritance and method overriding, a skill that merely says "understand object-oriented programming" captures the topic but does not preserve the important capability.

* If A tests multiple important programming capabilities, B should capture the important capabilities that are central to successfully completing A.

* Do NOT require B to capture every trivial detail or every piece of information appearing in A.

* Do NOT reward B for capabilities that are not supported by A. Only the important capabilities actually tested or required by A contribute to informativeness.

* Do NOT evaluate whether B contains unsupported capabilities. That is the purpose of FAITHFULNESS.

* Do NOT penalize B merely because it is more general than A. Generalization is expected in a hierarchical knowledge taxonomy.

* Do NOT evaluate whether B is the most faithful possible generalization. Evaluate how much of A's important programming capability is retained by B.

* A knowledge component can therefore be highly faithful but poorly informative. For example, "solve Java programming problems" may genuinely apply to a Java exercise but fail to capture the specific programming knowledge being tested.

* Distinguish between a PROGRAMMING TOPIC and a PROGRAMMING KNOWLEDGE COMPONENT/SKILL.

  * A topic identifies a broad area of programming knowledge.
  * A knowledge component or skill describes a meaningful capability that a learner can apply.
  * For example, "arrays" is a topic, whereas "traverse an array while maintaining the maximum value encountered" is a more informative skill.

* Prefer knowledge components that preserve the important programming construct, relationship, algorithm, procedure, or reasoning pattern tested by A.

* Do not reward B merely for mentioning a Java feature that appears in A. The feature should represent an important capability required by the exercise.

* If a Java feature is incidental to the exercise, B does not need to preserve it.

* For code-tracing or output-prediction exercises, focus on the knowledge required to reason about program execution and determine the result.

* For debugging exercises, focus on the knowledge required to identify and correct the relevant programming error.

* For code-completion exercises, focus on the knowledge required to generate the missing code.

* For exercises involving APIs or libraries, focus on the meaningful knowledge required to use the relevant API or library correctly.

* For exercises involving several steps, consider the underlying knowledge components required across those steps rather than simply counting individual actions.

* Do not require B to preserve every implementation detail if those details are incidental to the underlying knowledge component.

Base the score on the proportion of A's IMPORTANT PROGRAMMING KNOWLEDGE COMPONENTS and CAPABILITIES that are captured or meaningfully implied by B.

Scoring guidance:

* 0.85-1.0 (Excellent Coverage / High Recall):
  B captures essentially all of the important programming knowledge and capabilities required by A. It preserves the central programming concepts, constructs, relationships, algorithms, procedures, or reasoning patterns while appropriately abstracting away incidental exercise-specific details.

* 0.6-0.85 (Good Coverage):
  B captures the main programming capability of A but loses one or more important aspects of the reasoning, relationship, procedure, algorithm, or conceptual content.

* 0.3-0.6 (Partial / Vague Coverage):
  B captures a broad aspect of what A requires but misses much of the important programming capability. It may identify the general programming activity or topic without preserving the specific knowledge component.

* 0.0-0.3 (Very Low Coverage / Mismatched):
  B captures little or none of the important programming knowledge or capabilities tested by A, or represents a substantially different capability.

## Examples

Exercise A:
Write a Java program that reads an integer and prints whether the number is positive, negative, or zero.

Knowledge Component B:
Use conditional statements to classify a value according to mutually exclusive conditions.

Score: 1.0

Reason:
B captures the central programming capability required by A: applying conditional logic to classify an input according to different conditions. The specific integer values and output messages are incidental.

---

Exercise A:
Write a Java program that reads an integer and prints whether the number is positive, negative, or zero.

Knowledge Component B:
Process input and produce output in Java.

Score: 0.55

Reason:
B captures the general input/output activity but misses the important conditional reasoning required to classify the value.

---

Exercise A:
Write a Java program that reads an integer and prints whether the number is positive, negative, or zero.

Knowledge Component B:
Write Java programs.

Score: 0.25

Reason:
B identifies the general programming activity but is too broad to capture the important conditional capability tested by A.

---

Exercise A:
Write a Java program that reads an integer and prints whether the number is positive, negative, or zero.

Knowledge Component B:
Use inheritance and polymorphism to organize related Java classes.

Score: 0.05

Reason:
B concerns a Java programming topic but does not capture the conditional reasoning required by A.

---

Exercise A:
Write a Java method that receives an integer array and returns the largest element.

Knowledge Component B:
Traverse an array while maintaining the maximum value encountered so far.

Score: 1.0

Reason:
B captures the central algorithmic capability required by A: traversing the array and maintaining the current maximum.

---

Exercise A:
Write a Java method that receives an integer array and returns the largest element.

Knowledge Component B:
Process arrays of numerical data.

Score: 0.5

Reason:
B captures the general array-processing capability but does not preserve the important algorithmic knowledge of tracking the maximum value.

---

Exercise A:
Write a Java method that receives an integer array and returns the largest element.

Knowledge Component B:
Perform computations in Java.

Score: 0.25

Reason:
B captures only the broad computational activity. It does not capture array traversal or the maximum-finding procedure.

---

Exercise A:
Predict the output of a Java program containing a for loop that increments a variable from 0 to 4 and prints the variable during each iteration.

Knowledge Component B:
Trace iterative control flow to determine how program state changes across successive iterations.

Score: 1.0

Reason:
B captures the important program-tracing capability required by A: following loop execution and tracking changes to the loop variable.

---

Exercise A:
Predict the output of a Java program containing a for loop that increments a variable from 0 to 4 and prints the variable during each iteration.

Knowledge Component B:
Understand Java loops.

Score: 0.45

Reason:
B identifies the relevant topic but does not sufficiently preserve the important capability of tracing execution to determine the output.

---

Exercise A:
Predict the output of a Java program containing a for loop that increments a variable from 0 to 4 and prints the variable during each iteration.

Knowledge Component B:
Write iterative Java programs using loops.

Score: 0.35

Reason:
B concerns loops but describes program construction rather than the output-tracing capability required by A. It therefore captures only part of the important capability.

---

Exercise A:
Create a Java class with a private field and public getter and setter methods.

Knowledge Component B:
Apply encapsulation by controlling access to object state through methods.

Score: 1.0

Reason:
B captures the central object-oriented knowledge component required by A: encapsulation and controlled access to private state.

---

Exercise A:
Create a Java class with a private field and public getter and setter methods.

Knowledge Component B:
Use object-oriented programming to create Java classes.

Score: 0.45

Reason:
B captures the broad object-oriented aspect but does not preserve the important knowledge of encapsulation and controlled access.

---

Exercise A:
Create a Java superclass Shape and subclasses Circle and Rectangle that override an area() method.

Knowledge Component B:
Use inheritance and method overriding to implement specialized behavior in subclasses.

Score: 1.0

Reason:
B captures the central object-oriented capabilities required by A: inheritance and method overriding.

---

Exercise A:
Create a Java superclass Shape and subclasses Circle and Rectangle that override an area() method.

Knowledge Component B:
Use object-oriented programming.

Score: 0.3

Reason:
B identifies the broad topic but fails to preserve the important capabilities of inheritance and method overriding.

---

Exercise A:
Create a Java superclass Shape and subclasses Circle and Rectangle that override an area() method.

Knowledge Component B:
Use inheritance, method overriding, interfaces, exception handling, and multithreading.

Score: 0.45

Reason:
B captures inheritance and method overriding but introduces several additional capabilities that are not relevant to A. Because informativeness measures how much of A's important capability B captures, the supported inheritance and overriding components provide partial coverage, while the unsupported additions are not counted as coverage.

---

Exercise A:
Write a recursive Java method that calculates the factorial of a positive integer.

Knowledge Component B:
Use recursion by solving a smaller instance of the problem and terminating with a base case.

Score: 1.0

Reason:
B captures the central recursive reasoning required by A, including recursive decomposition and termination at a base case.

---

Exercise A:
Write a recursive Java method that calculates the factorial of a positive integer.

Knowledge Component B:
Perform mathematical calculations in Java.

Score: 0.25

Reason:
B captures the broad computational aspect but does not preserve the important recursion capability.

---

Exercise A:
Write a Java program that stores student records in an ArrayList and prints records whose scores are above 80.

Knowledge Component B:
Use collections to store multiple elements, iterate through them, access relevant values, and filter elements according to a condition.

Score: 1.0

Reason:
B captures the important capabilities required by A: collection usage, iteration, accessing record information, and conditional filtering.

---

Exercise A:
Write a Java program that stores student records in an ArrayList and prints records whose scores are above 80.

Knowledge Component B:
Use ArrayLists in Java.

Score: 0.6

Reason:
B captures the collection-related capability but does not preserve the additional important capabilities of iteration and conditional filtering.

---

Exercise A:
Write a Java program that catches an exception caused by invalid integer input and displays an error message.

Knowledge Component B:
Use exception handling to detect and respond to runtime errors.

Score: 1.0

Reason:
B captures the central exception-handling capability required by A.

---

Exercise A:
Write a Java program that catches an exception caused by invalid integer input and displays an error message.

Knowledge Component B:
Write robust Java programs.

Score: 0.3

Reason:
B captures a broad programming goal but does not preserve the important knowledge component of exception handling.

---

Exercise A:
Fix a Java program that throws a NullPointerException when a method is called on an uninitialized object.

Knowledge Component B:
Identify null references and correct invalid dereferencing of null objects.

Score: 1.0

Reason:
B captures the central debugging knowledge required by A.

---

Exercise A:
Fix a Java program that throws a NullPointerException when a method is called on an uninitialized object.

Knowledge Component B:
Debug Java programs.

Score: 0.35

Reason:
B captures the general debugging activity but does not preserve the specific knowledge concerning null references and invalid dereferencing.

---

Exercise A:
Complete a Java method that takes a String and returns the number of vowels it contains.

Knowledge Component B:
Traverse a string character by character and classify characters according to a condition.

Score: 1.0

Reason:
B captures the central string-processing capability required by A: examining individual characters and determining whether they satisfy the vowel condition.

---

Exercise A:
Complete a Java method that takes a String and returns the number of vowels it contains.

Knowledge Component B:
Process textual data in Java.

Score: 0.45

Reason:
B captures the broad text-processing aspect but does not preserve the important character-level traversal and classification capability.

---

Exercise A:
Write a Java method that uses binary search to determine whether a target value exists in a sorted array.

Knowledge Component B:
Apply binary search by repeatedly narrowing the search interval based on comparisons with the target.

Score: 1.0

Reason:
B captures the central algorithmic capability explicitly required by A.

---

Exercise A:
Write a Java method that uses binary search to determine whether a target value exists in a sorted array.

Knowledge Component B:
Search arrays efficiently.

Score: 0.55

Reason:
B captures the general search capability but does not preserve the important binary-search strategy of repeatedly narrowing the search interval.

---

Exercise A:
Write a Java method that uses binary search to determine whether a target value exists in a sorted array.

Knowledge Component B:
Perform sorting and searching operations on arrays.

Score: 0.55

Reason:
B captures the general array-searching capability but does not preserve the specific binary-search procedure. Sorting is also not an important capability being tested by A.

## Multiple Important Capabilities

Exercise A:
Write a Java method that reads an integer array, calculates its average, and returns the elements whose values are above the average.

Knowledge Component B:
Traverse an array, calculate an aggregate statistic, and filter elements according to a computed condition.

Score: 1.0

Reason:
B captures the important capabilities required by A: array traversal, computation of an aggregate value, and filtering based on that computed value.

---

Exercise A:
Write a Java method that reads an integer array, calculates its average, and returns the elements whose values are above the average.

Knowledge Component B:
Traverse arrays and perform numerical calculations.

Score: 0.7

Reason:
B captures array traversal and numerical computation but misses the important capability of filtering elements according to the calculated average.

---

Exercise A:
Write a Java method that reads an integer array, calculates its average, and returns the elements whose values are above the average.

Knowledge Component B:
Process arrays in Java.

Score: 0.4

Reason:
B identifies the general domain of array processing but fails to preserve the important capabilities of calculating an aggregate statistic and filtering according to that statistic.

---

Exercise A:
Write a Java program that reads a list of integers, sorts them in ascending order, and removes duplicate values.

Knowledge Component B:
Sort a collection and eliminate duplicate elements.

Score: 1.0

Reason:
B captures both important capabilities required by A: sorting and duplicate removal.

---

Exercise A:
Write a Java program that reads a list of integers, sorts them in ascending order, and removes duplicate values.

Knowledge Component B:
Manipulate collections of numerical data.

Score: 0.5

Reason:
B captures the broad collection-processing capability but does not preserve the specific operations of sorting and duplicate removal.

---

Exercise A:
Write a Java program that reads a list of integers, sorts them in ascending order, and removes duplicate values.

Knowledge Component B:
Sort numerical data.

Score: 0.6

Reason:
B captures the sorting component but misses the additional important capability of identifying and removing duplicate values.

## Important Distinction Between INFORMATIVENESS and FAITHFULNESS

INFORMATIVENESS asks:

"How much of the important programming knowledge and capabilities required by A does B capture?"

FAITHFULNESS asks:

"Are the programming capabilities claimed by B actually supported by A?"

These dimensions must be evaluated independently.

For example:

Exercise A:
Write a Java method that returns the largest element in an integer array.

B1:
Traverse an array while maintaining the maximum value encountered so far.

→ High faithfulness and high informativeness.

B2:
Process arrays of integers.

→ High faithfulness but lower informativeness.

B3:
Understand Java programming.

→ High faithfulness but very low informativeness because the description is extremely broad.

B4:
Use binary search to find the largest element.

→ Low faithfulness and low informativeness if binary search is not required by A.

B5:
Use arrays and object-oriented programming.

→ Partial informativeness because arrays are relevant but object-oriented programming may not be an important capability required by A.

B6:
Use an array, calculate the maximum, and apply inheritance and polymorphism.

→ Partial informativeness because the array and maximum-finding components capture important capabilities in A, while inheritance and polymorphism do not contribute to the important capabilities tested by A.

Remember:

FAITHFULNESS = PRECISION / SUPPORT

INFORMATIVENESS = RECALL / COVERAGE

A candidate knowledge component should receive a high INFORMATIVENESS score when it captures most of the IMPORTANT PROGRAMMING KNOWLEDGE and CAPABILITIES required by exercise A.

A candidate knowledge component should receive a low INFORMATIVENESS score when it is merely a broad description of the programming topic or activity and fails to preserve the knowledge that makes A meaningful.

Do not require the knowledge component to reproduce exercise-specific details.

Do not reward lexical overlap.

Do not require every implementation detail to be preserved.

Focus on the underlying programming knowledge and capability that a learner must possess to successfully complete A.

The goal is to determine whether B is a useful higher-level knowledge component for representing exercise A in a hierarchical taxonomy of Java programming knowledge.
"""
    return evalPrompt_specificity_informativeness