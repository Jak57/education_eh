
def get_system_prompt():
    system_prompt = "You are a scientific skill taxonomy expert who identifies and describes the underlying scientific skills shared by two science exercises. Given exercises A and B, your task is to generate a concise common skill that generalizes the underlying scientific knowledge, reasoning, concepts, principles, laws, formulas, or procedures required by BOTH exercises. The common skill should represent the intersection of the skills required by A and B, not their union. Infer latent or implicit skills when they are genuinely required by both exercises, while avoiding exercise-specific details and unsupported relationships. The skill should be as specific as possible while remaining applicable to both exercises, and should be expressed as an action-oriented learner capability rather than merely a topic, scientific discipline, or summary of the exercises. No matter the content of exercises A and B, you must identify and write a common scientific skill."
    return system_prompt

def get_common_summary_prompt(story1, story2):
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
    return prompt14