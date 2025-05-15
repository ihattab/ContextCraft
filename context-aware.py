import google.generativeai as genai
import os
import json
from typing import Optional, Dict, List, Any

genai.configure(api_key="your_api_key_here")

# --- Configuration ---
# Make sure GOOGLE_API_KEY environment variable is set
# Or configure explicitly: genai.configure(api_key="YOUR_API_KEY")

# Use Gemini 2.5 Pro - Latest version with enhanced reasoning capabilities
model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

# Configure safety settings for responsible AI usage
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

# --- Feature 1: Intelligent Brief Analysis with Chain-of-Thought ---
def analyze_creative_brief(brief_text: str) -> Optional[Dict[str, Any]]:
    """
    Uses Gemini 2.5 Pro with explicit chain-of-thought reasoning to analyze a brief
    and extract structured information by understanding underlying intent.
    """
    prompt = f"""
    Let's analyze this creative brief step by step using chain-of-thought reasoning.

    Step 1: Initial Understanding
    - What is the core purpose of this brief?
    - What are the explicit and implicit requirements?
    - What emotions or reactions are they trying to evoke?

    Step 2: Target Audience Analysis
    - Who are they explicitly targeting?
    - What can we infer about their demographics and psychographics?
    - What are their likely pain points and motivations?

    Step 3: Brand Voice & Style Assessment
    - What tone is explicitly requested?
    - What underlying brand personality emerges?
    - How should the content make the audience feel?

    Step 4: Strategic Goals Evaluation
    - What are the primary and secondary objectives?
    - What specific actions should the content drive?
    - What key messages must be conveyed?

    Step 5: Constraints & Requirements
    - What are the explicit limitations?
    - What are the implicit boundaries?
    - What elements must be included?

    Now, based on this step-by-step analysis, provide a structured JSON response with the following:
    {{
        "target_audience": {{
            "demographics": "",
            "psychographics": "",
            "needs": "",
            "reasoning": ""
        }},
        "tone_and_style": {{
            "keywords": [],
            "feeling": "",
            "justification": ""
        }},
        "brand_voice": {{
            "characteristics": [],
            "reasoning": ""
        }},
        "communication_goals": {{
            "primary": "",
            "secondary": "",
            "reasoning": ""
        }},
        "key_messages": {{
            "core_ideas": [],
            "supporting_points": [],
            "reasoning": ""
        }},
        "emotional_impact": {{
            "desired_feelings": [],
            "reasoning": ""
        }},
        "constraints": {{
            "explicit": [],
            "implicit": [],
            "reasoning": ""
        }}
    }}

    Creative Brief:
    ---
    {brief_text}
    ---

    Provide ONLY the JSON object as your output. Ensure the JSON structure is valid.
    """
    try:
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        
        try:
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = response.text[json_start:json_end]
                analysis_json = json.loads(json_string)
                return analysis_json
            else:
                print("Warning: Could not find a valid JSON object in the response.")
                return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Model response text: {response.text}")
            return None

    except Exception as e:
        print(f"Error analyzing brief: {e}")
        return None

# --- Feature 2: Contextual Content Generation with Reasoning ---
def generate_content_with_context(brief_analysis: Dict[str, Any], previous_content: List[str], generation_task: str) -> str:
    """
    Generates content using chain-of-thought reasoning, considering the analyzed brief
    and project history.
    """
    brief_str = json.dumps(brief_analysis, indent=2)
    previous_content_str = "\n---\n".join(previous_content) if previous_content else "No previous content generated yet."

    prompt = f"""
    Let's generate content using a structured reasoning process:

    Step 1: Context Analysis
    - Review the brief analysis
    - Consider previous content
    - Understand the current task

    Step 2: Strategic Alignment
    - How does this content serve the brief's goals?
    - What key messages must be conveyed?
    - How should it build on previous content?

    Step 3: Tone & Style Planning
    - What tone is required?
    - How should the brand voice be expressed?
    - What emotional impact should it create?

    Step 4: Content Structure
    - What format best serves the purpose?
    - How should key messages be organized?
    - What elements must be included?

    Step 5: Generation
    Based on the above reasoning, generate the content.

    --- Creative Brief Analysis ---
    {brief_str}
    ---

    --- Previous Content Snippets ---
    {previous_content_str}
    ---

    Task: {generation_task}

    Generate the content below, ensuring it aligns with all the reasoning steps above.
    """
    try:
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config={
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 40
            }
        )
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return "Error generating content."

# --- Feature 3: Reasoned Feedback with Chain-of-Thought ---
def evaluate_content_alignment(brief_analysis: Dict[str, Any], content_snippet: str) -> str:
    """
    Evaluates content alignment using chain-of-thought reasoning.
    """
    brief_str = json.dumps(brief_analysis, indent=2)

    prompt = f"""
    Let's evaluate this content using a structured reasoning process:

    Step 1: Brief Requirements Analysis
    - What are the key requirements from the brief?
    - What tone and style are specified?
    - What are the communication goals?

    Step 2: Content Evaluation
    - How well does the content meet each requirement?
    - What elements align well?
    - What elements need improvement?

    Step 3: Specific Analysis
    - Tone & Style alignment
    - Brand voice consistency
    - Message clarity and impact
    - Goal achievement

    Step 4: Improvement Suggestions
    - What specific changes would improve alignment?
    - How would these changes better serve the brief?

    Creative Brief Analysis:
    ---
    {brief_str}
    ---

    Content Snippet to Evaluate:
    ---
    {content_snippet}
    ---

    Provide your evaluation and suggestions based on the above reasoning process.
    """
    try:
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        return response.text
    except Exception as e:
        print(f"Error evaluating content: {e}")
        return "Error evaluating content."

# --- Example Usage Flow ---
if __name__ == "__main__":
    creative_brief = """
    We need ad copy for our new sustainable coffee brand, 'EarthBloom'.
    Target audience are millennials aged 25-40, eco-conscious, appreciate quality and transparency.
    Tone should be warm, inviting, slightly premium, but approachable. Avoid corporate jargon.
    Goal is to drive website traffic to learn more about our sourcing practices.
    Mention our compostable packaging.
    """

    # 1. Analyze the brief using chain-of-thought reasoning
    print("--- Analyzing Brief with Chain-of-Thought Reasoning ---")
    analysis = analyze_creative_brief(creative_brief)

    if analysis:
        print("\n--- Brief Analysis (Reasoned) ---")
        print(json.dumps(analysis, indent=2))

        # 2. Generate content with reasoning
        project_context = {
            "brief": analysis,
            "generated_content": []
        }

        task1 = "Write 2 short ad copy variations (max 30 words each) for a social media campaign, strictly adhering to the brief's tone and goals."
        print(f"\n--- Generating Ad Copy with Reasoning: {task1} ---")
        ad_copy1 = generate_content_with_context(project_context["brief"], project_context["generated_content"], task1)
        print("\n--- Generated Ad Copy ---")
        print(ad_copy1)
        project_context["generated_content"].append(ad_copy1)

        # 3. Generate blog post idea with reasoning
        task2 = "Suggest a catchy title and a 3-sentence outline for a blog post expanding on EarthBloom's sourcing transparency. Ensure it maintains consistency with the tone and messaging established in the brief and previous content."
        print(f"\n--- Generating Blog Idea with Reasoning: {task2} ---")
        blog_idea = generate_content_with_context(project_context["brief"], project_context["generated_content"], task2)
        print("\n--- Generated Blog Idea ---")
        print(blog_idea)
        project_context["generated_content"].append(blog_idea)

        # 4. Evaluate content with chain-of-thought reasoning
        sample_snippet_for_eval = "Buy EarthBloom coffee today! Our supply chain is super efficient and we use green packaging."
        print(f"\n--- Evaluating Content with Chain-of-Thought: '{sample_snippet_for_eval}' ---")
        evaluation = evaluate_content_alignment(project_context["brief"], sample_snippet_for_eval)
        print("\n--- Content Evaluation (Reasoned) ---")
        print(evaluation)

    else:
        print("\n--- Brief analysis failed. Cannot proceed with generation/evaluation. ---")