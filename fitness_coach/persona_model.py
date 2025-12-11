"""
Fine-tuned persona model for generating coaching feedback

This module provides a local alternative to the Gemini API by using
fine-tuned transformer models for each persona.
"""

import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


class PersonaModel:
    """Fine-tuned model for generating persona-specific coaching feedback"""
    
    def __init__(self, persona_name, model_dir='./models'):
        """
        Initialize persona model
        
        Args:
            persona_name: Name of persona (e.g., "Hype Beast")
            model_dir: Directory containing fine-tuned models
        """
        self.persona_name = persona_name
        persona_safe = persona_name.lower().replace(' ', '_')
        model_path = Path(model_dir) / f'persona_{persona_safe}'
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please fine-tune the model first using scripts/fine_tune_persona.py"
            )
        
        print(f"Loading persona model: {persona_name} from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
    
    def generate_feedback(self, objective_report, max_length=512, temperature=0.7):
        """
        Generate coaching feedback for given objective report
        
        Args:
            objective_report: Objective performance report string
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
        
        Returns:
            Generated feedback string
        """
        prompt = f"<|persona|>{self.persona_name}<|input|>{objective_report}<|output|>"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids('<|endoftext|>') if '<|endoftext|>' in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract output part
        if '<|output|>' in generated_text:
            output = generated_text.split('<|output|>')[-1]
            if '<|endoftext|>' in output:
                output = output.split('<|endoftext|>')[0]
            return output.strip()
        
        return generated_text


def get_persona_feedback(objective_report, persona_name, use_fine_tuned=True, fallback_to_gemini=True):
    """
    Get persona feedback, using fine-tuned model if available, otherwise Gemini API
    
    Args:
        objective_report: Objective performance report
        persona_name: Name of persona
        use_fine_tuned: Try to use fine-tuned model first
        fallback_to_gemini: Fall back to Gemini API if fine-tuned model fails
    
    Returns:
        Feedback string
    """
    # Try fine-tuned model first
    if use_fine_tuned:
        try:
            model = PersonaModel(persona_name)
            feedback = model.generate_feedback(objective_report)
            print(f"✅ Generated feedback using fine-tuned model for {persona_name}")
            return feedback
        except FileNotFoundError:
            print(f"⚠️  Fine-tuned model not found for {persona_name}, falling back to Gemini")
        except Exception as e:
            print(f"⚠️  Error using fine-tuned model: {e}, falling back to Gemini")
    
    # Fall back to Gemini API
    if fallback_to_gemini:
        try:
            import google.generativeai as genai
            
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                config_path = Path(__file__).parent.parent / 'config.txt'
                if config_path.exists():
                    api_key = config_path.read_text().strip()
            
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found")
            
            genai.configure(api_key=api_key)
            
            # Persona system instructions
            PERSONAS = {
                "Hype Beast": "You are The Hype Beast. Your tone is ultra-motivational, energetic, and uses enthusiastic modern slang. You focus on confidence, confidence, confidence, bringing the energy, and framing corrections as leveling up. Use many emojis and exclamation points. Make the user feel like a star. DO NOT use LaTeX formatting like $...$ for scores or percentages.",
                "Data Scientist": "You are The Data Scientist. You speak in precise, objective terms. Your feedback uses specific metrics and quantified improvement. Translate complex biomechanical terms into clear, actionable, technical advice that a novice can understand. Use a formal, structured report format. DO NOT use LaTeX formatting like $...$ for scores or percentages.",
                "No-Nonsense Pro": "You are The No-Nonsense Pro. You are direct, challenging, and slightly impatient with wasted effort. Your language is concise and demanding. Emphasize immediate correction and demand a higher standard of execution. Focus on 'why' the bad form wastes energy and must be fixed NOW. DO NOT use LaTeX formatting like $...$ for scores or percentages.",
                "Mindful Aligner": "You are The Mindful Aligner. Your tone is calm, centered, and encouraging. You focus on connecting movement to breath, finding internal stability, and making gentle, internal adjustments to achieve proper alignment. Use soft, encouraging language. DO NOT use LaTeX formatting like $...$ for scores or percentages.",
            }
            
            system_instruction = PERSONAS.get(persona_name, PERSONAS["Hype Beast"])
            
            prompt = f"""Based on the following objective performance report for a workout, you must adopt the selected persona and provide detailed, actionable coaching feedback. Start your response with a clear, concise title using a markdown H1 heading (# TITLE). DO NOT use LaTeX commands like $...$ or \\frac{{}}{{}} for scores, percentages, or numbers. Use plain text and standard Unicode symbols (like the percent sign %).

**OBJECTIVE PERFORMANCE DATA:**
---
{objective_report}
---"""
            
            # Try available models
            model_names = [
                "gemini-1.5-flash-latest",
                "gemini-1.5-flash-002",
                "gemini-1.5-pro-latest",
                "gemini-pro",
            ]
            
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=system_instruction
                    )
                    response = model.generate_content(prompt)
                    print(f"✅ Generated feedback using Gemini API ({model_name})")
                    return response.text
                except Exception as e:
                    continue
            
            raise Exception("All Gemini models failed")
            
        except Exception as e:
            raise Exception(f"Failed to generate feedback: {e}")
    
    raise Exception("No feedback generation method available")


