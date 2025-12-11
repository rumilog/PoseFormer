"""
Fitness Coach Backend Server
Handles video upload, processing, scoring, and LLM feedback generation
"""

import os
import sys
import json
import tempfile
import shutil
import traceback
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import cgi
import requests
import google.generativeai as genai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Global cache for fine-tuned persona models
PERSONA_MODELS = {}
PERSONA_TOKENIZERS = {}

def load_persona_model(persona_name):
    """Load a fine-tuned persona model (cached). Tries HuggingFace first, then local."""
    if persona_name in PERSONA_MODELS:
        return PERSONA_MODELS[persona_name], PERSONA_TOKENIZERS[persona_name]
    
    # HuggingFace model repos
    hf_model_mapping = {
        "Hype Beast": "rlogh/fitness-coach-persona-hype-beast",
        "Data Scientist": "rlogh/fitness-coach-persona-data-scientist",
        "No-Nonsense Pro": "rlogh/fitness-coach-persona-no-nonsense-pro",
        "Mindful Aligner": "rlogh/fitness-coach-persona-mindful-aligner"
    }
    
    # Local model paths (fallback)
    local_model_mapping = {
        "Hype Beast": "persona_hype_beast",
        "Data Scientist": "persona_data_scientist",
        "No-Nonsense Pro": "persona_no-nonsense_pro",
        "Mindful Aligner": "persona_mindful_aligner"
    }
    
    hf_repo = hf_model_mapping.get(persona_name)
    local_dir = local_model_mapping.get(persona_name)
    
    if not hf_repo and not local_dir:
        return None, None
    
    try:
        print(f"Loading {persona_name} model...")
        
        # Try HuggingFace first
        model_source = None
        try:
            print(f"  Trying HuggingFace: {hf_repo}")
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                hf_repo,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model_source = "HuggingFace"
        except Exception as e_hf:
            print(f"  HuggingFace failed: {e_hf}")
            
            # Try local path
            local_path = PROJECT_ROOT / "models" / local_dir
            if local_path.exists():
                print(f"  Trying local: {local_path}")
                tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                model_source = "local"
            else:
                raise e_hf
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        PERSONA_MODELS[persona_name] = model
        PERSONA_TOKENIZERS[persona_name] = tokenizer
        print(f"✓ Loaded {persona_name} from {model_source}")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load {persona_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


class FitnessAPIHandler(BaseHTTPRequestHandler):
    """Handle API requests for fitness coaching"""
    
    def _set_headers(self, status=200, content_type='application/json'):
        """Set response headers"""
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self._set_headers()
    
    def do_GET(self):
        """Serve static files"""
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            
            # Serve index.html (upload page)
            if path == '/' or path == '/index.html':
                try:
                    html_path = PROJECT_ROOT / 'public_html (1) (1)' / 'index_video_upload.html'
                    with open(html_path, 'rb') as f:
                        content = f.read()
                    self._set_headers(content_type='text/html')
                    self.wfile.write(content)
                    return
                except Exception as e:
                    print(f"Error serving HTML: {e}")
                    self._set_headers(404)
                    self.wfile.write(json.dumps({'error': str(e)}).encode())
                    return
            
            # Serve persona page
            elif path == '/persona.html':
                try:
                    html_path = PROJECT_ROOT / 'public_html (1) (1)' / 'index.html'
                    with open(html_path, 'rb') as f:
                        content = f.read()
                    self._set_headers(content_type='text/html')
                    self.wfile.write(content)
                    return
                except Exception as e:
                    print(f"Error serving persona page: {e}")
                    self._set_headers(404)
                    self.wfile.write(json.dumps({'error': str(e)}).encode())
                    return
        
            # Serve video files
            elif path.endswith('.mp4'):
                try:
                    video_path = PROJECT_ROOT / path.lstrip('/')
                    if video_path.exists():
                        with open(video_path, 'rb') as f:
                            content = f.read()
                        self._set_headers(content_type='video/mp4')
                        self.wfile.write(content)
                    else:
                        self._set_headers(404)
                        self.wfile.write(b'Video not found')
                    return
                except Exception as e:
                    print(f"Error serving video: {e}")
                    self._set_headers(500)
                    self.wfile.write(json.dumps({'error': str(e)}).encode())
                    return
            
            # Serve JSON reports
            elif path.endswith('.json'):
                try:
                    json_path = PROJECT_ROOT / path.lstrip('/')
                    with open(json_path, 'r') as f:
                        content = f.read()
                    self._set_headers()
                    self.wfile.write(content.encode())
                    return
                except Exception as e:
                    print(f"Error serving JSON: {e}")
                    self._set_headers(404)
                    self.wfile.write(json.dumps({'error': str(e)}).encode())
                    return
            
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Not found'}).encode())
        
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError) as e:
            # Client disconnected - this is normal, just log and continue
            print(f"Client disconnected: {e}")
        except Exception as e:
            print(f"Error in do_GET: {e}")
            try:
                self._set_headers(500)
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            except:
                pass  # Connection already closed, can't send error
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            
            print(f"POST request to: {path}")
            
            if path == '/api/upload-and-analyze':
                self.handle_video_upload()
            elif path == '/api/gemini':
                print("Routing to fine-tuned model handler (legacy Gemini endpoint)...")
                self.handle_persona_feedback()
            elif path == '/api/generate_feedback':
                print("Routing to fine-tuned model handler...")
                self.handle_persona_feedback()
            else:
                print(f"404: Unknown endpoint {path}")
                self._set_headers(404)
                self.wfile.write(json.dumps({'error': 'Endpoint not found'}).encode())
        except Exception as e:
            print(f"Error in do_POST: {e}")
            traceback.print_exc()
            try:
                self._set_headers(500)
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            except:
                pass
    
    def handle_video_upload(self):
        """Handle video upload and processing"""
        try:
            print("\n" + "="*60)
            print("RECEIVING VIDEO UPLOAD")
            print("="*60)
            
            # Parse multipart form data
            content_type = self.headers.get('Content-Type', '')
            print(f"Content-Type: {content_type}")
            
            if 'multipart/form-data' not in content_type:
                print("ERROR: Invalid content type")
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'Must be multipart/form-data'}).encode())
                return
            
            # Parse form
            print("Parsing form data...")
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': content_type,
                }
            )
            print("Form parsed successfully")
            
            # Get video file
            if 'video' not in form:
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'No video file provided'}).encode())
                return
            
            video_file = form['video']
            exercise_type = form.getvalue('exercise', 'pushup')
            
            # Save video temporarily
            temp_dir = tempfile.mkdtemp()
            video_filename = f"upload_{os.path.basename(video_file.filename)}"
            video_path = os.path.join(temp_dir, video_filename)
            
            with open(video_path, 'wb') as f:
                f.write(video_file.file.read())
            
            print(f"\n{'='*60}")
            print(f"Processing uploaded video: {video_filename}")
            print(f"Exercise type: {exercise_type}")
            print(f"{'='*60}\n")
            
            # Process video and get scores
            from fitness_coach.comparison import score_exercise
            
            results = score_exercise(
                user_video_path=video_path,
                reference_id=exercise_type,
                references_dir='references',
                use_dtw=True,
                scoring_method='statistical',
                force_reprocess=True,
                use_pose_invariant=True
            )
            
            # Generate LLM feedback
            print("\nGenerating personalized AI feedback...")
            llm_feedback = self.generate_llm_feedback(results, exercise_type)
            
            # Save results to file for the persona page to load
            import time
            timestamp = int(time.time())
            result_filename = f"analysis_{timestamp}.json"
            result_path = PROJECT_ROOT / result_filename
            
            # Prepare full report
            report = {
                'status': 'success',
                'exercise': {
                    'type': exercise_type,
                    'filename': video_filename
                },
                'scores': {
                    'overall': results['overall_score'],
                    'relevant': results['relevant_score'],
                    'body_parts': results['body_part_scores']
                },
                'feedback': results['feedback'],
                'metrics': {
                    'frames': {
                        'user': results['num_frames_user'],
                        'reference': results['num_frames_ref'],
                        'aligned': results['num_frames_aligned']
                    },
                    'alignment_quality': results.get('details', {}).get('alignment_score', 0),
                    'body_part_details': results.get('body_part_details', {})
                },
                'llm_context': {
                    'description': f"User performed {exercise_type} exercise",
                    'scoring_method': 'pose-invariant',
                    'interpretation': {
                        'score_range': '0-100, where 100 is perfect form matching the reference',
                        'joint_angles': 'Compared joint angles with 15° tolerance (70% weight)',
                        'bone_proportions': 'Compared relative bone lengths (30% weight)'
                    }
                }
            }
            
            # Save to file
            with open(result_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"✅ Results saved to: {result_filename}")
            
            # Cleanup temporary directory
            shutil.rmtree(temp_dir)
            
            # Return redirect response
            response = {
                'status': 'success',
                'redirect': f'/persona.html?results={result_filename}',
                'result_file': result_filename
            }
            
            self._set_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"\nERROR: {e}")
            traceback.print_exc()
            self._set_headers(500)
            self.wfile.write(json.dumps({
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }).encode())
    
    def generate_llm_feedback(self, results, exercise_type, persona="Hype Beast"):
        """Generate personalized feedback using fine-tuned persona model"""
        try:
            # Load the fine-tuned model
            model, tokenizer = load_persona_model(persona)
            
            if model is None:
                print(f"Warning: Could not load {persona} model, using basic feedback")
                return self.generate_basic_feedback(results, exercise_type)
            
            # Prepare input report
            report = f"""Exercise Analysis Report

Overall Score: {results['overall_score']:.1f}/100
Relevant Body Parts Score: {results['relevant_score']:.1f}/100

Body Part Breakdown:
"""
            for part, score in results['body_part_scores'].items():
                report += f"- {part.replace('_', ' ').title()}: {score:.1f}/100\n"
            
            report += f"\nKey Issues:\n"
            for feedback in results['feedback']:
                report += f"- {feedback}\n"
            
            # Format prompt as trained
            prompt = f"<|persona|>{persona}<|input|>{report}<|output|>"
            
            print(f"Generating feedback with {persona} model...")
            
            # Generate feedback
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode and extract only the generated part
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the output part after <|output|>
            if "<|output|>" in full_text:
                feedback = full_text.split("<|output|>")[-1].strip()
            else:
                feedback = full_text
            
            print(f"✓ Generated {len(feedback)} chars of feedback")
            return feedback
            
        except Exception as e:
            print(f"Error generating fine-tuned feedback: {e}")
            import traceback
            traceback.print_exc()
            return self.generate_basic_feedback(results, exercise_type)
    
    def generate_basic_feedback(self, results, exercise_type):
        """Generate basic feedback without LLM"""
        score = results['overall_score']
        
        if score >= 80:
            opening = "Excellent work! Your form is looking great."
        elif score >= 60:
            opening = "Good effort! You're on the right track with solid fundamentals."
        elif score >= 40:
            opening = "Nice start! There's definitely potential here."
        else:
            opening = "Thanks for trying! Everyone starts somewhere, and form takes practice."
        
        # Find weakest area
        weakest = min(results['body_part_scores'].items(), key=lambda x: x[1])
        tip = f"Focus particularly on your {weakest[0].replace('_', ' ')}, which needs the most attention."
        
        closing = "Keep practicing and you'll see improvement. Consistency is key!"
        
        return f"{opening} {tip} {closing}"
    
    def handle_persona_feedback(self):
        """Generate feedback using fine-tuned persona models"""
        try:
            print("\n============================================================")
            print("PERSONA MODEL REQUEST RECEIVED (using fine-tuned models)")
            print("============================================================")
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Extract persona and content
            system_instruction = request_data.get('systemInstruction', '')
            contents = request_data.get('contents', [])
            
            # Parse persona from system instruction
            persona = "Hype Beast"  # Default
            if "Hype Beast" in system_instruction:
                persona = "Hype Beast"
            elif "Data Scientist" in system_instruction:
                persona = "Data Scientist"
            elif "No-Nonsense Pro" in system_instruction:
                persona = "No-Nonsense Pro"
            elif "Mindful Aligner" in system_instruction:
                persona = "Mindful Aligner"
            
            # Extract user message
            user_message = ""
            if contents and len(contents) > 0:
                parts = contents[0].get('parts', [])
                if parts and len(parts) > 0:
                    full_message = parts[0].get('text', '')
                    # Remove the persona prefix and instructions, keep only from SUMMARY onwards
                    if "SUMMARY:" in full_message:
                        user_message = full_message.split("SUMMARY:", 1)[1].strip()
                        user_message = "SUMMARY: " + user_message
                    else:
                        user_message = full_message
            
            print(f"Persona: {persona}")
            print(f"Message length: {len(user_message)} chars")
            
            # Load model
            model, tokenizer = load_persona_model(persona)
            
            if model is None:
                raise Exception(f"Failed to load {persona} model")
            
            # Format prompt
            prompt = f"<|persona|>{persona}<|input|>{user_message}<|output|>"
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            
            # Force CPU to avoid token ID mismatch issues with CUDA
            # (Special tokens added during training may cause CUDA assertions)
            device = "cpu"
            model = model.cpu()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,  # Reduced to prevent repetition
                    temperature=0.9,  # Higher temp = more creative, less repetitive
                    top_p=0.95,  # Nucleus sampling
                    top_k=50,  # Top-k sampling to avoid repetition
                    do_sample=True,
                    repetition_penalty=1.2,  # Penalize repeated phrases
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract output
            if "<|output|>" in full_text:
                feedback = full_text.split("<|output|>")[-1].strip()
            else:
                feedback = full_text
            
            print(f"✅ Generated {len(feedback)} chars with {persona}")
            
            # Format response to match Gemini API format
            result = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": feedback}]
                    }
                }]
            }
            
            self._set_headers(200)
            self.wfile.write(json.dumps(result).encode())
            print("✅ Response sent to client")
            
        except Exception as e:
            print(f"❌ Persona model error: {e}")
            import traceback
            traceback.print_exc()
            self._set_headers(500)
            self.wfile.write(json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc()
            }).encode())
    
    def handle_gemini_proxy_LEGACY(self):
        """LEGACY: Proxy requests to Gemini API using Google Generative AI SDK (simplified!)"""
        try:
            print("="*60)
            print("GEMINI PROXY REQUEST RECEIVED (using google.generativeai)")
            print("="*60)
            
            # Get API key from environment or config file
            api_key = os.environ.get('GEMINI_API_KEY', '')
            if not api_key:
                config_path = Path(__file__).parent / 'config.txt'
                if config_path.exists():
                    api_key = config_path.read_text().strip()
                    print(f"✅ Loaded API key from config.txt (length: {len(api_key)})")
                else:
                    print("❌ config.txt not found!")
            else:
                print(f"✅ Using API key from environment (length: {len(api_key)})")
            
            if not api_key:
                print("❌ ERROR: No API key available!")
                self._set_headers(500)
                self.wfile.write(json.dumps({'error': 'API key not configured'}).encode())
                return
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            print(f"Content-Length: {content_length}")
            
            if content_length == 0:
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'No content provided'}).encode())
                return
            
            body = self.rfile.read(content_length)
            data = json.loads(body)
            print(f"Request data keys: {list(data.keys())}")
            
            # Extract system instruction and user message
            system_instruction = None
            if "systemInstruction" in data:
                si = data["systemInstruction"]
                if isinstance(si, str):
                    system_instruction = si
                elif isinstance(si, dict):
                    if "text" in si:
                        system_instruction = si["text"]
                    elif "parts" in si and len(si["parts"]) > 0:
                        system_instruction = si["parts"][0].get("text", "")
            
            # Extract user message from contents
            user_message = ""
            contents = data.get("contents", [])
            if contents and len(contents) > 0:
                # Get the last user message
                for content in reversed(contents):
                    if isinstance(content, dict) and "parts" in content:
                        for part in content.get("parts", []):
                            if "text" in part:
                                user_message = part["text"]
                                break
                        if user_message:
                            break
            
            print(f"System instruction: {system_instruction[:80] if system_instruction else 'None'}...")
            print(f"User message: {user_message[:80] if user_message else 'None'}...")
            
            # Use Google Generative AI SDK - much simpler!
            # First, try to list available models to find a working one
            available_model = None
            try:
                print("Listing available Gemini models...")
                models = genai.list_models()
                # Look for flash or pro models that support generateContent
                for model in models:
                    model_name = model.name.split('/')[-1]  # Extract just the model name
                    if 'generateContent' in model.supported_generation_methods:
                        if 'flash' in model_name.lower():
                            available_model = model_name
                            print(f"✅ Found available model: {available_model}")
                            break
                        elif 'pro' in model_name.lower() and available_model is None:
                            available_model = model_name  # Keep as fallback
                if available_model:
                    print(f"Using model: {available_model}")
            except Exception as e:
                print(f"Could not list models: {e}, using fallback list")
            
            # Fallback list of common model names to try
            if not available_model:
                model_names = [
                    "gemini-1.5-flash-latest",  # Latest stable 1.5 version
                    "gemini-1.5-flash-002",     # Specific version
                    "gemini-1.5-flash",         # Basic name
                    "gemini-1.5-pro-latest",     # Latest stable pro version
                    "gemini-1.5-pro-002",       # Specific pro version
                    "gemini-1.5-pro",           # Basic pro name
                    "gemini-pro",               # Basic model
                ]
            else:
                model_names = [available_model]
            
            last_error = None
            for model_name in model_names:
                try:
                    print(f"Trying model: {model_name}...")
                    # Initialize model
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=system_instruction if system_instruction else None
                    )
                    
                    print(f"Calling Gemini API...")
                    response = model.generate_content(user_message)
                    
                    # Format response to match expected format
                    result = {
                        "candidates": [{
                            "content": {
                                "parts": [{"text": response.text}]
                            }
                        }]
                    }
                    
                    print(f"✅ Success with {model_name}! Response length: {len(response.text)}")
                    self._set_headers(200)
                    self.wfile.write(json.dumps(result).encode())
                    print("✅ Response sent to client")
                    return  # Success, exit the function
                    
                except Exception as e:
                    print(f"❌ {model_name} failed: {e}")
                    last_error = e
                    if available_model:  # If we got this from list_models, don't try others
                        break
                    continue  # Try next model
            
            # If all models failed, raise the last error
            print(f"❌ All models failed. Last error: {last_error}")
            raise last_error if last_error else Exception("All Gemini models failed")
            
            print("="*60)
            
        except KeyError as e:
            print(f"❌ Missing header: {e}")
            traceback.print_exc()
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': f'Missing required header: {str(e)}'}).encode())
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': 'Invalid JSON in request body'}).encode())
        except Exception as e:
            print(f"❌ Gemini proxy error: {e}")
            traceback.print_exc()
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': str(e)}).encode())


def main():
    """Start the server"""
    port = 8000
    server_address = ('', port)
    
    # Check for API key
    api_key = os.environ.get('GEMINI_API_KEY')
    loaded_from = 'environment' if api_key else None
    
    if not api_key:
        config_path = Path(__file__).parent / 'config.txt'
        if config_path.exists():
            api_key = config_path.read_text().strip()
            os.environ['GEMINI_API_KEY'] = api_key  # Set it for this process
            loaded_from = 'config.txt'
    
    print("\n" + "="*60)
    print("🏋️  FITNESS COACH BACKEND SERVER")
    print("="*60)
    print(f"\n✅ Server running on http://localhost:{port}")
    print(f"\n📋 Available endpoints:")
    print(f"   GET  /                          - Frontend UI")
    print(f"   POST /api/upload-and-analyze    - Upload video & get analysis")
    print(f"   POST /api/gemini                - Gemini API proxy")
    print(f"\n💡 API Key Status:")
    if api_key:
        print(f"   ✅ Loaded from {loaded_from}")
    else:
        print(f"   ❌ Not found (create config.txt with your API key)")
    print(f"\n🛑 Press Ctrl+C to stop\n")
    print("="*60 + "\n")
    
    httpd = HTTPServer(server_address, FitnessAPIHandler)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down server...")
        httpd.shutdown()
        print("✅ Server stopped\n")


if __name__ == '__main__':
    main()

