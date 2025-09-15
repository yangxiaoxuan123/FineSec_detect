import csv
import tiktoken
from openai import OpenAI
from typing import Dict, Any
import time
import os

class MultiAgentDataDistillation:
    def __init__(self, api_key: str = None, base_url: str = "https://api.openai.com/v1"):
        # Use environment variable or provided API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide api_key parameter.")
            
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.encoding = tiktoken.encoding_for_model("gpt-4o")
        self.max_tokens = 3700
        
        # Load professional knowledge
        self.professional_knowledge = self._load_professional_knowledge("professional_knowledge.txt")
        
        # Agent configuration
        self.agents = {
            "scenario_extractor": {
                "model": "gpt-4o-mini",
                "system_prompt": """Your primary task is to extract the brief scenario from code snippets in 50 words or less.
                Focus on the main functionality and purpose of the code."""
            },
            "vulnerability_analyzer": {
                "model": "gpt-4o",
                "system_prompt": f"""You are a security expert analyzing code for vulnerabilities. 
                Follow these steps for each vulnerability:
                1. Identify and label the vulnerability (not limited to provided knowledge)
                2. List the specific code lines containing the vulnerability
                3. Provide detailed rationale including exploit mechanisms
                
                Professional Knowledge: {self.professional_knowledge[:1500]}..."""
            },
            "code_generator": {
                "model": "gpt-4o",
                "system_prompt": """You are a code generation specialist. Create C functions that intentionally 
                include specific vulnerabilities with explanatory comments. Generate only the core function without 
                main() or #include statements."""
            }
        }

    def _load_professional_knowledge(self, file_path: str) -> str:
        """Load professional knowledge from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                print(f"Loaded professional knowledge: {len(content)} characters")
                return content
        except FileNotFoundError:
            print(f"Warning: Professional knowledge file {file_path} not found.")
            return "Common vulnerabilities include buffer overflows, integer overflows, use-after-free, format string vulnerabilities, and improper input validation."

    def scenario_extractor_agent(self, code_snippet: str) -> str:
        """Scenario Extractor Agent"""
        try:
            print("  Running Scenario Extractor Agent...")
            completion = self.client.chat.completions.create(
                model=self.agents["scenario_extractor"]["model"],
                messages=[
                    {
                        "role": "system",
                        "content": self.agents["scenario_extractor"]["system_prompt"]
                    },
                    {
                        "role": "user",
                        "content": f"Extract the brief scenario (max 50 words) from this code:\n\n{code_snippet}"
                    }
                ],
                max_tokens=100,
                temperature=0.1,
                stream=False,
            )
            scenario = completion.choices[0].message.content.strip()
            print(f"  Scenario extracted: {scenario[:50]}...")
            return scenario
        except Exception as e:
            print(f"Scenario extractor error: {e}")
            return ""

    def vulnerability_analyzer_agent(self, code_snippet: str) -> str:
        """Vulnerability Analyzer Agent"""
        try:
            print("  Running Vulnerability Analyzer Agent...")
            completion = self.client.chat.completions.create(
                model=self.agents["vulnerability_analyzer"]["model"],
                messages=[
                    {
                        "role": "system",
                        "content": self.agents["vulnerability_analyzer"]["system_prompt"]
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze this C/C++ code for security vulnerabilities:

{code_snippet}

Provide your analysis in the following format:
1. Vulnerability: [Label]
   Lines: [Line numbers]
   Rationale: [Detailed explanation]

2. Vulnerability: [Label]
   Lines: [Line numbers]
   Rationale: [Detailed explanation]"""
                    }
                ],
                max_tokens=800,
                temperature=0.2,
                stream=False,
            )
            analysis = completion.choices[0].message.content.strip()
            print(f"  Vulnerabilities analyzed: {len(analysis)} characters")
            return analysis
        except Exception as e:
            print(f"Vulnerability analyzer error: {e}")
            return ""

    def code_generator_agent(self, rationale: str, scenario: str) -> str:
        """Code Generator Agent"""
        try:
            print("  Running Code Generator Agent...")
            completion = self.client.chat.completions.create(
                model=self.agents["code_generator"]["model"],
                messages=[
                    {
                        "role": "system",
                        "content": self.agents["code_generator"]["system_prompt"]
                    },
                    {
                        "role": "user",
                        "content": f"""Generate a C function that intentionally includes the vulnerabilities described below.

Vulnerability Rationale:
{rationale}

Function Scenario:
{scenario}

Requirements:
- Generate only the function code (no main() or #include)
- Include comments explaining each vulnerability
- Make the vulnerabilities clearly visible but realistic"""
                    }
                ],
                max_tokens=600,
                temperature=0.3,
                stream=False,
            )
            generated_code = completion.choices[0].message.content.strip()
            print(f"  Code generated: {len(generated_code)} characters")
            return generated_code
        except Exception as e:
            print(f"Code generator error: {e}")
            return ""

    def process_single_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process single row through the complete agent pipeline"""
        result_row = row.copy()
        code_snippet = row.get('code', '')
        
        # Check token limit
        code_tokens = len(self.encoding.encode(code_snippet))
        if code_tokens > self.max_tokens:
            print(f"Skipping row due to token limit: {code_tokens} > {self.max_tokens}")
            result_row['sce'] = ""
            result_row['result'] = ""
            result_row['last'] = ""
            return result_row
        
        print(f"Processing code snippet ({code_tokens} tokens)...")
        
        # Agent pipeline execution
        scenario = self.scenario_extractor_agent(code_snippet)
        analysis = self.vulnerability_analyzer_agent(code_snippet)
        
        result_row['sce'] = scenario
        result_row['result'] = analysis
        
        # Generate code if we have sufficient information
        if scenario and analysis:
            generated_code = self.code_generator_agent(analysis, scenario)
            result_row['last'] = generated_code
        else:
            result_row['last'] = ""
            print("  Skipping code generation due to missing scenario or analysis")
        
        return result_row

    def process_csv(self, input_file: str, output_file: str,
                   start_index: int = None, end_index: int = None,
                   delay: float = 1.0):
        """Process CSV file through the complete pipeline"""
        
        results = []
        
        with open(input_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            original_fields = reader.fieldnames
            all_fields = list(original_fields) + ['sce', 'result', 'last']
            
            for index, row in enumerate(reader):
                # Handle index range
                if start_index is not None and index < start_index:
                    continue
                if end_index is not None and index >= end_index:
                    break
                
                print(f"\n=== Processing Row {index} ===")
                
                # Process single row
                processed_row = self.process_single_row(row)
                results.append(processed_row)
                
                print(f"Completed Row {index}")
                
                # Rate limiting
                if delay > 0:
                    time.sleep(delay)
        
        # Write results
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nProcessing complete! Results saved to {output_file}")
        print(f"Total rows processed: {len(results)}")

# Usage example
if __name__ == "__main__":
    # Initialize multi-agent system
    # Option 1: Set OPENAI_API_KEY environment variable
    # Option 2: Pass API key directly (not recommended for production)
    
    agent_system = MultiAgentDataDistillation()
    
    # Process files
    input_file = "input.csv"
    output_file = "distilled_output.csv"
    
    # Execute processing
    agent_system.process_csv(
        input_file=input_file,
        output_file=output_file,
        start_index=0,      # Start from row 0
        end_index=5,        # Process first 5 rows (for testing)
        delay=1.0           # 1 second delay between API calls
    )
