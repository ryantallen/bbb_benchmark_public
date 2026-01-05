# AI Strategy Benchmarker - Back Bay Battery Simulation
# This script runs AI models through Harvard Business School's Back Bay Battery simulation
# to evaluate their strategic decision-making capabilities across multiple runs and years.

# Core system imports
import os
import time
import json
from litellm import completion, cost_per_token  # LiteLLM for multi-provider AI API calls
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Custom modules for simulation management
from hbsp_login import HBSPLoginManager        # Handles login to HBS Publishing platform
from backbay_driver import BackBayDriver       # Main simulation interaction driver
from reset_backbay import SimulationResetter   # Resets simulation between runs
from masking import mask_messages

# Selenium imports for web automation
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === CONFIGURATION AND CREDENTIALS ===

# HBS Publishing login STUDENT credentials (stored in environment variables for security)
username = os.environ.get("HBSP_USERNAME")
password = os.environ.get("HBSP_PASSWORD")

# API keys for various AI providers (all stored in environment variables)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")     # For GPT models
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") # For Claude models
XAI_API_KEY = os.environ.get("XAI_API_KEY")           # For Grok models
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")     # For Google Gemini models
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") # For OpenRouter models
OPENROUTER_API_BASE="https://openrouter.ai/api/v1"

# Simulation configuration
coursepack_url = os.environ.get("HBSP_COURSEPACK_URL")  # HBS coursepack URL for Back Bay Battery
if not coursepack_url:
    raise ValueError(
        "Environment variable HBSP_COURSEPACK_URL is not set. Add it to your .env file."
    )
market_segment = 6 # The (Legacy) market segment for the simulation - determines market dynamics
print_all_io = False

# === PROMPT TEMPLATE FILES ===

# Text content file names - modify these to use different prompt files
SCRIPT_DIR = Path(__file__).resolve().parent
# Assumes this script folder lives at repo root (e.g., `simulation/` or `@simulation/`)
REPO_ROOT = SCRIPT_DIR.parent

FORMAT_INSTRUCTIONS_FILE = SCRIPT_DIR / "format_instructions.txt"  # Specifies response format for AI models

# Load format instructions from external file to allow easy modification without code changes
with FORMAT_INSTRUCTIONS_FILE.open("r", encoding="utf-8") as f:
    FORMAT_INSTRUCTIONS = f.read().strip()   # Formatting requirements for AI responses

# === EXPERIMENT PARAMETERS ===

# Dictionary of instruction sets - different strategic objectives to test AI decision-making
# Each instruction set gives the AI a different goal to optimize for during the simulation
instruction_sets = {
    "default_instructions": "Please make the requested strategic decisions for the upcoming year based on the information provided.",
    "emerging_tech_instructions": "Please make the requested strategic decisions for the upcoming year based on the information provided. **IMPORTANT**: your goal is to make investments to maximize *revenue from the emerging technology* by the end of year 10, while still maintaining profitability in existing markets.",

}

# Dictionary of background sets - different simulation contexts to test AI decision-making
# Each background set provides different company context or scenario details
background_sets = {
    "masked_background": "background_masked.txt",
    # "background_original": "background_original.txt"
}

# Experiment configurations - specify which models to test and how many runs per model
# Each configuration is [model_name, num_runs, instruction_set_name, background_set_name, difficulty]
# Uncomment/modify configurations to run different experiments
configurations = [
    # Format: [model, num_runs, instruction_set, background_set, mask_all_input, difficulty]
   
#     ############### ADVANCED MODE MODELS ###############
#     # # === OPENAI MODELS === 
#     ["openai/gpt-3.5-turbo-0125", 30, 'default_instructions', 'masked_background', "masked", "Advanced"], 
#     ["openai/gpt-4-turbo-2024-04-09", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["openai/gpt-4o-2024-08-06", 30, 'default_instructions', 'masked_background', "masked", "Advanced"], 
#     ["openai/o3-mini-2025-01-31", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["openai/o4-mini-2025-04-16", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["openai/o3-2025-04-16", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["openai/gpt-5-2025-08-07", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["openai/gpt-5.2-2025-12-11", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],

#     # === ANTHROPIC MODELS ===
#     ["anthropic/claude-3-5-sonnet-20240620", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["anthropic/claude-3-5-haiku-20240307", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["anthropic/claude-3-7-sonnet-20250219", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["anthropic/claude-sonnet-4-20250514", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["anthropic/claude-sonnet-4-5-20250929", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["anthropic/claude-opus-4-5-20251101", 15, 'default_instructions', 'masked_background', "masked", "Advanced"],

#     # # === XAI (GROK) MODELS ===
#     ["xai/grok-3-mini", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["xai/grok-3", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["xai/grok-4-0709", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
    
#     # # === GOOGLE GEMINI MODELS ===
#     ["gemini/gemini-1.5-flash", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["gemini/gemini-2.0-flash", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["gemini/gemini-2.5-flash", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["gemini/gemini-2.5-pro", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
#     ["gemini/gemini-3-pro-preview", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],

# ############### BASIC MODE MODELS ###############
#     # === OPENAI MODELS === 
#     ["openai/gpt-3.5-turbo-0125", 30, 'default_instructions', 'masked_background', "masked", "Basic"], 
#     ["openai/gpt-4-turbo-2024-04-09", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["openai/gpt-4o-2024-08-06", 30, 'default_instructions', 'masked_background', "masked", "Basic"], 
#     ["openai/o3-mini-2025-01-31", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["openai/o4-mini-2025-04-16", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["openai/o3-2025-04-16", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["openai/gpt-5-2025-08-07", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["openai/gpt-5.2-2025-12-11", 30, 'default_instructions', 'masked_background', "masked", "Basic"],

#     # === ANTHROPIC MODELS ===
#     ["anthropic/claude-3-5-sonnet-20240620", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["anthropic/claude-3-5-haiku-20240307", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["anthropic/claude-3-7-sonnet-20250219", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["anthropic/claude-sonnet-4-20250514", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["anthropic/claude-sonnet-4-5-20250929", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["anthropic/claude-opus-4-5-20251101", 15, 'default_instructions', 'masked_background', "masked", "Basic"],

#     # # === XAI (GROK) MODELS ===
#     ["xai/grok-3-mini", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["xai/grok-3", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["xai/grok-4-0709", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
    
#     # # === GOOGLE GEMINI MODELS ===
#     ["gemini/gemini-1.5-flash", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["gemini/gemini-2.0-flash", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["gemini/gemini-2.5-flash", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["gemini/gemini-2.5-pro", 30, 'default_instructions', 'masked_background', "masked", "Basic"],
#     ["gemini/gemini-3-pro-preview", 30, 'default_instructions', 'masked_background', "masked", "Basic"],

#     ############### EMERGING TECH REVENUE INSTRUCTIONS ###############
#     # # # === OPENAI MODELS === 
#     ["openai/gpt-3.5-turbo-0125", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"], 
#     ["openai/gpt-4-turbo-2024-04-09", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
#     ["openai/gpt-4o-2024-08-06", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"], 
#     ["openai/o3-mini-2025-01-31", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
#     ["openai/o4-mini-2025-04-16", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
#     ["openai/o3-2025-04-16", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
#     ["openai/gpt-5-2025-08-07", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],

#     # === ANTHROPIC MODELS ===
#     #["anthropic/claude-3-5-haiku-20240307", 30, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
#     ["anthropic/claude-3-5-sonnet-20240620", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
#     ["anthropic/claude-3-7-sonnet-20250219", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
#     ["anthropic/claude-sonnet-4-20250514", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],

#     # # === XAI (GROK) MODELS ===
#     ["xai/grok-3-mini", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
#     ["xai/grok-3", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
#     ["xai/grok-4-0709", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
    
#     # # === GOOGLE GEMINI MODELS ===
#     ["gemini/gemini-1.5-flash", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
#     ["gemini/gemini-2.0-flash", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
#     ["gemini/gemini-2.5-flash", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],
#     ["gemini/gemini-2.5-pro", 20, 'emerging_tech_instructions', 'masked_background', "masked", "Advanced"],

    ########### OPEN-WEIGHT VIA OPENROUTER ###########
    # === GEMMA (Google) ===
    # ["openrouter/google/gemma-3-27b-it", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
    # ["openrouter/google/gemma-3-12b-it", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
    # ["openrouter/google/gemma-2-27b-it", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],

    # # === DEEPSEEK ===
    # ["openrouter/deepseek/deepseek-chat-v3-0324", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
    # ["openrouter/deepseek/deepseek-chat-v3.1",     30, 'default_instructions', 'masked_background', "masked", "Advanced"],
    # ["openrouter/deepseek/deepseek-r1-0528",       30, 'default_instructions', 'masked_background', "masked", "Advanced"],

    # # === QWEN (Alibaba) ===
    # ["openrouter/qwen/qwen3-235b-a22b-thinking-2507", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
    # ["openrouter/qwen/qwen3-32b",                     30, 'default_instructions', 'masked_background', "masked", "Advanced"],
    # ["openrouter/qwen/qwen-2.5-72b-instruct",   30, 'default_instructions', 'masked_background', "masked", "Advanced"],

    # # === LLAMA (Meta) ===
    # ["openrouter/meta-llama/llama-3.1-70b-instruct", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
    # ["openrouter/meta-llama/llama-3.3-70b-instruct", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
    # ["openrouter/meta-llama/llama-4-maverick",       30, 'default_instructions', 'masked_background', "masked", "Advanced"],

    # # === GPT-OSS (OpenAI) ===
    # ["openrouter/openai/gpt-oss-120b", 30, 'default_instructions', 'masked_background', "masked", "Advanced"],
]

# === OUTPUT DIRECTORIES ===

# Define directories for storing experiment results and logs
results_dir = REPO_ROOT / "raw_results"  # JSON files with detailed run data and decisions
logs_dir = REPO_ROOT / "logs"            # Text files with detailed execution logs

# Create log directory if it doesn't exist (results dir created per model to organize by provider)
logs_dir.mkdir(parents=True, exist_ok=True)

# === UTILITY FUNCTIONS ===

# Helper function to reset the WebDriver when browser issues occur
# This is called when web automation fails and we need a fresh browser session
def reset_driver(current_driver, headless=False):
    """
    Safely closes the current WebDriver and creates a new one with fresh login.
    Used to recover from browser errors or session timeouts.
    """
    try:
        current_driver.quit()  # Close the problematic browser session
    except Exception as e:
        logging.error("Error closing WebDriver during reset: %s", e)
    time.sleep(2)  # Brief pause to ensure clean shutdown
    
    # Create new login manager and get fresh browser session
    login_manager = HBSPLoginManager(username=username, password=password, headless=headless)
    new_driver = login_manager.complete_login_process(
        coursepack_url=coursepack_url
    )
    if new_driver:
        new_driver.delete_all_cookies()  # Clear any lingering session data
    logging.info("WebDriver has been reset.")
    return new_driver

# === COMMAND LINE INTERFACE ===

# Setup argument parser for command line options
parser = argparse.ArgumentParser(description="Run Back Bay Battery simulation experiments.")
parser.add_argument("--headless", action="store_true", help="Run the browser in headless mode.")
args = parser.parse_args()

# === MAIN EXPERIMENT EXECUTION ===

# Outer loop: iterate over each configuration (model, num_runs, instruction_type, background_type, difficulty)
# Each configuration represents a complete experiment (multiple runs of the same model/settings)
for config in configurations:
    MODEL, NUM_RUNS, INSTRUCTION_SET, BACKGROUND_SET, MASK_MODE, difficulty = config
    
    # === FILE AND LOGGING SETUP ===
    
    # Create a unique timestamp for this experiment to avoid file conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate filenames that include all experiment parameters for easy identification
    results_filename = f"all_results_{MODEL}_{MASK_MODE}_{timestamp}.json"
    log_filename = f"log_{MODEL}_{MASK_MODE}_{timestamp}.txt"

    results_path = results_dir / Path(results_filename)
    log_path = logs_dir / Path(log_filename)

    # Ensure the log directory exists (create any subdirectories if model name contains slashes)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # === LOGGING CONFIGURATION ===
    
    # Configure dedicated logger for this experiment (prevents log mixing between models)
    logger = logging.getLogger(f"{MODEL}_{timestamp}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent messages from being passed to root logger

    # Create file handler which logs even debug messages to experiment-specific file
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler for consistent log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)

    # Log experiment start
    logger.info(f"Starting simulation run for model: {MODEL}")
    logger.info(f"Difficulty set to: {difficulty}")

    # === EXPERIMENT METADATA ===
    
    # Load background for this configuration
    BACKGROUND_FILE = background_sets[BACKGROUND_SET]
    background_path = SCRIPT_DIR / BACKGROUND_FILE
    with background_path.open("r", encoding="utf-8") as f:
        BACKGROUND = f.read().strip()            # Multi-paragraph context about the simulation
    
    # Define header to store all experiment parameters for the current model
    # This metadata is saved with results to enable later analysis and reproducibility
    header = {
        "MODEL": MODEL,                                    # AI model being tested
        "NUM_RUNS": NUM_RUNS,                             # Number of simulation runs
        "difficulty": difficulty,                          # Simulation difficulty setting
        "market_segment": market_segment,                  # Market segment for simulation
        "instructions": instruction_sets[INSTRUCTION_SET], # Strategic instructions given to AI
        "instruction_set": INSTRUCTION_SET,               # Name of instruction set used
        "background_set": BACKGROUND_SET,                 # Name of background set used                   
        "format_instructions": str(FORMAT_INSTRUCTIONS_FILE),  # Response format file
        "timestamp": timestamp                            # When experiment was run
    }

    logger.debug(f"Header configuration: {json.dumps({k: v for k, v in header.items() if k not in ['background']})}")

    # === RESULTS STRUCTURE INITIALIZATION ===
    
    # Initialize the results dictionary with metadata header and container for run-specific results
    all_results = {
        "header": header,  # Experiment configuration and metadata
        "runs": {}        # Will be filled as each run completes - stores decisions and outcomes
    }

    # === INITIAL SIMULATION RESET ===
    
    # Reset simulation to ensure clean starting state before any AI runs begin
    # This ensures all models start from the same baseline conditions
    print("\n=== Initial simulation reset to ensure fresh start ===")
    logger.info("Performing initial simulation reset before runs")
    init_resetter = SimulationResetter(difficulty=difficulty, market_segment=market_segment, headless=args.headless,coursepack_url=coursepack_url)
    init_resetter.reset_simulation(collect_data=False)  # Reset without collecting data (just clean slate)
    init_resetter.close_browser()
    logger.info("Initial simulation reset completed")

    # === INDIVIDUAL SIMULATION RUNS ===
    
    # Execute the specified number of runs for this model configuration
    # Each run is a complete 8-year simulation (years 3-10) with independent AI decisions
    for run in range(1, NUM_RUNS + 1):
        print(f"\n\n\n=== Model: {MODEL} | Starting Run {run}/{NUM_RUNS} ===")
        logger.info(f"Starting run {run}/{NUM_RUNS}")
        
        # Initialize container for storing this run's results (decisions, outcomes, errors)
        run_results = {}

        try:
            # === BROWSER SETUP AND LOGIN ===
            
            # Login to HBS Publishing platform and get WebDriver instance for simulation access
            print("\n=== Logging in to HBS Publishing ===")
            logger.info("Initiating login to HBS Publishing")
            login_manager = HBSPLoginManager(username=username, password=password, headless=args.headless)
            web_driver = login_manager.complete_login_process(
                coursepack_url=coursepack_url
            )
            
            # Handle login failures gracefully - skip this run but continue with others
            if not web_driver:
                error_msg = f"Failed to login or launch HBSP simulation for run {run}. Skipping to next run."
                print(error_msg)
                logger.error(error_msg)
                run_results["error"] = "Login failure."
                all_results["runs"][f"run_{run}"] = run_results
                continue  # Move to the next run
            
            # === SIMULATION DRIVER INITIALIZATION ===
            
            # Initialize the Back Bay Driver for web automation and data extraction
            print(f"\n=== Starting simulation for run {run} ===")
            logger.info("Initializing BackBayDriver")
            backbay = BackBayDriver(web_driver, difficulty=difficulty)
            
            # === YEARLY SIMULATION LOOP ===
            
            # Execute strategic decision-making for simulation years 3-10
            # Years 1-2 are pre-set; AI makes decisions starting from year 3
            for year in range(3, 11):
                print(f"\n\n== Model: {MODEL} | Run {run}, Year {year} ==")
                logger.info(f"Processing year {year}")
                
                try:
                    # === INSTRUCTION PREPARATION ===
                    
                    # Get the instruction template and format it with the current year if it contains {year}
                    # This allows year-specific instructions (e.g., "for year 5, focus on...")
                    instruction_template = instruction_sets[INSTRUCTION_SET]
                    current_instructions = instruction_template.format(year=year) if "{year}" in instruction_template else instruction_template
                    
                    # === DATA COLLECTION FROM SIMULATION ===
                    
                    print("\n= Collecting data =")
                    logger.debug(f"Calling get_data_for_step for year {year}")
                    
                    # Attempt to get simulation data with retry logic for browser issues
                    # Sometimes the web automation fails due to page loading or network issues
                    max_attempts = 2
                    attempt = 0
                    while attempt < max_attempts:
                        try:
                            data = backbay.get_data_for_step(year)  # Extract current market data and company status
                            break  # Success - exit retry loop
                        except Exception as e:
                            logger.error(f"Error in get_data_for_step for year {year}: {e}")
                            attempt += 1
                            if attempt < max_attempts:
                                # Reset browser and try again before giving up
                                logger.info(f"Resetting WebDriver and retrying get_data_for_step for year {year}.")
                                web_driver = reset_driver(web_driver, headless=args.headless)
                                backbay.driver = web_driver
                                time.sleep(2)
                            else:
                                raise e  # Give up after max attempts
                    
                    # === DATA VALIDATION ===
                    
                    # Check if data collection encountered any errors (network issues, page changes, etc.)
                    if data.get("errors"):
                        error_msg = f"Encountered error(s) for Year {year}: {data['errors']}"
                        print(error_msg)
                        logger.error(error_msg)
                        run_results[f"year_{year}"] = {"error": data["errors"]}
                        continue  # Skip to the next year
                    
                    # === DATA FORMATTING FOR AI ===
                    
                    print("\n= Formatting data =")
                    logger.debug(f"Calling format_data_for_ai for year {year}")
                    # Convert raw simulation data into structured format that AI can understand
                    formatted_data = backbay.format_data_for_ai(data, year)
                    
                    # === PAST DECISIONS CONTEXT ===
                    
                    # Build a string summarizing past decisions to provide AI with strategic context
                    # This helps the AI maintain consistency and learn from previous year outcomes
                    past_decisions_str = ""
                    if year > 3:  # Only include past decisions starting from year 4
                        last_year_key = f"year_{year - 1}"
                        if last_year_key in run_results:
                            past_decision = run_results[last_year_key]
                            past_decisions_str = f"Decisions for Year {year - 1}:\n```\n{past_decision}\n```\n"
                    if not past_decisions_str:
                        past_decisions_str = "None"  # First year or no previous decisions available
                    
                    # === PROMPT PREPARATION ===
                    
                    # Prepare the messages list with separate roles for AI API call
                    # Two versions: one with system role, one without (some models don't support system role)
                    prompt = [
                        {"role": "system", "content": f"# BACKGROUND\n{BACKGROUND}\n\n# FORMATTING INSTRUCTIONS\n{FORMAT_INSTRUCTIONS}"},
                        {"role": "user", "content": f"# INSTRUCTIONS\n{current_instructions}\n\n# DATA\n{formatted_data}\n\n# PAST DECISIONS\n{past_decisions_str}"}
                    ]
                    prompt_no_roles = [  # Fallback version if system role isn't supported
                        {"role": "user", "content": f"# BACKGROUND\n{BACKGROUND}\n\n# FORMATTING INSTRUCTIONS\n{FORMAT_INSTRUCTIONS}"},
                        {"role": "user", "content": f"# INSTRUCTIONS\n{current_instructions}\n\n# DATA\n{formatted_data}\n\n# PAST DECISIONS\n{past_decisions_str}"}
                    ]
                    # Log the user prompt content for debugging (excluding lengthy background)
                    logger.debug(f"Year {year} prompt: {prompt[1]['content']}")
                    
                    # === AI API CALL ===
                    
                    # Submit prompt to AI model and get strategic decision response
                    print("\n= Requesting AI response =")
                    logger.info(f"Requesting AI response for year {year}")
                    no_roles = False
                    try:
                        # Try with system role first (preferred format)
                        prompt_to_send = mask_messages(prompt, mode=MASK_MODE)
                        
                        if print_all_io:
                            print("\n" + "="*50)
                            print("LLM INPUT (masked messages):")
                            print("="*50)
                            for i, msg in enumerate(prompt_to_send):
                                print(f"Message {i+1} [{msg.get('role', 'unknown')}]:")
                                print(msg.get('content', ''))
                                print("-" * 30)
                        
                        response = completion(model=MODEL, messages=prompt_to_send,stream=False)
                        print("\n" + "="*50)
                        print("RESPONSE:")
                        print("="*50)
                        print(response["choices"][0]["message"]["content"])
                        print("="*50)
                        
                    except Exception as e:
                        # Fallback to user-only format if system role fails
                        no_roles = True
                        logger.warning(f"Failed with system role, retrying without: {str(e)}")
                        prompt_no_roles_to_send = mask_messages(prompt_no_roles, mode=MASK_MODE)
                        
                        if print_all_io:
                            print("\n" + "="*50)
                            print("LLM INPUT (masked messages - no roles fallback):")
                            print("="*50)
                            for i, msg in enumerate(prompt_no_roles_to_send):
                                print(f"Message {i+1} [{msg.get('role', 'unknown')}]:")
                                print(msg.get('content', ''))
                                print("-" * 30)
                        
                        response = completion(model=MODEL, messages=prompt_no_roles_to_send,stream=False)
                        print("\n" + "="*50)
                        print("RESPONSE:")
                        print("="*50)
                        print(response["choices"][0]["message"]["content"])
                        print("="*50)
                    
                    # Extract the AI's strategic decision text
                    response_text = response["choices"][0]["message"]["content"]
                    
                    if print_all_io:
                        print("\n" + "="*50)
                        print("LLM OUTPUT:")
                        print("="*50)
                        print(response_text)
                        print("="*50)
                    
                    # === TOKEN COUNTING AND COST CALCULATION ===

                    usage = response.get("usage", {}) or {}
                    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    output_tokens = int(usage.get("completion_tokens", 0) or 0)
                    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)

                    # Defaults in case pricing lookup fails
                    prompt_cost = 0.0
                    completion_cost = 0.0
                    estimated_cost = 0.0

                    try:
                        prompt_cost, completion_cost = cost_per_token(
                            model=MODEL,
                            prompt_tokens=input_tokens,
                            completion_tokens=output_tokens,
                        )
                        estimated_cost = prompt_cost + completion_cost
                    except Exception as e:
                        logger.warning(f"Skipping cost calc for {MODEL}: {e}")

                    # Console output (always)
                    print(f"Input tokens: {input_tokens}")
                    print(f"Output tokens: {output_tokens}")
                    print(f"Total tokens: {total_tokens}")
                    print(f"Estimated total cost: ${estimated_cost:.6f}")

                    # Structured logs
                    logger.info(
                        f"[{MODEL}] Tokens → input={input_tokens}, output={output_tokens}, total={total_tokens}"
                    )
                    logger.info(
                        f"[{MODEL}] Cost   → prompt=${prompt_cost:.6f}, completion=${completion_cost:.6f}, total=${estimated_cost:.6f}"
                    )

                    
                    
                    # === RESPONSE PARSING AND VALIDATION ===
                    
                    print("\n= Parsing AI response =")
                    logger.debug("Parsing AI response")
                    
                    # Extract R&D budget limit from simulation data for validation
                    rd_budget_limit = data.get("rd_budget_limit")
                    if rd_budget_limit is None:
                        error_msg = f"ERROR: No R&D budget limit found in data for Year {year}, Run {run}. Cannot continue."
                        print(error_msg)
                        logger.error(error_msg)
                        run_results[f"year_{year}"] = {"error": error_msg}
                        break  # End simulation for this run
                        
                    # Parse the AI's text response into structured decision data
                    # This extracts production volumes, prices, and R&D allocations from the response
                    decisions = backbay.parse_ai_response(response_text, rd_budget_limit)
                    logger.debug(f"Parsed decisions: {json.dumps(decisions)}")
                    
                    # === PARSING FAILURE RETRY LOGIC ===
                    
                    # Retry parsing if AI response format was incorrect or couldn't be parsed
                    # This gives the AI multiple chances to provide properly formatted responses
                    parsing_retry_attempts = 0
                    max_parsing_retries = 3
                    while (decisions["rationale"] == "Error parsing AI response." and 
                           parsing_retry_attempts < max_parsing_retries):
                        error_msg = "ERROR: AI response parsing failed. Requesting a new complete response with stricter formatting adherence."
                        print(error_msg)
                        logger.error(error_msg)
                        
                        # Send clarified formatting instructions to the AI
                        parsing_error_message = f'''
Your previous response did not follow the requested formatting correctly.
Please provide a NEW COMPLETE RESPONSE strictly adhering to the following formatting. First, print the following 14 numbers, separated by commas:
AGM units (in Millions), SC units (in Millions), AGM price (in $/unit), SC price (in $/unit), R&D dollar allocations for 5 AGM areas (in $M), R&D dollar allocations for 5 SC areas (in $M)

After listing just those 14 numbers (no units or words), provide a brief written rationale for your decisions. Ensure that your response exactly follows this format.
'''
                        # Append the error message to the existing conversation
                        retry_prompt_parsing = prompt_no_roles.copy() if no_roles else prompt.copy()
                        retry_prompt_parsing.append({"role": "user", "content": parsing_error_message})
                        
                        # Log the retry attempt for debugging
                        logger.debug(f"Retry parsing prompt: {parsing_error_message}")
                        retry_prompt_to_send = mask_messages(retry_prompt_parsing, mode=MASK_MODE)
                        response = completion(model=MODEL, messages=retry_prompt_to_send)
                        response_text = response["choices"][0]["message"]["content"]
                        logger.debug(f"Retry parsing response: {response_text}")
                        
                        # Try to parse the revised response
                        decisions = backbay.parse_ai_response(response_text, rd_budget_limit)
                        parsing_retry_attempts += 1
                        print(f"Parsing retry attempt {parsing_retry_attempts}: Checking parsed response.")
                        logger.info(f"Parsing retry attempt {parsing_retry_attempts}")
                    
                    # Give up if parsing continues to fail after multiple attempts
                    if decisions["rationale"] == "Error parsing AI response.":
                        error_msg = f"ERROR: Unable to parse AI response after {max_parsing_retries} attempts."
                        print(error_msg)
                        logger.error(error_msg)
                        run_results[f"year_{year}"] = {"error": error_msg}
                        break  # End simulation for this run

                    # === R&D BUDGET VALIDATION ===
                    
                    # Calculate total R&D spending from AI's allocations to check against budget limit
                    total_rd = sum(decisions['agm_rd_allocations']) + sum(decisions['supercapacitor_rd_allocations'])
                    retry_attempts = 0
                    max_retries = 3  # Maximum number of retries for R&D budget issues
                    
                    # Retry if AI exceeded the R&D budget (common issue with AI strategic decisions)
                    while total_rd > rd_budget_limit and retry_attempts < max_retries:
                        error_msg = f"WARNING: Total R&D budget (${total_rd} million) exceeds the limit (${rd_budget_limit} million)!"
                        print(error_msg)
                        logger.warning(error_msg)
                        print("Requesting a new response from the AI within budget limit...")
                        
                        # Explain the budget constraint clearly to the AI
                        budget_error_message = f'''
Your R&D allocations totaled ${total_rd} million, which exceeds the budget limit of ${rd_budget_limit} million.

Please provide a NEW COMPLETE RESPONSE with R&D allocations that stay within the ${rd_budget_limit} million budget limit.

Remember to follow the format:
AGM units (in Millions), SC units (in Millions), AGM price (in $/unit), SC price (in $/unit), R&D dollar allocations for 5 AGM areas (in $M), R&D dollar allocations for 5 SC areas (in $M)

Ensure the total of all 10 R&D allocations is EXACTLY ${rd_budget_limit} million or less.
'''
                        # Append budget correction to conversation
                        retry_prompt_rd = prompt_no_roles.copy() if no_roles else prompt.copy()
                        retry_prompt_rd.append({"role": "user", "content": budget_error_message})
                        
                        # Log the budget retry attempt for debugging
                        logger.debug(f"Budget retry prompt: {budget_error_message}")
                        retry_prompt_to_send = mask_messages(retry_prompt_rd, mode=MASK_MODE)
                        response = completion(model=MODEL, messages=retry_prompt_to_send)
                        response_text = response["choices"][0]["message"]["content"]
                        logger.debug(f"Budget retry response: {response_text}")
                        
                        # Parse the corrected response and check budget again
                        decisions = backbay.parse_ai_response(response_text, rd_budget_limit)
                        total_rd = sum(decisions['agm_rd_allocations']) + sum(decisions['supercapacitor_rd_allocations'])
                        retry_attempts += 1
                        print(f"Retry attempt {retry_attempts}: New total R&D budget: ${total_rd} million")
                        logger.info(f"Budget retry attempt {retry_attempts}: New total R&D budget: ${total_rd} million")
                    
                    # Give up if budget issues persist after multiple attempts
                    if total_rd > rd_budget_limit:
                        error_msg = f"\nERROR: After {max_retries} attempts, total R&D budget (${total_rd} million) still exceeds limit (${rd_budget_limit} million)."
                        print(error_msg)
                        logger.error(error_msg)
                        run_results[f"year_{year}"] = {"error": error_msg}
                        break  # End simulation for this run
                    
                    # === DECISION SUBMISSION ===
                    
                    print("\n= Submitting decisions =")
                    logger.info(f"Submitting decisions for year {year} with difficulty={difficulty}")
                    
                    # Submit decisions to simulation with retry logic for dynamic budget updates
                    # Sometimes the simulation updates budget limits based on the pricing decisions
                    submit_attempts = 0
                    max_submit_retries = 3
                    while submit_attempts < max_submit_retries:
                        ok, live_limit = backbay.submit_decision(decisions, difficulty)
                        
                        logger.debug(f"Submit attempt {submit_attempts+1}: ok={ok}, live_limit={live_limit}")
                        submit_attempts += 1

                        if ok:
                            break                 # success!
                        if not live_limit or total_rd <= live_limit:
                            break                 # button failed for some other reason → give up
                        # === DYNAMIC BUDGET LIMIT HANDLING ===
                        
                        # Simulation dynamically updated R&D budget based on revenue forecasts

                        budget_update_msg = f"Surpassed updated R&D budget limit. Updated from ${rd_budget_limit} million to ${live_limit} million based on sales forecasts and prices."
                        print(budget_update_msg)
                        logger.warning(budget_update_msg)

                        # Inform the AI about the updated budget limit and request new allocations
                        warn = (f"Your total R&D (${total_rd} M) exceeds the UPDATED budget "
                                f"limit of ${live_limit} M. The budget limit was updated based on the sales forecasts and unit prices you determined."
                                f" Send a new 14‑number line within that limit.")

                        retry_prompt_rd2 = prompt_no_roles.copy() if no_roles else prompt.copy()
                        retry_prompt_rd2.append({"role": "user", "content": warn})
                        retry_prompt_to_send = mask_messages(retry_prompt_rd2, mode=MASK_MODE)
                        response = completion(model=MODEL, messages=retry_prompt_to_send)
                        decisions = backbay.parse_ai_response(response.choices[0].message.content,
                                                            live_limit)
                        total_rd = sum(decisions['agm_rd_allocations']) + \
                                sum(decisions['supercapacitor_rd_allocations'])
                        

                    # Handle submission failure after all retries
                    if not ok:
                        run_results[f"year_{year}"] = {"error": "Could not submit after "
                                                                f"{max_submit_retries} attempts."}
                        break      # Abandon this run

                    # === DECISION STORAGE AND SIMULATION CHECK ===
                    
                    # Store the successful decisions for this year
                    run_results[f"year_{year}"] = decisions
                    
                    # Check if simulation ended early (e.g., company bankruptcy)
                    print("\n= Checking if simulation ended =")
                    logger.debug("Checking if simulation ended")
                    if backbay.check_simulation_ended():
                        end_msg = f"Simulation ended after year {year} for run {run}."
                        print(end_msg)
                        logger.info(end_msg)
                        break  # Exit yearly loop - this run is complete
                        
                except Exception as e:
                    # === YEAR-LEVEL ERROR HANDLING ===
                    
                    # Handle unexpected errors during year processing - log and continue to next year
                    error_msg = f"Error in run {run}, year {year}: {e}"
                    print(error_msg)
                    logger.error(f"Exception in year {year}: {str(e)}", exc_info=True)
                    run_results[f"year_{year}"] = {"error": error_msg}
                    continue  # Continue with the next year
            
        except Exception as e:
            # === RUN-LEVEL ERROR HANDLING ===
            
            # Handle catastrophic errors that prevent the entire run from completing
            error_msg = f"Unexpected error in run {run}: {e}"
            print(error_msg)
            logger.error(f"Unexpected exception in run {run}: {str(e)}", exc_info=True)
            run_results["error"] = error_msg
            
        finally:
            # === RUN CLEANUP AND RESULTS COLLECTION ===
            
            # Reset simulation and get final results regardless of success/failure
            logger.info("Finalizing run and resetting simulation")
            time.sleep(2)
            
            # === FINAL SUBMISSION ATTEMPT ===
            
            # Try to click any remaining submit button to finalize the simulation
            try:
                submit_button = WebDriverWait(backbay.driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Submit') or contains(@value, 'Submit')]"))
                )
                submit_button.click()
                logger.debug("Final submit button clicked")
            except Exception as e:
                logger.warning(f"No submit button found: {str(e)}")
                print("No submit button found.")
            
            time.sleep(2)
            
            # === SIMULATION RESET AND RESULTS COLLECTION ===
            
            print(f"\n\n=== Resetting simulation and saving results for run {run} ===")
            try:
                # Create fresh SimulationResetter to collect final scores and reset for next run
                logger.info(f"Creating SimulationResetter with difficulty={difficulty} and market_segment={market_segment}")
                resetter = SimulationResetter(difficulty=difficulty, market_segment=market_segment, headless=args.headless,coursepack_url=coursepack_url)
                best_scores, simulation_summary = resetter.reset_simulation()  # Collect performance data
                resetter.close_browser()
                logger.debug("Simulation reset completed")
                
                # Store the final simulation outcomes with this run's results
                run_results["best_scores"] = best_scores          # Performance metrics and rankings
                run_results["simulation_summary"] = simulation_summary  # Detailed outcome summary
            except Exception as e:
                error_msg = f"Error during simulation reset in run {run}: {e}"
                print(error_msg)
                logger.error(f"Error during simulation reset: {str(e)}", exc_info=True)
            
            # === RESULTS STORAGE ===
            
            # Save final run results to the all_results structure
            all_results["runs"][f"run_{run}"] = run_results
            logger.info(f"Run {run} completed, saving results")
            
            # === FILE SYSTEM OPERATIONS ===
            
            # Write complete results to JSON file for analysis and backup
            try:
                # Ensure the results directory exists
                results_path.parent.mkdir(parents=True, exist_ok=True)
                # Write all results (header + all completed runs) to JSON file
                with results_path.open("w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2)
                logger.debug(f"Results saved to {results_path}")
            except Exception as e:
                error_msg = f"Error writing results for run {run}: {e}"
                print(error_msg)
                logger.error(f"Error writing results: {str(e)}", exc_info=True)
            
            # === BROWSER CLEANUP ===
            
            # Close the WebDriver browser session to free resources
            try:
                if web_driver:
                    web_driver.quit()
                    logger.debug("WebDriver closed")
            except Exception as e:
                logger.warning(f"Error closing WebDriver: {str(e)}")
                pass

    # === EXPERIMENT COMPLETION SUMMARY ===
    
    # Display completion status and file locations for this model's experiment
    print(f"\n=== All {NUM_RUNS} runs completed for model {MODEL} with {difficulty} difficulty ===")
    print(f"Full results saved to {results_path}")
    logger.info(f"All {NUM_RUNS} runs completed for model {MODEL} with {difficulty} difficulty")
    logger.info(f"Full log saved to {log_path}")
    
    # === LOGGER CLEANUP ===
    
    # Remove handlers to avoid duplicated logs in the next model iteration
    # This prevents log messages from being written to multiple files when testing multiple models
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
