import re
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class BackBayDriver:
    """
    Selenium WebDriver automation for the Harvard Business School "Back Bay Battery" simulation.
    
    This class handles all interactions with the web-based business simulation including:
    - Navigating between Analyze and Decide tabs
    - Extracting market data, financial reports, and news
    - Parsing R&D budget constraints
    - Submitting pricing and R&D allocation decisions
    - Monitoring simulation completion status
    
    The simulation involves managing two battery product lines (AGM and Supercapacitor)
    with decisions around pricing, unit sales forecasts, and R&D investments across
    five categories: Energy Density, Recharge Cycles, Self Discharge, Recharge Time, 
    and Process Improvement/Pricing.
    """
    
    def __init__(self, driver, difficulty="Intermediate"):
        """
        Initialize the Back Bay Battery simulation driver.
        
        Args:
            driver: Selenium WebDriver instance (already logged in and on simulation page)
            difficulty: Difficulty level of the simulation ("Basic", "Intermediate", or "Advanced")
                        Basic mode has fewer data tabs and simplified inputs
        """
        self.driver = driver
        self.difficulty = difficulty
        
        # Configure which data subtabs are available based on simulation difficulty
        # Sales Variance tab is only available in Intermediate and Advanced modes
        if difficulty == "Basic":
            self.SUBTABS = ["Unit Sales", "Features", "Performance", "Customers", "Income Statement", "R&D Investments"]
        else:
            self.SUBTABS = ["Unit Sales", "Features", "Performance", "Customers", "Income Statement", "Sales Variance", "R&D Investments"]

        # Maps subtab names to their corresponding HTML table container IDs
        # These IDs are used to locate and extract data from each section
        self.TABLE_ID_MAP = {
            "Unit Sales":           "unit-sales-tables",
            "Features":             "features-tables", 
            "Performance":          "performance-tables",
            "Customers":            "customers-tables",
            "Income Statement":     "income-statement-table",
            "Sales Variance":       "sales-variance-tables",
            "R&D Investments":      "rnd-investments-table",
        }

    ## Helper function for current R&D budget limit
    def current_rd_budget_limit(self) -> float | None:
        """
        Extracts the current R&D budget limit from the simulation interface.
        
        The simulation dynamically updates the R&D budget limit based on projected
        revenues after price/unit changes. This method reads the current limit value
        that appears as "Not to Exceed $X Million" text on the page.
        
        Returns:
            float: The R&D budget limit in millions of dollars, or None if not found
        """
        time.sleep(0.5)  # Brief pause to allow UI updates after user input

        # Primary method: Look for "Not to Exceed" text in the main decisions table
        try:
            WebDriverWait(self.driver,10).until(
                EC.text_to_be_present_in_element((By.ID,"decisions-table"),"Not to Exceed")
            )
            tbl = self.driver.find_element(By.ID, "decisions-table").text
            m = re.search(r'Not to Exceed\s*\$?\s*([\d.]+)\s*Million', tbl, re.I)
            if m:
                return float(m.group(1))
        except Exception:
            pass  # Fall through to backup method

        # Backup method: Check tutorial highlight element (sometimes shows budget limit)
        try:
            hdr = self.driver.find_element(
                By.CSS_SELECTOR, "#tutorial-highlight-14"
            ).text
            m = re.search(r'\$\s*([\d.]+)', hdr)
            if m:
                return float(m.group(1))
        except Exception:
            pass  # Budget limit not found
        
        return None
    
    def get_data_for_step(self, year):
        """
        Collect all available data for the current simulation year.
        
        This is the main data extraction method that navigates through the simulation
        interface to gather market data, financial reports, news, and decision inputs.
        The method switches between the Analyze tab (historical data) and Decide tab
        (current year inputs) to collect comprehensive information.
        
        Args:
            year: Current simulation year (e.g., 2025)
            
        Returns:
            dict: Complete data package including:
                - subtabs: Data from all analysis subtabs (Unit Sales, Features, etc.)
                - news: Current year news items
                - decisions_table: Current decision inputs and constraints
                - rd_budget_limit: R&D spending constraint for this year
                - errors: Any extraction errors encountered
        """
        prior_year = year - 1  # Analyze tab shows data from the previous completed year
        
        # Initialize the data structure to collect all simulation information
        data = {
            "year": year,
            "analyze_year": prior_year,  # The year shown in the Analyze tab
            "subtabs": {},
            "news": [],
            "decisions_table": {},
            "errors": []
        }
        
        try:
            time.sleep(1)
            # Ensure we're working with the main simulation window
            if len(self.driver.window_handles) > 1:
                self.driver.switch_to.window(self.driver.window_handles[-1])
            
            # === ANALYZE TAB NAVIGATION ===
            # Navigate to the Analyze tab to collect historical market data
            expected_analyze_year = prior_year
            analyze_tab = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@aria-label, 'Analyze')]"))
            )
            time.sleep(1)
            
            # Wait for any loading animations to complete before proceeding
            wait = WebDriverWait(self.driver, 15, poll_frequency=0.5)
            wait.until(lambda d: d.execute_script(
                "return document.querySelector('.loading .accessibility-text')?.textContent.trim() === 'Finished Loading.'"
            ))
            time.sleep(1)

            # Click the Analyze tab and wait for data to load
            analyze_tab.click()
            wait.until(lambda d: d.execute_script(
                "return document.querySelector('.loading .accessibility-text')?.textContent.trim() === 'Finished Loading.'"
            ))
            time.sleep(1)
            
            # === YEAR VALIDATION ===
            # Verify we're looking at data from the correct year (prior_year)
            expected_text_fragment = f"year {expected_analyze_year}"
            timeline_span = WebDriverWait(self.driver, 10).until(
                EC.text_to_be_present_in_element((By.CSS_SELECTOR, "#timeline span"), expected_text_fragment)
                )
            # Retrieve the full timeline text to validate the year
            timeline_span_element = self.driver.find_element(By.CSS_SELECTOR, "#timeline span")
            analyze_text = timeline_span_element.text
            
            # Check multiple possible year formats in the timeline text
            valid_year = False
            if (str(expected_analyze_year) in analyze_text or 
                f'Year {expected_analyze_year}' in analyze_text or 
                f'year {expected_analyze_year}' in analyze_text.lower() or
                f'from year {expected_analyze_year}' in analyze_text.lower()):
                valid_year = True
            
            # If we're not looking at the right year's data, abort this data collection
            if not valid_year:
                error_msg = f"ERROR: Expected to see 'Year {expected_analyze_year}' in Analyze tab timeline, but got: '{analyze_text}'"
                print(error_msg)
                data.setdefault("errors", []).append(error_msg)
                return data

            # === SUBTAB DATA EXTRACTION ===
            # Iterate through each analysis subtab to collect market and financial data
            for subtab in self.SUBTABS:
                try:
                    # Click on the specific subtab (Unit Sales, Features, Performance, etc.)
                    WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, f"//a[contains(@aria-label, '{subtab}')]"))
                    ).click()
                    
                    # For most subtabs, switch to detailed table view (third view toggle option)
                    # Income Statement and R&D Investments already show in table format
                    if subtab not in ["Income Statement", "R&D Investments"]:
                        WebDriverWait(self.driver, 3).until(
                            EC.element_to_be_clickable((By.XPATH, "//*[@id='view-toggle']/span[3]"))
                        ).click()

                    # Locate the data table using the predefined ID mapping
                    table_div_id = self.TABLE_ID_MAP[subtab]
                    table = self.driver.find_element(By.ID, table_div_id).find_element(By.TAG_NAME, "table")

                    # Extract all table data row by row
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    lines = []
                    for row in rows:
                        # Get both header (th) and data (td) cells from each row
                        cells = row.find_elements(By.XPATH, ".//th|.//td")
                        texts = [c.text.strip() for c in cells]
                        lines.append("\t".join(texts))  # Tab-separated for easy parsing

                    # Convert table data to string format and validate
                    data_str = "\n".join(lines)
                    data["subtabs"][subtab] = data_str if len(data_str) > 10 else "ERROR: Insufficient data"
                    
                except Exception as e:
                    # Log any errors but continue with other subtabs
                    error_msg = f"Error in subtab '{subtab}': {e}"
                    data["errors"].append(error_msg)
                    data["subtabs"][subtab] = f"ERROR: {e}"
            
            # === NEWS EXTRACTION ===
            # Open the news modal to get current market events and updates
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//*[@id='news-toggle-label']"))
            ).click()

            # Wait for news modal to appear
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "news-modal"))
            )

            # Extract all news items using JavaScript for reliable text extraction
            news_texts = self.driver.execute_script(
                "return Array.from(document.querySelectorAll('#news-modal .news-item-container'))"
                ".map(el => el.innerText.trim());"
            )

            # Store news items, with fallback to modal content if individual items not found
            if news_texts:
                data["news"].extend(news_texts)
            else:
                data["news"].append(self.driver.execute_script(
                    "return document.getElementById('news-modal').innerText.trim();"
                ))

            # === DECISION TAB NAVIGATION ===
            # Switch to Decide tab to collect current year's decision inputs and constraints
            decide_tab = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@aria-label, 'Decide')]"))
            )
            decide_tab.click()

            # Wait for decisions table to load and capture its full content
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "decisions-table"))
            )
            time.sleep(0.5)
            decisions_table = self.driver.find_element(By.ID, "decisions-table")
            data["decisions_table_content"] = decisions_table.text

            # === R&D BUDGET LIMIT EXTRACTION ===
            # The simulation dynamically calculates R&D budget based on projected revenues.
            # We need to extract this critical constraint using multiple fallback methods.
            budget_limit = None

            # Method 1: Check tutorial highlight bubble (fastest and most reliable)
            try:
                highlight = self.driver.find_element(
                    By.CSS_SELECTOR, "#tutorial-highlight-14 span"
                ).text
                m = re.search(r'\$\s*([\d.]+)', highlight)
                if m:
                    budget_limit = float(m.group(1))
            except Exception:
                pass  # Continue to next method if highlight not found

            # Method 2: Search for explicit "Not to Exceed $XX Million" text in decisions table
            if budget_limit is None:
                m = re.search(
                    r'Not to Exceed\s*\$?\s*([\d.]+)\s*Million',
                    data["decisions_table_content"],
                    flags=re.IGNORECASE
                )
                if m:
                    budget_limit = float(m.group(1))

            # Method 3: Extract from R&D section - find the last dollar amount
            # The R&D section shows category totals, with the final line being the budget limit
            if budget_limit is None:
                r_and_d_block = re.search(
                    r'R&D Investment[^\n]*?\n(.*?Total R&D Costs[^\n]*?)\n',
                    data["decisions_table_content"],
                    flags=re.IGNORECASE | re.DOTALL
                )
                if r_and_d_block:
                    # Find all dollar amounts and take the last one (budget limit)
                    dollars = re.findall(r'\$\s*([\d.]+)\s*M', r_and_d_block.group(1))
                    if dollars:
                        budget_limit = float(dollars[-1])

            # Store the budget limit or report error if extraction failed
            if budget_limit is not None:
                data["rd_budget_limit"] = budget_limit
            else:
                err = f"ERROR: Could not detect R&D budget limit for Year {year}."
                data.setdefault("errors", []).append(err)
                print(err)
                return data


            # === DECISION INPUT FIELD EXTRACTION ===
            # Extract current values from price and R&D allocation input fields
            data.setdefault("decisions_table", {})

            # Find price input fields - try specific selectors first, then broader search
            price_fields = self.driver.find_elements(By.CSS_SELECTOR, "input[id='Price[AGM]'], input[id='Price[Supercapacitor]']")
            if not price_fields:
                # Fallback: search for any price-related input fields
                price_fields = self.driver.find_elements(By.CSS_SELECTOR, "input[type='text'][id*='price'], input[id*='Price']")

            # Find R&D allocation input fields
            rd_fields = self.driver.find_elements(By.CSS_SELECTOR, "input[id*='R_and_D_Spending']")
            if not rd_fields:
                # Fallback: search for any R&D-related input fields
                rd_fields = self.driver.find_elements(By.CSS_SELECTOR, "input[type='text'][id*='rd'], input[id*='allocation']")

            # Extract current values from all found input fields
            for field in price_fields + rd_fields:
                field_id = field.get_attribute("id")
                
                # Try to find associated label for better field identification
                label_elements = self.driver.find_elements(By.CSS_SELECTOR, f"label[for='{field_id}'], label[id='label-{field_id}']")
                label_text = label_elements[0].text if label_elements else field_id.replace('[', ' ').replace(']', '').replace('_', ' ')
                
                # Get current field value and clean up currency symbols
                field_value = field.get_attribute("value")
                if field_value and field_value.startswith('$'):
                    field_value = field_value[1:]  # Remove dollar sign
                    
                data["decisions_table"][label_text] = field_value

            return data

        except Exception as e:
            # Handle any unexpected errors gracefully
            error_msg = f"ERROR: Could not get data for year {year}: {e}"
            print(error_msg)
            data.setdefault("errors", []).append(error_msg)
            # Return partial data rather than failing completely
            return data


    
    def format_data_for_ai(self, data, year):
        """
        Format the collected simulation data into a comprehensive prompt for AI decision-making.
        
        This method transforms the raw simulation data into a structured, readable format
        that provides the AI with all necessary context including historical performance,
        market conditions, financial constraints, and current input values.
        
        Args:
            data: Dictionary containing all collected simulation data from get_data_for_step()
            year: Current simulation year for which decisions need to be made
            
        Returns:
            str: Formatted prompt text ready for AI consumption, including:
                 - R&D budget constraints (prominently featured)
                 - Historical market and financial data
                 - Current news and market conditions
                 - Existing decision input values
        """
        prior_year = year - 1  # Historical data is from the previous year
        
        formatted_data = f"Data from Year {prior_year}.\n\n"
        
        # === R&D BUDGET CONSTRAINT EXTRACTION ===
        # Critical constraint that must be prominently featured for the AI
        rd_budget = None
        if "rd_budget_limit" in data:
            rd_budget = data["rd_budget_limit"]
        else:
            # Backup: try to extract from decision table labels
            for key, value in data["decisions_table"].items():
                if "R&D Investment" in key:
                    budget_match = re.search(r'\$(\d+\.?\d*)', key)
                    if budget_match:
                        rd_budget = float(budget_match.group(1))
                        break

        # Feature the R&D budget constraint prominently at the top
        if rd_budget is not None:
            formatted_data += f"## IMPORTANT: R&D BUDGET LIMIT FOR YEAR {year}: ${rd_budget} MILLION\n"
            formatted_data += "The total R&D allocation across all categories for both AGM and Supercapacitor MUST NOT exceed this limit.\n\n"
        
        # === HISTORICAL ANALYSIS DATA ===
        formatted_data += "## ANALYSIS DATA\n\n"
        for subtab, content in data["subtabs"].items():
            formatted_data += f"### {subtab}\n"
            formatted_data += f"```\n{content}\n```\n\n"
        
        # === CURRENT MARKET NEWS ===
        formatted_data += "## NEWS\n\n"
        for news_item in data["news"]:
            formatted_data += f"{news_item}\n\n"

        # === DECISION CONTEXT AND CONSTRAINTS ===
        formatted_data += "## SALES FORECAST AND FINANCIAL DATA FROM DECIDE TAB\n\n"
        
        if "decisions_table_content" in data:
            formatted_data += "### Decision Table Content\n"
            formatted_data += f"```\n{data['decisions_table_content']}\n```\n\n"
        
        # Include any additional decision-related tables
        for key in sorted([k for k in data.keys() if k.startswith("decide_table_")]):
            formatted_data += f"### Table {key.replace('decide_table_', '')}\n"
            formatted_data += f"{data[key]}\n\n"
        
        if "sales_forecast_table" in data:
            formatted_data += "### Sales Forecast\n"
            formatted_data += f"```\n{data['sales_forecast_table']}\n```\n\n"
        
        if "rd_budget_info" in data:
            formatted_data += "### R&D Budget Information\n"
            formatted_data += f"```\n{data['rd_budget_info']}\n```\n\n"
        
        # === CURRENT INPUT VALUES ===
        formatted_data += "## CURRENT DECISION INPUTS\n\n"
        if rd_budget is not None:
            formatted_data += f"### R&D BUDGET FOR THIS YEAR: ${rd_budget} MILLION\n\n"
        
        # Show current values in all input fields
        for key, value in data["decisions_table"].items():
            formatted_data += f"- {key}: {value}\n"
                
        # === FINAL BUDGET REMINDER ===
        # Emphasize the budget constraint one more time at the end
        if rd_budget is not None:
            formatted_data += f"\n## FINAL REMINDER\n"
            formatted_data += f"R&D BUDGET LIMIT FOR YEAR {year}: ${rd_budget} MILLION\n"
            formatted_data += "Your total R&D allocations across all categories for both products MUST NOT exceed this limit.\n"
        
        return formatted_data

    def parse_ai_response(self, response_text, rd_budget_limit=None):
        """
        Parse the AI's response text to extract structured decision values.
        
        The AI is expected to respond with a comma-separated line of numbers representing
        business decisions (prices, units, R&D allocations) followed by a rationale.
        This method uses robust parsing to handle various response formats and extract
        the numerical decisions while preserving the strategic reasoning.
        
        Args:
            response_text: The raw text response from the AI containing decisions and rationale
            rd_budget_limit: The R&D budget constraint (currently unused but available for validation)
            
        Returns:
            dict: Structured decisions containing:
                - agm_price/supercapacitor_price: Product pricing decisions
                - agm_units/supercapacitor_units: Unit sales forecasts (0 if not provided)
                - agm_rd_allocations/supercapacitor_rd_allocations: R&D spending by category [5 values each]
                - rationale: AI's explanation of the strategic reasoning
        """
        # Initialize decision structure with default values
        decisions = {
            "agm_price": 0,
            "supercapacitor_price": 0,
            "agm_units": 0,
            "supercapacitor_units": 0,
            "agm_rd_allocations": [0, 0, 0, 0, 0],  # 5 R&D categories
            "supercapacitor_rd_allocations": [0, 0, 0, 0, 0],  # 5 R&D categories
            "rationale": ""
        }
        
        try:
            lines = response_text.strip().split('\n')
            numbers_line = ""
            
            # === FIND THE DECISION NUMBERS LINE ===
            # Look for a line containing comma-separated numerical values
            for line in lines:
                # Heuristic: line should have many commas and sufficient digit content
                if line.count(',') >= 11 and sum(c.isdigit() for c in line.replace('.', '', 1)) > 10:
                    numbers_line = line.strip()
                    break
            
            if not numbers_line:
                # No valid numbers line found - trigger retry
                decisions["rationale"] = "Error parsing AI response."
                return decisions

            # === EXTRACT AND CLEAN NUMERICAL VALUES ===
            number_values_str = [n.strip() for n in numbers_line.split(',') if n.strip()]
            number_values = []
            for n_str in number_values_str:
                try:
                    # Clean string: remove any non-numeric characters except decimal points and minus signs
                    cleaned_n_str = re.sub(r'[^\d\.\-]', '', n_str)
                    if cleaned_n_str:  # Ensure not empty after cleaning
                        number_values.append(float(cleaned_n_str))
                except ValueError:
                    # Skip invalid values - will cause length mismatch and trigger retry
                    pass
            
            # === MAP VALUES TO DECISION FIELDS ===
            # Support two formats: 14-value (with units) or 12-value (prices + R&D only)
            if len(number_values) == 14:
                # Full format: units, prices, R&D allocations
                decisions["agm_units"] = number_values[0]
                decisions["supercapacitor_units"] = number_values[1]
                decisions["agm_price"] = number_values[2]
                decisions["supercapacitor_price"] = number_values[3]
                decisions["agm_rd_allocations"] = number_values[4:9]
                decisions["supercapacitor_rd_allocations"] = number_values[9:14]
            elif len(number_values) == 12:
                # Compact format: prices and R&D allocations only
                decisions["agm_price"] = number_values[0]
                decisions["supercapacitor_price"] = number_values[1]
                decisions["agm_rd_allocations"] = number_values[2:7]
                decisions["supercapacitor_rd_allocations"] = number_values[7:12]
                # Units default to 0 when not provided
                decisions["agm_units"] = 0
                decisions["supercapacitor_units"] = 0
            else:
                # Unexpected number count - trigger retry
                decisions["rationale"] = "Error parsing AI response."
                return decisions

            # === EXTRACT RATIONALE TEXT ===
            # Capture all text that comes after the numbers line as strategic reasoning
            rationale_lines = []
            capture_rationale = False
            
            for line in lines:
                if capture_rationale:
                    rationale_lines.append(line)
                elif line.strip() == numbers_line:
                    capture_rationale = True
            
            # Find the first non-empty line to start the rationale
            for i, line in enumerate(rationale_lines):
                if line.strip():
                    decisions["rationale"] = '\n'.join(rationale_lines[i:])
                    break
            
            # Provide default rationale if none found
            if not decisions["rationale"]:
                decisions["rationale"] = "AI's strategic reasoning for these decisions."
            return decisions

        except Exception as e:
            # Any unexpected parsing errors should trigger retry
            decisions["rationale"] = "Error parsing AI response."
            return decisions

    
    def submit_decision(self, decisions, difficulty="Intermediate"):
        """
        Submit the parsed business decisions to the simulation interface.
        
        This method takes the structured decisions and enters them into the appropriate
        input fields on the simulation's Decide tab. It handles different difficulty
        levels and provides feedback on submission success.
        
        Args:
            decisions: Dictionary containing structured decisions from parse_ai_response()
            difficulty: Simulation difficulty level ("Basic", "Intermediate", or "Advanced")
                       Basic mode doesn't include unit sales forecast inputs
            
        Returns:
            tuple: (success_boolean, updated_rd_budget_limit)
                   - success_boolean: True if submission succeeded, False if rejected
                   - updated_rd_budget_limit: New R&D budget after submission (may change based on inputs)
        """
        time.sleep(1)
        try:
            # Wait for the decisions table to be available
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "decisions-table"))
            )
            
            # === UNIT SALES FORECASTS ===
            # Only available in Intermediate and Advanced difficulty modes
            if difficulty != "Basic":
                if "agm_units" in decisions and decisions["agm_units"] > 0:
                    try:
                        agm_units_input = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.ID, "Pro_Forma_Total_Unit_Sales[AGM]"))
                        )
                        agm_units_input.clear()
                        agm_units_input.send_keys(str(decisions["agm_units"]))
                    except Exception as e:
                        pass  # Continue if unit input fails
                
                if "supercapacitor_units" in decisions and decisions["supercapacitor_units"] > 0:
                    try:
                        sc_units_input = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.ID, "Pro_Forma_Total_Unit_Sales[Supercapacitor]"))
                        )
                        sc_units_input.clear()
                        sc_units_input.send_keys(str(decisions["supercapacitor_units"]))
                    except Exception as e:
                        pass  # Continue if unit input fails
            
            # === PRODUCT PRICING ===
            # Set prices for both product lines
            try:
                agm_price_input = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.ID, "Price[AGM]"))
                )
                agm_price_input.clear()
                agm_price_input.send_keys(str(decisions["agm_price"]))
            except Exception as e:
                pass  # Continue if price input fails
            
            try:
                sc_price_input = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.ID, "Price[Supercapacitor]"))
                )
                sc_price_input.clear()
                sc_price_input.send_keys(str(decisions["supercapacitor_price"]))
            except Exception as e:
                pass  # Continue if price input fails
            
            # === R&D ALLOCATION INPUTS ===
            # Map logical R&D categories to HTML form field names
            rd_categories = ["Energy_Density", "Recharge_Cycles", "Self_Discharge", "Recharge_Time", "Process_Improvement"]
            id_categories = ["Energy_Density", "Recharge_Cycles", "Self_Discharge", "Recharge_Time", "Pricing"]
            
            # Set R&D allocations for AGM product line
            for i, category in enumerate(rd_categories):
                try:
                    html_category = id_categories[i]  # Map to HTML field naming
                    input_id = f"R_and_D_Spending[AGM,_{html_category}]"
                    rd_input = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.ID, input_id))
                    )
                    rd_input.clear()
                    rd_input.send_keys(str(decisions["agm_rd_allocations"][i]))
                except Exception as e:
                    pass  # Continue if R&D input fails
            
            # Set R&D allocations for Supercapacitor product line
            for i, category in enumerate(rd_categories):
                try:
                    html_category = id_categories[i]  # Map to HTML field naming
                    input_id = f"R_and_D_Spending[Supercapacitor,_{html_category}]"
                    rd_input = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.ID, input_id))
                    )
                    rd_input.clear()
                    rd_input.send_keys(str(decisions["supercapacitor_rd_allocations"][i]))
                except Exception as e:
                    pass  # Continue if R&D input fails
            
            # === STRATEGY RATIONALE ===
            # Enter the AI's strategic reasoning into the strategy text field
            try:
                strategy_input = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.ID, "strategy"))
                )
                strategy_input.clear()
                strategy_input.send_keys(decisions["rationale"])
            except Exception as e:
                pass  # Continue if strategy input fails
            
            # === SUBMISSION ATTEMPT ===
            # Try to submit the decisions and handle potential rejection
            try:
                btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, "#decisions-submit > button")
                    )
                )
                time.sleep(1)
                btn.click()
                time.sleep(2)
                # Success: return True with updated budget limit
                return True, self.current_rd_budget_limit()
            except Exception:
                # Submission failed (likely budget constraint violation)
                # Return False with current budget limit for retry logic
                return False, self.current_rd_budget_limit()
                
        except Exception as e:
            pass  # Any other errors during submission
        return False, None  # Complete failure: return tuple of False and None
        
    def check_simulation_ended(self):
        """
        Check if the simulation has reached a terminal state (ended or player fired).
        
        This method performs fast polling to detect various end-game conditions
        including successful completion, being fired for poor performance, or 
        other terminal states. It uses optimized WebDriver settings for quick
        response detection without long waits.
        
        Returns:
            bool: True if simulation has ended (any terminal state), False if still active
        """
        import time
        max_wait_time = 3.0  # Maximum time to spend checking for end conditions
        check_interval = 0.5  # How frequently to poll for changes
        waited = 0.0
        
        # Optimize WebDriver for fast polling by disabling implicit waits temporarily
        original_wait = self.driver.timeouts.implicit_wait
        self.driver.implicitly_wait(0)  # Set to 0 for immediate response
        
        try:
            while waited < max_wait_time:
                # Check for explicit "game over" div (clean simulation end)
                if self.driver.find_elements(By.XPATH, "//div[@id='game-over']"):
                    return True

                # Check for termination messages (fired or simulation completed)
                # Use case-insensitive text matching for various end states
                if self.driver.find_elements(
                    By.XPATH,
                    "//h1[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'fired')]"
                ):
                    print("fired")
                    return True
                
                if self.driver.find_elements(
                    By.XPATH,
                    "//h1[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'simulation completed')]"
                ):
                    print("completed")
                    return True

                # Check for active Analyze tab (indicates simulation is still running)
                # If this tab is active, the simulation is definitely not ended
                if self.driver.find_elements(
                    By.XPATH,
                    "//a[@aria-label='Analyze' and @aria-current='page']"
                ):
                    return False

                # Brief pause before next check to avoid overwhelming the browser
                time.sleep(check_interval)
                waited += check_interval
        finally:
            # Always restore the original implicit wait setting
            self.driver.implicitly_wait(original_wait)
        
        # If we couldn't definitively determine the state, assume simulation continues
        return False
