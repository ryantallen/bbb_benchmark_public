"""
HBSP Automation Login Manager

This module provides automated login and navigation for Harvard Business Publishing (HBSP)
coursepacks and simulations. It uses Selenium WebDriver to automate the browser interaction
required to:

1. Log into HBSP with user credentials
2. Navigate to specific coursepack URLs  
3. Launch business simulations from coursepacks
4. Handle common popup overlays that interfere with automation

Key features:
- Robust overlay/popup handling (especially Qualtrics surveys)
- Intelligent clicking that retries after dismissing overlays
- Network-level blocking of Qualtrics domains
- Browser window positioning for development visibility
- Complete error handling and resource cleanup

Usage:
    Set HBSP_USERNAME and HBSP_PASSWORD environment variables, then:
    
    login_manager = HBSPLoginManager()
    driver = login_manager.complete_login_process("https://coursepack-url")
    # Use driver for further simulation automation...
    login_manager.close_browser()

Requirements:
    - Chrome browser installed
    - Valid HBSP account credentials
    - Environment variables or direct credential passing
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
from subprocess import DEVNULL 

# Load environment variables from .env file to get HBSP credentials
load_dotenv()

class HBSPLoginManager:
    """
    Manages automated login to Harvard Business Publishing (HBSP) and coursepack navigation.
    
    This class automates the process of:
    1. Starting a Chrome browser instance
    2. Logging into HBSP with provided credentials
    3. Navigating to specific coursepack URLs
    4. Launching simulations from within coursepacks
    5. Handling various popup overlays that may interfere with automation
    """
    
    # CSS selectors for popup overlays that commonly appear on HBSP and interfere with automation
    # These overlays need to be dismissed to allow clicks on page elements
    OVERLAY_SELECTORS = [
        ".QSIWebResponsiveShadowBox",            # Qualtrics survey popup shadow/fade overlay
        "div[class*='QSIWebResponsiveDialog']",  # The main Qualtrics dialog container
        ".QSIWebResponsive-creative-container",  # Qualtrics creative content container
        "div[role='dialog'].modal-backdrop",     # Generic HBS in-site modal backdrops
    ]
    
    def __init__(self, headless=False, username=None, password=None):
        """
        Initialize the login manager with credentials and browser settings.
        
        Args:
            headless (bool): Whether to run browser in headless mode (no visible window)
            username (str): HBS Publishing username (if None, uses HBSP_USERNAME env variable)
            password (str): HBS Publishing password (if None, uses HBSP_PASSWORD env variable)
            
        Raises:
            ValueError: If credentials are not provided via parameters or environment variables
        """
        # Get credentials from parameters or fall back to environment variables
        self.username = username or os.environ.get("HBSP_USERNAME")
        self.password = password or os.environ.get("HBSP_PASSWORD")
        
        # Validate that we have the required credentials
        if not self.username or not self.password:
            raise ValueError("HBSP credentials not provided. Set HBSP_USERNAME and HBSP_PASSWORD environment variables or provide them as parameters.")
        
        # Store browser configuration
        self.headless = headless
        self.driver = None  # Will hold the WebDriver instance once browser is started
    
    def start_browser(self):
        """
        Start a Chrome browser instance with optimized settings for HBSP automation.
        
        Configures Chrome with settings to:
        - Suppress logging and developer messages
        - Block Qualtrics survey popups at the network level
        - Position window for visibility (if not headless)
        - Set appropriate timeouts for reliable automation
        
        Returns:
            WebDriver: The configured Chrome WebDriver instance
        """
        chrome_options = Options()
        
        # Security and performance options for automation
        chrome_options.add_argument("--no-sandbox")                # Disable sandbox for automation
        chrome_options.add_argument("--disable-dev-shm-usage")     # Overcome limited resource problems
        chrome_options.add_argument("--log-level=3")               # Suppress Chrome info/warning messages
        chrome_options.add_argument("--silent")                    # Minimize console output
        chrome_options.add_argument("--disable-logging")           # Disable Chrome logging
        chrome_options.add_argument("--host-resolver-rules=MAP *.qualtrics.com 127.0.0.1")  # Block Qualtrics survey domains at network level
        chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])        # Further disable logging
        chrome_options.add_experimental_option("detach", True)     # Keep browser open after script ends

        # Enable headless mode if requested (no visible browser window)
        if self.headless:
            chrome_options.add_argument("--headless=new")
        
        # Set page load strategy to wait for full page load
        chrome_options.page_load_strategy = 'normal'
        
        # Create Chrome service with suppressed output and auto-managed driver
        service = Service(ChromeDriverManager().install(), log_output=DEVNULL)
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Set default timeout for finding elements (10 seconds)
        self.driver.implicitly_wait(10)
        
        # Position browser window on right side of screen for visibility (if not headless)
        if not self.headless:
            self._dock_to_right()

        return self.driver

    def _dock_to_right(self, min_width: int = 1280) -> None:
        """
        Position the browser window on the right half of the screen for easy viewing.
        
        This helps with development and debugging by keeping the browser visible
        while allowing space for other applications on the left side.
        
        Args:
            min_width (int): Minimum window width in pixels (default 1280)
        """
        # Get the actual screen dimensions using JavaScript to query the monitor
        screen_w = self.driver.execute_script("return window.screen.availWidth;")
        screen_h = self.driver.execute_script("return window.screen.availHeight;")

        # Calculate desired window size: half screen width or minimum width (whichever is larger)
        width  = max(screen_w // 2, min_width)
        height = screen_h

        # Handle case where monitor is too narrow for our minimum width requirement
        if width >= screen_w:
            x = 0                         # Position at left edge
            width = screen_w              # Use full screen width
        else:
            x = screen_w - width          # Position on right side of screen

        # Apply the calculated position and size to the browser window
        self.driver.set_window_rect(x=x, y=0, width=width, height=height)
    
    def _dismiss_overlays(self):
        """
        Dismiss popup overlays that interfere with automation.
        
        Uses a two-step approach:
        1. First tries to properly close overlays by clicking their close buttons
        2. Falls back to forcibly removing overlay elements from the DOM
        
        This is necessary because HBSP often shows Qualtrics survey popups and
        other overlays that can block clicks on page elements.
        """
        try:
            # Step 1: Try to find and click the close button on Qualtrics survey popups
            # Qualtrics modals have dynamic class names, so we use a partial match
            q_close_button_selector = "button[aria-label='Close'][class*='QSIWebResponsiveDialog']"
            close_button = self.driver.find_element(By.CSS_SELECTOR, q_close_button_selector)

            # If we found a visible close button, click it
            if close_button and close_button.is_displayed():
                close_button.click()

                # Wait for the fadeout animation to complete before proceeding
                WebDriverWait(self.driver, 15).until(
                    EC.invisibility_of_element_located((By.CSS_SELECTOR, ".QSIWebResponsiveShadowBox"))
                )
                time.sleep(0.5)  # Extra pause to ensure animation is fully complete
                return # Successfully dismissed via close button

        except Exception:
            # If the close button approach fails, continue to fallback method
            pass

        # Step 2: Fallback method - forcibly remove overlay elements from the DOM
        # This is more aggressive but ensures overlays are gone
        for sel in self.OVERLAY_SELECTORS:
            self.driver.execute_script(
                """
                const elements = document.querySelectorAll(arguments[0]);
                elements.forEach(el => el.remove());
                """,
                sel,
            )
    
    def login_to_hbs(self):
        """
        Perform the automated login process to Harvard Business Publishing.
        
        Steps performed:
        1. Navigate to HBSP sign-in page
        2. Handle cookie consent notice if present
        3. Fill in username and password credentials
        4. Submit the login form
        5. Verify successful login by checking for user navigation elements
        
        Returns:
            bool: True if login successful, False otherwise
            
        Raises:
            ValueError: If browser has not been started yet
        """
        if not self.driver:
            raise ValueError("Browser not started. Call start_browser() first.")
        
        # Navigate to the HBSP sign-in page
        self.driver.get("https://hbsp.harvard.edu/signin")
        
        # Step 1: Handle cookie consent notice if it appears
        # Some users see a cookie notice that must be dismissed before login
        try:
            cookie_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.cookie-notice__close-button"))
            )
            self._click(cookie_button)
        except TimeoutException:
            # Cookie notice not present, continue with login
            pass
        
        # Step 2: Wait for the login form to fully load
        WebDriverWait(self.driver, 20).until(
            EC.presence_of_element_located((By.ID, "sign-in-form-username"))
        )
        
        # Step 3: Enter username credentials
        username_field = self.driver.find_element(By.ID, "sign-in-form-username")
        username_field.clear()  # Clear any existing text
        username_field.send_keys(self.username)
        
        # Step 4: Enter password credentials
        password_field = self.driver.find_element(By.ID, "sign-in-form-password")
        password_field.clear()  # Clear any existing text
        password_field.send_keys(self.password)
        
        # Step 5: Submit the login form
        submit_button = self.driver.find_element(By.CSS_SELECTOR, ".signin-form__signin-button")
        self._click(submit_button)  # Use our overlay-aware click method
        
        # Step 6: Verify successful login by waiting for post-login navigation elements
        # The 'My Coursepacks' link appears only when user is successfully logged in
        try:
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a.qa-navigation__my-coursepacks"))
            )
            print("Successfully logged in to HBS Publishing")
            return True
        except TimeoutException:
            print("Login failed or timed out.")
            return False
    
    def navigate_to_coursepack_url(self, coursepack_url):
            """
            Navigate directly to a specific coursepack URL after login.
            
            This method assumes the user is already logged in and takes them
            directly to the coursepack page where they can access simulations.
            
            Args:
                coursepack_url (str): The full URL of the coursepack to navigate to
                
            Returns:
                bool: True when navigation completes (always succeeds)
                
            Raises:
                ValueError: If browser has not been started
            """
            if not self.driver:
                raise ValueError("Browser not started or not logged in.")
            
            # Navigate directly to the provided coursepack URL
            self.driver.get(coursepack_url)
            
            # Wait for coursepack page elements to load
            # Look for either student launch buttons or simulation management buttons
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "button.student-product-launch-button, button.coursepack-item__manage-simulation-button"))
                )
            except TimeoutException:
                # If expected elements aren't found, continue anyway
                # The page may have loaded but with different elements
                pass
            
            return True
    
    def launch_simulation(self):
        """
        Launch the simulation from the coursepack page.
        
        Handles the complete simulation launch process:
        1. Clicks the 'Run Simulation' button
        2. Handles new tab/window switching if needed
        3. Dismisses any popup overlays
        4. Switches to iframe context if simulation uses iframes
        5. Waits for simulation to fully load
        
        Returns:
            bool: True if launch successful
            
        Raises:
            ValueError: If browser has not been started
        """
        if not self.driver:
            raise ValueError("Browser not started or not logged in.")
        
        # Step 1: Wait for and click the 'Run Simulation' button
        WebDriverWait(self.driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "button.student-product-launch-button"))
        )
        run_simulation_button = self.driver.find_element(By.CSS_SELECTOR, "button.student-product-launch-button")
        self._click(run_simulation_button)  # Use overlay-aware click
        
        # Step 2: Allow initial load time
        time.sleep(2)
        
        # Step 3: Handle case where simulation opens in a new browser tab/window
        if len(self.driver.window_handles) > 1:
            # Switch to the newest tab (the simulation)
            self.driver.switch_to.window(self.driver.window_handles[-1])
        
        # Step 4: Dismiss any overlays that might interfere with simulation interaction
        self._dismiss_overlays()
        
        # Step 5: Many simulations run inside iframes - switch to iframe context if present
        iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
        if iframes:
            # Switch to the first iframe (usually the simulation container)
            self.driver.switch_to.frame(iframes[0])
        
        # Step 6: Wait for simulation to fully initialize before returning control
        time.sleep(5)
        
        print("Successfully launched simulation")
        return True

    def _click(self, element):
        """
        Intelligent click method that handles overlay interference.
        
        Attempts to click an element, and if the click is blocked by an overlay,
        automatically dismisses overlays and retries the click. This is more
        reliable than regular clicking in environments with popup interference.
        
        Args:
            element: The WebElement to click
            
        Raises:
            ElementClickInterceptedException: If click fails even after overlay removal
        """
        try:
            # Attempt normal click first
            element.click()
        except ElementClickInterceptedException:
            # Click was blocked by an overlay - dismiss overlays and retry
            print("Click intercepted, dismissing overlays and retrying.")
            self._dismiss_overlays()
            time.sleep(0.5)  # Brief pause to let overlay removal complete
            element.click()  # Retry the click

    def complete_login_process(self, coursepack_url):
        """
        Execute the complete automated workflow from browser start to simulation launch.
        
        This is the main orchestrator method that combines all individual steps:
        1. Starts a configured Chrome browser instance
        2. Logs into Harvard Business Publishing with stored credentials
        3. Navigates to the specified coursepack URL
        4. Launches the simulation from the coursepack
        
        This method provides a single entry point for the entire automation workflow.
        
        Args:
            coursepack_url (str): The full URL of the coursepack containing the simulation
            
        Returns:
            WebDriver: The browser driver instance if successful, None if any step fails
        """
        try:
            print("Starting login process...")
            self.start_browser()                        # Step 1: Launch and configure browser
            self.login_to_hbs()                         # Step 2: Authenticate with HBSP
            self.navigate_to_coursepack_url(coursepack_url)  # Step 3: Go to coursepack
            self.launch_simulation()                    # Step 4: Start the simulation
            return self.driver                          # Return driver for further automation
        except Exception as e:
            # If any step fails, clean up and return None
            print(f"Error during login process: {e}")
            if self.driver:
                self.driver.quit()  # Close browser to prevent resource leaks
            return None
    
    def close_browser(self):
        """
        Safely close the browser and clean up resources.
        
        This method should be called when automation is complete to ensure
        the browser process is properly terminated and system resources are freed.
        """
        if self.driver:
            self.driver.quit()     # Terminate the browser process
            self.driver = None     # Clear the driver reference

# Example usage and testing
if __name__ == "__main__":
    # Create login manager instance (will read credentials from environment variables)
    login_manager = HBSPLoginManager()
    try:
        # Execute complete workflow: start browser -> login -> navigate -> launch simulation
        driver = login_manager.complete_login_process("Simulation Experiment")
        print("Successfully logged in and launched simulation")
        
        # At this point, the driver can be used for further automation
        # such as interacting with the simulation, filling forms, etc.
        # Your simulation automation code would go here...
        
    except Exception as e:
        print(f"Error in login process: {e}")
    finally:
        # Always clean up browser resources, even if an error occurred
        login_manager.close_browser() 