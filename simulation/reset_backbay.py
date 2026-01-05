# reset_backbay.py  ← FULL REPLACEMENT
#
# Harvard Business School Publishing (HBSP) "Back Bay Battery" Simulation Reset Tool
# 
# This script automates the process of resetting HBSP simulation runs and optionally
# collecting performance data. It uses Selenium WebDriver to interact with the HBSP
# web interface, handles authentication, manages simulation configurations, and 
# extracts tabular data from results pages.
#
# Key capabilities:
# - Reset simulation runs with specified difficulty levels and market segments
# - Extract "Best Scores" and "Simulation Summary" data as TSV format
# - Handle various UI overlays and modals that might interfere with automation
# - Support both headless and visible browser modes
# - Maintain backward compatibility with legacy code interfaces
import os
import time
from contextlib import contextmanager
from typing import Callable, Optional, Tuple

from selenium.webdriver import Chrome            # type: ignore
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    ElementClickInterceptedException,
)

from dotenv import load_dotenv
from hbsp_login import HBSPLoginManager

# --------------------------------------------------------------------------- #
#  Environment Setup
# --------------------------------------------------------------------------- #
# Load environment variables from .env file for HBSP credentials and configuration
load_dotenv()


def wait_until_loaded(driver, timeout: int = 30):
    """Block until the global loading overlay reports “Finished Loading.”"""
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script(
            "return (document.querySelector('.loading .accessibility-text')?.textContent ?? '').trim() === 'Finished Loading.';"
        )
    )


class SimulationResetter:
    """
    Reset an HBSP “Back Bay Battery” simulation run and (optionally) download
    summary data.

    **Back-compat notes**

    * Old code may construct with `market_segment=` instead of `sim_version=`.
    * Call-sites may pass `reset_simulation(collect_data=…)`.
    * Call-sites expect the return order **(best_scores, simulation_summary)**.
    * Some code still invokes `close_browser()`.

    All of those continue to work unchanged.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        difficulty: str,
        sim_version: Optional[int] = None,
        *,
        headless: bool = False,
        gather_data: bool = True,
        # legacy alias --------------------------------------------------- #
        market_segment: Optional[int] = None,
        # credentials ---------------------------------------------------- #
        admin_username: Optional[str] = None,
        admin_password: Optional[str] = None,
        coursepack_url: Optional[str] = None,
        default_wait: int = 30,
    ):
        self.difficulty = difficulty
        # accept either keyword
        self.sim_version = sim_version if sim_version is not None else market_segment
        self.headless = headless
        self.gather_data = gather_data
        self.default_wait = default_wait

        self.admin_username = (
            admin_username if admin_username else os.environ.get("ADMIN_USERNAME")
        )
        self.admin_password = (
            admin_password if admin_password else os.environ.get("ADMIN_PASSWORD")
        )
        self.coursepack_url = coursepack_url or os.environ.get(
            "HBSP_COURSEPACK_URL"
        )

        if not self.admin_username or not self.admin_password:
            raise ValueError(
                "ADMIN_USERNAME / ADMIN_PASSWORD not set in env or kwargs."
            )

        self.driver: Optional[Chrome] = None
        self.login_manager: Optional[HBSPLoginManager] = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    # CSS selectors for modal overlays that can interfere with automation
    # These include Qualtrics survey popups and HBS site modals that must be dismissed
    OVERLAY_SELECTORS = [
        ".QSIWebResponsiveShadowBox",            # Qualtrics shadow/fade
        "div[class*='QSIWebResponsiveDialog']",  # Whole Qualtrics dialog
        ".QSIWebResponsive-creative-container",  # Qualtrics main div
        "div[role='dialog'].modal-backdrop",     # HBS in-site modals
    ]

    @staticmethod
    def _dismiss_overlays(driver):
        """
        Dismiss modal overlays that can block automation interactions.
        
        First attempts to properly close modals by clicking their close buttons.
        If that fails, falls back to forcefully removing the overlay elements
        from the DOM using JavaScript.
        
        This handles both Qualtrics survey popups and HBS site modals that
        commonly appear during simulation management.
        
        Args:
            driver: Selenium WebDriver instance
        """
        try:
            q_close = driver.find_element(
                By.CSS_SELECTOR,
                "button[aria-label='Close'][class*='QSIWebResponsiveDialog']",
            )
            if q_close.is_displayed():
                q_close.click()
                WebDriverWait(driver, 20).until(
                    EC.invisibility_of_element_located(
                        (By.CSS_SELECTOR, ".QSIWebResponsiveShadowBox")
                    )
                )
                time.sleep(0.5)
                return
        except Exception:
            pass  # fall through to hard-remove

        # If graceful close failed, forcefully remove overlay elements from DOM
        for sel in SimulationResetter.OVERLAY_SELECTORS:
            driver.execute_script(
                """
                document.querySelectorAll(arguments[0]).forEach(el => el.remove());
                """,
                sel,
            )

    @staticmethod
    def _sleep(seconds: float = 0.5):
        """Convenience wrapper for time.sleep with default pause duration."""
        time.sleep(seconds)

    @staticmethod
    def _until(
        driver: Chrome,
        condition: Callable,
        timeout: int = 20,
        poll: float = 0.2,
    ):
        """Convenience wrapper for WebDriverWait with configurable polling frequency."""
        return WebDriverWait(driver, timeout, poll_frequency=poll).until(condition)

    @classmethod
    def _click_when_ready(
        cls, driver, by, selector, timeout: int = 20, tries: int = 3
    ):
        """
        Reliably click an element, handling overlays and retries.
        
        Waits for the element to be clickable, scrolls it into view, and attempts
        to click it. If clicking is intercepted by modal overlays, dismisses the
        overlays and retries.
        
        Args:
            driver: Selenium WebDriver instance
            by: Selenium locator type (e.g., By.CSS_SELECTOR)
            selector: Element selector string
            timeout: Seconds to wait for element to be clickable
            tries: Number of retry attempts if click is intercepted
            
        Returns:
            The clicked element
            
        Raises:
            ElementClickInterceptedException: If all retry attempts fail
        """
        last_err = None
        for _ in range(tries):
            try:
                elem = WebDriverWait(driver, timeout).until(
                    EC.element_to_be_clickable((by, selector))
                )
                driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center'});", elem
                )
                elem.click()
                return elem
            except ElementClickInterceptedException as e:
                last_err = e
                cls._dismiss_overlays(driver)
                time.sleep(0.3)
        raise last_err

    @contextmanager
    def _managed_browser(self):
        """
        Context manager for browser lifecycle management.
        
        Creates and configures an HBSPLoginManager, starts the browser,
        and ensures proper cleanup even if exceptions occur. Sets instance
        variables for driver and login_manager during the context.
        
        Yields:
            Tuple[Chrome, HBSPLoginManager]: Browser driver and login manager
        """
        login_manager = HBSPLoginManager(
            headless=self.headless,
            username=self.admin_username,
            password=self.admin_password,
        )
        driver = login_manager.start_browser()
        self.driver = driver
        self.login_manager = login_manager
        try:
            yield driver, login_manager
        finally:
            driver.quit()
            self.driver = None
            self.login_manager = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset_simulation(
        self,
        *,  # force keyword
        collect_data: Optional[bool] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Perform the reset; returns **(best_scores_tsv, simulation_summary_tsv)**.

        Old code may pass `collect_data`; if supplied it overrides the
        constructor’s `gather_data` flag.
        """
        if collect_data is not None:
            self.gather_data = bool(collect_data)

        try:
            with self._managed_browser() as (driver, login):
                print("Resetting simulation…")

                # ---- login & nav ---------------------------------------- #
                # Authenticate with HBSP and navigate to the coursepack page
                login.login_to_hbs()
                login.navigate_to_coursepack_url(self.coursepack_url)

                # Click "Manage Simulation" button - this opens a new browser tab
                time.sleep(1)
                self._click_when_ready(
                    driver,
                    By.CSS_SELECTOR,
                    "button.coursepack-item__manage-simulation-button",
                )
                self._sleep()
                # Switch to the new simulation management tab (latest opened window)
                driver.switch_to.window(driver.window_handles[-1])

                # ---- optional data extraction -------------------------- #
                # Extract performance data from completed simulation runs if requested
                if self.gather_data:
                    sim_summary, best_scores = self._collect_data()
                else:
                    sim_summary, best_scores = ("", "")

                # ---- configure & open a fresh run ---------------------- #
                # Set up new simulation run with specified difficulty and market segment
                try:
                    self._configure_run()
                except Exception as e:
                    print(f"Error configuring run: {e}")
                # Return order (best_scores, sim_summary) kept for legacy callers
                return best_scores, sim_summary

        except Exception as exc:  # pragma: no cover
            import traceback

            print(f"❌ Error resetting simulation ➜ {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return None, None

    # ------------------------------------------------------------------ #
    # Data-collection helpers
    # ------------------------------------------------------------------ #
    def _collect_table_tsv(self, wrapper_selector: str) -> Optional[str]:
        """
        Extract tabular data from an HBSP page and convert to TSV format.
        
        Locates a table within the specified wrapper element, extracts headers
        and row data, and formats as tab-separated values. Handles cases where
        no data is available or tables haven't loaded yet.
        
        Args:
            wrapper_selector: CSS selector for the container element holding the table
            
        Returns:
            Optional[str]: TSV-formatted string with headers and data rows,
                          or None if no table data is found
        """
        driver = self.driver
        assert driver

        wrapper = self._until(
            driver, lambda d: d.find_element(By.CSS_SELECTOR, wrapper_selector), 3
        )

        try:
            # Wait for either a data table to appear or "no data" message
            self._until(
                driver,
                lambda d: wrapper.find_elements(By.TAG_NAME, "table")
                or wrapper.find_elements(
                    By.XPATH, ".//h2[contains(., 'no data to show')]"
                ),
                3,
            )
        except TimeoutException:
            return None

        # Check if page shows "no data to show" message instead of table
        if wrapper.find_elements(By.XPATH, ".//h2[contains(., 'no data to show')]"):
            return None

        # Extract table headers and data rows
        table = wrapper.find_element(By.TAG_NAME, "table")
        headers = [th.text for th in table.find_elements(By.TAG_NAME, "th")]
        rows = [
            [td.text for td in tr.find_elements(By.TAG_NAME, "td")]
            for tr in table.find_elements(By.CSS_SELECTOR, "tbody tr")
        ]

        if headers and headers[-1] == "View":  # drop trailing “View” column
            headers = headers[:-1]
            rows = [r[:-1] for r in rows]

        # Format as TSV with Windows line endings  
        lines = ["\t".join(headers)] + ["\t".join(r) for r in rows]
        return "\r\n".join(lines)

    def _collect_data(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract performance data from HBSP simulation management interface.
        
        Navigates to the Class Results section and extracts data from two tables:
        1. Simulation Summary - overall performance metrics
        2. Best Scores - top performing teams/individuals
        
        Returns:
            Tuple[Optional[str], Optional[str]]: (simulation_summary_tsv, best_scores_tsv)
            Both strings in TSV format, or None if data extraction fails
        """
        driver = self.driver
        assert driver

        wait_until_loaded(driver)

        # Navigate to Class Results section and extract simulation summary data
        self._click_when_ready(
            driver, By.CSS_SELECTOR, 'a.nav-item[aria-label="Class Results"]'
        )
        wait_until_loaded(driver)
        self._click_when_ready(
            driver, By.CSS_SELECTOR, 'a.nav-item[aria-label="Simulation Summary"]'
        )
        self._sleep(0.1)
        sim_tsv = self._collect_table_tsv("div#sim-summary .table-wrapper")

        # Navigate to Best Scores section and extract top performer data
        self._click_when_ready(
            driver, By.CSS_SELECTOR, 'a.nav-item[aria-label="Best Scores"]'
        )
        self._sleep(0.1)
        best_tsv = self._collect_table_tsv("div#best-scores .table-wrapper")

        return sim_tsv, best_tsv

    # ------------------------------------------------------------------ #
    # Run-configuration helpers
    # ------------------------------------------------------------------ #
    def _configure_run(self):
        """
        Configure and start a new simulation run with specified parameters.
        
        This method:
        1. Navigates to the Simulation Setup page
        2. Creates a new simulation run if needed
        3. Sets the difficulty level (Basic/Intermediate/Advanced)
        4. Selects the market segment version if specified
        5. Opens the simulation for student participation
        
        The method handles UI interactions like hiding headers during
        radio button selection to avoid click interception issues.
        """
        driver = self.driver
        assert driver

        # Navigate to simulation setup page
        wait_until_loaded(driver)
        self._click_when_ready(
            driver, By.CSS_SELECTOR, 'a.nav-item[aria-label="Simulation Setup"]'
        )
        wait_until_loaded(driver)
        self._sleep()

        # Create a new simulation run if the button is available (not disabled)
        new_run_btn = self._until(
            driver, lambda d: d.find_element(By.CSS_SELECTOR, "button.new-run-button")
        )
        wait_until_loaded(driver)
        self._sleep()

        if not new_run_btn.get_attribute("disabled"):
            new_run_btn.click()
            self._sleep()

        wait_until_loaded(driver)
        self._sleep()

        # Configure difficulty level (Intermediate is default, skip if already set)
        # Difficulty radio buttons – IDs: 0=Basic, 1=Intermediate, 2=Advanced
        diff_map = {
            "Basic": "difficulty-level_0",
            "Intermediate": "difficulty-level_1",
            "Advanced": "difficulty-level_2",
        }

        if self.difficulty in diff_map and self.difficulty != "Intermediate":
            self._toggle_header(False)  # Hide header to prevent click interception
            input_id = diff_map[self.difficulty]
            driver.find_element(By.CSS_SELECTOR, f'label[for="{input_id}"]').click()
            self._toggle_header(True)   # Restore header
            self._sleep()

        # Configure market segment version if specified
        if self.sim_version is not None:
            self._toggle_header(False)  # Hide header to prevent click interception
            input_id = f"market-segment_{self.sim_version}"
            input_elem = self._until(
                driver, EC.presence_of_element_located((By.ID, input_id)), 20
            )
            # Scroll element into view and click via JavaScript for reliability
            driver.execute_script(
                "arguments[0].scrollIntoView({block:'center'});", input_elem
            )
            self._sleep(0.3)
            driver.execute_script("arguments[0].click();", input_elem)
            self._toggle_header(True)   # Restore header
            self._sleep(0.3)

        # Open the simulation for student access
        wait_until_loaded(driver)
        self._sleep()
        self._click_when_ready(driver, By.CSS_SELECTOR, 'label[for="status_open"]')
        self._sleep()
        wait_until_loaded(driver)

        # Verify that the "Open" status was successfully selected
        self._until(
            driver,
            lambda d: d.find_element(By.ID, "status_open").is_selected(),
            30,
        )
        wait_until_loaded(driver)
        self._sleep()



    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #
    def _toggle_header(self, show: bool):
        """
        Show or hide the page header to prevent click interception issues.
        
        Some radio buttons and form elements can be blocked by fixed headers.
        This method temporarily hides the header during interactions.
        
        Args:
            show: True to show header, False to hide it
        """
        driver = self.driver
        assert driver
        display = "" if show else "none"
        driver.execute_script(
            "document.getElementById('header').style.display = arguments[0];", display
        )

    # ---------- legacy public helper ----------------------------------- #
    def close_browser(self):
        """
        Legacy method for manually closing the browser.
        
        Maintained for backward compatibility with existing code.
        Note: The context manager _managed_browser() automatically
        handles browser cleanup, so this method is typically not needed.
        """
        if self.driver:
            self.driver.quit()
            self.driver = None
