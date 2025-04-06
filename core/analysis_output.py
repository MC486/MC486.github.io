from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime
import logging

class AnalysisOutput(ABC):
    """
    Abstract base class for analysis output handling.
    Defines the interface for different types of analysis output.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.enabled = True
        self.detail_level = 2  # 1: Basic, 2: Detailed, 3: Debug
        
    @abstractmethod
    def process_analysis(self, data: Dict[str, Any]) -> None:
        """Process and output analysis data"""
        pass
        
    @abstractmethod
    def clear(self) -> None:
        """Clear any stored analysis data"""
        pass

class DeveloperAnalysis(AnalysisOutput):
    """
    Handles analysis output for VS Code development environment.
    Provides detailed technical information for debugging and development.
    """
    def __init__(self):
        super().__init__()
        self.detail_level = 3  # Maximum detail for developers
        self.analysis_buffer = []
        
    def process_analysis(self, data: Dict[str, Any]) -> None:
        """
        Process and format analysis data for developer view.
        
        Args:
            data: Analysis data including AI decisions, probabilities, etc.
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_data = self._format_dev_data(data)
        
        # Structure for VS Code output
        entry = {
            'timestamp': timestamp,
            'component': data.get('component', 'Unknown'),
            'analysis': formatted_data,
            'debug_info': data.get('debug_data', {})
        }
        
        self.analysis_buffer.append(entry)
        self._output_to_dev_console(entry)
        
    def _format_dev_data(self, data: Dict[str, Any]) -> str:
        """Format data with technical details for developer view"""
        # Detailed formatting including memory usage, execution time, etc.
        return f"[{data.get('component', 'Unknown')}] {data.get('message', '')}\n" \
               f"Details: {data.get('debug_data', {})}"
               
    def _output_to_dev_console(self, entry: Dict[str, Any]) -> None:
        """Output formatted data to VS Code console"""
        # Implementation would connect to VS Code output channel
        pass
        
    def clear(self) -> None:
        """Clear analysis buffer"""
        self.analysis_buffer.clear()

class UserAnalysis(AnalysisOutput):
    """
    Handles analysis output for game UI.
    Provides user-friendly information about AI decisions.
    """
    def __init__(self):
        super().__init__()
        self.detail_level = 2  # Standard detail for users
        self.current_analysis = []
        
    def process_analysis(self, data: Dict[str, Any]) -> None:
        """
        Process and format analysis data for user view.
        
        Args:
            data: Analysis data to be displayed to users
        """
        if not self.enabled:
            return
            
        formatted_data = self._format_user_data(data)
        self.current_analysis.append(formatted_data)
        self._update_ui_display()
        
    def _format_user_data(self, data: Dict[str, Any]) -> str:
        """Format data in user-friendly way"""
        return f"AI is thinking about: {data.get('message', '')}"
        
    def _update_ui_display(self) -> None:
        """Update game UI with current analysis"""
        # Implementation would update game UI
        pass
        
    def clear(self) -> None:
        """Clear current analysis"""
        self.current_analysis.clear()

class HistoricalAnalysis(AnalysisOutput):
    """
    Handles analysis output for historical logging.
    Stores detailed analysis data for future reference and training.
    """
    def __init__(self, log_file_path: str = "analysis_history.log"):
        super().__init__()
        self.log_file_path = log_file_path
        self.detail_level = 3  # Full detail for historical record
        
    def process_analysis(self, data: Dict[str, Any]) -> None:
        """
        Process and log analysis data for historical record.
        
        Args:
            data: Analysis data to be logged
        """
        if not self.enabled:
            return
            
        formatted_data = self._format_historical_data(data)
        self._write_to_log(formatted_data)
        
    def _format_historical_data(self, data: Dict[str, Any]) -> str:
        """Format data for historical logging"""
        timestamp = datetime.now().isoformat()
        return f"[{timestamp}] {data.get('component', 'Unknown')}: " \
               f"{data.get('message', '')}\n" \
               f"Full Data: {data}"
               
    def _write_to_log(self, formatted_data: str) -> None:
        """Write formatted data to log file"""
        # Implementation would write to log file
        pass
        
    def clear(self) -> None:
        """Clear log file"""
        # Implementation would clear log file
        pass