"""
Service layer for interfacing with the VEX U analysis Python modules
"""

import os
import sys
import subprocess
import json
import tempfile
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class VEXAnalysisService:
    """Service for executing Push Back VEX U analysis operations"""
    
    def __init__(self, vex_path: str, python_path: str = "python3"):
        """
        Initialize the VEX analysis service for Push Back
        
        Args:
            vex_path: Path to the VEX U analysis directory
            python_path: Path to Python executable
        """
        self.vex_path = Path(vex_path).resolve()
        self.python_path = python_path
        self.main_script = self.vex_path / "vex_analysis" / "main.py"
        
        # Validate that the Push Back analysis system exists
        push_back_analyzer = self.vex_path / "vex_analysis" / "analysis" / "push_back_strategy_analyzer.py"
        push_back_engine = self.vex_path / "vex_analysis" / "core" / "simulator.py"
        
        if not push_back_analyzer.exists():
            logger.warning(f"Push Back strategy analyzer not found at {push_back_analyzer}")
        if not push_back_engine.exists():
            logger.warning(f"Push Back scoring engine not found at {push_back_engine}")
        
        # Add VEX path to Python path for imports
        if str(self.vex_path) not in sys.path:
            sys.path.insert(0, str(self.vex_path))
    
    def _execute_command(self, command: List[str], timeout: int = 300) -> Dict[str, Any]:
        """
        Execute a VEX analysis command
        
        Args:
            command: Command arguments to execute
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Change to VEX directory for execution
            result = subprocess.run(
                [self.python_path] + command,
                cwd=self.vex_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout} seconds")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "return_code": -1
            }
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            }
    
    def run_demo_analysis(self, strategy_count: int = 10) -> Dict[str, Any]:
        """
        Run quick demo analysis
        
        Args:
            strategy_count: Number of strategies to analyze
            
        Returns:
            Analysis results
        """
        command = ["vex_analysis/main.py", "demo", "--strategies", str(strategy_count)]
        result = self._execute_command(command)
        
        if result["success"]:
            return self._parse_analysis_output(result["stdout"], "demo")
        else:
            raise RuntimeError(f"Demo analysis failed: {result['stderr']}")
    
    def run_full_analysis(self, 
                         strategy_count: int = 50,
                         simulation_count: int = 1000,
                         complexity: str = "intermediate") -> Dict[str, Any]:
        """
        Run comprehensive analysis
        
        Args:
            strategy_count: Number of strategies to analyze
            simulation_count: Number of simulations to run
            complexity: Analysis complexity level
            
        Returns:
            Analysis results
        """
        command = [
            "vex_analysis/main.py", "analyze", 
            "--strategies", str(strategy_count),
            "--simulations", str(simulation_count),
            "--complexity", complexity
        ]
        
        result = self._execute_command(command, timeout=600)  # Longer timeout for full analysis
        
        if result["success"]:
            return self._parse_analysis_output(result["stdout"], "full")
        else:
            raise RuntimeError(f"Full analysis failed: {result['stderr']}")
    
    def run_statistical_analysis(self, 
                                sample_size: int = 1000,
                                method: str = "descriptive") -> Dict[str, Any]:
        """
        Run statistical analysis
        
        Args:
            sample_size: Sample size for analysis
            method: Statistical method to use
            
        Returns:
            Statistical analysis results
        """
        command = [
            "vex_analysis/main.py", "statistical",
            "--sample-size", str(sample_size),
            "--method", method
        ]
        
        result = self._execute_command(command)
        
        if result["success"]:
            return self._parse_analysis_output(result["stdout"], "statistical")
        else:
            raise RuntimeError(f"Statistical analysis failed: {result['stderr']}")
    
    def generate_visualization(self, 
                              analysis_data: Dict[str, Any],
                              chart_types: List[str] = None) -> Dict[str, Any]:
        """
        Generate visualizations
        
        Args:
            analysis_data: Data to visualize
            chart_types: Types of charts to generate
            
        Returns:
            Visualization data
        """
        # Create temporary file for analysis data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(analysis_data, f)
            data_file = f.name
        
        try:
            command = ["vex_analysis/main.py", "visualize", "--data", data_file]
            if chart_types:
                command.extend(["--charts"] + chart_types)
            
            result = self._execute_command(command)
            
            if result["success"]:
                return self._parse_visualization_output(result["stdout"])
            else:
                raise RuntimeError(f"Visualization generation failed: {result['stderr']}")
        finally:
            # Clean up temporary file
            os.unlink(data_file)
    
    def generate_report(self, 
                       analysis_data: Dict[str, Any],
                       report_type: str = "strategic",
                       output_format: str = "html") -> Dict[str, Any]:
        """
        Generate strategic report
        
        Args:
            analysis_data: Analysis data for report
            report_type: Type of report to generate
            output_format: Output format (html, pdf, json)
            
        Returns:
            Report generation results
        """
        # Create temporary file for analysis data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(analysis_data, f)
            data_file = f.name
        
        try:
            command = [
                "vex_analysis/main.py", "report",
                "--data", data_file,
                "--type", report_type,
                "--format", output_format
            ]
            
            result = self._execute_command(command, timeout=300)
            
            if result["success"]:
                return self._parse_report_output(result["stdout"])
            else:
                raise RuntimeError(f"Report generation failed: {result['stderr']}")
        finally:
            # Clean up temporary file
            os.unlink(data_file)
    
    def train_ml_model(self, 
                      model_type: str,
                      training_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train ML model
        
        Args:
            model_type: Type of ML model to train
            training_params: Training parameters
            
        Returns:
            Training results
        """
        command = ["vex_analysis/main.py", "ml-train", "--model", model_type]
        
        if training_params:
            # Create temporary file for training parameters
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(training_params, f)
                params_file = f.name
            
            command.extend(["--params", params_file])
            
            try:
                result = self._execute_command(command, timeout=1800)  # 30 minute timeout for training
                
                if result["success"]:
                    return self._parse_ml_output(result["stdout"], "training")
                else:
                    raise RuntimeError(f"ML training failed: {result['stderr']}")
            finally:
                # Clean up temporary file
                os.unlink(params_file)
        else:
            result = self._execute_command(command, timeout=1800)
            
            if result["success"]:
                return self._parse_ml_output(result["stdout"], "training")
            else:
                raise RuntimeError(f"ML training failed: {result['stderr']}")
    
    def ml_predict(self, 
                   model_type: str,
                   input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make ML prediction
        
        Args:
            model_type: Type of ML model to use
            input_data: Input data for prediction
            
        Returns:
            Prediction results
        """
        # Create temporary file for input data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            data_file = f.name
        
        try:
            command = [
                "vex_analysis/main.py", "ml-predict",
                "--model", model_type,
                "--data", data_file
            ]
            
            result = self._execute_command(command)
            
            if result["success"]:
                return self._parse_ml_output(result["stdout"], "prediction")
            else:
                raise RuntimeError(f"ML prediction failed: {result['stderr']}")
        finally:
            # Clean up temporary file
            os.unlink(data_file)
    
    def optimize_strategy(self, 
                         strategy_data: Dict[str, Any],
                         optimization_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize strategy using ML
        
        Args:
            strategy_data: Strategy to optimize
            optimization_params: Optimization parameters
            
        Returns:
            Optimization results
        """
        # Create temporary file for strategy data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(strategy_data, f)
            strategy_file = f.name
        
        try:
            command = ["vex_analysis/main.py", "ml-optimize", "--strategy", strategy_file]
            
            if optimization_params:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(optimization_params, f)
                    params_file = f.name
                
                command.extend(["--params", params_file])
                
                try:
                    result = self._execute_command(command, timeout=600)
                    
                    if result["success"]:
                        return self._parse_ml_output(result["stdout"], "optimization")
                    else:
                        raise RuntimeError(f"Strategy optimization failed: {result['stderr']}")
                finally:
                    os.unlink(params_file)
            else:
                result = self._execute_command(command, timeout=600)
                
                if result["success"]:
                    return self._parse_ml_output(result["stdout"], "optimization")
                else:
                    raise RuntimeError(f"Strategy optimization failed: {result['stderr']}")
        finally:
            # Clean up temporary file
            os.unlink(strategy_file)
    
    def generate_scenarios(self, 
                          scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate scenarios using ML
        
        Args:
            scenario_params: Scenario generation parameters
            
        Returns:
            Generated scenarios
        """
        # Create temporary file for scenario parameters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(scenario_params, f)
            params_file = f.name
        
        try:
            command = ["vex_analysis/main.py", "scenario-evolution", "--params", params_file]
            
            result = self._execute_command(command, timeout=900)  # 15 minute timeout
            
            if result["success"]:
                return self._parse_scenario_output(result["stdout"])
            else:
                raise RuntimeError(f"Scenario generation failed: {result['stderr']}")
        finally:
            # Clean up temporary file
            os.unlink(params_file)
    
    def _parse_analysis_output(self, output: str, analysis_type: str) -> Dict[str, Any]:
        """Parse analysis output from stdout"""
        try:
            # Look for JSON output in the stdout
            lines = output.strip().split('\n')
            for line in lines:
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    return json.loads(line)
            
            # If no JSON found, create a basic result structure
            return {
                "analysis_id": str(uuid.uuid4()),
                "analysis_type": analysis_type,
                "title": f"{analysis_type.title()} Analysis",
                "summary": "Analysis completed successfully",
                "created_at": datetime.utcnow().isoformat(),
                "raw_output": output
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "analysis_id": str(uuid.uuid4()),
                "analysis_type": analysis_type,
                "title": f"{analysis_type.title()} Analysis",
                "summary": "Analysis completed successfully",
                "created_at": datetime.utcnow().isoformat(),
                "raw_output": output
            }
    
    def _parse_visualization_output(self, output: str) -> Dict[str, Any]:
        """Parse visualization output from stdout"""
        try:
            # Look for JSON output
            lines = output.strip().split('\n')
            for line in lines:
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    return json.loads(line)
            
            return {
                "visualization_id": str(uuid.uuid4()),
                "charts": [],
                "created_at": datetime.utcnow().isoformat(),
                "raw_output": output
            }
        except json.JSONDecodeError:
            return {
                "visualization_id": str(uuid.uuid4()),
                "charts": [],
                "created_at": datetime.utcnow().isoformat(),
                "raw_output": output
            }
    
    def _parse_report_output(self, output: str) -> Dict[str, Any]:
        """Parse report output from stdout"""
        try:
            lines = output.strip().split('\n')
            for line in lines:
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    return json.loads(line)
            
            return {
                "report_id": str(uuid.uuid4()),
                "report_type": "strategic",
                "created_at": datetime.utcnow().isoformat(),
                "raw_output": output
            }
        except json.JSONDecodeError:
            return {
                "report_id": str(uuid.uuid4()),
                "report_type": "strategic",
                "created_at": datetime.utcnow().isoformat(),
                "raw_output": output
            }
    
    def _parse_ml_output(self, output: str, operation_type: str) -> Dict[str, Any]:
        """Parse ML operation output from stdout"""
        try:
            lines = output.strip().split('\n')
            for line in lines:
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    return json.loads(line)
            
            return {
                "operation_id": str(uuid.uuid4()),
                "operation_type": operation_type,
                "created_at": datetime.utcnow().isoformat(),
                "raw_output": output
            }
        except json.JSONDecodeError:
            return {
                "operation_id": str(uuid.uuid4()),
                "operation_type": operation_type,
                "created_at": datetime.utcnow().isoformat(),
                "raw_output": output
            }
    
    def _parse_scenario_output(self, output: str) -> Dict[str, Any]:
        """Parse scenario generation output from stdout"""
        try:
            lines = output.strip().split('\n')
            for line in lines:
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    return json.loads(line)
            
            return {
                "scenario_set_id": str(uuid.uuid4()),
                "scenarios": [],
                "created_at": datetime.utcnow().isoformat(),
                "raw_output": output
            }
        except json.JSONDecodeError:
            return {
                "scenario_set_id": str(uuid.uuid4()),
                "scenarios": [],
                "created_at": datetime.utcnow().isoformat(),
                "raw_output": output
            }
    
    # Push Back Specific Methods
    def run_push_back_analysis(self, robot_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run comprehensive Push Back strategy analysis
        
        Args:
            robot_specs: List of robot specifications
            
        Returns:
            Push Back analysis results
        """
        # Create temporary file for robot specs
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(robot_specs, f)
            specs_file = f.name
        
        try:
            command = [
                "-c",
                f"""
import sys
import json
sys.path.append('{self.vex_path}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer

# Load robot specs
with open('{specs_file}', 'r') as f:
    robot_specs = json.load(f)

analyzer = PushBackStrategyAnalyzer()
analysis = analyzer.run_comprehensive_analysis(robot_specs)

# Output results as JSON
print(json.dumps({{
    "analysis_id": "pb_" + str(hash(str(robot_specs))),
    "analysis_type": "push_back_comprehensive",
    "robot_specs": robot_specs,
    "block_flow_optimization": analysis.block_flow_optimization.to_dict() if hasattr(analysis, 'block_flow_optimization') else {{}},
    "autonomous_decision": analysis.autonomous_decision.to_dict() if hasattr(analysis, 'autonomous_decision') else {{}},
    "goal_priority_analysis": analysis.goal_priority_analysis.to_dict() if hasattr(analysis, 'goal_priority_analysis') else {{}},
    "parking_decision_analysis": analysis.parking_decision_analysis.to_dict() if hasattr(analysis, 'parking_decision_analysis') else {{}},
    "offense_defense_balance": analysis.offense_defense_balance.to_dict() if hasattr(analysis, 'offense_defense_balance') else {{}},
    "recommended_archetype": analysis.recommended_archetype,
    "recommendations": analysis.recommendations,
    "created_at": "{datetime.utcnow().isoformat()}"
}}))
"""
            ]
            
            result = self._execute_command(command, timeout=300)
            
            if result["success"]:
                return self._parse_analysis_output(result["stdout"], "push_back_comprehensive")
            else:
                raise RuntimeError(f"Push Back analysis failed: {result['stderr']}")
        finally:
            # Clean up temporary file
            os.unlink(specs_file)
    
    def calculate_push_back_score(self, field_state: Dict[str, Any], alliance: str = "red") -> Dict[str, Any]:
        """
        Calculate Push Back score for given field state
        
        Args:
            field_state: Current field state
            alliance: Alliance to calculate score for
            
        Returns:
            Scoring results
        """
        # Create temporary file for field state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(field_state, f)
            state_file = f.name
        
        try:
            command = [
                "-c",
                f"""
import sys
import json
sys.path.append('{self.vex_path}')
from vex_analysis.core.simulator import PushBackScoringEngine

# Load field state
with open('{state_file}', 'r') as f:
    field_state = json.load(f)

engine = PushBackScoringEngine()
score, breakdown = engine.calculate_push_back_score(field_state, "{alliance}")

print(json.dumps({{
    "score": score,
    "breakdown": breakdown,
    "alliance": "{alliance}",
    "field_state": field_state
}}))
"""
            ]
            
            result = self._execute_command(command)
            
            if result["success"]:
                return json.loads(result["stdout"].strip().split('\n')[-1])
            else:
                raise RuntimeError(f"Push Back scoring failed: {result['stderr']}")
        finally:
            # Clean up temporary file
            os.unlink(state_file)
    
    def run_push_back_monte_carlo(self, strategy: Dict[str, Any], num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for Push Back strategy
        
        Args:
            strategy: Strategy to simulate
            num_simulations: Number of simulations to run
            
        Returns:
            Simulation results
        """
        # Create temporary file for strategy
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(strategy, f)
            strategy_file = f.name
        
        try:
            command = [
                "-c",
                f"""
import sys
import json
sys.path.append('{self.vex_path}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer

# Load strategy
with open('{strategy_file}', 'r') as f:
    strategy = json.load(f)

analyzer = PushBackStrategyAnalyzer()
simulation = analyzer.run_monte_carlo_simulation(strategy, {num_simulations})

print(json.dumps({{
    "win_rate": simulation.win_rate,
    "avg_score": simulation.avg_score,
    "score_std": simulation.score_std,
    "scoring_breakdown": simulation.scoring_breakdown,
    "opponent_matchups": simulation.opponent_matchups,
    "performance_confidence": simulation.performance_confidence,
    "num_simulations": {num_simulations}
}}))
"""
            ]
            
            result = self._execute_command(command, timeout=600)
            
            if result["success"]:
                return json.loads(result["stdout"].strip().split('\n')[-1])
            else:
                raise RuntimeError(f"Push Back Monte Carlo simulation failed: {result['stderr']}")
        finally:
            # Clean up temporary file
            os.unlink(strategy_file)
    
    def get_push_back_archetypes(self) -> Dict[str, Any]:
        """
        Get available Push Back strategy archetypes
        
        Returns:
            Available archetypes
        """
        command = [
            "-c",
            f"""
import sys
sys.path.append('{self.vex_path}')
from vex_analysis.analysis.push_back_strategy_analyzer import PushBackStrategyAnalyzer

analyzer = PushBackStrategyAnalyzer()
archetypes = analyzer.get_strategy_archetypes()

print(json.dumps(archetypes))
"""
        ]
        
        result = self._execute_command(command)
        
        if result["success"]:
            return json.loads(result["stdout"].strip().split('\n')[-1])
        else:
            raise RuntimeError(f"Failed to get Push Back archetypes: {result['stderr']}")