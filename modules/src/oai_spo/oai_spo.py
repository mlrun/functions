# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import os
import datetime as dt
from pathlib import Path
from typing import Optional
import pandas as pd
import mlrun
from config.config import load_config


class OaiHub:
    """
    OAI Hub class for managing MLRun project setup and artifact logging.
    
    This class handles:
    - Environment configuration
    - Project creation/retrieval
    - Input data processing and logging
    - Configuration dataset logging
    """
    
    def __init__(
        self,
        project_name: str,
        data_dir: str,
        default_env_file: str,
        local_env_file: str,
        pipeline_config_path: str,
        default_image: str,
        source: str,
    ):
        """
        Initialize OaiHub instance.
        
        Args:
            project_name: Name of the MLRun project
            data_dir: Directory containing data files
            default_env_file: Default environment file path
            local_env_file: Local environment file path (takes precedence)
            pipeline_config_path: Path to pipeline configuration YAML
            default_image: Default Docker image for the project
            source: Source location for the project (S3 path)
        """
        self.project_name = project_name
        self.data_dir = data_dir
        self.default_env_file = default_env_file
        self.local_env_file = local_env_file
        self.pipeline_config_path = pipeline_config_path
        self.default_image = default_image
        self.source = source
        
        self.project: Optional[mlrun.projects.MlrunProject] = None
        self.pipeline_config = None
    
    def setup_environment(self) -> str:
        """
        Load environment variables from env file.
        Prefers local env file if it exists, otherwise uses default.
        
        Returns:
            Path to the env file that was loaded
        """
        env_file = (
            self.local_env_file
            if os.path.exists(self.local_env_file)
            else self.default_env_file
        )
        print(f"Loading environment from: {env_file}")
        mlrun.set_env_from_file(env_file)
        return env_file
    
    def load_configuration(self):
        """Load pipeline configuration from YAML file."""
        print(f"Loading configuration from: {self.pipeline_config_path}")
        self.pipeline_config = load_config(self.pipeline_config_path)
    
    def get_or_create_project(self) -> mlrun.projects.MlrunProject:
        """
        Get or create the MLRun project.
        
        Returns:
            MLRun project instance
        """
        print(f"Getting or creating project '{self.project_name}'...")
        self.project = mlrun.get_or_create_project(
            self.project_name,
            parameters={
                "source": self.source,
                "pipeline_config_path": self.pipeline_config_path,
                "default_image": self.default_image,
            },
        )
        return self.project
    
    def process_and_log_input_data(self) -> Optional[str]:
        """
        Process input data (shift dates to current time) and log as artifact.
        
        Returns:
            Artifact key if successful, None otherwise
        """
        print("Processing 'input_data'...")
        input_data_path = os.path.join(self.data_dir, "01_raw/sample_input_data.csv")
        
        if not os.path.exists(input_data_path):
            print(f"Warning: {input_data_path} not found.")
            return None
        
        orig_input_data = pd.read_csv(input_data_path)
        
        # Shift dates to current time
        if "date" in orig_input_data.columns:
            # Convert to datetime if not already
            orig_input_data["date"] = pd.to_datetime(orig_input_data["date"])
            
            # Calculate delta to shift max date to now
            max_date = orig_input_data["date"].max()
            delta = dt.datetime.now() - max_date
            
            # Apply shift and floor to hour
            orig_input_data["date"] = orig_input_data["date"] + delta
            orig_input_data["date"] = orig_input_data["date"].dt.floor("h")
        
        print("Logging 'input_data' artifact...")
        artifact = self.project.log_dataset(
            key="input_data",
            df=orig_input_data,
            format="csv"
        )
        return artifact.key if artifact else None
    
    def log_config_dataset(
        self, key: str, filename: str, label_schema: str
    ) -> Optional[str]:
        """
        Log a configuration dataset from a CSV file.
        
        Args:
            key: Artifact key name
            filename: Name of the CSV file (relative to data_dir/01_raw/)
            label_schema: Schema label for the artifact
            
        Returns:
            Artifact key if successful, None otherwise
        """
        file_path = os.path.join(self.data_dir, f"01_raw/{filename}")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping {key}.")
            return None
        
        print(f"Logging '{key}' from {filename}...")
        df = pd.read_csv(file_path, sep=";")
        artifact = self.project.log_dataset(
            key=key,
            df=df,
            format="csv",
            labels={"parameters_schema": label_schema},
        )
        return artifact.key if artifact else None
    
    def log_all_config_datasets(self):
        """Log all configuration datasets."""
        config_datasets = [
            ("sample_tags_raw", "sample_tags_raw_config.csv", "raw"),
            ("sample_tags_meta", "sample_tags_meta_config.csv", "meta"),
            ("sample_tags_outliers", "sample_tags_outliers_config.csv", "outliers"),
            ("sample_tags_imputation", "sample_tags_imputation_config.csv", "impute"),
            (
                "sample_tags_on_off_dependencies",
                "sample_tags_on_off_dependencies_config.csv",
                "on_off",
            ),
            ("sample_tags_resample", "sample_tags_resample_config.csv", "resample"),
        ]
        
        for key, filename, label_schema in config_datasets:
            self.log_config_dataset(key, filename, label_schema)
    
    def setup(self):
        """
        Complete setup process:
        1. Setup environment
        2. Load configuration
        3. Get or create project
        4. Process and log input data
        5. Log all configuration datasets
        """
        self.setup_environment()
        self.load_configuration()
        self.get_or_create_project()
        self.process_and_log_input_data()
        self.log_all_config_datasets()
        print("Artifact logging completed.")

