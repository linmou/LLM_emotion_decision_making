#!/usr/bin/env python3
"""
Test Data Version Control System

Manages versioned test datasets for regression testing and ensures
consistent test data across development cycles.
"""

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class DatasetVersion:
    """Metadata for a versioned test dataset"""
    version: str
    created_at: str
    description: str
    file_hash: str
    file_size: int
    test_categories: List[str]
    compatibility: str  # e.g., ">=1.0.0"


class TestDataVersionManager:
    """Manages versioned test data for regression testing"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent
        self.versions_dir = self.data_dir / "versions"
        self.current_dir = self.data_dir / "current"  
        self.metadata_file = self.data_dir / "dataset_versions.json"
        
        # Ensure directories exist
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.current_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_metadata()
    
    def _load_metadata(self):
        """Load dataset version metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                data = json.load(f)
                self.versions = {
                    name: {
                        version: DatasetVersion(**version_data)
                        for version, version_data in versions.items()
                    }
                    for name, versions in data.items()
                }
        else:
            self.versions = {}
    
    def _save_metadata(self):
        """Save dataset version metadata"""
        data = {}
        for name, versions in self.versions.items():
            data[name] = {
                version: asdict(version_data)
                for version, version_data in versions.items()
            }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def create_version(self, dataset_name: str, source_file: Path, 
                      version: str, description: str = "",
                      test_categories: List[str] = None,
                      compatibility: str = ">=1.0.0") -> DatasetVersion:
        """Create a new version of a test dataset"""
        
        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")
        
        # Create version directory
        version_dir = self.versions_dir / dataset_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy file to versioned location
        dest_file = version_dir / source_file.name
        shutil.copy2(source_file, dest_file)
        
        # Create metadata
        dataset_version = DatasetVersion(
            version=version,
            created_at=datetime.now().isoformat(),
            description=description,
            file_hash=self._calculate_file_hash(dest_file),
            file_size=dest_file.stat().st_size,
            test_categories=test_categories or [],
            compatibility=compatibility
        )
        
        # Update metadata
        if dataset_name not in self.versions:
            self.versions[dataset_name] = {}
        self.versions[dataset_name][version] = dataset_version
        
        self._save_metadata()
        
        print(f"✅ Created dataset version: {dataset_name} v{version}")
        print(f"   File: {dest_file}")
        print(f"   Hash: {dataset_version.file_hash[:16]}...")
        print(f"   Size: {dataset_version.file_size} bytes")
        
        return dataset_version
    
    def get_version(self, dataset_name: str, version: str) -> Optional[Path]:
        """Get path to specific dataset version"""
        
        if dataset_name not in self.versions:
            return None
        
        if version not in self.versions[dataset_name]:
            return None
        
        # Find the file in version directory
        version_dir = self.versions_dir / dataset_name / version
        if not version_dir.exists():
            return None
        
        # Return first file found (assumes one file per version)
        for file_path in version_dir.iterdir():
            if file_path.is_file():
                return file_path
        
        return None
    
    def get_latest_version(self, dataset_name: str) -> Optional[Path]:
        """Get path to latest version of dataset"""
        
        if dataset_name not in self.versions:
            return None
        
        # Get versions sorted by creation date
        versions = list(self.versions[dataset_name].items())
        if not versions:
            return None
        
        versions.sort(key=lambda x: x[1].created_at, reverse=True)
        latest_version = versions[0][0]
        
        return self.get_version(dataset_name, latest_version)
    
    def set_current_version(self, dataset_name: str, version: str):
        """Set current version for a dataset (symlink)"""
        
        source_file = self.get_version(dataset_name, version)
        if not source_file:
            raise ValueError(f"Version {version} of {dataset_name} not found")
        
        # Create symlink in current directory
        current_file = self.current_dir / f"{dataset_name}.{source_file.suffix}"
        
        # Remove existing symlink/file
        if current_file.exists() or current_file.is_symlink():
            current_file.unlink()
        
        # Create relative symlink
        relative_path = source_file.relative_to(self.current_dir)
        current_file.symlink_to(relative_path)
        
        print(f"✅ Set current version: {dataset_name} -> v{version}")
    
    def list_versions(self, dataset_name: str = None) -> Dict[str, List[DatasetVersion]]:
        """List all versions for dataset(s)"""
        
        if dataset_name:
            if dataset_name in self.versions:
                return {dataset_name: list(self.versions[dataset_name].values())}
            else:
                return {}
        
        return {
            name: list(versions.values())
            for name, versions in self.versions.items()
        }
    
    def verify_integrity(self, dataset_name: str = None) -> Dict[str, bool]:
        """Verify integrity of versioned datasets"""
        
        results = {}
        datasets_to_check = [dataset_name] if dataset_name else list(self.versions.keys())
        
        for name in datasets_to_check:
            if name not in self.versions:
                results[name] = False
                continue
            
            all_valid = True
            for version, metadata in self.versions[name].items():
                file_path = self.get_version(name, version)
                
                if not file_path or not file_path.exists():
                    print(f"❌ Missing file: {name} v{version}")
                    all_valid = False
                    continue
                
                # Verify hash
                current_hash = self._calculate_file_hash(file_path)
                if current_hash != metadata.file_hash:
                    print(f"❌ Hash mismatch: {name} v{version}")
                    print(f"   Expected: {metadata.file_hash}")
                    print(f"   Actual:   {current_hash}")
                    all_valid = False
                
                # Verify size
                current_size = file_path.stat().st_size
                if current_size != metadata.file_size:
                    print(f"❌ Size mismatch: {name} v{version}")
                    print(f"   Expected: {metadata.file_size}")
                    print(f"   Actual:   {current_size}")
                    all_valid = False
            
            results[name] = all_valid
            if all_valid:
                print(f"✅ {name}: All versions verified")
        
        return results
    
    def create_baseline_datasets(self):
        """Create baseline test datasets for regression testing"""
        
        baseline_datasets = {
            "minimal_passkey": {
                "data": [
                    {"id": 0, "input": "What is the passkey?", "answer": "12345", 
                     "context": "The secret passkey is 12345", "task_name": "passkey"},
                    {"id": 1, "input": "What is the passkey?", "answer": "67890",
                     "context": "The secret passkey is 67890", "task_name": "passkey"}
                ],
                "description": "Minimal passkey test data for InfiniteBench",
                "categories": ["unit", "regression", "infinitebench"]
            },
            
            "minimal_qa": {
                "data": [
                    {"id": "test_0", "input": "What is machine learning?", 
                     "answers": ["Machine learning is a subset of AI"],
                     "context": "Machine learning is a branch of artificial intelligence.", 
                     "task_name": "narrativeqa"},
                    {"id": "test_1", "input": "What is deep learning?",
                     "answers": ["Deep learning uses neural networks"], 
                     "context": "Deep learning is based on neural network architectures.",
                     "task_name": "narrativeqa"}
                ],
                "description": "Minimal QA test data for LongBench",
                "categories": ["unit", "regression", "longbench"]
            },
            
            "emotion_validation": {
                "data": [
                    {"id": 1, "input": "How are you feeling?", 
                     "ground_truth": ["happy", "joyful", "content"], "category": "happiness"},
                    {"id": 2, "input": "What's your emotional state?",
                     "ground_truth": ["angry", "mad", "furious"], "category": "anger"},
                    {"id": 3, "input": "How do you feel right now?",
                     "ground_truth": ["neutral", "calm", "fine"], "category": "neutral"}
                ],
                "description": "Emotion validation test data",
                "categories": ["unit", "regression", "emotion_check"]
            }
        }
        
        created_versions = []
        
        for dataset_name, config in baseline_datasets.items():
            # Write data to temporary file
            temp_file = self.data_dir / f"temp_{dataset_name}.jsonl"
            
            with open(temp_file, 'w') as f:
                for item in config["data"]:
                    f.write(json.dumps(item) + '\n')
            
            try:
                # Create version
                version_info = self.create_version(
                    dataset_name=dataset_name,
                    source_file=temp_file,
                    version="1.0.0",
                    description=config["description"],
                    test_categories=config["categories"]
                )
                
                # Set as current
                self.set_current_version(dataset_name, "1.0.0")
                
                created_versions.append((dataset_name, version_info))
                
            finally:
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()
        
        return created_versions


def main():
    """Command-line interface for test data version management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Data Version Manager")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Create version command
    create_parser = subparsers.add_parser('create', help='Create new dataset version')
    create_parser.add_argument('dataset_name', help='Dataset name')
    create_parser.add_argument('source_file', type=Path, help='Source data file')
    create_parser.add_argument('version', help='Version string (e.g., 1.0.0)')
    create_parser.add_argument('--description', default='', help='Version description')
    create_parser.add_argument('--categories', nargs='+', help='Test categories')
    
    # List versions command
    list_parser = subparsers.add_parser('list', help='List dataset versions')
    list_parser.add_argument('dataset_name', nargs='?', help='Dataset name (optional)')
    
    # Set current command  
    current_parser = subparsers.add_parser('current', help='Set current dataset version')
    current_parser.add_argument('dataset_name', help='Dataset name')
    current_parser.add_argument('version', help='Version to set as current')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify dataset integrity')
    verify_parser.add_argument('dataset_name', nargs='?', help='Dataset name (optional)')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize baseline datasets')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = TestDataVersionManager()
    
    if args.command == 'create':
        manager.create_version(
            args.dataset_name, 
            args.source_file, 
            args.version,
            args.description,
            args.categories
        )
    
    elif args.command == 'list':
        versions = manager.list_versions(args.dataset_name)
        
        for dataset_name, version_list in versions.items():
            print(f"\n📁 Dataset: {dataset_name}")
            print("=" * 40)
            
            for version in sorted(version_list, key=lambda x: x.created_at, reverse=True):
                print(f"  v{version.version} - {version.created_at[:10]}")
                print(f"    {version.description}")
                print(f"    Categories: {', '.join(version.categories)}")
                print(f"    Hash: {version.file_hash[:16]}...")
                print()
    
    elif args.command == 'current':
        manager.set_current_version(args.dataset_name, args.version)
    
    elif args.command == 'verify':
        results = manager.verify_integrity(args.dataset_name)
        
        all_good = all(results.values())
        if all_good:
            print("✅ All datasets verified successfully")
        else:
            failed = [name for name, success in results.items() if not success]
            print(f"❌ Verification failed for: {', '.join(failed)}")
    
    elif args.command == 'init':
        print("🎯 Creating baseline test datasets...")
        versions = manager.create_baseline_datasets()
        
        print(f"\n✅ Created {len(versions)} baseline datasets:")
        for name, version_info in versions:
            print(f"  - {name} v{version_info.version}")


if __name__ == "__main__":
    main()