#!/usr/bin/env python3
"""
U-CogNet Project Organizer
Automatically organizes files and maintains project structure
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import argparse

class ProjectOrganizer:
    """Automated project organization system"""

    def __init__(self, base_path="/mnt/c/Users/desar/Documents/Science/UCogNet"):
        self.base_path = Path(base_path)
        self.config_file = self.base_path / ".project_config.json"
        self.load_config()

    def load_config(self):
        """Load organization configuration"""
        default_config = {
            "file_patterns": {
                "documentation": ["*.md", "*.txt", "*.pdf"],
                "results": ["*results.json", "*analysis.json"],
                "checkpoints": ["*checkpoint*.json", "*model*.json"],
                "visualizations": ["*.png", "*.jpg", "*.svg"],
                "videos": ["*.mp4", "*.avi", "*.webm"]
            },
            "folder_mappings": {
                "docs": ["README.md", "CHANGELOG.md", "ARCHITECTURE_GUIDE.md"],
                "docs/architecture": ["ARCHITECTURE_GUIDE.md", "API_DOCUMENTATION.md"],
                "docs/experiments": ["*EXPERIMENT*.md", "*RESULT*.md"],
                "docs/research": ["RESEARCH_ROADMAP.md", "CHANGELOG.md"],
                "results/experiments": ["*results.json", "*_results/"],
                "results/analysis": ["*analysis.json"],
                "results/visualizations": ["*.png"],
                "checkpoints/training": ["*checkpoint*.json"],
                "checkpoints/models": ["*model*.json"]
            },
            "auto_clean": {
                "remove_empty_dirs": True,
                "archive_old_results": False,
                "max_result_files": 50
            }
        }

        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = {**default_config, **json.load(f)}
        else:
            self.config = default_config
            self.save_config()

    def save_config(self):
        """Save current configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def organize_files(self):
        """Organize files according to configuration"""
        print("🔄 Organizing U-CogNet project files...")

        moved_files = 0

        # Organize by file patterns
        for category, patterns in self.config["file_patterns"].items():
            for pattern in patterns:
                for file_path in self.base_path.glob(f"**/{pattern}"):
                    if file_path.is_file() and file_path.parent != self.base_path:
                        target_dir = self._get_target_directory(category, file_path)
                        if target_dir and self._move_file(file_path, target_dir):
                            moved_files += 1

        # Organize by specific folder mappings
        for target_folder, file_patterns in self.config["folder_mappings"].items():
            target_path = self.base_path / target_folder
            target_path.mkdir(parents=True, exist_ok=True)

            for pattern in file_patterns:
                for file_path in self.base_path.glob(f"**/{pattern}"):
                    if file_path.is_file():
                        if self._move_file(file_path, target_path):
                            moved_files += 1

        # Clean up empty directories
        if self.config["auto_clean"]["remove_empty_dirs"]:
            self._remove_empty_dirs()

        print(f"✅ Organization complete! Moved {moved_files} files.")
        return moved_files

    def _get_target_directory(self, category, file_path):
        """Determine target directory for a file"""
        category_map = {
            "documentation": "docs",
            "results": "results/experiments",
            "checkpoints": "checkpoints/training",
            "visualizations": "results/visualizations",
            "videos": "results/videos"
        }

        if category in category_map:
            return self.base_path / category_map[category]
        return None

    def _move_file(self, source_path, target_dir):
        """Move file to target directory"""
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / source_path.name

            # Handle duplicate names
            if target_path.exists():
                stem = target_path.stem
                suffix = target_path.suffix
                counter = 1
                while target_path.exists():
                    target_path = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            shutil.move(str(source_path), str(target_path))
            print(f"  📁 Moved: {source_path.name} → {target_dir.name}/")
            return True
        except Exception as e:
            print(f"  ❌ Error moving {source_path.name}: {e}")
            return False

    def _remove_empty_dirs(self):
        """Remove empty directories"""
        removed = 0
        for dir_path in sorted(self.base_path.rglob("*"), key=lambda x: len(str(x)), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    print(f"  🗑️  Removed empty directory: {dir_path.name}")
                    removed += 1
                except:
                    pass
        if removed > 0:
            print(f"Removed {removed} empty directories.")

    def generate_file_inventory(self):
        """Generate inventory of all project files"""
        inventory = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "files": {}
        }

        # Count files by type
        file_counts = {}
        for file_path in self.base_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                file_counts[ext] = file_counts.get(ext, 0) + 1

        inventory["summary"] = file_counts

        # Detailed file listing by directory
        for dir_path in sorted(self.base_path.rglob("*")):
            if dir_path.is_dir():
                rel_path = dir_path.relative_to(self.base_path)
                files = []
                for file_path in dir_path.glob("*"):
                    if file_path.is_file():
                        files.append({
                            "name": file_path.name,
                            "size": file_path.stat().st_size,
                            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        })

                if files:
                    inventory["files"][str(rel_path)] = files

        # Save inventory
        inventory_file = self.base_path / "reports" / "progress" / "file_inventory.json"
        inventory_file.parent.mkdir(parents=True, exist_ok=True)

        with open(inventory_file, 'w') as f:
            json.dump(inventory, f, indent=2)

        print(f"✅ File inventory generated: {inventory_file}")
        return inventory

    def create_backup(self, backup_name=None):
        """Create backup of important files"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_dir = self.base_path / "backups" / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Files to backup
        backup_targets = [
            "src/",
            "docs/",
            "results/experiments/",
            "checkpoints/",
            "*.py",
            "*.md",
            "*.json",
            "*.toml"
        ]

        backed_up = 0
        for target in backup_targets:
            for file_path in self.base_path.glob(target):
                if file_path.is_file():
                    rel_path = file_path.relative_to(self.base_path)
                    target_path = backup_dir / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, target_path)
                    backed_up += 1

        print(f"✅ Backup created: {backup_dir} ({backed_up} files)")
        return backup_dir

def main():
    """Main organization function"""
    parser = argparse.ArgumentParser(description="U-CogNet Project Organizer")
    parser.add_argument("--organize", action="store_true", help="Organize project files")
    parser.add_argument("--inventory", action="store_true", help="Generate file inventory")
    parser.add_argument("--backup", action="store_true", help="Create backup")
    parser.add_argument("--all", action="store_true", help="Run all organization tasks")

    args = parser.parse_args()

    organizer = ProjectOrganizer()

    if args.all or args.organize:
        organizer.organize_files()

    if args.all or args.inventory:
        organizer.generate_file_inventory()

    if args.backup:
        organizer.create_backup()

    if not any([args.organize, args.inventory, args.backup, args.all]):
        print("U-CogNet Project Organizer")
        print("Usage: python organize_project.py [--organize] [--inventory] [--backup] [--all]")
        print("\nOptions:")
        print("  --organize    Organize project files")
        print("  --inventory   Generate file inventory")
        print("  --backup      Create backup")
        print("  --all         Run all tasks")

if __name__ == "__main__":
    main()</content>
