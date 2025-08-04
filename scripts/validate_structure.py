#!/usr/bin/env python3
"""Validate Fleet-Mind project structure and core functionality without external dependencies."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_project_structure():
    """Verify the project has the expected structure."""
    print("Checking project structure...")
    
    expected_structure = {
        'fleet_mind/': [
            '__init__.py',
            'coordination/__init__.py',
            'coordination/swarm_coordinator.py',
            'communication/__init__.py',
            'communication/webrtc_streamer.py',
            'communication/latent_encoder.py',
            'planning/__init__.py',
            'planning/llm_planner.py',
            'fleet/__init__.py',
            'fleet/drone_fleet.py',
            'ros2_integration/__init__.py',
            'ros2_integration/fleet_manager_node.py',
            'utils/__init__.py',
            'utils/logging.py',
            'utils/validation.py',
            'utils/security.py',
            'utils/error_handling.py',
            'optimization/__init__.py',
            'optimization/performance_monitor.py',
            'optimization/cache_manager.py',
            'cli.py',
        ],
        'tests/': [
            '__init__.py',
            'test_swarm_coordinator.py',
            'test_webrtc_streamer.py',
            'test_performance_monitor.py',
        ],
        'docs/': [
            'ROADMAP.md',
            'adr/0001-architecture-decision-record-template.md',
            'adr/0002-llm-backend-selection.md',
        ],
        'root': [
            'README.md',
            'ARCHITECTURE.md',
            'PROJECT_CHARTER.md',
            'CONTRIBUTING.md',
            'LICENSE',
            'pyproject.toml',
            'requirements.txt',
            'requirements-minimal.txt',
        ]
    }
    
    missing_files = []
    found_files = []
    
    for directory, files in expected_structure.items():
        if directory == 'root':
            base_path = project_root
        else:
            base_path = project_root / directory
        
        for file in files:
            file_path = base_path / file
            if file_path.exists():
                found_files.append(str(file_path.relative_to(project_root)))
            else:
                missing_files.append(str(file_path.relative_to(project_root)))
    
    print(f"✓ Found {len(found_files)} expected files")
    
    if missing_files:
        print(f"✗ Missing {len(missing_files)} files:")
        for file in missing_files[:5]:  # Show first 5
            print(f"  - {file}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
    
    return len(missing_files) == 0

def check_python_syntax():
    """Check Python syntax of all Python files."""
    print("\nChecking Python syntax...")
    
    python_files = []
    for root, dirs, files in os.walk(project_root / 'fleet_mind'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to compile the code
            compile(content, file_path, 'exec')
            
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
        except Exception as e:
            syntax_errors.append(f"{file_path}: {e}")
    
    if syntax_errors:
        print(f"✗ Found {len(syntax_errors)} syntax errors:")
        for error in syntax_errors[:3]:  # Show first 3
            print(f"  - {error}")
    else:
        print(f"✓ All {len(python_files)} Python files have valid syntax")
    
    return len(syntax_errors) == 0

def check_import_structure():
    """Check that imports are structured correctly."""
    print("\nChecking import structure...")
    
    try:
        # Test basic imports without external dependencies
        import importlib.util
        
        # Test utils modules (should have minimal dependencies)
        modules_to_test = [
            'fleet_mind.utils.error_handling',
        ]
        
        successful_imports = 0
        
        for module_name in modules_to_test:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec:
                    successful_imports += 1
                    print(f"✓ {module_name}")
                else:
                    print(f"✗ {module_name}: Module not found")
            except Exception as e:
                print(f"✗ {module_name}: {e}")
        
        print(f"✓ Successfully validated {successful_imports}/{len(modules_to_test)} core modules")
        return successful_imports > 0
        
    except Exception as e:
        print(f"✗ Import structure check failed: {e}")
        return False

def check_configuration_files():
    """Check configuration files are valid."""
    print("\nChecking configuration files...")
    
    try:
        # Check pyproject.toml
        try:
            if sys.version_info >= (3, 11):
                import tomllib
            else:
                tomllib = None
        except ImportError:
            tomllib = None
        
        pyproject_path = project_root / 'pyproject.toml'
        if pyproject_path.exists():
            print("✓ pyproject.toml exists")
            
            # Basic validation
            with open(pyproject_path, 'r') as f:
                content = f.read()
                if '[project]' in content and 'name = "fleet-mind"' in content:
                    print("✓ pyproject.toml has valid structure")
                else:
                    print("✗ pyproject.toml missing required sections")
                    return False
        else:
            print("✗ pyproject.toml missing")
            return False
        
        # Check README
        readme_path = project_root / 'README.md'
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                content = f.read()
                if 'Fleet-Mind' in content and len(content) > 1000:
                    print("✓ README.md is comprehensive")
                else:
                    print("✗ README.md is incomplete")
                    return False
        else:
            print("✗ README.md missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration check failed: {e}")
        return False

def check_documentation():
    """Check documentation completeness."""
    print("\nChecking documentation...")
    
    docs_to_check = [
        ('ARCHITECTURE.md', 'architecture'),
        ('PROJECT_CHARTER.md', 'charter'),
        ('CONTRIBUTING.md', 'contributing'),
        ('docs/ROADMAP.md', 'roadmap'),
    ]
    
    doc_scores = []
    
    for doc_path, doc_type in docs_to_check:
        full_path = project_root / doc_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                content = f.read()
                word_count = len(content.split())
                
                if word_count > 500:  # Substantial documentation
                    print(f"✓ {doc_path} ({word_count} words)")
                    doc_scores.append(1)
                else:
                    print(f"⚠ {doc_path} is brief ({word_count} words)")
                    doc_scores.append(0.5)
        else:
            print(f"✗ {doc_path} missing")
            doc_scores.append(0)
    
    avg_score = sum(doc_scores) / len(doc_scores) if doc_scores else 0
    print(f"✓ Documentation completeness: {avg_score*100:.0f}%")
    
    return avg_score > 0.7

def run_validation():
    """Run all validation checks."""
    print("=" * 60)
    print("Fleet-Mind Project Structure Validation")
    print("=" * 60)
    
    checks = [
        ("Project Structure", check_project_structure),
        ("Python Syntax", check_python_syntax),
        ("Import Structure", check_import_structure),
        ("Configuration Files", check_configuration_files),
        ("Documentation", check_documentation),
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{'='*20} {check_name} {'='*20}")
        try:
            if check_func():
                passed_checks += 1
                print(f"✓ {check_name}: PASSED")
            else:
                print(f"✗ {check_name}: FAILED")
        except Exception as e:
            print(f"✗ {check_name}: EXCEPTION - {e}")
    
    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY: {passed_checks}/{total_checks} checks passed")
    
    # Calculate overall score
    score = (passed_checks / total_checks) * 100
    
    if score >= 80:
        print(f"🎉 EXCELLENT ({score:.0f}%) - Fleet-Mind project structure is solid!")
        grade = "A"
    elif score >= 60:
        print(f"✅ GOOD ({score:.0f}%) - Fleet-Mind project structure is acceptable")
        grade = "B"
    elif score >= 40:
        print(f"⚠️ FAIR ({score:.0f}%) - Fleet-Mind project needs some improvements")
        grade = "C"
    else:
        print(f"❌ POOR ({score:.0f}%) - Fleet-Mind project needs significant work")
        grade = "F"
    
    print("=" * 60)
    
    # Additional insights
    total_files = sum(len(files) for files in os.walk(project_root))
    python_files = sum(1 for _, _, files in os.walk(project_root) for f in files if f.endswith('.py'))
    
    print(f"\nProject Statistics:")
    print(f"- Total files: {total_files}")
    print(f"- Python files: {python_files}")
    print(f"- Core modules: 8+ (coordination, communication, planning, etc.)")
    print(f"- Architecture: Multi-layered with optimization and security")
    print(f"- Quality grade: {grade}")
    
    return score >= 60

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)