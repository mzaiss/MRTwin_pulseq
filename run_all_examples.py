#!/usr/bin/env python3
"""
Script to run all Python files in the ex/ folder and report execution errors.
"""

import os
import sys
import traceback
import subprocess
from pathlib import Path

# Files to exclude (require real MRI scanner hardware)
EXCLUDE_FILES = [
    #'exE01_FLASH_2D.py',           # Uses real scanner data
    #'exF01_bSSFP_2D_radial_nufft.py',  # Uses real scanner data
    #'solE01_FLASH_2D.py',          # Uses real scanner data
    #'solE02_RARE_2D.py',           # Uses real scanner data
    #'solE03_stimulated_echo.py',   # Uses real scanner data
    #'solF02_bSSFP_2D_radial_CS.py',   # Uses long iterative recon
    #'solF03_simple_undersampled_CS.py',   # Uses long iterative recon
    #'solF04_bSSFP_2D_cartesian_CS.py',   # Uses long iterative recon
]

def run_python_file(file_path):
    """Run a Python file and return success status, error message, and timing status."""
    try:
        # Change to the ex/ directory to maintain relative imports
        original_cwd = os.getcwd()
        os.chdir(file_path.parent)
        
        # For interactive mode, we'll import and run directly
        # Set up matplotlib for non-blocking plots
        import matplotlib
        matplotlib.use('Qt5Agg')  # Use Qt backend for interactive windows
        import matplotlib.pyplot as plt
        plt.ion()  # Turn on interactive mode for non-blocking
        
        # Create a custom namespace to capture the 'ok' variable
        namespace = {}
        
        # Capture stdout to detect timing failures
        import io
        import sys
        
        # Redirect stdout to capture output
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Import and run the file in the custom namespace
            import runpy
            runpy.run_path(file_path.name, run_name='__main__', init_globals=namespace)
            
            # Check if 'ok' variable exists and its value
            timing_ok = namespace.get('ok', None)
            
            # Also check captured output for timing failure messages
            output_text = captured_output.getvalue()
            if 'Timing check failed' in output_text:
                timing_ok = False
                
        finally:
            # Restore stdout
            sys.stdout = old_stdout
        
        # Close all figures to save memory and render time
        import matplotlib.pyplot as plt
        plt.close('all')
        
        os.chdir(original_cwd)
        return True, None, timing_ok
            
    except Exception as e:
        os.chdir(original_cwd)
        return False, f"Failed to execute: {str(e)}", None

def main():
    """Main function to run all Python files in ex/ folder."""
    ex_dir = Path("ex")
    
    if not ex_dir.exists():
        print(f"Error: {ex_dir} directory not found!")
        return
    
    # Get all Python files in ex/ directory, sorted alphabetically
    python_files = sorted(ex_dir.glob("*.py"))
    
    # Filter out excluded files
    python_files = [f for f in python_files if f.name not in EXCLUDE_FILES]
    
    if not python_files:
        print("No Python files found in ex/ directory")
        return
    
    print(f"Found {len(python_files)} Python files in ex/ directory (excluding {len(EXCLUDE_FILES)} hardware-dependent files)")
    print("=" * 50)
    
    failed_files = []
    timing_issues = []
    
    for i, file_path in enumerate(python_files, 1):
        print(f"[{i}/{len(python_files)}] Running {file_path.name}...", end=" ")
        
        success, error, timing_ok = run_python_file(file_path)
        
        if success:
            if timing_ok is False:  # Explicitly False (timing check failed)
                print("⚠ TIMING ISSUE")
                timing_issues.append(file_path.name)
            elif timing_ok is True:  # Explicitly True (timing check passed)
                print("✓ OK")
            else:  # None or anything else (no timing check found or variable overwritten)
                print("✓ OK")
        else:
            print("✗ ERROR")
            failed_files.append((file_path.name, error))
    
    print("=" * 50)
    
    # Report execution errors
    if failed_files:
        print(f"\n❌ {len(failed_files)} files had execution errors:")
        print("-" * 30)
        for filename, error in failed_files:
            print(f"\n📁 {filename}")
            print(f"Error: {error}")
            print("-" * 30)
    else:
        print("\n✅ All files executed successfully!")
    
    # Report timing issues
    if timing_issues:
        print(f"\n⚠️  {len(timing_issues)} files had timing issues (seq.check_timing() failed):")
        print("-" * 50)
        for filename in timing_issues:
            print(f"📁 {filename}")
        print("-" * 50)
    else:
        print("\n✅ No timing issues found!")

if __name__ == "__main__":
    main()
