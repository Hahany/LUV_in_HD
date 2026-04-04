from pathlib import Path

def get_sub_project_root():
    """
    Locates the sub-project root by searching for the 'hd2d_src' directory.
    """
    # 1. Get the absolute path of the current file
    current_path = Path(__file__).resolve()

    # 2. Trace upwards to find the directory containing 'hd2d_src'
    for parent in [current_path] + list(current_path.parents):
        # Check if 'hd2d_src' exists and is a directory within the current parent
        if (parent / "hd2d_src").is_dir():
            return parent
            
    # 3. Return None or raise an error if the root cannot be found after the loop
    return None