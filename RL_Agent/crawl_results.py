import os
import json
from collections import defaultdict


def detailed_directory_crawler(root_dir):
    """
    Crawls through a directory and all its subdirectories,
    recording all files found in each directory.
    """
    directory_contents = {}

    # Check if the directory exists
    if not os.path.exists(root_dir):
        print(f"Directory not found: {root_dir}")
        return directory_contents

    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Get the relative path to make output more readable
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == '.':
            rel_path = os.path.basename(root_dir)

        # Create a list of all files in this directory
        directory_contents[rel_path] = []

        # Add all files in this directory to the list
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_size = os.path.getsize(file_path)

            # Add file info to the list
            directory_contents[rel_path].append({
                "filename": filename,
                "size_bytes": file_size,
                "full_path": file_path
            })

    return directory_contents


# Path to crawl
root_directory = r"C:\Users\USER\PycharmProjects\fyp-rnd\RL_Agent\run_results"

# Get directory contents
print(f"Crawling directory: {root_directory}")
directory_listing = detailed_directory_crawler(root_directory)

# Save to JSON
listing_file = "run_results/directory_listing.json"
with open(listing_file, 'w') as f:
    json.dump(directory_listing, f, indent=2)
print(f"Directory listing saved to {listing_file}")

# Additionally create a simple text listing for easy reading
text_listing = "directory_listing.txt"
with open(text_listing, 'w') as f:
    for directory, files in directory_listing.items():
        f.write(f"\nDIRECTORY: {directory}\n")
        f.write("=" * 80 + "\n")
        for file_info in files:
            f.write(f"  - {file_info['filename']} ({file_info['size_bytes']} bytes)\n")
        f.write("\n")

print(f"Text listing saved to {text_listing}")