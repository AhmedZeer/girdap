#!/bin/bash

# Recursively find all files ending in .rc.lib
find . -type f -name "*.rc.lib" | while IFS= read -r file; do
    # Remove the .rc.lib suffix and append .c.lib
    new_name="${file%.rc.lib}.c.lib"
    
    # Print what is happening for visibility
    echo "Renaming: $file -> $new_name"
    
    # Perform the rename
    mv "$file" "$new_name"
done

echo "Renaming complete!"