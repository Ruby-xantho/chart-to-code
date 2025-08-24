import os
import glob

def reassemble_model():
    """Reassemble the split model file"""
    # Find all model parts in order
    parts = sorted(glob.glob('model_part_*'))

    if not parts:
        print("No model parts found!")
        return False

    print(f"Found {len(parts)} model parts, reassembling...")

    with open('model.safetensors', 'wb') as output_file:
        for part in parts:
            print(f"Adding {part}...")
            with open(part, 'rb') as part_file:
                output_file.write(part_file.read())

    print("Model reassembled successfully!")
    return True

if __name__ == "__main__":
    reassemble_model()
