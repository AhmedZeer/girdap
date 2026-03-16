import struct

def parse_bfloat16_hex(hex_string):
    """
    Parses a continuous hex string of bfloat16 values into Python floats.
    Assumes Big-Endian byte order.
    """
    chunk_size = 4 # 4 hex characters = 16 bits
    parsed_floats = []
    
    # Process the string in chunks of 4 characters
    for i in range(0, len(hex_string), chunk_size):
        bfloat_hex = hex_string[i:i+chunk_size]
        
        # Pad with 16 bits of zeros to reconstruct a 32-bit float
        float32_hex = bfloat_hex + "0000"
        
        # Convert hex string to bytes
        float32_bytes = bytes.fromhex(float32_hex)
        
        # Unpack as a standard 32-bit float
        # '>f' specifies Big-Endian, standard size float
        float_val = struct.unpack('>f', float32_bytes)[0]
        parsed_floats.append((bfloat_hex, float_val))
        
    return parsed_floats

# Your hex sequence
# hex_sequence = "3F00C020BFC0C080404040003F800000"
hex_sequence = "FF80404000000000000000003F8C0000"

# Parse and print
print(f"{'Hex Code':<10} | {'Float Value'}")
print("-" * 35)

results = parse_bfloat16_hex(hex_sequence)
for hex_chunk, value in results:
    print(f"{hex_chunk:<10} | {value}")
