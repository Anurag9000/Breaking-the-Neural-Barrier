import sys
import math
import argparse

def compute_text_params(depth, width):
    vocab_size = 50257
    ff_dim = width * 4
    embed_params = vocab_size * width
    # Layer norm (approx)
    ln_params = 4 * width
    # Attention (QKV + proj)
    attn_params = 4 * (width * width)
    # FFN
    ffn_params = 2 * (width * ff_dim)
    layer_params = ln_params + attn_params + ffn_params
    total = embed_params + (depth * layer_params) + embed_params # lm_head
    return total

def compute_vision_params(depth, width):
    patches = 196
    ff_dim = width * 4
    embed_params = (16 * 16 * 3) * width + (patches + 1) * width # patch embed + pos embed
    # Layer norm
    ln_params = 4 * width
    # Attention
    attn_params = 4 * (width * width)
    # FFN
    ffn_params = 2 * (width * ff_dim)
    layer_params = ln_params + attn_params + ffn_params
    total = embed_params + (depth * layer_params) + (width * 1000) # classifier head
    return total

def binary_search_width(depth, target_params, arch, max_width=4096):
    compute_fn = compute_text_params if arch == 'text' else compute_vision_params
    low = 16
    high = max_width
    best_w = low
    best_diff = float('inf')

    while low <= high:
        mid = (low + high) // 2
        p = compute_fn(depth, mid)
        diff = abs(p - target_params)
        
        if diff < best_diff:
            best_diff = diff
            best_w = mid

        if p < target_params:
            low = mid + 1
        else:
            high = mid - 1
            
    return best_w

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=['text', 'vision'], required=True)
    parser.add_argument("--min-band", type=int, default=1)
    parser.add_argument("--max-band", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--depths", type=str, default="1,2,4,8,12")
    parser.add_argument("--max-width", type=int, default=4096)
    args = parser.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    
    # Generate log-spaced parameter targets
    targets = []
    for band in range(args.min_band, args.max_band):
        low_val = 10**band
        high_val = 10**(band+1)
        # linear interpolate in log space or linear space?
        # Standard in your MLPS is linear interpolation inside the decade
        for i in range(args.samples):
            ratio = i / max(1, (args.samples - 1))
            val = low_val + ratio * (high_val - low_val)
            targets.append(int(val))

    seen = set()
    configs = []
    
    for depth in depths:
        for t in targets:
            w = binary_search_width(depth, t, args.arch, max_width=args.max_width)
            # Ensure heads > 0
            if args.arch == 'text':
                w = max(64, w)
            else:
                w = max(64, w)
            
            # Snap to nearest 64 for multi-head attention compatibility
            w = max(64, (w // 64) * 64)
            
            p = compute_text_params(depth, w) if args.arch == 'text' else compute_vision_params(depth, w)
            if (depth, w) not in seen:
                seen.add((depth, w))
                configs.append((depth, w, p))

    # Print in bash parseable format
    for depth, w, p in sorted(configs, key=lambda x: (x[0], x[1])):
        print(f"{depth} {w}")

if __name__ == "__main__":
    main()
