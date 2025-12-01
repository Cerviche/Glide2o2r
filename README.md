# Texture Conversion Pipeline: Rice (Glide) → o2r (NextGen) Format for SoH

## Project Description

This pipeline automates the conversion of Ocarina of Time community texture packs from the legacy Rice/Glide format to the modern o2r format used by **Ship of Harkinian (SoH)**.

---

## Texture Formats

### 1. Rice/Glide Format (Legacy)
- **Used by:** Mupen64Plus, Project64 with GlideN64 plugin  
- **Naming:** Texture hash-based filenames (e.g., `THE_LEGEND_OF_ZELDA#92C1F51C#4#1_rgb.png`)  
- **Purpose:** Matches textures in N64 game RAM via hash lookup  
- **Structure:** Flat directory, no organization  

### 2. o2r Format (Modern)
- **Used by:** Ship of Harkinian (SoH)  
- **Naming:** Descriptive, organized names (e.g., `objects/gameplay_keep/gEffBombExplosion1Tex.png`)  
- **Purpose:** Direct filepath-based loading  
- **Structure:** Hierarchical directories mirroring game assets  

### 3. Ocarina Reloaded Bridge
- **Reloaded Glide Pack:** Rice format textures from Ocarina Reloaded project  
- **Reloaded o2r Pack:** Same textures in o2r format  
- **Key Insight:** Both packs contain the same textures with different naming schemes, so they can be used as a translation dictionary.

---

## The Conversion Challenge

Community texture packs (like **Hyrule Field HD** or character packs) exist only in Rice format. To use them with SoH, we need to:  

1. Match Rice-format community textures to Rice-format Reloaded textures (by filename hash)  
2. Find the corresponding o2r-format texture in the Reloaded o2r pack (by content matching)  
3. Convert community textures to NG format with proper directory structure  

---

## Technical Implementation

### Stage 1: Hash-to-Hash Matching
```text
Community Rice: "THE_LEGEND_OF_ZELDA#92C1F51C#4#1_rgb.png"
Reloaded Rice:  "THE_LEGEND_OF_ZELDA#92C1F51C#4#1_rgb.png"  # Exact match!
Result: Copy matching Reloaded Rice file to working directory

Stage 2: Rice-to-o2r Content Matching

Reloaded Rice: "THE_LEGEND_OF_ZELDA#92C1F51C#4#1_rgb.png"
Calculate perceptual hash → "a1b2c3d4e5f6..."
Find matching hash in Reloaded o2r pack
Reloaded o2r: "objects/gameplay_keep/gEffBombExplosion1Tex.png" → same hash!
Build mapping: COM_hash_filename → NG_descriptive_path

Stage 3: Community Conversion

Community Rice: "THE_LEGEND_OF_ZELDA#92C1F51C#4#1_rgb.png"
Lookup in mapping → "objects/gameplay_keep/gEffBombExplosion1Tex.png"
Copy to: comout/objects/gameplay_keep/gEffBombExplosion1Tex.png

Why Perceptual Hashing?

    pHashing Algorithm: Creates a 64-bit fingerprint of visual content

    Content Matching: Finds identical textures regardless of filename

    Hamming Distance: Measures similarity (0=identical, 1-2=very similar, 3+=similar)

Setup Instructions
Directory Structure

/wip/          # Root directory (will prompt)
com/           # Community Rice pack (flat, hash-named PNGs)
glide/         # Reloaded Rice pack (flat, hash-named PNGs)
ng/            # Reloaded o2r pack (hierarchical, descriptive PNGs)

Output Directories (Created Automatically)

glideout/      # Matched Reloaded Rice textures
comout/        # Converted community textures in o2r format

Recommended Start

python texture_converter_hamming2.py

More Aggressive Matching (Test Thoroughly)

python texture_converter_hamming3.py

Performance Metrics

    Match Rate: ~53% (2329/4400 textures) at Hamming distance 2

    Processing Time: ~30 seconds for 45,000+ textures

    Accuracy: High quality matches with minimal errors at Hamming ≤2

Use Cases

    Texture Pack Porting: Convert existing Rice-format community packs to SoH

    Hybrid Packs: Combine multiple community packs in o2r format

    Legacy Preservation: Make old texture packs compatible with modern emulators

    Quality Assurance: Verify texture matches between format conversions

Limitations & Considerations

    Partial Coverage: Not all community textures have matches in Reloaded pack

    Hash Collisions: Rare cases where different textures have similar hashes

    Manual Review Needed: For Hamming distance >2 matches

    Texture Updates: Requires re-running when new packs are released

Collaboration Notes

This pipeline enables the OoT modding community to:

    Share texture packs across different emulator/port ecosystems

    Preserve years of community texture work

    Standardize on the modern o2r format while maintaining compatibility

    Automate what would otherwise be thousands of hours of manual work

Credits

    Ocarina Reloaded Team: For providing both Rice and o2r format packs

    Ship of Harkinian Team: For the o2r format specification

    GlideN64 Developers: For the Rice format specification

    Community Texture Artists: For the original texture packs

This pipeline represents a bridge between emulator generations, ensuring that years of community texture work remain accessible as the Ocarina of Time modding ecosystem evolves.
