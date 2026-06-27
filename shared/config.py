import os as _os
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '.env'))

class Config:

    # APP CONFIG
    # -----------------------------------------------------------------------------
    APP_NAME = "ARTIST"
    INPUT_FOLDER  = r"../../input"
    OUTPUT_FOLDER  = r"../../output"
    WINDOW_W, WINDOW_H = 1920, 1080#1024, 1024#2048, 2028#2048, 2048#1024, 1024#1152, 2048#512, 512#2160, 2160
    FPS = 10
    VIDEO_SECONDS = 10 # rolling frame buffer depth for USER_VIDEO gif


    # SIM CONFIG
    # -----------------------------------------------------------------------------
    IMAGE_W = (WINDOW_W // 4) // 16 * 16
    IMAGE_H = (WINDOW_H // 4) // 16 * 16
    GRID_SIZE = 128
    PIXELS_PER_CELL = 2#max(IMAGE_W, IMAGE_H) // GRID_SIZE
    aspect = WINDOW_W / WINDOW_H
    if aspect >= 1.0:
        GX = GRID_SIZE
        GY = int(GRID_SIZE / aspect)
    else:
        GX = int(GRID_SIZE * aspect)
        GY = GRID_SIZE
    GZ = GRID_SIZE // 4
    G = [GX, GY, GZ]
    LAYERS = 1
    world_center = [0.5 * GX / max(GX, GY), 0.5 * GY / max(GX, GY), 0.5 * GZ / GRID_SIZE]
    world_radius = [0.4 * GX / max(GX, GY), 0.4 * GY / max(GX, GY), 0.4 * GZ / GRID_SIZE]
    SIM_SPEED = 0.1  # 1.0 = real-time, 0.5 = half-speed (slower particles, better stream detail)
    FPS_SIM = FPS * 2
    MAX_SIM_STEPS_PER_LOOP = 2 * FPS_SIM / FPS


    # RAY CONFIG
    # -----------------------------------------------------------------------------
    camera = [0.5 * GX / max(GX, GY), 0.5 * GY / max(GX, GY), 1.0]
    target = [0.5 * GX / max(GX, GY), 0.5 * GY / max(GX, GY), 0.0]
    light = [0.4 * GX / max(GX, GY), 0.4 * GY / max(GX, GY), 0.6]    
    fov = 1.0
    samples = 1
    background = [0.0, 0.0, 0.0]
    ambient = 0.6
    shadow = 0.3


    # AI CONFIG
    # -----------------------------------------------------------------------------
    # Google
    GEMINI_API_KEY = __import__('os').environ.get('GEMINI_API_KEY', '')
    GEMINI_TEXT_MODEL = "gemini-2.5-flash"
    GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
    GEMINI_STT_MODEL = "gemini-2.5-flash"
    CONTEXT_SIZE = 20  # max conversation turns

    # Stable Diffusion
    SD_MODEL = r"C:\Users\NAS\Models\stable-diffusion-3.5-medium"
    SD_INFERENCE_STEPS = 10
    SD_GUIDANCE_SCALE = 3.0
    SD_SEED = 80367253

    # AnimateDiff
    AD_CONTROLNET_ID   = "guoyww/animatediff-sparsectrl-rgb"
    AD_MOTION_ADAPTER  = "guoyww/animatediff-motion-adapter-v1-5-2"
    AD_SD_BASE         = "Lykon/dreamshaper-8"
    AD_INFERENCE_STEPS = 8
    AD_GUIDANCE_SCALE  = 7.5
    AD_NUM_FRAMES      = 16
    AD_CONTROLNET_SCALE   = 0.5
    AD_SEED            = 80367253
    CLIPS_PER_SCENE  = 2
    RAW_QUEUE_MAXSIZE = 64
    DISPLAY_MAXFRAMES = 500
    GLOBAL_STYLE    = "minimalist"
    GLOBAL_NEGATIVE = "close-up, indoor, blurry, watermark, text"
    MOTION_LORAS = [
        # (lora_name, weight, hint, repo)  — all repos use diffusion_pytorch_model.safetensors
        ("zoom-in",  0.8, "approaching, growing larger", "guoyww/animatediff-motion-lora-zoom-in"),
        ("zoom-out", 0.8, "receding into distance",      "guoyww/animatediff-motion-lora-zoom-out"),
        ("pan-left", 0.8, "drifting left",               "guoyww/animatediff-motion-lora-pan-left"),
        ("pan-right",0.8, "drifting right",              "guoyww/animatediff-motion-lora-pan-right"),
        ("tilt-up",  0.8, "rising upward",               "guoyww/animatediff-motion-lora-tilt-up"),
        ("tilt-down",0.8, "falling downward",            "guoyww/animatediff-motion-lora-tilt-down"),
        ("roll-cw",  0.8, "slowly rotating",             "guoyww/animatediff-motion-lora-rolling-clockwise"),
        ("roll-ccw", 0.8, "slowly rotating",             "guoyww/animatediff-motion-lora-rolling-anticlockwise"),
        (None,       None, "sways gently in the wind",    None),
    ]
    SUBJECTS = [
        "colossal shell",
        #"tiny abandoned house floating",
        #"jellyfish beneath a hot air balloon",
        #"tiny seagulls in the horizon",
        "colossal ivory white sculptural on the beach",
        #"a giant ship stuck in the sand",
        "giant wave",
        "person with pomegranate diving helmet and scarf on the desert",
        #"person with thick black goggles and indigo blue scarf on the desert",
        #"person with thick black goggles and covid mask on orange desert",
        "colossal barnacled shell",
        #"kinetic sculpture on the beach",
        #"crowd of people rushing on the orange subway",
        #"crowd of people with red kart in supermarket",
        #"crowd of people with shopping carts in crowded supermarket",
        #"birds eye of beach",
        #"birds eye of crowded city with red bridges",
        #"colossal red coral sculpture on the beach",
        #"colossal sea anemone in the desert",
        "pufferfish flying in the sky like a balloon",
        #"waterfall in forest",
        #"alien ship in the sea",
        #"rusty toaster on the beach",
    ]

    # Frame interpolation (temporal smoothing between SD keyframes)
    OF_FRAMES = 7
    # "rife" = Practical-RIFE (fastest); "film" = Google FILM (better rotation,
    # scaling, large translation); "amt" = AMT (sharper than RIFE on large motion,
    # lighter than FILM). FILM needs film_net_fp16.pt from
    # dajes/frame-interpolation-pytorch dropped into MODELS_FOLDER.
    VFI_MODEL = "film"
    # AMT (used when VFI_MODEL="amt"): needs the MCG-NKU/AMT repo checked out (for
    # its `networks` package + cfgs/) and the matching checkpoint
    # (amt-s.pth / amt-l.pth / amt-g.pth) in MODELS_FOLDER.
    AMT_REPO  = r"..\..\models\AMT"
    AMT_MODEL = "amt-s"

    # SuperResolution
    MODELS_FOLDER  = r"../../models"

    # FramePack (image-to-video next-frame prediction via HunyuanVideo backbone)
    FP_TRANSFORMER_PATH = r"C:\Users\NAS\Models\FramePack_F1_I2V_HY_20250503"
    FP_BASE_PATH        = r"C:\Users\NAS\Models\HunyuanVideo"
    FP_LATENT_WINDOW    = 9        # bars per batch = ceil(NUM_FRAMES / LATENT_WINDOW); set equal for 1 bar
    FP_NUM_FRAMES       = 9        # latent frames to generate; output pixel frames = 4×N+1 = 37
    FP_SEED             = 42
    FP_PROMPT           = "melting, dripping downward, viscous liquid flowing slowly, wax melting, paint dripping under gravity, forms elongating and falling"
    FP_GUIDANCE_SCALE   = 9.0
    FP_TRUE_CFG_SCALE   = 3.5
    FP_INFERENCE_STEPS  = 6       # 5-8 is enough; resolution is the quality limit at small sizes

    # Wan2.2 FLF2V-14B (first-last-frame-to-video: true interpolation between two stills)
    # TI2V-5B accepts last_image but wasn't trained on it — video drifts from the target.
    # FLF2V-14B was trained on first/last pairs so it reliably lands on the last frame.
    WAN_MODEL            = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
    WAN_SIZE             = 480     # 480px; 720p needs ~3× more VRAM than 480p
    WAN_INFERENCE_STEPS  = 30      # FLF2V needs 30-50 for coherent intermediate frames
    WAN_GUIDANCE_SCALE   = 5.5
    WAN_NUM_FRAMES       = 33      # must be 4k+1 (Wan temporal factor 4); ~2s at 16fps
    WAN_FPS              = 16
    WAN_SEED             = 42
    # int8 quantization (bitsandbytes) shrinks the 14B transformer from ~28 GB
    # bfloat16 to ~16 GB with near-lossless quality, ending the shared-memory
    # spill that throttled full-residency bf16 on 32 GB. None = full bf16.
    WAN_QUANTIZE         = "8bit"  # "8bit" | None
    # Quantized pipelines can't take .to("cuda") (bnb places the transformer
    # itself), so the Wan loader always uses cpu-offload when quantized — which
    # still keeps the transformer GPU-resident across the whole denoising loop.
    WAN_OFFLOAD          = False

    # Wan2.2 TI2V-5B — single-image I2V (no first/last pair). Drives the dog
    # head-turn via the WanI2V class. Unlike FLF2V it has no CLIP image encoder
    # and conditions on the input frame's VAE latent, so one image + a motion
    # prompt is its native mode. At ~5B (~10 GB bf16) it fits full GPU residency
    # on 32 GB, so QUANTIZE=None (bf16) is faster than int8 here.
    WANI2V_MODEL            = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    WANI2V_SIZE             = 704
    WANI2V_INFERENCE_STEPS  = 30      # 30-50; TI2V-5B is coherent from ~30
    WANI2V_GUIDANCE_SCALE   = 5.0     # TI2V-5B's recommended range
    WANI2V_NUM_FRAMES       = 33      # 4k+1; ~1.4s at 24fps
    WANI2V_FPS              = 24      # TI2V-5B is trained at 24fps
    WANI2V_SEED             = 42
    WANI2V_QUANTIZE         = None    # bf16 full residency (5B fits 32 GB easily)
    WANI2V_OFFLOAD          = False

    # LTX-Video (Lightricks) — fast I2V: ~2B params, much lighter than Wan.
    # Animates from the first frame only (no FLF2V). Great for quick iteration.
    LTX_MODEL           = "Lightricks/LTX-Video"
    LTX_WIDTH           = 512
    LTX_HEIGHT          = 512
    LTX_NUM_FRAMES      = 49      # (n-1)%8==0; 25 = ~1s clip at 24fps
    LTX_INFERENCE_STEPS = 25
    LTX_GUIDANCE_SCALE  = 3.0
    LTX_FPS             = 24
    LTX_SEED            = 42
    LTX_OFFLOAD         = False

    STYLE = {
        "None": {
            "name": "None",
            "short": "",
            "long": "faithful photograph, natural colors, accurate detail, realistic lighting, no artistic style or transformation"
        },
        "Botanical Lithograph": {
            "name": "Botanical Lithograph",
            "short": "in style of botanical lithograph, engraved linework, pale salmon, dusty rose, muted sage green",
            "long": "In style of a high-detail vintage botanical lithograph with delicate engraved contour lines, fine crosshatching, and precise scientific plate composition. Pale salmon petals, dusty rose shading, muted sage-green foliage, cream highlights, and a matte black background with crisp paper texture."
        },
        "Art Nouveau Engraving": {
            "name": "Art Nouveau Engraving",
            "short": "in style of Art Nouveau engraving, gold, ivory, amber filigree, ornamental linework",
            "long": "In style of Art Nouveau metal engraving with flowing ornamental linework, elegant whiplash curves, and dense filigree. Burnished gold, ivory, and deep amber accents, subtle metallic sheen, refined embossed texture, and a solid black background."
        },
        "Deep-Sea Specimen": {
            "name": "Deep-Sea Specimen",
            "short": "in style of deep-sea specimen illustration, glowing cyan, electric violet, phosphorescent mint",
            "long": "In style of a deep-sea scientific specimen illustration with translucent anatomy, bioluminescent glow, and precise natural-history detailing. Glowing cyan, electric violet, and phosphorescent mint with luminous edge highlights, dark inky shadows, and a solid black background."
        },
        "Edo Woodblock": {
            "name": "Edo Woodblock",
            "short": "in style of Edo woodblock print, vermillion, indigo, aged cream, flat ink shapes",
            "long": "In style of an Edo-period Japanese woodblock print with flat ink shapes, carved linework, subtle woodgrain texture, and elegant negative space. Vermillion, indigo, and aged cream with restrained gradients and a solid black background."
        },
        "Medieval Manuscript": {
            "name": "Medieval Manuscript",
            "short": "in style of medieval illuminated manuscript, ultramarine, gold leaf, crimson, ornate borders",
            "long": "In style of a medieval illuminated manuscript with intricate border ornament, hand-painted initials, and luminous gilded detailing. Ultramarine blue, burnished gold leaf, and crimson with parchment texture, fine ink outlines, and a solid black background."
        },
        "Bauhaus Geometric": {
            "name": "Bauhaus Geometric",
            "short": "in style of Bauhaus geometric abstraction, red, white, yellow, black grid",
            "long": "In style of Bauhaus geometric abstraction with strict modular forms, hard edges, and disciplined modernist composition. Pure red, white, and yellow primary shapes arranged with black grid structure, flat color fields, and a solid black background."
        },
        "Electron Microscope": {
            "name": "Electron Microscope",
            "short": "in style of electron microscope scanning photography, silver, platinum, graphite, microtexture",
            "long": "In style of electron microscope scanning photography with extreme microtexture, crisp surface relief, and clinical scientific contrast. Monochromatic silver, platinum, and graphite tones with metallic specular detail and deep black shadows."
        },
        "Cyanotype": {
            "name": "Cyanotype",
            "short": "in style of cyanotype photogram, Prussian blue, white silhouettes, botanical shadows",
            "long": "In style of a cyanotype photogram with crisp white silhouettes, delicate botanical shadow forms, and high-contrast sun-print textures. Deep Prussian blue paper, pale white exposures, subtle chemical bloom, and a solid black background."
        },
        "Charred Woodblock": {
            "name": "Charred Woodblock",
            "short": "in style of charred woodblock relief print, burnt sienna, ash white, ember orange, rough grain",
            "long": "In style of a charred woodblock relief print with scorched carved marks, rough grain, and smoky layered textures. Burnt sienna, ash white, and ember orange with soot-dark edges and a solid black background."
        },
        "Aztec Circuit": {
            "name": "Aztec Circuit",
            "short": "in style of Aztec circuit diagram, jade green, obsidian, blood red, coded geometry",
            "long": "In style of an Aztec codex fused with circuit board diagrams, combining symbolic geometry, ritual patterning, and technological linework. Jade green, obsidian, and blood red with etched glyph-like traces and a solid black background."
        },
        "Soviet Constructivist": {
            "name": "Soviet Constructivist",
            "short": "in style of Soviet Constructivist poster, red, steel grey, bone white, bold diagonals",
            "long": "In style of Soviet Constructivist propaganda design fused with anatomical diagram aesthetics, using bold diagonals, sharp angles, and monumental graphic tension. Stark red, steel grey, and bone white with poster-paper texture and a solid black background."
        },
        "Alchemical Manuscript": {
            "name": "Alchemical Manuscript",
            "short": "in style of alchemical manuscript illustration, copper green, silver, sulfur yellow, occult symbols",
            "long": "In style of an alchemical manuscript illustration with esoteric symbols, handwritten annotations, and antique mystical plate design. Oxidized copper green, tarnished silver, and sulfur yellow with vellum texture, delicate ink detail, and a solid black background."
        },
        "Light Painting": {
            "name": "Light Painting",
            "short": "in style of light painting photography, magenta, blue, gold trails, long exposure",
            "long": "In style of long-exposure light painting photography with flowing motion trails, luminous arcs, and dense layered light streaks. Neon magenta, electric blue, and molten gold with glowing edge bloom and a solid black background."
        },
        "Laser-Etched Glass": {
            "name": "Laser-Etched Glass",
            "short": "in style of laser-etched glass engraving, white, ice blue, refracted lines, translucent",
            "long": "In style of laser-etched glass engraving with translucent surfaces, crisp refracted linework, and precise etched highlights. Pure white and ice blue reflections with subtle frosted texture and a solid black background."
        },
        "Pre-Columbian Textile": {
            "name": "Pre-Columbian Textile",
            "short": "in style of Pre-Columbian textile, terracotta, turquoise, maize yellow, woven geometry",
            "long": "In style of Pre-Columbian textile weaving translated into illustration, with woven geometry, tactile fiber structure, and ceremonial pattern rhythm. Terracotta, turquoise, and maize yellow with thread-like texture and a solid black background."
        },
        "Dutch Mezzotint": {
            "name": "Dutch Mezzotint",
            "short": "in style of Dutch mezzotint, velvety blacks, silver highlights, sepia, soft tonal depth",
            "long": "In style of 17th-century Dutch Golden Age mezzotint with velvety blacks, soft tonal transitions, and luminous silver highlights. Rich sepia mid-tones, subtle plate grain, and a solid black background."
        },
        "Islamic Geometric": {
            "name": "Islamic Geometric",
            "short": "in style of Islamic geometric tilework, lapis, copper, ivory, repeating star patterns",
            "long": "In style of Islamic geometric tilework translated to fine ink illustration, with repeating star polygons, interlaced symmetry, and precise ornamental rhythm. Deep lapis lazuli, burnished copper, and ivory white with crisp linework and a solid black background."
        },
        "Alhambra Geometry": {
            "name": "Alhambra Geometry",
            "short": "in style of Alhambra geometry, rose gold, silver, warm white, palace ornament",
            "long": "In style of a modern architectural interpretation of Alhambra palace geometry with arabesque rhythm, carved ornament, and elegant symmetry. Rose gold, matte silver, and warm white with polished stone texture and a solid black background."
        },
        "Bauhaus Textile": {
            "name": "Bauhaus Textile",
            "short": "in style of Bauhaus textile weave, black and white blocks, gold threads, woven geometry",
            "long": "In style of Bauhaus textile weaving translated to fine illustration, combining strict black-and-white block forms with rhythmic woven geometry. Metallic gold threads run diagonally like stitched wires of light through matte textile texture on a solid black background."
        },
        "Kusama Infinity Dots": {
            "name": "Kusama Infinity Dots",
            "short": "in style of Kusama infinity dots, white polka dots, gold outlines, dense repetition",
            "long": "In style of Yayoi Kusama-inspired infinity dot obsession with dense repetition, circular rhythm, and obsessive optical patterning. Overlapping white polka dot fields with fine 24-karat gold thread outlines on a solid black background."
        },
        "Inflated Textiles": {
            "name": "Inflated Textiles",
            "short": "in style of inflated textiles, ivory organza, black polka dots, sculptural volumes",
            "long": "In style of haute couture inflated textiles with sculptural balloon-like volumes, crisp dimensional folds, and exaggerated softness. Ivory silk organza with perfect black polka dots, matte and satin contrasts, and a solid black background."
        },
        "Cinematic B&W": {
            "name": "Cinematic B&W",
            "short": "in style of cinematic black and white photography, high contrast, orange accent, film grain",
            "long": "In style of cinematic black-and-white photography with high contrast lighting, dramatic tonal separation, and subtle film grain. Deep monochrome blacks and whites with a single restrained orange accent, realistic lens response, and moody atmospheric depth."
        },
        "Van Gogh": {
            "name": "Van Gogh",
            "short": "in style of Van Gogh, impasto strokes, swirling paint, vivid yellows, cobalt blues",
            "long": "In style of Van Gogh with thick impasto strokes, swirling directional brushwork, and emotionally charged color movement. Vivid yellows, cobalt blues, and warm earth tones with visible pigment texture and a lively painted surface."
        },
        "Theo Jansen": {
            "name": "Theo Jansen",
            "short": "in style of Theo Jansen, kinetic machine forms, wind-driven structures, pale sand, skeletal motion",
            "long": "In style of Theo Jansen with kinetic machine structures, skeletal beach-walker forms, and wind-driven mechanical motion. Pale sand, weathered beige, oxidized metal, and translucent plastic membranes with engineering precision and wind-swept atmosphere."
        },
        "Mario Giacomelli": {
            "name": "Mario Giacomelli",
            "short": "in style of Mario Giacomelli, stark black and white, graphic contrast, textured grain",
            "long": "In style of Mario Giacomelli with stark black-and-white contrast, expressive aerial-like composition, and intense graphic simplification. Heavy textured grain, deep blacks, bright whites, and rough tonal edges with poetic abstraction."
        },
        "Karl Blossfeldt": {
            "name": "Karl Blossfeldt",
            "short": "in style of Karl Blossfeldt, botanical photography, sculptural plants, monochrome detail",
            "long": "In style of Karl Blossfeldt botanical photography with sculptural plant forms, precise natural detail, and formal specimen-like presentation. Monochrome tonal rendering, crisp texture, and elegant botanical structure on a dark field."
        },
        "Hiroshi Sugimoto": {
            "name": "Hiroshi Sugimoto",
            "short": "in style of Hiroshi Sugimoto, minimalist monochrome, meditative light, calm horizons",
            "long": "In style of Hiroshi Sugimoto with minimalist monochrome composition, meditative stillness, and refined tonal gradients. Subtle silver-grey light, smooth atmospheric transitions, and deep quiet negative space."
        },
        "Ellsworth Kelly": {
            "name": "Ellsworth Kelly",
            "short": "in style of Ellsworth Kelly, vibrant color fields, crisp edges, bold minimal forms",
            "long": "In style of Ellsworth Kelly with vibrant color-field minimalism, crisp edge geometry, and bold flat forms. Clean primary and secondary color blocks, spacious composition, and matte-paint simplicity."
        },
        "Edward Hopper": {
            "name": "Edward Hopper",
            "short": "in style of Edward Hopper, quiet color composition, loneliness, geometric light",
            "long": "In style of Edward Hopper with quiet architectural composition, emotional stillness, and geometric daylight. Muted urban colors, hard-edged shadows, pale interior light, and an atmosphere of solitude."
        },
        "James Turrell": {
            "name": "James Turrell",
            "short": "in style of James Turrell, minimalist light art, glowing gradients, immersive space",
            "long": "In style of James Turrell with immersive minimalist light art, soft-edged luminous fields, and spatial color gradients. Saturated glow, seamless light transitions, and an ethereal atmosphere of pure perception."
        },
        "Giorgio Morandi": {
            "name": "Giorgio Morandi",
            "short": "in style of Giorgio Morandi, muted still life, quiet bottles, dusty neutrals",
            "long": "In style of Giorgio Morandi with muted still-life minimalism, restrained object repetition, and contemplative balance. Dusty neutrals, chalky surfaces, soft shadows, and delicate tonal harmony."
        },
        "Lewis Baltz": {
            "name": "Lewis Baltz",
            "short": "in style of Lewis Baltz, new topographics, industrial color photography, plain architecture",
            "long": "In style of Lewis Baltz with New Topographics sensibility, plain industrial structures, and detached color photography. Faded concrete, muted sky tones, sparse framing, and objective documentary clarity."
        },
        "Georgia O'Keeffe": {
            "name": "Georgia O'Keeffe",
            "short": "in style of Georgia O'Keeffe, minimalist abstract color, enlarged forms, flowing curves",
            "long": "In style of Georgia O'Keeffe with minimalist abstract color, enlarged organic forms, and flowing sensual contours. Soft desert hues, luminous petal-like surfaces, and a balance of intimacy and spaciousness."
        },
        "Patrick Caulfield": {
            "name": "Patrick Caulfield",
            "short": "in style of Patrick Caulfield, graphic minimalism, flat color, sharp outlines",
            "long": "In style of Patrick Caulfield with graphic minimalism, flat color blocks, and sharp simplified outlines. Clean visual hierarchy, bold contouring, and restrained but vivid decorative color."
        },
        "Michael Kenna": {
            "name": "Michael Kenna",
            "short": "in style of Michael Kenna, minimal color photography, quiet atmosphere, long exposure",
            "long": "In style of Michael Kenna with minimal color photography, serene long exposure, and vast quiet negative space. Subtle tonal layering, delicate horizon light, and contemplative atmospheric softness."
        },
        "Fashion": {
            "name": "Fashio",
            "short": "in style of high-fashion editorial photography, sculptural wardrobe, theatrical lighting, couture silhouette",
            "long": "In style of high-fashion editorial photography featuring architectural couture, sculptural silhouettes, and dramatic presentation. Structured slate-grey or charcoal base garments layered with precision-pleated overlays in soft taupe, beige, terracotta, rust-orange, and burnt sienna; matte and glossy fabric contrasts; oversized statement accessories; and museum-quality theatrical lighting with sharp rim light and soft shadow falloff."
        },
        "Barcode": {
            "name": "Barcode",
            "short": "in style of barcode abstraction, black and white stripes, sumi-e faces, fragmented geometry",
            "long": "In style of barcode abstraction with vertical stripe fields, distorted barcode rhythms, and fragmented figurative integration. Pure black and white bands with occasional gray gradients, sumi-e ink wash facial forms, brushy edges, and a solid black background."
        },
        "Renaissance Painting": {
            "name": "Renaissance Painting",
            "short": "in style of Renaissance oil painting, earth tones, chiaroscuro, classical portraiture",
            "long": "In style of Renaissance oil painting with classical portrait composition, rich earth tones, and dramatic chiaroscuro. Thick oil brushwork, luminous skin modeling, deep shadow depth, and a formal noble atmosphere."
        },
        "Japanese Ink Wash": {
            "name": "Japanese Ink Wash",
            "short": "in style of Japanese ink wash painting, misty mountains, cherry blossoms, black and gray tones",
            "long": "In style of traditional Japanese ink wash painting with misty mountain forms, soft atmospheric washes, and expressive brush control. Black and gray ink tones, delicate cherry blossoms, paper absorbency, and a calm negative-space composition."
        },
        "Impressionist Garden": {
            "name": "Impressionist Garden",
            "short": "in style of Impressionist garden painting, Monet palette, visible brushstrokes, dappled light",
            "long": "In style of Impressionist garden painting with short visible brushstrokes, dappled sunlight, and color observed through atmosphere. Vibrant purples, yellows, and greens inspired by Monet, with soft focus, shimmering reflections, and plein-air freshness."
        },
        "World War I": {
            "name": "World War I",
            "short": "in style of World War I poster art, primary color fields, goggles, raincoats, graphic tension",
            "long": "In style of World War I-era poster art fused with bold geometric color fields, energetic brushstrokes, and visual propaganda tension. Contrasting primaries, leather hoods, fur-lined goggles, long raincoats, and a stark graphic atmosphere."
        },
        "Watercolor Skies": {
            "name": "Watercolor Skies",
            "short": "in style of watercolor skies, blue ocean clouds, silhouette linework, airy wash",
            "long": "In style of watercolor skies with translucent washes, blue atmospheric layers, and airy cloud movement. Oceanic blues, diluted grays, silhouette linework, soft bleeding edges, and a luminous paper surface."
        },
        "Rennaissance Tricycle": {
            "name": "Rennaissance Tricycle",
            "short": "in style of Raphael, Renaissance portrait, tiny tricycle, warm reds and golds, regal absurdity",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons posed majestically in three-quarter view riding a tiny children's old tricycle. Create a grand Renaissance oil painting portrait in the style of Raphael. Rich warm Renaissance palette with deep reds, burnished golds, umbers, and soft ivory highlights. Painted clouds, subtle cherubs, classical drapery, and museum-grade fresco atmosphere. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in elaborate Renaissance nobleman attire with velvet doublet, gold embroidery, ruffled collar, cape, jeweled rings, and serious formal expression. Add one foot on the pedal, the other extended for balance. Include a tiny metal tricycle with polished highlights, absurdly small against regal scale."
        },
        "Rennaissance Duck": {
            "name": "Rennaissance Duck",
            "short": "in style of Raphael, Renaissance portrait, giant inflatable duck, warm reds and golds, regal absurdity",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons posed majestically in three-quarter view riding an enormous inflatable yellow rubber duck with black sunglasses. Create a grand Renaissance oil painting portrait in the style of Raphael. Rich warm Renaissance palette with deep reds, burnished golds, warm browns, and ivory highlights, contrasted with glossy duck-yellow plastic. Painted clouds, cherubs, and a classical courtly atmosphere. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in elaborate Renaissance nobleman attire with velvet doublet, gold embroidery, ruffled collar, and cape flowing dramatically. Add serious, dignified expression. Position on the giant duck with realistic weight and glossy reflections."
        },
        "Rennaissance Angels": {
            "name": "Rennaissance Angels",
            "short": "in style of Michelangelo, Baroque ceiling fresco, angels, duck, divine clouds, dramatic foreshortening",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons floating majestically through swirling clouds surrounded by angels and cherubs, but riding an enormous inflatable yellow rubber duck instead of a divine chariot. Create a dramatic Baroque ceiling fresco in the style of Michelangelo's Sistine Chapel. Dynamic diagonal composition with foreshortening, billowing drapery, and golden divine light rays breaking through storm clouds. Trompe-l'oeil architectural framing, celestial depth, and monumental fresco atmosphere. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in flowing classical robes in white and gold toga-style garments billowing in divine wind. Add laurel wreath crown, ecstatic transcendent expression, one arm pointing upward, the other gripping the giant rubber duck. Add cherubs playing with smaller rubber ducks and baroque angels looking bewildered. Ensure the duck is enormous, glossy, and has the signature orange beak."
        },
        "Monet Flamingo": {
            "name": "Monet Flamingo",
            "short": "in style of Monet, Impressionist garden, pink flamingo floatie, water lilies, pastel brushwork",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons floating serenely on a giant inflatable flamingo pool floatie in the middle of a water-lily pond, painted with loose Impressionist brushstrokes and dappled light. Create a Monet-style garden scene with soft focus, pastel pinks, powder blues, fresh greens, and shimmering reflections. Weeping willow reflections, Japanese bridge, plein-air atmosphere, and painterly surface texture. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in white Edwardian summer clothing with parasol and sun hat. Add relaxed reclining pose on the bright pink inflatable flamingo. Surround with water lilies, mirrorlike water, and an oversized cartoonishly pink floatie integrated into the painting."
        },
        "Dali": {
            "name": "Dali",
            "short": "in style of Dalí, surreal desert, melting clocks, impossible shadows, hyper-real detail",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons standing in a barren desert with impossibly long shadows, wearing enormous melting clocks draped over head and shoulders like a hat. Create a Salvador Dalí-style surrealist desert landscape with distorted perspective, dreamlike clarity, and hyper-real detail in impossible scenarios. Warm sand, pale sky, soft shadow gradients, ants crawling across clocks, a dead tree branch, strange elephant legs in the distance, and a fragile horizon. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in a formal suit that appears to melt at the edges like the clocks. Add multiple pocket watches draped over head, shoulders, and arms. Add a serious contemplative expression and a shadow stretched impossibly long across desert sand."
        },
        "Rococo Unicorn": {
            "name": "Rococo Unicorn",
            "short": "in style of Fragonard, Rococo garden party, inflatable unicorn, pastel romance, ornate decoration",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons on an ornate swing, but the swing is actually a giant inflatable unicorn pool toy suspended from flowering trees. Create an elaborate Rococo garden party scene in the style of Fragonard with decorative excess, playful elegance, and soft romantic light. Pastel pinks, mint greens, pale golds, creamy whites, and floral accents; silk, lace, ribbons, and powdered textures; lush garden atmosphere with rose petals falling and aristocrats in the background. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in elaborate Rococo attire with silk gowns, panniers, embroidered coats, lace cuffs, and powdered wigs with ribbons. Add one dainty shoe kicked off mid-swing, a delighted expression, the oversized rainbow inflatable unicorn suspended by silk ribbons, and cherubs holding the ribbons."
        },
        "New Year": {
            "name": "New Year",
            "short": "in style of Jackson Pollock, abstract expressionism, champagne, gold splatter, kinetic chaos",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons standing in the center of the canvas holding a crystal champagne flute, surrounded by explosive splatter patterns of gold leaf fragments, champagne spray, and liquid-gold paint flung across the composition. Create a Jackson Pollock-style Abstract Expressionist action painting with energetic gestural marks, dense drips, and chaotic overlapping metallics. Gold, bronze, copper, deep black, and rich burgundy with a large-scale canvas feeling and raw celebratory luxury. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in elegant black formal attire with gold paint splatters. Add a crystal champagne flute held at chest level, champagne mid-spray, a tilted bottle pouring into the chaos, floating gold leaf, liquid drips, scattered pearls, and a dynamic celebratory stance."
        },
        "Rembrandt selfie stick": {
            "name": "Rembrandt selfie stick",
            "short": "in style of Rembrandt, chiaroscuro, selfie stick, Dutch interior, warm candlelight",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons in the center of a wealthy merchant group painting, all formally posed in a dark interior with dramatic window light, but one person is holding a modern selfie stick extended toward the viewer. Create a Rembrandt-style Dutch Golden Age group portrait with chiaroscuro lighting, deep shadows, warm golden highlights, and a quiet aristocratic interior. Add a map on the wall, dark wood, subtle fabric sheen, and candlelit atmosphere. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in 17th-century Dutch merchant clothing with black doublet, white lace collar, and feathered black hat. Add a modern smartphone on an extended selfie stick with a glowing screen, and other period-dressed figures behind looking confused at the device."
        }
    }


    # BUS CONFIG
    # ----------------------------------------------------------------------------- 
    redis_host     = "tfnca-redis.redis.cache.windows.net"
    redis_port     = 6380
    redis_password = __import__('os').environ.get('REDIS_PASSWORD', '')
    redis_ssl      = True


    # DIRECTOR CONFIG
    # -----------------------------------------------------------------------------
    DIRECTOR_PROMPT_INTERVAL = 20    # seconds between AI-generated prompts


    # PAINTER CONFIG
    # -----------------------------------------------------------------------------     
    MAX_DIM = 512
    MAX_STROKES = 1000
    IMPASTO_Z = 0.4
    
    
    # SESSION CONFIG
    # -----------------------------------------------------------------------------
    URL = "https://tfnca.com"
    MAX_USERS = 5
    ADMIN_NICKNAME = "NAS"
    ADMIN_SESSION_ID = "1234"  # change this whenever you want; all apps accept it


    # STREAMING CONFIG
    # -----------------------------------------------------------------------------    
    stream_on = True
    STREAM_MAX_SIDE = 0           # 0 is for the window resolution
    STREAM_BITRATE = 80_000_000   # VP9 target bitrate ceiling — WebRTC CC reduces this on limited links
    HOST_IP = "192.168.68.60"
    # TURN relay — required for viewers on mobile data (CGNAT blocks STUN-only).
    # Preferred: Cloudflare's free TURN service (dynamic credentials). Create a
    # key under Cloudflare dashboard → Realtime → TURN Server, then set both
    # env vars on the PC and on Railway (backend).
    CF_TURN_KEY_ID    = _os.environ.get('CF_TURN_KEY_ID', '')
    CF_TURN_API_TOKEN = _os.environ.get('CF_TURN_API_TOKEN', '')
    CF_TURN_TTL       = 86400  # credential lifetime (s); auto-refreshed at 80%
    # Fallback: static self-hosted relay, e.g. TURN_URL="turn:<vm-ip>:3478".
    TURN_URL      = _os.environ.get('TURN_URL', '')
    TURN_USERNAME = _os.environ.get('TURN_USERNAME', '')
    TURN_PASSWORD = _os.environ.get('TURN_PASSWORD', '')
    VIEWER_HTML = r"../tests/viewer.html"
    GIF_DIFF_THRESHOLD = 3.0  # mean abs pixel diff (0-255) required to add a frame
    # Phone backend (for direct HTTP uploads that bypass the bus)
    PHONE_BACKEND_URL = __import__('os').environ.get('PHONE_BACKEND_URL', 'https://phoneapp-production-48e4.up.railway.app')


    # LIVE CONFIG
    # -----------------------------------------------------------------------------
    # Gemini Live API (voice conversation via PC mic + speakers)
    LIVE_VERTEX_LOCATION = _os.environ.get("VERTEX_LOCATION", "us-central1")
    LIVE_MODEL           = "gemini-live-2.5-flash-native-audio"

    LIVE_CHANNELS     = 1
    LIVE_SEND_RATE    = 16000   # Microfone → Gemini: 16 kHz
    LIVE_RECEIVE_RATE = 24000   # Gemini → colunas: 24 kHz
    LIVE_CHUNK        = 1024
    LIVE_MIC_DEVICE_INDEX     = None
    LIVE_SPEAKER_DEVICE_INDEX = None
    LIVE_PTT_KEY = "space"      # PUSH-TO-TALK (carrega para falar)

    LIVE_BANDPASS_LOW_HZ  = 200   # corta rumor/zumbido grave (DC, 50/60 Hz, passos, vento)
    LIVE_BANDPASS_HIGH_HZ = 3800  # corta hiss/sibilância aguda acima da fala (banda telefónica)
    LIVE_BANDPASS_ORDER   = 4


    # UI CONFIG
    # -----------------------------------------------------------------------------
    UI_POSE_MODEL = r"..\..\models\pose_landmarker_lite.task"
    UI_CHANNELS = ["mouse", "cam", "mic"]
    UI_CHANNEL = UI_CHANNELS[2]

