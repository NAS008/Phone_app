import os as _os
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '.env'))

class Config:
    # App
    URL = "https://tfnca.com"
    ADMIN_SESSION_ID = "1234"  # change this whenever you want; all apps accept it
    APP_NAME = "RTX4070"
    PHONE_NAME = "PHONE"
    INPUT_FOLDER  = r"../../input"
    OUTPUT_FOLDER  = r"../../output"
    WINDOW_W, WINDOW_H = 2044, 2048
    PHONE_W, PHONE_H = 512,512#512, 1024
    IMAGE_SIZE = 512
    GRID_SIZE = 128
    PIXELS_PER_CELL = 2#IMAGE_SIZE // GRID_SIZE
    aspect = WINDOW_W / WINDOW_H
    if aspect >= 1.0:
        GX = GRID_SIZE
        GY = int(GRID_SIZE / aspect)
    else:
        GX = int(GRID_SIZE * aspect)
        GY = GRID_SIZE
    GZ = GRID_SIZE // 8
    G = [GX, GY, GZ]
    camera = [0.5 * GX / max(GX, GY), 0.5 * GY / max(GX, GY), 1.0]
    target = [0.5 * GX / max(GX, GY), 0.5 * GY / max(GX, GY), 0.0]
    light = [0.5 * GX / max(GX, GY), 1.0 * GY / max(GX, GY), 0.5]    
    fov = 1.2
    samples = 1
    background = [0.0, 0.0, 0.0]
    ambient = 0.4
    shadow = 0.4
    FPS = 12

    # UI
    POSE_MODEL = r"..\..\models\pose_landmarker_lite.task"
    
    # Context
    CONTEXT_SIZE = 20  # max conversation turns kept per session

    # Bus
    redis_host     = "tfnca-redis.redis.cache.windows.net"
    redis_port     = 6380
    redis_password = __import__('os').environ.get('REDIS_PASSWORD', '')
    redis_ssl      = True

    # Google
    GEMINI_API_KEY = __import__('os').environ.get('GEMINI_API_KEY', '')
    GEMINI_TEXT_MODEL = "gemini-2.5-flash"
    GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
    GEMINI_STT_MODEL = "gemini-2.5-flash"

    # Stable Diffusion
    SD_MODEL = r"C:\Users\NAS\Models\stable-diffusion-3.5-medium"
    SD_INFERENCE_STEPS = 12
    SD_GUIDANCE_SCALE = 3.5
    SD_SEED = 80367253

    # AnimateDiff
    AD_CONTROLNET_ID   = "guoyww/animatediff-sparsectrl-rgb"
    AD_MOTION_ADAPTER  = "guoyww/animatediff-motion-adapter-v1-5-2"
    AD_SD_BASE         = "Lykon/dreamshaper-8"
    AD_INFERENCE_STEPS = 10
    AD_GUIDANCE_SCALE  = 7.5
    AD_NUM_FRAMES      = 16
    CONTROLNET_SCALE   = 0.5
    AD_SEED            = 80367253

    # Optical flow
    OF_FRAMES = 24

    STYLE = [
        "In style of high-detail vintage botanical lithograph. Pale salmon, dusty rose, and muted sage green, solid black background.",
        "In the style of Art Nouveau metal engraving. Burnished gold, ivory, and deep amber filigree on a solid black background.",
        "In the style of deep-sea scientific specimen illustration. Glowing cyan, electric violet, and phosphorescent mint on a solid black background.",
        "In the style of Edo-period Japanese woodblock print. Vermillion, indigo, and aged cream on a solid black background.",
        "In the style of a medieval illuminated manuscript. Ultramarine blue, burnished gold leaf, and crimson on a solid black background.",
        "In the style of Bauhaus geometric abstraction. Pure red, white, and yellow primary forms on a solid black background.",
        "In the style of electron microscope scanning photography. Monochromatic silver, platinum, and graphite on solid black.",
        "In the style of cyanotype photogram. Prussian blue and white silhouettes on solid black.",
        "In the style of charred woodblock relief print. Burnt sienna, ash white, and ember orange on solid black.",
        "In the style of Aztec codex fused with circuit board diagrams. Jade green, obsidian, and blood red on solid black.",
        "In the style of Soviet Constructivist propaganda poster fused with biological anatomy. Stark red, steel grey, and bone white on solid black.",
        "In the style of alchemical manuscript illustration. Oxidized copper green, tarnished silver, and sulfur yellow on solid black.",
        "In the style of long-exposure light painting photography. Neon magenta, electric blue, and molten gold trails on solid black.",
        "In the style of laser-etched glass engraving. Pure white and ice blue refraction lines on solid black.",
        "In the style of pre-Columbian textile weaving translated to illustration. Terracotta, turquoise, and maize yellow geometric motifs on solid black.",
        "In the style of 17th century Dutch Golden Age mezzotint. Rich velvety blacks, silver-white highlights, and deep sepia mid-tones on solid black.",
        "In the style of Islamic geometric tilework translated to fine ink illustration. Deep lapis lazuli, burnished copper, and ivory white on solid black background.",
        "In the style of modern architectural interpretation of Alhambra palace geometry. Rose gold, matte silver, and warm white on solid black.",
        "In the style of Bauhaus textile weave translated to fine illustration. Strict alternating black and white rectangular block. Metallic gold threads run diagonally like wires of light stitched through woven darkness. Black background.",
        "In the style of Japanese Yayoi Kusama infinity dot obsession. Dense overlapping white polka dot fields with fine 24-karat gold thread outlines tracing the edges of each circle on solid black background.",
        "In the style of haute couture inflated textiles. Ivory silk organza fabric printed with perfect black polka dots, dramatically puffed into sculptural balloon-like volumes against solid black.",
        "Cinematic style, photography, ultra detailed, black and white with some NY cabs orange."
    ]
    MOTION_LORAS = [
        # (lora_name, weight, hint, repo)  — all repos use diffusion_pytorch_model.safetensors
        ("zoom-in",  0.80, "approaching, growing larger", "guoyww/animatediff-motion-lora-zoom-in"),
        ("zoom-out", 0.80, "receding into distance",      "guoyww/animatediff-motion-lora-zoom-out"),
        ("pan-left", 0.75, "drifting left",               "guoyww/animatediff-motion-lora-pan-left"),
        ("pan-right",0.75, "drifting right",              "guoyww/animatediff-motion-lora-pan-right"),
        ("tilt-up",  0.75, "rising upward",               "guoyww/animatediff-motion-lora-tilt-up"),
        ("tilt-down",0.50, "falling downward",            "guoyww/animatediff-motion-lora-tilt-down"),
        ("roll-cw",  0.60, "slowly rotating",             "guoyww/animatediff-motion-lora-rolling-clockwise"),
        ("roll-ccw", 0.60, "slowly rotating",             "guoyww/animatediff-motion-lora-rolling-anticlockwise"),
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