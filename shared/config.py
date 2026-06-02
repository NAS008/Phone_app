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
    WINDOW_W, WINDOW_H = 1080, 1080#3840, 2560
    IMAGE_SIZE = 512
    GRID_SIZE = 128
    PIXELS_PER_CELL = IMAGE_SIZE // GRID_SIZE
    aspect = WINDOW_W / WINDOW_H
    if aspect >= 1.0:
        GX = GRID_SIZE
        GY = int(GRID_SIZE / aspect)
    else:
        GX = int(GRID_SIZE * aspect)
        GY = GRID_SIZE
    GZ = GRID_SIZE // 4
    G = [GX, GY, GZ]
    camera = [0.5 * GX / max(GX, GY), 0.5 * GY / max(GX, GY), 1.0]
    target = [0.5 * GX / max(GX, GY), 0.5 * GY / max(GX, GY), 0.0]
    light = [0.5 * GX / max(GX, GY), 1.0 * GY / max(GX, GY), 0.5]    
    fov = 1.2
    samples = 1
    background = [0.0, 0.0, 0.0]
    ambient = 0.6
    shadow = 0.0
    FPS = 12
    VIDEO_SECONDS = 10 # rolling frame buffer depth for USER_VIDEO gif

    # UI
    POSE_MODEL = r"..\..\models\pose_landmarker_lite.task"
    
    # Context
    CONTEXT_SIZE = 20  # max conversation turns kept per session

    # Phone backend (for direct HTTP uploads that bypass the bus)
    PHONE_BACKEND_URL = __import__('os').environ.get('PHONE_BACKEND_URL', 'https://phoneapp-production-48e4.up.railway.app')

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

    STYLE = {
        "Botanical Lithograph": {
            "name": "Botanical Lithograph",
            "short": "botanical lithograph, pale salmon, dusty rose, sage green, black background",
            "long": "In style of high-detail vintage botanical lithograph. Pale salmon, dusty rose, and muted sage green, solid black background.",
        },
        "Art Nouveau Engraving": {
            "name": "Art Nouveau Engraving",
            "short": "Art Nouveau engraving, gold, ivory, amber filigree, black background",
            "long": "In the style of Art Nouveau metal engraving. Burnished gold, ivory, and deep amber filigree on a solid black background.",
        },
        "Deep-Sea Specimen": {
            "name": "Deep-Sea Specimen",
            "short": "deep-sea specimen, cyan, violet, mint, black background",
            "long": "In the style of deep-sea scientific specimen illustration. Glowing cyan, electric violet, and phosphorescent mint on a solid black background.",
        },
        "Edo Woodblock": {
            "name": "Edo Woodblock",
            "short": "Edo woodblock, vermillion, indigo, cream, black background",
            "long": "In the style of Edo-period Japanese woodblock print. Vermillion, indigo, and aged cream on a solid black background.",
        },
        "Medieval Manuscript": {
            "name": "Medieval Manuscript",
            "short": "medieval manuscript, ultramarine, gold leaf, crimson, black background",
            "long": "In the style of a medieval illuminated manuscript. Ultramarine blue, burnished gold leaf, and crimson on a solid black background.",
        },
        "Bauhaus Geometric": {
            "name": "Bauhaus Geometric",
            "short": "Bauhaus geometric abstraction, red, white, yellow, black background",
            "long": "In the style of Bauhaus geometric abstraction. Pure red, white, and yellow primary forms on a solid black background.",
        },
        "Electron Microscope": {
            "name": "Electron Microscope",
            "short": "electron microscope, silver, platinum, graphite, black background",
            "long": "In the style of electron microscope scanning photography. Monochromatic silver, platinum, and graphite on solid black.",
        },
        "Cyanotype": {
            "name": "Cyanotype",
            "short": "cyanotype, Prussian blue, white silhouettes, black background",
            "long": "In the style of cyanotype photogram. Prussian blue and white silhouettes on solid black.",
        },
        "Charred Woodblock": {
            "name": "Charred Woodblock",
            "short": "charred woodblock, burnt sienna, ash white, ember orange, black background",
            "long": "In the style of charred woodblock relief print. Burnt sienna, ash white, and ember orange on solid black.",
        },
        "Aztec Circuit": {
            "name": "Aztec Circuit",
            "short": "Aztec circuit, jade green, obsidian, blood red, black background",
            "long": "In the style of Aztec codex fused with circuit board diagrams. Jade green, obsidian, and blood red on solid black.",
        },
        "Soviet Constructivist": {
            "name": "Soviet Constructivist",
            "short": "Soviet Constructivist, red, steel grey, bone white, black background",
            "long": "In the style of Soviet Constructivist propaganda poster fused with biological anatomy. Stark red, steel grey, and bone white on solid black.",
        },
        "Alchemical Manuscript": {
            "name": "Alchemical Manuscript",
            "short": "alchemical manuscript, copper green, silver, sulfur yellow, black background",
            "long": "In the style of alchemical manuscript illustration. Oxidized copper green, tarnished silver, and sulfur yellow on solid black.",
        },
        "Light Painting": {
            "name": "Light Painting",
            "short": "light painting, magenta, blue, gold trails, black background",
            "long": "In the style of long-exposure light painting photography. Neon magenta, electric blue, and molten gold trails on solid black.",
        },
        "Laser-Etched Glass": {
            "name": "Laser-Etched Glass",
            "short": "laser-etched glass, white, ice blue, black background",
            "long": "In the style of laser-etched glass engraving. Pure white and ice blue refraction lines on solid black.",
        },
        "Pre-Columbian Textile": {
            "name": "Pre-Columbian Textile",
            "short": "Pre-Columbian textile, terracotta, turquoise, maize yellow, black background",
            "long": "In the style of pre-Columbian textile weaving translated to illustration. Terracotta, turquoise, and maize yellow geometric motifs on solid black.",
        },
        "Dutch Mezzotint": {
            "name": "Dutch Mezzotint",
            "short": "Dutch mezzotint, velvety blacks, silver highlights, sepia, black background",
            "long": "In the style of 17th century Dutch Golden Age mezzotint. Rich velvety blacks, silver-white highlights, and deep sepia mid-tones on solid black.",
        },
        "Islamic Geometric": {
            "name": "Islamic Geometric",
            "short": "Islamic geometric, lapis, copper, ivory, black background",
            "long": "In the style of Islamic geometric tilework translated to fine ink illustration. Deep lapis lazuli, burnished copper, and ivory white on solid black background.",
        },
        "Alhambra Geometry": {
            "name": "Alhambra Geometry",
            "short": "Alhambra geometry, rose gold, silver, warm white, black background",
            "long": "In the style of modern architectural interpretation of Alhambra palace geometry. Rose gold, matte silver, and warm white on solid black.",
        },
        "Bauhaus Textile": {
            "name": "Bauhaus Textile",
            "short": "Bauhaus textile, black and white blocks, gold threads, black background",
            "long": "In the style of Bauhaus textile weave translated to fine illustration. Strict alternating black and white rectangular block. Metallic gold threads run diagonally like wires of light stitched through woven darkness. Black background.",
        },
        "Kusama Infinity Dots": {
            "name": "Kusama Infinity Dots",
            "short": "Kusama dots, white polka dots, gold outlines, black background",
            "long": "In the style of Japanese Yayoi Kusama infinity dot obsession. Dense overlapping white polka dot fields with fine 24-karat gold thread outlines tracing the edges of each circle on solid black background.",
        },
        "Inflated Textiles": {
            "name": "Inflated Textiles",
            "short": "inflated textiles, ivory organza, black polka dots, sculptural volumes, black background",
            "long": "In the style of haute couture inflated textiles. Ivory silk organza fabric printed with perfect black polka dots, dramatically puffed into sculptural balloon-like volumes against solid black.",
        },
        "Cinematic B&W": {
            "name": "Cinematic B&W",
            "short": "cinematic black and white, orange accents",
            "long": "Cinematic style, photography, ultra detailed, black and white with some NY cabs orange.",
        },
        "Van Gogh": {
            "name": "Van Gogh",
            "short": "Van Gogh, impasto strokes",
            "long": "by Van Gogh, impasto strokes",
        },
        "Theo Jansen": {
            "name": "Theo Jansen",
            "short": "Theo Jansen, kinetic machines",
            "long": "by Theo Jansen, kinetic machines",
        },
        "Mario Giacomelli": {
            "name": "Mario Giacomelli",
            "short": "Mario Giacomelli, black and white",
            "long": "by Mario Giacomelli, black and white",
        },
        "Karl Blossfeldt": {
            "name": "Karl Blossfeldt",
            "short": "Karl Blossfeldt, botanical photography",
            "long": "by Karl Blossfeldt, botanical photography",
        },
        "Hiroshi Sugimoto": {
            "name": "Hiroshi Sugimoto",
            "short": "Hiroshi Sugimoto, minimalist polarized color",
            "long": "by Hiroshi Sugimoto, minimalist polarized color",
        },
        "Ellsworth Kelly": {
            "name": "Ellsworth Kelly",
            "short": "Ellsworth Kelly, vibrant color fields",
            "long": "by Ellsworth Kelly, vibrant color field minimalism",
        },
        "Edward Hopper": {
            "name": "Edward Hopper",
            "short": "Edward Hopper, minimal color composition",
            "long": "by Edward Hopper, minimal color composition",
        },
        "James Turrell": {
            "name": "James Turrell",
            "short": "James Turrell, neon light art",
            "long": "by James Turrell, minimalist neon light art",
        },
        "Giorgio Morandi": {
            "name": "Giorgio Morandi",
            "short": "Giorgio Morandi, muted still life",
            "long": "by Giorgio Morandi, muted color still life minimalism",
        },
        "Lewis Baltz": {
            "name": "Lewis Baltz",
            "short": "Lewis Baltz, new topographics color",
            "long": "by Lewis Baltz, new topographics color photography",
        },
        "Georgia O'Keeffe": {
            "name": "Georgia O'Keeffe",
            "short": "Georgia O'Keeffe, minimalist abstract color",
            "long": "by Georgia O'Keeffe, minimalist abstract color",
        },
        "Patrick Caulfield": {
            "name": "Patrick Caulfield",
            "short": "Patrick Caulfield, graphic minimalism",
            "long": "by Patrick Caulfield, graphic minimalism color",
        },
        "Michael Kenna": {
            "name": "Michael Kenna",
            "short": "Michael Kenna, minimal color photography",
            "long": "by Michael Kenna, minimal color photography",
        },
        "Fashion": {
            "name": "Fashio",
            "short": "high-fashion editorial, sculptural wardrobe, theatrical lighting",
            "long": "Transform subject into high-fashion editorial photography featuring architectural, sculptural wardrobe with theatrical presentation. Subject wears a structured slate-grey or charcoal base garment (tailored jacket) layered with precision-pleated sculptural overlays in soft taupe or beige creating dramatic fan-like or accordion-folded dimensional forms at shoulders and neckline, resembling fine paper or feather structures. Terracotta, rust-orange, or burnt sienna, neckline details as theatrical makeup elements on cheekbones and chest. Add minimal mustard-gold or deep jewel-tone accents sparingly in geometric patterns or accessory highlights. Wardrobe materials appear to be stiffened silk organza, structured neoprene, or precision-pleated cotton-synthetic blends with matte base layers contrasting against highly textured sculptural overlays. Include oversized statement accessories — dramatic sunglasses, ornate metallic jewelry—functioning as significant design objects. Position subject in museum-quality theatrical lighting.",
        },
        "Barcode": {
            "name": "Barcode",
            "short": "barcode stripes, sumi-e faces, black and white",
            "long": "Replace background with vertical stripes of varying widths resembling distorted barcodes, alternating pure black and white with occasional gray gradients. Transform faces into vertical bars painted in traditional sumi-e ink wash painting, black ink gradients, minimal color accents, fluid brush strokes. Integrate faces in the barcode background while stripes fragment and scatter around edges for artistic integration. Avoid horizontal lines.",
        },
        "Renaissance Painting": {
            "name": "Renaissance Painting",
            "short": "Renaissance oil painting, earth tones, chiaroscuro",
            "long": "Replace background with thick oil painting strokes in warm earth tones and dramatic chiaroscuro lighting, Renaissance style. Preserve faces but integrate them with painterly atmospheric effects and dress people with Renaissance clothes.",
        },
        "Japanese Ink Wash": {
            "name": "Japanese Ink Wash",
            "short": "Japanese ink wash, misty mountains, cherry blossoms",
            "long": "Replace background with traditional Japanese ink wash painting featuring misty mountains and cherry blossoms in black and gray tones. Preserve faces but integrate them with subtle ink splatter effects around edges and dress people with traditional Japanese clothes.",
        },
        "Impressionist Garden": {
            "name": "Impressionist Garden",
            "short": "Impressionist garden, Monet palette, visible brushstrokes",
            "long": "Replace background with impressionist-style garden scene using short, visible brushstrokes in vibrant purples, yellows, and greens inspired by Monet's color palette. Preserve faces but with visible brushstrokes in vibrant purples, yellows, and greens inspired by Monet's color palette and dress man and women with appropriate elegant fashions of late 19th-century Paris.",
        },
        "World War I": {
            "name": "World War I",
            "short": "World War I, primary color fields, goggles, rain coats",
            "long": "Replace background with bold geometric shapes and sweeping color fields in contrasting primary colors with energetic brushstrokes. Maintain faces but add World War I leather hoods with fur-lined goggles and dress people with primary colors long rain coats.",
        },
        "Watercolor Skies": {
            "name": "Watercolor Skies",
            "short": "watercolor skies, blue ocean clouds, silhouette linework",
            "long": "Replace background with water colors in tones of blue representing the ocean and clouds. Preserve faces while adding hand-drawn linework around silhouettes.",
        },
        "Rennaissance Tricycle": {
            "name": "Rennaissance Tricycle",
            "short": "Renaissance portrait, tiny tricycle, warm reds and golds",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons posed majestically in three-quarter view riding a tiny children's old tricycle. Create a grand Renaissance oil painting portrait in the style of Raphael. Rich, warm Renaissance color palette with deep reds, golds, and earth tones. Painted clouds and cherubs in background. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in elaborate Renaissance nobleman attire (velvet doublet with gold embroidery, ruffled collar, cape flowing dramatically behind). Add jeweled rings on hands gripping tiny tricycle handles. Add serious, dignified expression as if sitting for formal royal portrait. Add one foot on tricycle pedal, other extended for balance. Add small bell on tricycle handlebars catching light. Position on bright metalic tiny tricycle creating absurd contrast with regal attire.",
        },
        "Rennaissance Duck": {
            "name": "Rennaissance Duck",
            "short": "Renaissance portrait, giant yellow duck, warm reds and golds",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons posed majestically in three-quarter view riding an enormous inflatable yellow rubber duck with black sunglasses. Create a grand Renaissance oil painting portrait in the style of Raphael. Rich, warm Renaissance color palette with deep reds, golds, and earth tones constrasting with the wasted rubber yellow from the duck. Painted clouds and cherubs in background. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in elaborate Renaissance nobleman attire (velvet doublet with gold embroidery, ruffled collar, cape flowing dramatically behind). Add serious, dignified expression as if sitting for formal royal portrait. Add one foot on the floor, other extended for balance.",
        },
        "Rennaissance Angels": {
            "name": "Rennaissance Angels",
            "short": "Baroque ceiling fresco, angels, duck, divine clouds",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons floating majestically through swirling clouds surrounded by angels and cherubs, but riding an enormous inflatable yellow rubber duck instead of divine chariot. Create a dramatic Baroque ceiling fresco in the style of Michelangelo's Sistine Chapel. Dynamic diagonal composition with foreshortening. Billowing dramatic drapery. Golden divine light rays breaking through storm clouds. Trompe-l'oeil architectural elements framing the scene. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in flowing classical robes (white and gold toga-style garments) billowing in divine wind. Add laurel wreath crown. Add one arm pointing dramatically upward, other gripping giant rubber duck. Add ecstatic, transcendent expression. Add cherubs playing with smaller rubber ducks around scene. Add baroque angels looking bewildered. Ensure rubber duck is enormous, glossy, and has the signature orange beak.",
        },
        "Monet Flamingo": {
            "name": "Monet Flamingo",
            "short": "Monet garden, pink flamingo floatie, water lilies",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons floating serenely on a giant inflatable flamingo pool floatie in the middle of a water lily pond, painted with loose brushstrokes and dappled light. Create a Monet-style Impressionist garden scene. Soft focus with emphasis on light and color over detail. Pastel palette of pinks, blues, and greens. Weeping willow reflections in water. Japanese bridge visible in background. Plein air painting aesthetic. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in white Edwardian summer clothing (flowing dress or linen suit) with parasol. Add sun hat with ribbons. Add relaxed reclining pose on bright pink inflatable flamingo. Add serene, contemplative expression. Add water lilies surrounding the floatie. Add impressionist brushstroke effect to entire scene. Ensure flamingo pool toy is oversized and cartoonishly pink against painterly environment.",
        },
        "Dali": {
            "name": "Dali",
            "short": "Dalí surreal desert, melting clocks, impossible shadows",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons standing in barren desert with impossibly long shadows, wearing enormous melting clocks draped over their head and shoulders like a hat. Create a Salvador Dalí-style Surrealist desert landscape. Distorted perspective with dreamlike quality. Ants crawling across some clocks. Dead tree branch with more melting clocks. Strange elephant legs in distant background. Hyper-realistic detail in impossible scenarios. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in formal suit that appears to be melting at the edges like the clocks. Add multiple pocket watches and clocks draped over head, shoulders, and arms, all soft and melting. Add serious, contemplative expression oblivious to absurdity. Add shadow stretched impossibly long across desert sand. Add small ants marching across melting clock faces. Add desert landscape with impossible horizon.",
        },
        "Rococo Unicorn": {
            "name": "Rococo Unicorn",
            "short": "Rococo garden party, inflatable unicorn, pastel romance",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons on an ornate swing but the swing is actually a giant inflatable unicorn pool toy suspended from flowering trees. Create an elaborate Rococo garden party scene in the style of Fragonard. Pastel paradise with pinks, mint greens, and soft golds. Other aristocrats in background having pool party with various inflatable creatures. Frivolous, decorative, overly romantic composition. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in elaborate Rococo attire (silk gown with panniers or embroidered coat with lace, powdered wig with ribbons). Add one dainty shoe kicked off mid-swing. Add delighted, playful expression. Add seated on enormous rainbow inflatable unicorn suspended by silk ribbons. Add cherubs holding the ribbons. Add other aristocrats in powdered wigs lounging on inflatable flamingos and dolphins. Add rose petals falling.",
        },
        "New Year": {
            "name": "New Year",
            "short": "Pollock abstract expressionism, champagne, gold splatter",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons standing in center of canvas holding a crystal champagne flute, surrounded by explosive splatter patterns of gold leaf fragments, champagne spray, and liquid gold paint flung across the composition. Create a Jackson Pollock-style Abstract Expressionist action painting. Energetic gestural marks and drips. Chaotic overlapping metallics - gold, bronze, copper - mixed with deep blacks and rich burgundy. Large-scale canvas feeling. Raw, spontaneous luxury aesthetic. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in elegant black formal attire (tuxedo or evening gown) with gold paint splatters creating striking contrast. Add crystal champagne flute held at chest level with champagne mid-spray creating dramatic arc. Add bottle of Dom Pérignon in other hand tilted, pouring champagne into the chaos. Add confident, celebratory artistic expression. Add gold leaf sheets floating and fragmenting throughout composition like confetti. Add liquid gold and champagne drips running down face and clothing. Add scattered pearl necklace breaking apart with pearls embedded in paint splatters. Add dynamic powerful stance with champagne pooling at feet.",
        },
        "Rembrandt selfie stick": {
            "name": "Rembrandt selfie stick",
            "short": "Rembrandt portrait, chiaroscuro, selfie stick, Dutch interior",
            "long": "SCENE: Transform *MAIN SUBJECTS* in [Image] into persons in center of wealthy merchant group painting, all formally posed in dark interior with dramatic window light, but person is holding a modern selfie stick extended toward viewer. Create a Rembrandt-style Dutch Golden Age group portrait. Chiaroscuro lighting with dark background and illuminated faces. Rich blacks and warm golden highlights. Classical Dutch interior with map on wall. *MAIN SUBJECTS* FACES: Preserve original facial proportions, bone structure, and identity. You may freely modify bodies, poses, clothing, hairstyles, background, and all other elements to fit the scene perfectly. Maintain realistic lighting and shadows that match the scene. Ensure enhancements look natural and photorealistic. ADD ELEMENTS: Add elements from {transcript}. Dress in 17th century Dutch merchant clothing (black doublet with white lace collar, black hat with feather). Add elaborate ruff collar. Add serious, dignified expression typical of Dutch portraits. Add modern smartphone on extended selfie stick held prominently with screen glowing. Add other period-dressed figures arranged formally behind looking confused at device. Add dramatic Rembrandt lighting from window highlighting face and selfie stick.",
        },
    }
    # Director (auto-play)
    DIRECTOR_PROMPT_INTERVAL = 30    # seconds between AI-generated prompts
    DIRECTOR_SCENE_INTERVAL  = 120   # seconds between shape-change / pointer sweep
    DIRECTOR_SPLINE_POINTS   = 6     # random control points for the pointer spline
    DIRECTOR_SPLINE_SUBSTEPS = 8     # Catmull-Rom interpolation steps per segment

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
